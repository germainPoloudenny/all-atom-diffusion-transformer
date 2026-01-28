"""LightningDataModule for XtoCIF HDF5 datasets."""

import os
import sys
from pathlib import Path
from typing import Optional, Sequence

from lightning import LightningDataModule
from omegaconf import DictConfig
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence

_XTOCIF_ROOT = Path(__file__).resolve().parents[3]
if str(_XTOCIF_ROOT) not in sys.path:
    sys.path.append(str(_XTOCIF_ROOT))

from decifer.decifer_dataset import DeciferDataset
from decifer.tokenizer import Tokenizer
from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


class XtoCifDataModule(LightningDataModule):
    """Load tokenized CIF + optional PXRD data from HDF5 splits."""

    def __init__(
        self,
        dataset_root: str,
        batch_size: DictConfig,
        num_workers: DictConfig,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        prefetch_factor: Optional[int] = None,
        condition: bool = False,
        train_split: str = "train",
        val_split: str = "val",
        test_split: str = "test",
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self._tokenizer = Tokenizer()
        self._pad_id = self._tokenizer.padding_id

    def _collate_fn(self, batch):
        batch_data = {}
        for key in batch[0].keys():
            field_data = [item[key] for item in batch]
            if "xrd" in key:
                padded_seqs = pad_sequence(field_data, batch_first=True, padding_value=0.0)
                batch_data[key] = padded_seqs
            elif "cif" in key:
                padded_seqs = pad_sequence(
                    field_data, batch_first=True, padding_value=self._pad_id
                )
                batch_data[key] = padded_seqs
            else:
                batch_data[key] = field_data
        return batch_data

    def _build_dataset(self, split: str, desc: str) -> DeciferDataset:
        h5_path = os.path.join(self.hparams.dataset_root, "serialized", f"{split}.h5")
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"Missing HDF5 split: {h5_path}")
        keys = ["cif_tokens"]
        if self.hparams.condition:
            keys.extend(["xrd.q", "xrd.iq"])
        return DeciferDataset(h5_path, keys, progress_desc=desc, show_progress=False)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage is None or stage in ["fit", "validate"]:
            self.train_dataset = self._build_dataset(self.hparams.train_split, "train dataset")
            self.val_dataset = self._build_dataset(self.hparams.val_split, "validation dataset")
            log.info(
                f"Training dataset: {len(self.train_dataset)} samples, "
                f"validation dataset: {len(self.val_dataset)} samples"
            )
        if stage is None or stage in ["test", "predict"]:
            self.test_dataset = self._build_dataset(self.hparams.test_split, "test dataset")
            log.info(f"Test dataset: {len(self.test_dataset)} samples")

    def train_dataloader(self) -> DataLoader:
        return self._build_loader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size.train,
            num_workers=self.hparams.num_workers.train,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self) -> Sequence[DataLoader]:
        return [
            self._build_loader(
                dataset=self.val_dataset,
                batch_size=self.hparams.batch_size.val,
                num_workers=self.hparams.num_workers.val,
                shuffle=False,
                drop_last=False,
            )
        ]

    def test_dataloader(self) -> Sequence[DataLoader]:
        return [
            self._build_loader(
                dataset=self.test_dataset,
                batch_size=self.hparams.batch_size.test,
                num_workers=self.hparams.num_workers.test,
                shuffle=False,
                drop_last=False,
            )
        ]

    def _build_loader(
        self,
        dataset: DeciferDataset,
        batch_size: int,
        num_workers: int,
        shuffle: bool,
        drop_last: bool,
    ) -> DataLoader:
        sampler = self._get_distributed_sampler(dataset, shuffle=shuffle, drop_last=drop_last)
        if sampler is not None:
            shuffle = False
            log.info(
                f"DataLoader uses DistributedSampler (world_size={self._ddp_world_size}, "
                f"rank={self._ddp_rank}, samples={len(sampler)}, batch_size={batch_size})."
            )
        else:
            log.info(
                f"DataLoader uses default sampler (world_size={self._ddp_world_size}, "
                f"rank={self._ddp_rank}, batch_size={batch_size})."
            )
        loader_kwargs = {
            "dataset": dataset,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": bool(self.hparams.pin_memory),
            "shuffle": shuffle,
            "drop_last": drop_last,
            "collate_fn": self._collate_fn,
        }
        if sampler is not None:
            loader_kwargs["sampler"] = sampler
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = bool(self.hparams.persistent_workers)
            if self.hparams.prefetch_factor is not None:
                loader_kwargs["prefetch_factor"] = int(self.hparams.prefetch_factor)
        return DataLoader(**loader_kwargs)

    def _get_distributed_sampler(
        self,
        dataset: DeciferDataset,
        *,
        shuffle: bool,
        drop_last: bool,
    ) -> Optional[DistributedSampler]:
        world_size: Optional[int] = None
        rank: Optional[int] = None
        if getattr(self, "trainer", None) is not None:
            world_size = int(getattr(self.trainer, "world_size", 1))
            rank = int(getattr(self.trainer, "global_rank", 0))
        elif dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
        else:
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
            rank = int(os.environ.get("RANK", "0"))
        self._ddp_world_size = world_size
        self._ddp_rank = rank
        if world_size <= 1:
            return None
        return DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank or 0,
            shuffle=shuffle,
            drop_last=drop_last,
        )
