"""Latent diffusion module that uses the XtoCIF latent autoencoder as a first-stage model."""

import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from lightning import LightningModule
from omegaconf import DictConfig, OmegaConf
from torch.nn import ModuleDict
from torchmetrics import MeanMetric

_XTOCIF_ROOT = Path(__file__).resolve().parents[3]
if str(_XTOCIF_ROOT) not in sys.path:
    sys.path.append(str(_XTOCIF_ROOT))

from decifer.latent_ae_inference import build_augmentation_kwargs, load_model
from decifer.tokenizer import Tokenizer
from decifer.utility import discrete_to_continuous_xrd


class LatentAEDiffusionLitModule(LightningModule):
    """LightningModule for latent diffusion over latent_ae states."""

    def __init__(
        self,
        latent_ae_ckpt: str,
        denoiser: torch.nn.Module,
        interpolant: DictConfig,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        scheduler_frequency: str,
        compile: bool,
        latent_ae_config: Optional[str] = None,
        prefer_best: bool = True,
        latent_tokens: Optional[int] = None,
        latent_dim: Optional[int] = None,
        ae_max_input_tokens: Optional[int] = None,
        condition: Optional[bool] = None,
        xrd_augmentation: Optional[DictConfig] = None,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        # Load latent autoencoder on CPU; Lightning will move it to device.
        latent_ae, model_config, ae_cfg = load_model(
            latent_ae_ckpt,
            latent_ae_config,
            device=torch.device("cpu"),
            prefer_best=prefer_best,
        )
        latent_ae.requires_grad_(False)
        latent_ae.eval()
        self.latent_ae = latent_ae
        self.latent_tokens = int(model_config["latent_tokens"])
        self.latent_dim = int(model_config["latent_dim"])
        self.block_size = int(model_config["block_size"])
        self.condition_size = int(model_config.get("condition_size", 0) or 0)

        if latent_tokens is not None and int(latent_tokens) != self.latent_tokens:
            raise ValueError(
                f"latent_tokens mismatch: ckpt={self.latent_tokens}, config={latent_tokens}"
            )
        if latent_dim is not None and int(latent_dim) != self.latent_dim:
            raise ValueError(f"latent_dim mismatch: ckpt={self.latent_dim}, config={latent_dim}")

        if ae_max_input_tokens is None:
            ae_max_input_tokens = (self.block_size + 1 - self.latent_tokens) // 2
        if ae_max_input_tokens <= 0:
            raise ValueError("Invalid AE sizing: block_size too small for latent_tokens.")
        self.ae_max_input_tokens = int(ae_max_input_tokens)
        if self.ae_max_input_tokens < 2:
            raise ValueError("ae_max_input_tokens must be >= 2 to append newline tokens.")

        if condition is None:
            condition = bool(model_config.get("condition", False))
        self.use_condition = bool(condition)

        aug_base = build_augmentation_kwargs(ae_cfg)
        if xrd_augmentation is not None:
            aug_override = OmegaConf.to_container(xrd_augmentation, resolve=True)
            if isinstance(aug_override, dict):
                for key, value in aug_override.items():
                    if value is not None:
                        aug_base[key] = value
        self.augmentation_kwargs = aug_base

        self.tokenizer = Tokenizer()
        self.pad_id = self.tokenizer.padding_id
        self.latent_token_id = self.tokenizer.latent_id
        self.newline_id = self.tokenizer.token_to_id["\n"]

        self.denoiser = denoiser
        self.interpolant = interpolant

        self.train_metrics = ModuleDict(
            {
                "loss": MeanMetric(),
                "x_loss": MeanMetric(),
                "t_avg": MeanMetric(),
            }
        )
        self.val_metrics = ModuleDict(
            {
                "loss": MeanMetric(),
                "x_loss": MeanMetric(),
                "t_avg": MeanMetric(),
            }
        )
        self.test_metrics = ModuleDict(
            {
                "loss": MeanMetric(),
                "x_loss": MeanMetric(),
                "t_avg": MeanMetric(),
            }
        )

    def _build_condition(self, batch: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        if not self.use_condition:
            return None
        xrd_q = batch["xrd.q"].to(self.device)
        xrd_iq = batch["xrd.iq"].to(self.device)
        cond = discrete_to_continuous_xrd(xrd_q, xrd_iq, **self.augmentation_kwargs)["iq"]
        cond = cond.to(dtype=self.denoiser.x_embedder.weight.dtype)
        if self.condition_size and cond.size(-1) != self.condition_size:
            raise ValueError(
                f"Condition size mismatch: expected {self.condition_size}, got {cond.size(-1)}"
            )
        return cond

    def _build_encode_batch(self, sequences: torch.Tensor) -> torch.Tensor:
        encode_sequences = []
        for seq in sequences:
            seq = seq[seq != self.pad_id]
            if seq.numel() == 0:
                raise RuntimeError("Empty CIF sequence after removing padding.")
            max_payload_tokens = self.ae_max_input_tokens - 2
            if seq.numel() > max_payload_tokens:
                seq = seq[:max_payload_tokens]
            seq = torch.cat(
                [
                    seq,
                    torch.tensor(
                        [self.newline_id, self.newline_id], dtype=seq.dtype, device=seq.device
                    ),
                ]
            )
            latent_tokens = torch.full(
                (self.latent_tokens,),
                self.latent_token_id,
                dtype=seq.dtype,
                device=seq.device,
            )
            encode_seq = torch.cat([seq, latent_tokens], dim=0)
            encode_sequences.append(encode_seq)

        max_len = max(int(seq.numel()) for seq in encode_sequences)
        batch_size = len(encode_sequences)
        encode_batch = torch.full(
            (batch_size, max_len),
            fill_value=self.pad_id,
            dtype=encode_sequences[0].dtype,
            device=encode_sequences[0].device,
        )
        for i, encode_seq in enumerate(encode_sequences):
            encode_batch[i, : int(encode_seq.numel())] = encode_seq
        return encode_batch

    def _encode_latents(
        self, sequences: torch.Tensor, cond_vec: Optional[torch.Tensor]
    ) -> torch.Tensor:
        encode_batch = self._build_encode_batch(sequences)
        with torch.no_grad():
            _, _, latent_state = self.latent_ae(
                encode_batch, cond_vec=cond_vec, return_latents=True
            )
        return latent_state

    def forward(self, batch: Dict[str, torch.Tensor]):
        sequences = batch["cif_tokens"].to(self.device)
        cond_vec = self._build_condition(batch)

        latent_state = self._encode_latents(sequences, cond_vec)
        latent_state = latent_state.to(dtype=self.denoiser.x_embedder.weight.dtype)
        batch_size, latent_tokens = latent_state.shape[:2]
        token_mask = torch.ones(
            batch_size, latent_tokens, device=latent_state.device, dtype=torch.bool
        )
        dense_encoded_batch = {
            "x_1": latent_state,
            "token_mask": token_mask,
            "diffuse_mask": token_mask,
        }

        self.interpolant.device = latent_state.device
        noisy_batch = self.interpolant.corrupt_batch(dense_encoded_batch)

        dataset_idx = torch.zeros(batch_size, device=latent_state.device, dtype=torch.long)
        spacegroup = torch.zeros(batch_size, device=latent_state.device, dtype=torch.long)

        if self.interpolant.self_condition and random.random() < self.interpolant.self_condition_prob:
            with torch.no_grad():
                x_sc = self.denoiser(
                    x=noisy_batch["x_t"],
                    t=noisy_batch["t"],
                    dataset_idx=dataset_idx,
                    spacegroup=spacegroup,
                    mask=token_mask,
                    x_sc=None,
                    cond_vec=cond_vec,
                )
        else:
            x_sc = None

        pred_x = self.denoiser(
            x=noisy_batch["x_t"],
            t=noisy_batch["t"],
            dataset_idx=dataset_idx,
            spacegroup=spacegroup,
            mask=token_mask,
            x_sc=x_sc,
            cond_vec=cond_vec,
        )
        return pred_x, noisy_batch

    def criterion(
        self,
        noisy_batch: Dict[str, torch.Tensor],
        pred_x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        gt_x_1 = noisy_batch["x_1"]
        norm_scale = 1 - torch.min(
            noisy_batch["t"].unsqueeze(-1),
            torch.tensor(0.9, device=gt_x_1.device, dtype=noisy_batch["t"].dtype),
        )
        x_error = (gt_x_1 - pred_x) / norm_scale
        loss_mask = noisy_batch["token_mask"] * noisy_batch["diffuse_mask"]
        loss_denom = torch.sum(loss_mask, dim=-1) * pred_x.size(-1)
        x_loss = torch.sum(x_error**2 * loss_mask[..., None], dim=(-1, -2)) / loss_denom
        loss_dict = {"loss": x_loss.mean(), "x_loss": x_loss, "t_avg": noisy_batch["t"].mean()}
        return loss_dict

    def on_train_epoch_start(self) -> None:
        for metric in self.train_metrics.values():
            metric.reset()

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        pred_x, noisy_batch = self.forward(batch)
        loss_dict = self.criterion(noisy_batch, pred_x)

        for k, v in loss_dict.items():
            self.train_metrics[k](v)
            self.log(
                f"train/{k}",
                self.train_metrics[k],
                on_step=True,
                on_epoch=False,
                prog_bar=(k == "loss"),
            )

        return loss_dict["loss"]

    def on_validation_epoch_start(self) -> None:
        for metric in self.val_metrics.values():
            metric.reset()

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        pred_x, noisy_batch = self.forward(batch)
        loss_dict = self.criterion(noisy_batch, pred_x)

        for k, v in loss_dict.items():
            self.val_metrics[k](v)
            self.log(
                f"val/{k}",
                self.val_metrics[k],
                on_step=False,
                on_epoch=True,
                prog_bar=(k == "loss"),
                sync_dist=True,
            )

    def on_test_epoch_start(self) -> None:
        for metric in self.test_metrics.values():
            metric.reset()

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        pred_x, noisy_batch = self.forward(batch)
        loss_dict = self.criterion(noisy_batch, pred_x)

        for k, v in loss_dict.items():
            self.test_metrics[k](v)
            self.log(
                f"test/{k}",
                self.test_metrics[k],
                on_step=False,
                on_epoch=True,
                prog_bar=(k == "loss"),
                sync_dist=True,
            )

    def setup(self, stage: Optional[str] = None) -> None:
        if self.hparams.compile and stage == "fit":
            self.denoiser = torch.compile(self.denoiser)
            self.latent_ae = torch.compile(self.latent_ae)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": self.hparams.scheduler_frequency,
                },
            }
        return {"optimizer": optimizer}
