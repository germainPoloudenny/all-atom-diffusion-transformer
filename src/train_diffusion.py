"""Copyright (c) Meta Platforms, Inc. and affiliates."""

from typing import Any, Dict, List, Optional, Tuple

import inspect
import os
import shutil
from pathlib import Path

import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


def _get_resume_run_dir(ckpt_path: str) -> Path:
    """Infer run directory from a checkpoint path."""
    path = Path(ckpt_path).expanduser()
    if path.is_dir():
        if path.name == "checkpoints":
            return path.parent
        return path
    if path.parent.name == "checkpoints":
        return path.parent.parent
    return path.parent


def _maybe_copy_resume_dir(cfg: DictConfig) -> None:
    """Optionally copy previous run directory into the current output dir."""
    if not cfg.get("ckpt_path"):
        return
    resume_cfg = cfg.get("resume")
    if not resume_cfg or not resume_cfg.get("copy_run_dir"):
        return
    if os.environ.get("RANK", "0") != "0" or os.environ.get("LOCAL_RANK", "0") != "0":
        return

    src_dir = _get_resume_run_dir(cfg.ckpt_path)
    dst_dir = Path(cfg.paths.output_dir).expanduser()

    try:
        if src_dir.resolve() == dst_dir.resolve():
            return
    except FileNotFoundError:
        # If the source doesn't exist yet, resolve() can fail; handle below.
        pass

    if not src_dir.exists():
        log.warning(
            f"Resume copy requested but source run dir not found: {src_dir} "
            f"(ckpt_path={cfg.ckpt_path})"
        )
        return

    log.info(f"Copying resume run dir from {src_dir} to {dst_dir} (resume.copy_run_dir=True)")
    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".hydra"))

# PyTorch 2.6 flips torch.load default to weights_only=True; Lightning resume needs full ckpt.
if "weights_only" in inspect.signature(torch.load).parameters:
    _torch_load = torch.load

    def _torch_load_compat(*args, **kwargs):
        # Force full checkpoint loading for Lightning resume (PyTorch 2.6+ defaults to weights_only=True).
        # Lightning may pass weights_only=None, which would still resolve to True inside torch.load.
        kwargs["weights_only"] = False
        return _torch_load(*args, **kwargs)

    torch.load = _torch_load_compat


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the diffusion model for generative modelling.

    Can additionally evaluate on a testset, using best weights obtained during training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    _maybe_copy_resume_dir(cfg)

    log.info(f"Instantiating datamodule <{cfg.data.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=False
    )
    # datamodule.setup()  # to save metadata the first time code is run

    log.info(f"Instantiating diffusion model <{cfg.diffusion_module._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.diffusion_module)

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(
        cfg.get("callbacks"), using_logger=(len(logger) > 0)
    )

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train_diffusion.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    try:
        import lovely_tensors as lt

        lt.monkey_patch()
    except ImportError:
        pass

    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)

    torch.set_float32_matmul_precision("high")
    main()
