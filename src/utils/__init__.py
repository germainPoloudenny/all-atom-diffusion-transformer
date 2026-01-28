"""Copyright (c) Meta Platforms, Inc. and affiliates."""

from pathlib import Path

from omegaconf import OmegaConf

from src.utils.instantiators import instantiate_callbacks, instantiate_loggers
from src.utils.joblib import joblib_map
from src.utils.logging_utils import log_hyperparameters
from src.utils.pylogger import RankedLogger
from src.utils.rich_utils import enforce_tags, print_config_tree
from src.utils.utils import extras, get_metric_value, task_wrapper


def _resume_run_dir(ckpt_path: str, default_dir: str, copy_on_resume: bool = False) -> str:
    """Resolve the run directory for resume-in-place behavior."""
    if not ckpt_path:
        return default_dir
    if copy_on_resume:
        return default_dir

    path = Path(ckpt_path).expanduser()
    if path.is_dir():
        if path.name == "checkpoints":
            return str(path.parent)
        return str(path)
    if path.parent.name == "checkpoints":
        return str(path.parent.parent)
    return str(path.parent)


if not OmegaConf.has_resolver("resume_run_dir"):
    OmegaConf.register_new_resolver("resume_run_dir", _resume_run_dir)
