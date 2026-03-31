
"""Base pipeline class for all training pipelines."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict
import pandas as pd


@dataclass
class PipelineConfig:
    """Immutable config for a pipeline."""

    model_name: str
    model_cfg: Dict
    dataset_cfg: Dict
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    global_config: Dict


class BasePipeline(ABC):
    """Base class for all training pipelines."""

    def __init__(self, config: PipelineConfig):
        self.cfg = config
        self.device = self._get_device()
        self.output_dir = self._build_output_dir()

    @abstractmethod
    def run(self) -> Dict:
        """Execute the complete pipeline. Return metrics dict."""
        pass

    def _get_device(self) -> str:
        """Get device from global config."""
        from utils.device import get_device
        return get_device(self.cfg.global_config.get("device", "cuda"))

    def _build_output_dir(self) -> str:
        """Build output directory path."""
        import os

        return os.path.join(
            self.cfg.global_config["output"]["dir"],
            self.cfg.dataset_cfg["name"],
            self.cfg.model_name,
        )
