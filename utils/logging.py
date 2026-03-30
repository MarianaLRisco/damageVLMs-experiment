"""
Logging utilities for VLM experiments.

Provides structured logging with timestamps, log levels, and file output.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logging(
    output_dir: str | Path,
    experiment_name: str,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Configure structured logging for an experiment.

    Args:
        output_dir: Directory to save log files
        experiment_name: Name for this experiment (used in log filename)
        level: Logging level (default: INFO)

    Returns:
        Configured logger instance

    Example:
        >>> logger = setup_logging("outputs/", "siglip_sigmoid")
        >>> logger.info("Starting training...")
        2026-03-30 12:34:56 | INFO | __main__ | Starting training...
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"{experiment_name}_{timestamp}.log"

    # Configure logging format
    log_format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Configure handlers
    handlers = [
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]

    logging.basicConfig(
        level=level,
        format=log_format,
        datefmt=date_format,
        handlers=handlers,
        force=True  # Reset any existing configuration
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.

    Args:
        name: Logger name (usually __name__ of the calling module)

    Returns:
        Logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing batch %d", batch_idx)
    """
    return logging.getLogger(name)


class ExperimentTracker:
    """
    Simple experiment tracker for logging metrics and results.

    Stores experiment metadata and metrics in a JSON file for later analysis.
    """

    def __init__(self, output_dir: str | Path, experiment_name: str):
        """
        Initialize experiment tracker.

        Args:
            output_dir: Directory to save experiment logs
            experiment_name: Name for this experiment
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name

        self.metadata = {}
        self.metrics = {}
        self.start_time = None
        self.end_time = None

    def start(self, config: dict):
        """
        Record experiment start time and configuration.

        Args:
            config: Experiment configuration dict
        """
        from datetime import datetime

        self.start_time = datetime.now()
        self.metadata = {
            "experiment_name": self.experiment_name,
            "start_time": self.start_time.isoformat(),
            "config": config
        }

    def log_metrics(self, metrics: dict, step: int | None = None):
        """
        Log metrics for a specific step (epoch).

        Args:
            metrics: Dict of metric names to values
            step: Step number (e.g., epoch number)
        """
        if step is not None:
            self.metrics[f"step_{step}"] = metrics
        else:
            self.metrics["final"] = metrics

    def finish(self):
        """Record experiment end time."""
        from datetime import datetime

        self.end_time = datetime.now()
        self.metadata["end_time"] = self.end_time.isoformat()

        if self.start_time:
            duration = (self.end_time - self.start_time).total_seconds()
            self.metadata["duration_seconds"] = duration

        self._save_log()

    def _save_log(self):
        """Save experiment log to JSON file."""
        import json

        log_data = {
            "metadata": self.metadata,
            "metrics": self.metrics
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.output_dir / f"{self.experiment_name}_experiment_{timestamp}.json"

        with open(log_file, "w") as f:
            json.dump(log_data, f, indent=2)

        return log_file
