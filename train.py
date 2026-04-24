import argparse
import os
import sys
import warnings
from typing import Any

warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)
warnings.filterwarnings("ignore", message="Setuptools is replacing distutils", category=UserWarning)

# CRÍTICO: Configurar PYTHONPATH para FuseLIP ANTES de cualquier import
# Esto asegura que se use la versión correcta de open_clip compatible con FuseLIP
fuselip_src = os.path.join(os.path.dirname(__file__), 'fuselip_repo', 'src')
if fuselip_src not in sys.path:
    sys.path.insert(0, fuselip_src)

# Enable MPS fallback for unsupported ops (Apple Silicon GPU)
# This fixes: NotImplementedError for aten::_upsample_bilinear2d_aa_backward.grad_input
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Allow MPS to use more unified memory (24GB on M4 Pro)
# 0.0 = no limit (recommended for M4 Pro with 16GB+)
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# Make engine/ and models/ importable from root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "engine"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models"))

# Config
from config_loaders.loader import load_config

# Data
from data.loader import load_damage_dataset, load_crisisMMD

# Utils
from utils.device import get_device
from utils.logging import setup_logging, get_logger

# Pipelines
from pipelines.registry import get_pipeline
from pipelines.base import PipelineConfig


def _load_env_file(path: str = ".env") -> None:
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as env_file:
        for line in env_file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"'")
            if key:
                os.environ.setdefault(key, value)


def _resolve_dataset_root(dataset_name: str, configured_root: str) -> str:
    env_key = f"{dataset_name.upper()}_ROOT"
    root = os.getenv(env_key) or os.getenv("DATASET_ROOT")
    if not root:
        return configured_root

    dataset_root = os.path.expanduser(root.strip())
    if not os.path.isdir(dataset_root):
        return dataset_root

    if os.path.basename(dataset_root) == dataset_name:
        return dataset_root

    # Dataset name may not match the real folder name (e.g. CrisisMMD_v2.0).
    # Try common fallback candidates before falling back to the provided root.
    fallback_names = {dataset_name}
    if dataset_name == "crisisMMD":
        fallback_names.update({"CrisisMMD_v2.0", "crisismmd_v2.0", "CrisisMMD"})

    for candidate_name in fallback_names:
        candidate = os.path.join(dataset_root, candidate_name)
        if os.path.isdir(candidate):
            return candidate

    # Fallback: if root contains dataset-like folders, pick the first matching case-insensitive one.
    for child in os.listdir(dataset_root):
        if child.lower() == dataset_name.lower():
            return os.path.join(dataset_root, child)

    return dataset_root


def _resolve_seed(cli_seed: int | None, config: dict[str, Any]) -> int:
    """Resolve the effective random seed for a run.

    CLI argument wins over config value. If neither is provided, return
    a safe default.
    """
    configured_seed = config.get("seed")
    if cli_seed is not None:
        return cli_seed

    if configured_seed is None:
        return 42

    try:
        return int(configured_seed)
    except (TypeError, ValueError) as err:
        raise ValueError(f"Invalid seed in config: {configured_seed!r}") from err


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run VLM disaster classification experiments")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--models", nargs="*", default=None, help="Run only these model names (optional filter)")
    parser.add_argument("--datasets", nargs="*", default=None, help="Run only these dataset names (optional filter)")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for reproducible runs (ConvNet only)")
    parser.add_argument("--eval-only", action="store_true", help="Skip training, run zero-shot evaluation only")
    args = parser.parse_args()

    # Load env and config
    _load_env_file()
    config = load_config(args.config)
    config["device"] = get_device(config.get("device", "cuda"))
    config["seed"] = _resolve_seed(args.seed, config)
    config["eval_only"] = args.eval_only

    # Setup logging (use first model/dataset combo for log directory)
    log_output_dir = config["output"]["dir"]
    log_experiment_name = f"run_{args.config.replace('/', '_').replace('.yaml', '')}"

    logger = setup_logging(log_output_dir, log_experiment_name)
    logger.info(f"Config loaded from: {args.config}")
    logger.info(f"Seed: {config['seed']}")
    logger.info(f"Device: {config['device']}")
    logger.info(f"Eval-only mode: {config['eval_only']}")

    # Get logger for this module
    logger = get_logger(__name__)

    for dataset_cfg in config["datasets"]:
        if args.datasets and dataset_cfg["name"] not in args.datasets:
            continue

        dataset_name = dataset_cfg["name"]
        root = _resolve_dataset_root(dataset_cfg["name"], dataset_cfg["root"])
        dataset_cfg["root"] = root

        logger.info("=" * 60)
        logger.info(f"Loading dataset: {dataset_name} from {root}")
        logger.info("=" * 60)

        if dataset_name == "damage_dataset":
            train_df, val_df, test_df = load_damage_dataset(root)
        elif dataset_name == "crisisMMD":
            train_df, val_df, test_df = load_crisisMMD(root)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        logger.info(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

        for model_cfg in config["models"]:
            model_name = model_cfg["name"]

            if args.models and model_name not in args.models:
                continue

            try:
                # Create pipeline config
                pipeline_config = PipelineConfig(
                    model_name=model_name,
                    model_cfg=model_cfg,
                    dataset_cfg=dataset_cfg,
                    train_df=train_df,
                    val_df=val_df,
                    test_df=test_df,
                    global_config=config,
                )

                # Get and run pipeline
                pipeline = get_pipeline(model_name, pipeline_config)
                logger.info("=" * 60)
                mode = "eval-only" if config.get("eval_only", False) else "training"
                logger.info(f"[{mode}] {model_name} | Dataset: {dataset_name}")
                logger.info("=" * 60)

                pipeline.run()

                logger.info(f" Completed: {model_name} on {dataset_name}")
            except Exception as e:
                logger.error(f"Error running {model_name} on {dataset_name}: {e}")
                import traceback
                logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()
