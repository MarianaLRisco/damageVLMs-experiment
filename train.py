import argparse
import os
import sys
import warnings

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


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run VLM disaster classification experiments")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--models", nargs="*", default=None, help="Run only these model names (optional filter)")
    parser.add_argument("--datasets", nargs="*", default=None, help="Run only these dataset names (optional filter)")
    parser.add_argument("--eval-only", action="store_true", help="Skip training, run zero-shot evaluation only")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    config["device"] = get_device(config.get("device", "cuda"))
    config["eval_only"] = args.eval_only

    # Setup logging (use first model/dataset combo for log directory)
    log_output_dir = config["output"]["dir"]
    log_experiment_name = f"run_{args.config.replace('/', '_').replace('.yaml', '')}"

    logger = setup_logging(log_output_dir, log_experiment_name)
    logger.info(f"Config loaded from: {args.config}")
    logger.info(f"Device: {config['device']}")
    logger.info(f"Eval-only mode: {config['eval_only']}")

    # Get logger for this module
    logger = get_logger(__name__)

    for dataset_cfg in config["datasets"]:
        if args.datasets and dataset_cfg["name"] not in args.datasets:
            continue

        dataset_name = dataset_cfg["name"]
        root = dataset_cfg["root"]

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

                logger.info(f"✅ Completed: {model_name} on {dataset_name}")
            except Exception as e:
                logger.error(f"❌ Error running {model_name} on {dataset_name}: {e}")
                import traceback
                logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()
