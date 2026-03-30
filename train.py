import argparse
import os
import sys

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
from config.loader import load_config

# Data
from data.loader import load_damage_dataset, load_crisisMMD

# Utils
from utils.device import get_device

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

    config = load_config(args.config)
    config["device"] = get_device(config.get("device", "cuda"))
    config["eval_only"] = args.eval_only
    print(f"[INFO] Using device: {config['device']}")
    if config["eval_only"]:
        print("[INFO] eval-only mode: skipping training, running zero-shot evaluation")

    for dataset_cfg in config["datasets"]:
        if args.datasets and dataset_cfg["name"] not in args.datasets:
            continue

        name = dataset_cfg["name"]
        root = dataset_cfg["root"]

        print(f"\n{'#'*60}")
        print(f"Loading dataset: {name} from {root}")
        print(f"{'#'*60}")

        if name == "damage_dataset":
            train_df, val_df, test_df = load_damage_dataset(root)
        elif name == "crisisMMD":
            train_df, val_df, test_df = load_crisisMMD(root)
        else:
            raise ValueError(f"Unknown dataset: {name}")

        print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

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
                pipeline.run()
            except Exception as e:
                print(f"[ERROR] {model_name} on {name}: {e}")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main()
