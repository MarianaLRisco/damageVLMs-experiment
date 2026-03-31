import argparse
import yaml
from dataloaders.dataloader import get_dataloader
from engine.trainer_base import Trainer
from engine.trainer_twoStage import TrainerFirstStep, TrainerSecondStep

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
<<<<<<< Updated upstream
    
=======
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

                logger.info(f"Completed: {model_name} on {dataset_name}")
            except Exception as e:
                logger.error(f"Error running {model_name} on {dataset_name}: {e}")
                import traceback
                logger.error(traceback.format_exc())
>>>>>>> Stashed changes

    
    # Evaluator(model).run(val_dl)

if __name__ == "__main__":
    main()
