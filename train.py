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
    

    
    # Evaluator(model).run(val_dl)

if __name__ == "__main__":
    main()
