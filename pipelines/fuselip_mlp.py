"""FuseLIP-MLP training pipeline."""

from torch.utils.data import DataLoader
from pipelines.base import BasePipeline, PipelineConfig
from data.utils import get_batch_size, get_num_workers
from engine.dataloaders.fuselip import ImageTextFuseLIP
from engine.trainers.mlp import TrainerMLP
from models.pretrained import fuselip_model_loader
from models.fuselip_mlp import FuseLIPMLPClassifier
from typing import Dict


class FuseLIPMLPPipeline(BasePipeline):
    """Pipeline for FuseLIP-MLP training."""

    def run(self) -> Dict:
        """Run FuseLIP-MLP training pipeline."""
        hp = self.cfg.model_cfg["hyperparams"]
        device = self.device
        checkpoint = self.cfg.model_cfg["checkpoint"]
        mode = hp["mode"]
        embed_dim = hp.get("embed_dim", 512)
        batch_size = get_batch_size(hp.get("batch_size", 64), device)
        nw = get_num_workers(device)
        classes = self.cfg.dataset_cfg["classes"]
        output_dir = self.output_dir

        print(f"\n{'='*60}")
        print(f"Experiment: {self.cfg.model_name} (mode={mode}) | Dataset: {self.cfg.dataset_cfg['name']}")
        print(f"{'='*60}")

        backbone, image_processor, text_tokenizer = fuselip_model_loader(device=device)
        num_classes = len(classes)

        model = FuseLIPMLPClassifier(
            backbone=backbone,
            num_classes=num_classes,
            mode=mode,
            embed_dim=embed_dim,
        ).to(device)

        train_dataset = ImageTextFuseLIP(self.cfg.train_df, image_processor, text_tokenizer)
        val_dataset = ImageTextFuseLIP(self.cfg.val_df, image_processor, text_tokenizer)
        test_dataset = ImageTextFuseLIP(self.cfg.test_df, image_processor, text_tokenizer)

        pin = device == "cuda"
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw, pin_memory=pin
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=pin
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=pin
        )

        trainer = TrainerMLP(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=hp["epochs"],
            lr=hp["lr"],
            device=device,
            weight_decay=hp.get("weight_decay", 1e-4),
        )
        trainer.fit()
        trainer.save(checkpoint, self.cfg.model_name)
        trainer.evaluate_test(test_loader, classes, output_dir, self.cfg.model_name)

        return {}
