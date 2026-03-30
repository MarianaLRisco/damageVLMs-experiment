"""Contrastive training pipeline for SigLIP models."""

import os
from torch.utils.data import DataLoader
from pipelines.base import BasePipeline, PipelineConfig
from data.utils import build_description_map, get_batch_size, get_num_workers
from engine.dataloaders.sigmoidCrossentropy import ImageTextDataset
from engine.trainers.base import Trainer
from evaluation.contrastive import evaluate_contrastive
from models.pretrained import load_siglip_pretrained
from models.siglip_crossentropy import SigLIPCrossentropy, SigLIP2Crossentropy
from transformers import AutoProcessor
from typing import Dict


class ContrastivePipeline(BasePipeline):
    """Pipeline for contrastive training (SigLIP sigmoid/crossentropy)."""

    def run(self) -> Dict:
        """Run contrastive training pipeline."""
        hp = self.cfg.model_cfg["hyperparams"]
        device = self.device
        pretrained = self.cfg.model_cfg["pretrained"]
        checkpoint = self.cfg.model_cfg["checkpoint"]
        loss_key = hp["loss_key"]
        batch_size = get_batch_size(hp.get("batch_size", 32), device)
        nw = get_num_workers(device)
        classes = self.cfg.dataset_cfg["classes"]
        prompts_en = self.cfg.dataset_cfg["prompts"]["english"]
        description_map = build_description_map(classes, prompts_en)
        output_dir = self.output_dir

        print(f"\n{'='*60}")
        print(f"Experiment: {self.cfg.model_name} | Dataset: {self.cfg.dataset_cfg['name']}")
        print(f"{'='*60}")
        print(f"batch_size={batch_size} | num_workers={nw} | device={device}")

        # Load model
        if loss_key == "sigmoid":
            model, processor = load_siglip_pretrained(pretrained, device=device)
        else:
            if "siglip2" in self.cfg.model_name:
                model = SigLIP2Crossentropy(pretrained)
            else:
                model = SigLIPCrossentropy(pretrained)
            if hasattr(model.base_model, "gradient_checkpointing_enable"):
                model.base_model.gradient_checkpointing_enable()
            processor = AutoProcessor.from_pretrained(pretrained)

        # Build datasets and loaders
        train_dataset = ImageTextDataset(self.cfg.train_df, processor)
        val_dataset = ImageTextDataset(self.cfg.val_df, processor)
        pin = device == "cuda"
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw, pin_memory=pin
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=pin
        )

        # Train
        trainer = Trainer(
            model=model,
            model_type=self.cfg.model_name,
            epochs=hp["epochs"],
            lr=hp["lr"],
            device=device,
            loss_key=loss_key,
            weight_decay=hp.get("weight_decay", 1e-4),
            factor=hp.get("factor", 0.5),
            patience=hp.get("patience", 2),
        )
        trainer.fit(train_loader, val_loader, hp["epochs"])
        trainer.save(model, processor, output_dir=checkpoint, model_type=self.cfg.model_name)

        # Evaluate on test set (zero-shot classification)
        evaluate_contrastive(
            model,
            self.cfg.test_df,
            classes,
            description_map,
            processor,
            device,
            output_dir,
            self.cfg.model_name,
        )

        # Return empty dict for consistency (metrics are saved internally)
        return {}
