"""Two-stage training pipeline for SigLIP models."""

import os
import torch
from torch.utils.data import DataLoader
from pipelines.base import BasePipeline, PipelineConfig
from data.utils import build_description_map, get_batch_size, get_num_workers
from engine.dataloaders.twoStage import (
    ImageTextTwoStage,
    generate_fewshot_dataframe,
)
from engine.trainers.two_stage import (
    TrainerFirstStep,
    TrainerSecondStep,
    freeze_all_except_layernorm,
)
from evaluation.classifier import evaluate_classifier
from models.pretrained import load_siglip_pretrained
from models.siglip_twostage import SigLIPLinearClassifier, SigLIP2LinearClassifier
from typing import Dict


class TwoStagePipeline(BasePipeline):
    """Pipeline for two-stage training (SigLIP twoStage)."""

    def run(self) -> Dict:
        """Run two-stage training pipeline."""
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
        is_siglip2 = "siglip2" in self.cfg.model_name
        is_fewshot = "fewshot" in self.cfg.model_name
        n_shots = hp.get("fewshot_samples_per_class", 16)

        print(f"\n{'='*60}")
        print(f"Experiment: {self.cfg.model_name} | Dataset: {self.cfg.dataset_cfg['name']}")
        print(f"{'='*60}")
        print(f"batch_size={batch_size} | num_workers={nw} | device={device}")

        # Load base model and processor
        base_model, processor = load_siglip_pretrained(pretrained, device=device)
        base_model = base_model.to(device)

        # ─── Stage 1: freeze all except LayerNorm, train contrastively ───
        print("\n--- Stage 1: Contrastive fine-tuning (LayerNorm only) ---")
        freeze_all_except_layernorm(base_model)

        train_dataset_s1 = ImageTextTwoStage(self.cfg.train_df, processor)
        val_dataset_s1 = ImageTextTwoStage(self.cfg.val_df, processor)
        pin = device == "cuda"
        train_loader_s1 = DataLoader(
            train_dataset_s1, batch_size=batch_size, shuffle=True, num_workers=nw, pin_memory=pin
        )
        val_loader_s1 = DataLoader(
            val_dataset_s1, batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=pin
        )

        trainer_s1 = TrainerFirstStep(
            model=base_model,
            lr=hp["stage1_lr"],
            device=device,
            loss_key=loss_key,
            weight_decay=hp.get("stage1_weight_decay", 1e-3),
        )
        trainer_s1.fit(train_loader_s1, val_loader_s1, epochs=hp["stage1_epochs"])
        trainer_s1.save(base_model, processor, output_dir=checkpoint, model_type=f"{self.cfg.model_name}_stage1")

        # ─── Stage 2: train linear classifier (frozen backbone) ───
        print("\n--- Stage 2: Linear classifier training ---")

        stage2_train_df = self.cfg.train_df
        if is_fewshot:
            stage2_train_df = generate_fewshot_dataframe(self.cfg.train_df, num_shots=n_shots)

        if is_siglip2:
            classifier = SigLIP2LinearClassifier(
                base_model=base_model,
                processor=processor,
                classnames=classes,
                description_map=description_map,
                device=device,
            ).to(device)
        else:
            classifier = SigLIPLinearClassifier(
                base_model=base_model,
                processor=processor,
                classnames=classes,
                description_map=description_map,
                device=device,
            ).to(device)

        train_dataset_s2 = ImageTextTwoStage(stage2_train_df, processor)
        train_loader_s2 = DataLoader(
            train_dataset_s2, batch_size=batch_size, shuffle=True, num_workers=nw, pin_memory=pin
        )

        optimizer_s2 = torch.optim.AdamW(
            [classifier.classifier], lr=hp["stage2_lr"], weight_decay=hp.get("stage2_weight_decay", 1e-3)
        )
        scheduler_s2 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_s2, T_max=hp.get("stage2_T_max", 70)
        )
        scaler_s2 = torch.cuda.amp.GradScaler() if device == "cuda" else torch.cuda.amp.GradScaler(enabled=False)

        trainer_s2 = TrainerSecondStep(
            model=classifier,
            train_loader=train_loader_s2,
            optimizer=optimizer_s2,
            scheduler=scheduler_s2,
            scaler=scaler_s2,
            device=device,
            num_epochs=hp["stage2_epochs"],
            loss_key=loss_key,
        )
        trainer_s2.train_epochs(hp["stage2_epochs"])
        os.makedirs(os.path.join(checkpoint, self.cfg.model_name), exist_ok=True)
        torch.save(classifier.state_dict(), os.path.join(checkpoint, self.cfg.model_name, "classifier.pth"))

        # Evaluate
        test_dataset = ImageTextTwoStage(self.cfg.test_df, processor)
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=pin
        )
        evaluate_classifier(classifier, test_loader, classes, device, output_dir, self.cfg.model_name, loss_key)

        return {}
