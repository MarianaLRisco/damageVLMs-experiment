"""Zero-shot evaluation pipeline (no training)."""

from pipelines.base import BasePipeline, PipelineConfig
from data.utils import build_description_map
from evaluation.contrastive import evaluate_contrastive
from models.pretrained import load_siglip_pretrained
from typing import Dict


class EvalOnlyPipeline(BasePipeline):
    """Pipeline for zero-shot evaluation only (no training)."""

    def run(self) -> Dict:
        """Run zero-shot evaluation pipeline."""
        device = self.device
        pretrained = self.cfg.model_cfg.get("pretrained", "google/siglip-base-patch16-256-multilingual")
        classes = self.cfg.dataset_cfg["classes"]
        prompts_en = self.cfg.dataset_cfg["prompts"]["english"]
        description_map = build_description_map(classes, prompts_en)
        output_dir = self.output_dir

        print(f"\n{'='*60}")
        print(f"[eval-only] {self.cfg.model_name} | Dataset: {self.cfg.dataset_cfg['name']}")
        print(f"{'='*60}")

        model, processor = load_siglip_pretrained(pretrained, device=device)
        model = model.to(device)
        model.eval()

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
        del model

        return {}