"""Pipeline registry for model-specific training pipelines."""

from typing import Dict, Type
from pipelines.base import BasePipeline, PipelineConfig

# Import all pipeline classes
from pipelines.contrastive import ContrastivePipeline
from pipelines.two_stage import TwoStagePipeline
from pipelines.fuselip_mlp import FuseLIPMLPPipeline
from pipelines.eval_only import EvalOnlyPipeline

# Simple registry dict
PIPELINE_REGISTRY: Dict[str, Type[BasePipeline]] = {
    # Contrastive models
    "siglip_sigmoid": ContrastivePipeline,
    "siglip_crossentropy": ContrastivePipeline,
    "siglip2_sigmoid": ContrastivePipeline,
    "siglip2_crossentropy": ContrastivePipeline,
    # Two-stage models
    "siglip_twoStage": TwoStagePipeline,
    "siglip2_twoStage": TwoStagePipeline,
    "siglip_twoStage_fewshot": TwoStagePipeline,
    "siglip2_twoStage_fewshot": TwoStagePipeline,
    # FuseLIP-MLP models
    "fuselip_mlp_image": FuseLIPMLPPipeline,
    "fuselip_mlp_text": FuseLIPMLPPipeline,
    "fuselip_mlp_multimodal": FuseLIPMLPPipeline,
    # Eval-only (any model)
    "eval_only": EvalOnlyPipeline,
}


def get_pipeline(model_name: str, config: PipelineConfig) -> BasePipeline:
    """Factory function to get pipeline instance."""
    # Check for eval-only mode first
    if config.global_config.get("eval_only", False):
        return PIPELINE_REGISTRY["eval_only"](config)

    # Direct lookup
    if model_name in PIPELINE_REGISTRY:
        pipeline_cls = PIPELINE_REGISTRY[model_name]
        return pipeline_cls(config)

    raise ValueError(f"No pipeline found for model: {model_name}")
