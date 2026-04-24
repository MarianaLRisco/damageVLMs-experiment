"""Pipeline registry for model-specific training pipelines."""

from typing import Dict, Type

from pipelines.base import BasePipeline, PipelineConfig


PIPELINE_MODULES: Dict[str, tuple[str, str]] = {
    # Contrastive models
    "siglip_sigmoid": ("pipelines.contrastive", "ContrastivePipeline"),
    "siglip_crossentropy": ("pipelines.contrastive", "ContrastivePipeline"),
    "siglip2_sigmoid": ("pipelines.contrastive", "ContrastivePipeline"),
    "siglip2_crossentropy": ("pipelines.contrastive", "ContrastivePipeline"),
    # Two-stage models
    "siglip_twoStage": ("pipelines.two_stage", "TwoStagePipeline"),
    "siglip2_twoStage": ("pipelines.two_stage", "TwoStagePipeline"),
    "siglip_twoStage_fewshot": ("pipelines.two_stage", "TwoStagePipeline"),
    "siglip2_twoStage_fewshot": ("pipelines.two_stage", "TwoStagePipeline"),
    # FuseLIP-MLP models
    "fuselip_mlp_image": ("pipelines.fuselip_mlp", "FuseLIPMLPPipeline"),
    "fuselip_mlp_text": ("pipelines.fuselip_mlp", "FuseLIPMLPPipeline"),
    "fuselip_mlp_multimodal": ("pipelines.fuselip_mlp", "FuseLIPMLPPipeline"),
    # ConvNet baselines
    "resnet50": ("pipelines.convnets", "ConvNetPipeline"),
    "efficientnet_b0": ("pipelines.convnets", "ConvNetPipeline"),
    "vgg16": ("pipelines.convnets", "ConvNetPipeline"),
    # Eval-only (any model)
    "eval_only": ("pipelines.eval_only", "EvalOnlyPipeline"),
}


def _load_pipeline_class(model_name: str) -> Type[BasePipeline]:
    """Load pipeline class lazily to avoid cross-flow dependency conflicts."""
    if model_name not in PIPELINE_MODULES:
        raise ValueError(f"No pipeline found for model: {model_name}")

    module_name, class_name = PIPELINE_MODULES[model_name]
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)


def get_pipeline(model_name: str, config: PipelineConfig) -> BasePipeline:
    """Factory function to get pipeline instance."""
    # Check for eval-only mode first
    if config.global_config.get("eval_only", False):
        return _load_pipeline_class("eval_only")(config)

    # Direct lookup
    if model_name in PIPELINE_MODULES:
        pipeline_cls = _load_pipeline_class(model_name)
        return pipeline_cls(config)

    raise ValueError(f"No pipeline found for model: {model_name}")
