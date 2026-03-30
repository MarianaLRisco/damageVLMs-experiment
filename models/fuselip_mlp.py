import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class FuseLIPMLPClassifier(nn.Module):
    """
    FuseLIP backbone (frozen) + MLP classification head.

    Modes:
        "image"      — classifies using image embeddings only
        "text"       — classifies using text embeddings only
        "multimodal" — classifies using concatenated image + text embeddings
    """

    VALID_MODES = ("image", "text", "multimodal")

    def __init__(self, backbone, num_classes: int, mode: str, embed_dim: int = 512):
        super().__init__()
        assert mode in self.VALID_MODES, f"mode must be one of {self.VALID_MODES}"
        self.backbone = backbone
        self.mode = mode

        input_dim = embed_dim * 2 if mode == "multimodal" else embed_dim
        self.mlp_head = MLPHead(input_dim, num_classes)

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def _extract(self, pixel_values, input_ids=None, attention_mask=None):
        image_embeds = self.backbone.encode_image(pixel_values, normalize=True)
        if isinstance(image_embeds, dict):
            image_embeds = image_embeds['fts']
        text_embeds = None
        if input_ids is not None:
            text_embeds = self.backbone.encode_text(input_ids, normalize=True)
            if isinstance(text_embeds, dict):
                text_embeds = text_embeds['fts']
        return image_embeds, text_embeds

    def forward(self, pixel_values, input_ids=None, attention_mask=None):
        image_embeds, text_embeds = self._extract(pixel_values, input_ids, attention_mask)

        if self.mode == "image":
            features = image_embeds
        elif self.mode == "text":
            features = text_embeds
        else:  # multimodal
            # Handle None text_embeds for multimodal mode
            if text_embeds is None:
                raise ValueError("text_embeds cannot be None in multimodal mode")
            features = torch.cat([image_embeds, text_embeds], dim=-1)

        return self.mlp_head(features)
