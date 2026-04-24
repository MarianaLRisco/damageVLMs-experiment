"""Image-only dataset utilities for ConvNet training pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Final, List, Mapping, Sequence

from PIL import Image, ImageFile, ImageOps
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms


ImageFile.LOAD_TRUNCATED_IMAGES = True


_SUPPORTED_PREPROCESS_MODES: Final[frozenset[str]] = frozenset({"legacy_rgb", "rgbsafe"})


def _normalize_to_rgb_before_model(image: Image.Image) -> Image.Image:
    """Convert input image to RGB using a safe, explicit alpha/palette path.

    For palette + transparency inputs we first pass through RGBA and composite
    against a white background. This avoids PIL palette/transparency conversion
    pitfalls and keeps RGB output deterministic.
    """
    if image.mode == "RGB":
        return image.copy()

    has_alpha = "A" in image.getbands()
    has_transparency = "transparency" in image.info

    if image.mode in {"P", "PA", "RGBA", "LA"} or has_alpha or has_transparency:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")

    return image.convert("RGB")


def _prepare_image_for_convnet(image: Image.Image, mode: str) -> Image.Image:
    """Prepare a single PIL image according to the selected preprocessing mode."""
    if mode == "legacy_rgb":
        return image.convert("RGB")

    if mode != "rgbsafe":
        supported = ", ".join(sorted(_SUPPORTED_PREPROCESS_MODES))
        raise ValueError(f"Unsupported preprocess_mode={mode!r}. Supported values: {supported}")

    normalized = ImageOps.exif_transpose(image)
    return _normalize_to_rgb_before_model(normalized)


def _build_train_augmentations(augmentation_cfg: Mapping[str, Any] | None) -> List[Callable]:
    """Build optional train-time augmentations from config.

    Accepts a small, explicit set of supported transforms:
    - random_rotation: numeric degrees or mapping with "degrees"
    - color_jitter: mapping with torchvision `ColorJitter` args
    """
    augmentations: List[Callable] = []

    if not augmentation_cfg:
        return augmentations

    random_rotation = augmentation_cfg.get("random_rotation")
    if random_rotation is not None:
        if isinstance(random_rotation, Mapping):
            degrees = random_rotation.get("degrees")
        else:
            degrees = random_rotation

        if isinstance(degrees, (int, float)):
            if degrees < 0:
                raise ValueError("random_rotation degrees must be >= 0")
            augmentations.append(transforms.RandomRotation(degrees=degrees))
        else:
            raise ValueError("train_augmentations.random_rotation must be a number or a mapping")

    color_jitter_cfg = augmentation_cfg.get("color_jitter")
    if color_jitter_cfg is not None:
        if not isinstance(color_jitter_cfg, Mapping):
            raise ValueError("train_augmentations.color_jitter must be a mapping")

        augmentations.append(
            transforms.ColorJitter(
                brightness=color_jitter_cfg.get("brightness", 0.0),
                contrast=color_jitter_cfg.get("contrast", 0.0),
                saturation=color_jitter_cfg.get("saturation", 0.0),
                hue=color_jitter_cfg.get("hue", 0.0),
            )
        )

    return augmentations


def build_image_transform(
    image_size: int,
    is_train: bool = False,
    augmentation_cfg: Mapping[str, Any] | None = None,
) -> transforms.Compose:
    """Build torchvision transform pipeline for image-only models.

    Keep augmentation minimal by default for stable local experimentation.
    """
    transforms_list: List[Callable] = []

    if is_train:
        transforms_list.extend(
            [
                transforms.RandomResizedCrop(image_size, antialias=True),
                transforms.RandomHorizontalFlip(),
            ]
        )
        transforms_list.extend(_build_train_augmentations(augmentation_cfg))
    else:
        transforms_list.append(transforms.Resize((image_size, image_size), antialias=True))

    transforms_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return transforms.Compose(transforms_list)


class ImageOnlyDataset(Dataset):
    """Simple image dataset returning (image_tensor, target) pairs."""

    def __init__(
        self,
        data: pd.DataFrame,
        classes: Sequence[str],
        transform=None,
        preprocess_mode: str = "legacy_rgb",
    ):
        self.data = data
        self.classes = list(classes)
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.transform = transform or build_image_transform(224)
        if preprocess_mode not in _SUPPORTED_PREPROCESS_MODES:
            supported = ", ".join(sorted(_SUPPORTED_PREPROCESS_MODES))
            raise ValueError(
                f"Unsupported preprocess_mode={preprocess_mode!r}. Supported values: {supported}"
            )
        self.preprocess_mode = preprocess_mode

    def __len__(self):
        return len(self.data)

    def _label_to_index(self, raw_label):
        if isinstance(raw_label, (int,)):
            return int(raw_label)

        if isinstance(raw_label, torch.Tensor):
            raw_value = raw_label.item()
            return int(raw_value)

        if isinstance(raw_label, str):
            if raw_label in self.class_to_idx:
                return self.class_to_idx[raw_label]
            if raw_label.isdigit():
                return int(raw_label)

        try:
            numeric = int(raw_label)
            return numeric
        except (TypeError, ValueError):
            raise ValueError(f"Unsupported label value: {raw_label!r}")

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = Path(row["image_path"])
        with Image.open(image_path) as image:
            image = _prepare_image_for_convnet(image, self.preprocess_mode)
        image_tensor = self.transform(image)
        target = torch.tensor(self._label_to_index(row["labels"]), dtype=torch.long)
        return image_tensor, target
