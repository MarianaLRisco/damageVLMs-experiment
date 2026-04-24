"""Image-only dataset utilities for ConvNet training pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Sequence

from PIL import Image, ImageFile
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms


ImageFile.LOAD_TRUNCATED_IMAGES = True


def build_image_transform(image_size: int, is_train: bool = False):
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

    def __init__(self, data: pd.DataFrame, classes: Sequence[str], transform=None):
        self.data = data
        self.classes = list(classes)
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.transform = transform or build_image_transform(224)

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
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)
        target = torch.tensor(self._label_to_index(row["labels"]), dtype=torch.long)
        return image_tensor, target
