from PIL import Image, ImageFile
from torch.utils.data import Dataset
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
import os


class ImageTextFuseLIP(Dataset):
    def __init__(self, data, image_processor, text_tokenizer):
        """
        Args:
            data: DataFrame with columns: image_path, post_text, labels
            image_processor: FuseLIP image processor
            text_tokenizer: FuseLIP text tokenizer
        """
        self.data = data.reset_index(drop=True)
        self.image_processor = image_processor
        self.text_tokenizer = text_tokenizer

        label_names = sorted(self.data["labels"].unique())
        self.label2id = {label: idx for idx, label in enumerate(label_names)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = row["image_path"]
        text = row["post_text"] if isinstance(row["post_text"], str) else ""
        label = self.label2id[row["labels"]]

        image = Image.open(image_path.replace("\\", "/")).convert("RGB")
        pixel_values = self.image_processor(image)

        input_ids = self.text_tokenizer(text).squeeze(0)

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "labels": torch.tensor(label, dtype=torch.long),
        }
