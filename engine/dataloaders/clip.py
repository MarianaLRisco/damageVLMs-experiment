from PIL import Image, ImageFile
from torch.utils.data import Dataset
import os


class ImageTextClip(Dataset):
    def __init__(self, data, tokenizer, preprocess):
        self.data = data
        self.tokenizer = tokenizer
        self.preprocess = preprocess

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = row['image_path']
        description = row['post_text']

        image = Image.open(image_path.replace("\\", "/")).convert('RGB')
        image_tensor = self.preprocess(image)
        tokens = self.tokenizer(description, return_tensors='pt', padding='max_length',
                                truncation=True, max_length=77)

        return image_tensor, tokens
