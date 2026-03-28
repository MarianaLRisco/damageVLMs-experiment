from PIL import Image, ImageFile
from torch.utils.data import Dataset
import os

class ImageTextMclip(Dataset):
    def __init__(self, data, tokenizer,preprocess):
        """
        Args:
            image_dir (str): Directory containing images
            text_file (str): Path to text file containing image descriptions
            processor: SigLIP2 processor for handling images and texts
        """
        self.data = data
        self.tokenizer = tokenizer
        self.preprocess = preprocess
        # self.image_text_pairs = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = row['image_path']
        description = row['post_text']

        # Load and process image
        image = Image.open(os.path.join(image_path).replace("\\", "/")).convert('RGB')

        image_tensor  = self.preprocess(image)
        tokens = self.tokenizer(description, return_tensors='pt', padding='max_length',
                                truncation=True, max_length=77)

        return image_tensor, tokens