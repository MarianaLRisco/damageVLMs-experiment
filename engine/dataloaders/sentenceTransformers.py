from PIL import Image, ImageFile
from torch.utils.data import Dataset
import os

class ImageTextSentenceTransformers(Dataset):
    def __init__(self, data, processor):
        """
        Args:
            data (pd.DataFrame): DataFrame with 'image_path', 'post_text' and 'label'
            processor: CLIPProcessor or similar
            label2id (dict): Diccionario que mapea clases de texto a IDs (ej. {'flood': 0, ...})
        """
        self.data = data.reset_index(drop=True)
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = row['image_path']
        text = row['post_text']

        # Carga de imagen
        image = Image.open(image_path.replace("\\", "/")).convert('RGB')

        return image, text

def custom_collate_fn(batch):
    images, texts = zip(*batch)  # batch es una lista de tuplas (img, txt)
    return list(images), list(texts)
