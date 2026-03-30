from PIL import Image, ImageFile
from torch.utils.data import Dataset
import os
import pandas as pd
import torch

def generate_fewshot_dataframe(df, label_column='labels', num_shots=16, repeat=True):
    """
    Crea un subset de few-shot desde un DataFrame.

    Args:
        df (pd.DataFrame): DataFrame con columnas, incluyendo una de etiquetas.
        label_column (str): Nombre de la columna que contiene las clases.
        num_shots (int): Número de instancias por clase.
        repeat (bool): Si True, repite instancias si hay menos que `num_shots`.

    Returns:
        pd.DataFrame: Subset del dataframe con `num_shots` por clase.
    """
    fewshot_data = []

    for label in df[label_column].unique():
        class_data = df[df[label_column] == label]
        if len(class_data) >= num_shots:
            sampled = class_data.sample(n=num_shots, random_state=42)
        else:
            if repeat:
                sampled = class_data.sample(n=num_shots, replace=True, random_state=42)
            else:
                sampled = class_data
        fewshot_data.append(sampled)

    return pd.concat(fewshot_data).reset_index(drop=True)


class ImageTextTwoStage(Dataset):
    def __init__(self, data, processor):
        """
        Args:
            image_dir (str): Directory containing images
            text_file (str): Path to text file containing image descriptions
            processor: SigLIP2 processor for handling images and texts
        """
        self.data = data
        self.processor = processor
        label_names = sorted(self.data['labels'].unique())
        label2id = {label: idx for idx, label in enumerate(label_names)}
        self.label2id = label2id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = row['image_path']
        description = row['post_text']
        label = row['labels']

        # Load and process image
        image = Image.open(image_path.replace("\\", "/")).convert('RGB')

        if not isinstance(description, str):
            description = ""

        label = self.label2id[label]  

        # Process image and text using the processor
        inputs = self.processor(
            text=description,
            images=image,
            padding="max_length",  # Important: use max_length padding
            max_length=64,
            truncation=True,
            return_tensors="pt"
        )

        # Remove batch dimension (from [1, ...] to [...])
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        return inputs, torch.tensor(label, dtype=torch.long)