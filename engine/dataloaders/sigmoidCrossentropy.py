from PIL import Image, ImageFile
import os

class ImageTextDataset(Dataset):
    def __init__(self, data, processor):
        """
        Args:
            image_dir (str): Directory containing images
            text_file (str): Path to text file containing image descriptions
            processor: SigLIP2 processor for handling images and texts
        """
        self.data = data
        self.processor = processor
        # self.image_text_pairs = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = row['image_path']
        description = row['post_text']

        # Load and process image
        image = Image.open(os.path.join(image_path).replace("\\", "/")).convert('RGB')

        if not isinstance(description, str):
            description = ""
        else:
            description = str(description)

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

        return inputs 