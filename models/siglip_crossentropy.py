import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor
import torch.nn as nn
import torch

class SigLIPCrossentropy(nn.Module):
    def __init__(self, model_name=str, dropout_prob=0.4, temperature=0.07):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(model_name, low_cpu_mem_usage=True)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.temperature = temperature
        self.patch_size = 16  # patch size que usa el modelo Naflex/SigLIP2

    def forward(
        self,
        input_ids,
        pixel_values,
        attention_mask=None,
        return_loss=False,
        labels=None,
    ):

        outputs = self.base_model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
        )

        image_embeds = self.dropout(outputs.image_embeds)
        text_embeds = self.dropout(outputs.text_embeds)

        # Normalizar embeddings para similitud coseno
        image_embeds = F.normalize(image_embeds, p=2, dim=-1)
        text_embeds = F.normalize(text_embeds, p=2, dim=-1)

        # Calcular logits de similitud
        logits_per_image = image_embeds @ text_embeds.t() / self.temperature
        logits_per_text = logits_per_image.t()

        if return_loss:
            batch_size = image_embeds.size(0)
            labels = torch.arange(batch_size, device=image_embeds.device)
            loss_img = F.cross_entropy(logits_per_image, labels)
            loss_txt = F.cross_entropy(logits_per_text, labels)
            loss = (loss_img + loss_txt) / 2
            return {
                "loss": loss,
                "logits_per_image": logits_per_image,
                "logits_per_text": logits_per_text,
            }
        else:
            return {
                "logits_per_image": logits_per_image,
                "logits_per_text": logits_per_text,
            }

class SigLIP2Crossentropy(nn.Module):
    def __init__(self, model_name="google/siglip2-base-patch16-naflex", dropout_prob=0.4, temperature=0.07):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(model_name, low_cpu_mem_usage=True)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.temperature = temperature
        self.patch_size = 16  # patch size que usa el modelo Naflex/SigLIP2

    def forward(
        self,
        input_ids,
        pixel_values,
        attention_mask=None,
        pixel_attention_mask=None,
        spatial_shapes=None,
        return_loss=False,
        labels=None,
    ):

        outputs = self.base_model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            pixel_attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
        )

        image_embeds = self.dropout(outputs.image_embeds)
        text_embeds = self.dropout(outputs.text_embeds)

        # Normalizar embeddings para similitud coseno
        image_embeds = F.normalize(image_embeds, p=2, dim=-1)
        text_embeds = F.normalize(text_embeds, p=2, dim=-1)

        # Calcular logits de similitud
        logits_per_image = image_embeds @ text_embeds.t() / self.temperature
        logits_per_text = logits_per_image.t()

        if return_loss:
            batch_size = image_embeds.size(0)
            labels = torch.arange(batch_size, device=image_embeds.device) # for CLIP 
            loss_img = F.cross_entropy(logits_per_image, labels)
            loss_txt = F.cross_entropy(logits_per_text, labels)
            loss = (loss_img + loss_txt) / 2
            return {
                "loss": loss,
                "logits_per_image": logits_per_image,
                "logits_per_text": logits_per_text,
            }
        else:
            return {
                "logits_per_image": logits_per_image,
                "logits_per_text": logits_per_text,
            }