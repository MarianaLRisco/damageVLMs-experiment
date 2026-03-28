import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from contextlib import nullcontext
import random
from collections import defaultdict
import pandas as pd
import random


# LORA
def freeze_all_except_layernorm(model):
    for name, param in model.named_parameters():
        # Trainable LayerNorm
        if 'layernorm' in name.lower():
            param.requires_grad = True
        else:
            param.requires_grad = False

class SigLIPLinearClassifier(nn.Module):
    def __init__(self, base_model, processor, classnames, temperature=0.07, description_map=None, device='cpu'):
        super().__init__()
        self.base_model = base_model
        self.visual = base_model.base_model.vision_model
        self.text = base_model.base_model.text_model
        self.processor = processor
        self.temperature = temperature

        self.classnames = classnames
        self.description_map = description_map
        self.cat2id = {cat: i for i, cat in enumerate(classnames)}

        self._init_classifier(classnames, processor, device)

    def _init_classifier(self, classnames, processor, device='cpu'):
        # prompts = [template.format(cls) for cls in classnames]
        prompts = [self.description_map[cls] for cls in classnames]
        with torch.no_grad():
            inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
            text_embeds = self.text(**inputs).last_hidden_state[:, 0, :]  # CLS token
            text_embeds = F.normalize(text_embeds, dim=-1)
        self.classifier = nn.Parameter(text_embeds, requires_grad=True)

    def forward(self, pixel_values, no_grad_backbone=True):
        context = torch.no_grad if no_grad_backbone else nullcontext #freeze base model
        with context():
            image_embeds = self.visual(pixel_values).pooler_output #image_embeds = self.visual(pixel_values).last_hidden_state[:, 0, :]  # CLS token
            image_embeds = F.normalize(image_embeds, dim=-1)

        classifier = F.normalize(self.classifier, dim=-1)
        logits = (1 / self.temperature) * (image_embeds @ classifier.t())
        return logits
    
    @torch.no_grad()
    def infer(self, x: Tensor, categories: list[str], template: dict = None, compute_classifier_once: bool = True, processor=None) :
        prompt_map = template if template is not None else self.description_map

        if compute_classifier_once and hasattr(self, "inference_classifier"):
            classifier = self.inference_classifier
        else:
            # Construye los prompts faltantes con el mapa proporcionado
            missing_classnames = [cat for cat in categories if cat not in self.cat2id]
            prompts = [prompt_map[cat] for cat in missing_classnames]

            if prompts:
                inputs = processor(text=prompts, return_tensors="pt", padding=True).to(x.device)
                device_type = "cuda" if str(x.device.type).startswith("cuda") else "cpu"
                with torch.amp.autocast(device_type=device_type, enabled=(device_type == "cuda")):
                    new_embeddings = self.text(**inputs).last_hidden_state[:, 0, :]
                new_embeddings = F.normalize(new_embeddings, dim=-1)
                cat2new = {cat: new for cat, new in zip(missing_classnames, new_embeddings)}
            else:
                cat2new = {}

            # Recuperar embeddings antiguos del clasificador existente
            cat2known = {cat: self.classifier[self.cat2id[cat], :] for cat in categories if cat in self.cat2id}

            # Combinar
            classifier = [cat2new[cat] if cat in cat2new else cat2known[cat] for cat in categories]
            classifier = torch.stack(classifier).to(x.device)
            classifier = F.normalize(classifier, dim=-1)


        # Embeddings visuales
        image_embeds = self.visual(x).pooler_output
        image_embeds = F.normalize(image_embeds, dim=-1)

        logits = (1 / self.temperature) * (image_embeds @ classifier.t())
        return logits

class SigLIP2LinearClassifier(nn.Module):
    def __init__(self, base_model, processor, classnames, temperature=0.07, description_map=None, device='cpu'):
        super().__init__()
        self.base_model = base_model
        self.visual = base_model.base_model.vision_model
        self.text = base_model.base_model.text_model
        self.processor = processor
        self.temperature = temperature

        self.classnames = classnames
        self.description_map = description_map
        self.patch_size = 16
        self.cat2id = {cat: i for i, cat in enumerate(classnames)}

        self._init_classifier(classnames, processor, device)

    def _init_classifier(self, classnames, processor, device='cpu'):
        # prompts = [template.format(cls) for cls in classnames]
        prompts = [self.description_map[cls] for cls in classnames]
        with torch.no_grad():
            inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
            text_embeds = self.text(**inputs).last_hidden_state[:, 0, :]  # CLS token
            text_embeds = F.normalize(text_embeds, dim=-1)
        self.classifier = nn.Parameter(text_embeds, requires_grad=True)

    def forward(self, pixel_values,
        attention_mask=None,
        spatial_shapes=None,
        no_grad_backbone=True):
        
        context = torch.no_grad if no_grad_backbone else nullcontext

        with context():
            outputs = self.visual(
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                spatial_shapes=spatial_shapes,
                return_dict=True
            )
            image_embeds = outputs.pooler_output
            image_embeds = F.normalize(image_embeds, dim=-1)

        classifier = F.normalize(self.classifier, dim=-1)
        logits = (1 / self.temperature) * (image_embeds @ classifier.t())
        return logits
        
    @torch.no_grad()
    def infer(self, x: Tensor, categories: list[str], template: dict = None, attention_mask=None, spatial_shapes=None, compute_classifier_once: bool = True, processor=None) :
        prompt_map = template if template is not None else self.description_map

        if compute_classifier_once and hasattr(self, "inference_classifier"):
            classifier = self.inference_classifier
        else:
            # Construye los prompts faltantes con el mapa proporcionado
            missing_classnames = [cat for cat in categories if cat not in self.cat2id]
            prompts = [prompt_map[cat] for cat in missing_classnames]

            if prompts:
                inputs = processor(text=prompts, return_tensors="pt", padding=True).to(x.device)
                device_type = "cuda" if str(x.device.type).startswith("cuda") else "cpu"
                with torch.amp.autocast(device_type=device_type, enabled=(device_type == "cuda")):
                    new_embeddings = self.text(**inputs).last_hidden_state[:, 0, :]
                new_embeddings = F.normalize(new_embeddings, dim=-1)
                cat2new = {cat: new for cat, new in zip(missing_classnames, new_embeddings)}
            else:
                cat2new = {}

            # Recuperar embeddings antiguos del clasificador existente
            cat2known = {cat: self.classifier[self.cat2id[cat], :] for cat in categories if cat in self.cat2id}

            # Combinar
            classifier = [cat2new[cat] if cat in cat2new else cat2known[cat] for cat in categories]
            classifier = torch.stack(classifier).to(x.device)
            classifier = F.normalize(classifier, dim=-1)
        
        # Embeddings visuales
        outputs = self.visual(
            pixel_values=x,
            attention_mask=attention_mask,  
            spatial_shapes=spatial_shapes,
            return_dict=True
        )
        image_embeds = outputs.pooler_output
        # image_embeds = self.visual(x).pooler_output
        image_embeds = F.normalize(image_embeds, dim=-1)

        logits = (1 / self.temperature) * (image_embeds @ classifier.t())
        return logits