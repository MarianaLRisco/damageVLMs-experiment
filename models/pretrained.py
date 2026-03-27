from sentence_transformers import SentenceTransformer, util
from multilingual_clip import pt_multilingual_clip
from transformers import AutoModel, AutoProcessor
import transformers
import open_clip
import torch
import clip

def load_siglip_pretrained(model_name=str):
    model = AutoModel.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor

# mclip
def mclip_model_loader(model_name: str, device: str = 'cuda'):
    # model_name = 'M-CLIP/XLM-Roberta-Large-Vit-B-32'

    text_model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(model_name)
    tokenizer  = transformers.AutoTokenizer.from_pretrained(model_name)

    model, preprocess = clip.load("ViT-B/32", device=device)

    return model, tokenizer, preprocess, text_model


# open_clip model 
def openclip_model_loader(model_name: str, device: str = 'cuda'):
    model_name = 'ViT-B-32'
    pretrained = 'laion2b_s34b_b79k'
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)

    model.to(device)

    return model, tokenizer, preprocess

# sentence transformer model
def sentence_transformer_model_loader(device: str = 'cuda'):
    img_model = SentenceTransformer('clip-ViT-B-32')
    text_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')

    return img_model, text_model

