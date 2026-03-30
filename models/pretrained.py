from sentence_transformers import SentenceTransformer, util
try:
    from multilingual_clip import pt_multilingual_clip  # type: ignore
except ImportError:
    pt_multilingual_clip = None  # type: ignore
from transformers import AutoModel, AutoProcessor
import transformers
import open_clip
import torch

try:
    import clip  # type: ignore
except ImportError:
    clip = None  # type: ignore

def load_siglip_pretrained(model_name: str, device: str = 'cpu'):
    model = AutoModel.from_pretrained(model_name, low_cpu_mem_usage=True)
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor

# mclip
def mclip_model_loader(model_name: str, device: str = 'cpu'):
    # model_name = 'M-CLIP/XLM-Roberta-Large-Vit-B-32'

    if pt_multilingual_clip is None:
        raise ImportError("multilingual_clip is not installed. Install it with: pip install multilingual-clip")
    
    # Type ignore due to third-party library type stubs issue
    text_model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(model_name)  # type: ignore
    text_model = text_model.to(device)  # type: ignore
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    if clip is None:
        raise ImportError("clip is not installed. Install it with: pip install git+https://github.com/openai/CLIP.git")
    model, _, preprocess = clip.load("ViT-B/32", device=device)

    return model, tokenizer, preprocess, text_model


# open_clip model 
def openclip_model_loader(model_name: str, device: str = 'cpu'):
    model_name = 'ViT-B-32'
    pretrained = 'laion2b_s34b_b79k'
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)

    model.to(device)

    return model, tokenizer, preprocess


# sentence transformer model
def sentence_transformer_model_loader(device: str = 'cpu'):
    img_model = SentenceTransformer('clip-ViT-B-32', device=device)
    text_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1', device=device)

    return img_model, text_model


def fuselip_model_loader(device: str = 'cpu'):
    try:
        from fuse_clip.fuse_clip_utils import load_model  # type: ignore
    except ImportError:
        raise ImportError(
            "fuse_clip is not installed. Install it before running fuselip_mlp experiments."
        )
    model, image_processor, text_tokenizer = load_model(
        "chs20/FuseLIP-S-CC3M-MM",
        device=device
    )

    model.eval()

    return model, image_processor, text_tokenizer
