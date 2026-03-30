"""
Wrapper simplificado para cargar modelos FuseLIP desde HuggingFace.
Evita las dependencias complejas de training local del repo original.

CRÍTICO: Este módulo debe importarse ANTES de cualquier uso de open_clip
para asegurar que se use la versión de fuselip_repo/src/open_clip/
"""

import sys
import os

# CRÍTICO: Agregar fuselip_repo/src al PYTHONPATH ANTES de cualquier import
# Esto asegura que se use la versión correcta de open_clip compatible con FuseLIP
fuselip_src = os.path.join(os.path.dirname(__file__), 'fuselip_repo', 'src')
if fuselip_src not in sys.path:
    sys.path.insert(0, fuselip_src)

import torch
import logging

def load_fuselip_from_huggingface(model_id: str = "chs20/FuseLIP-S-CC3M-MM", device: str = 'cpu'):
    """
    Carga un modelo FuseLIP desde HuggingFace Hub.

    Args:
        model_id: ID del modelo en HuggingFace (ej: "chs20/FuseLIP-S-CC3M-MM")
        device: Device donde cargar el modelo ('cpu', 'cuda', 'mps')

    Returns:
        tuple: (model, image_processor, tokenizer)

    Example:
        >>> model, processor, tokenizer = load_fuselip_from_huggingface("chs20/FuseLIP-S-CC3M-MM")
        >>> model.eval()
    """
    try:
        # Solo importamos lo necesario para HuggingFace
        from fuse_clip.fuse_clip_hub import FuseLIP
        from fuse_clip.fuse_clip_preprocess import get_fuse_clip_image_preprocess

        logging.info(f"Cargando FuseLIP desde Hugging Face Hub: {model_id}")

        # Cargar modelo desde HuggingFace
        model = FuseLIP.from_pretrained(model_id, device=device)
        model.eval()

        # Obtener processor y tokenizer
        image_processor = get_fuse_clip_image_preprocess(train=False)
        tokenizer = model.text_tokenizer

        logging.info(f"✅ FuseLIP cargado exitosamente en {device}")

        return model, image_processor, tokenizer

    except Exception as e:
        logging.error(f"❌ Error cargando FuseLIP: {e}")
        raise


# Para compatibilidad con el código existente
def load_model(model_id: str, device: str = 'cpu', ckpt_name: str = "epoch_final.pt"):
    """
    Wrapper compatible con la firma de load_model del repo original.
    Solo soporta modelos HuggingFace (model_id que empiezan con "chs20/")
    """
    if not model_id.startswith("chs20/"):
        raise ValueError(
            f"Solo se soportan modelos HuggingFace (chs20/*). "
            f"Modelos locales requieren el repo completo de FuseLIP con training setup."
        )

    return load_fuselip_from_huggingface(model_id, device)
