# Setup en Mac (Apple Silicon M1/M2/M3/M4)

## Requisitos previos

- Python 3.11: `brew install python@3.11`
- Git: `brew install git`

---

## 1. Crear entorno virtual

```bash
cd damageVLMs-experiment
python3.11 -m venv venv
```

---

## 2. Activar el entorno

```bash
source venv/bin/activate
```

Verás:
```
(venv) tu_usuario@mac %
```

---

## 3. Actualizar herramientas básicas

```bash
pip install --upgrade pip setuptools wheel
```

---

## 4. Instalar PyTorch (primero)

```bash
pip install torch torchvision torchaudio
```

Verificar que MPS funciona:
```python
import torch
print(torch.backends.mps.is_available())  # debe dar True
```

---

## 5. Configurar variables de entorno

Crear archivo `.env` en la raíz del proyecto:

```bash
touch .env
```

Contenido del `.env`:
```bash
# HuggingFace — token para descargar modelos privados o con acceso restringido
# Obtener en: https://huggingface.co/settings/tokens
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxx

# Carpeta donde se guardan los modelos descargados (evita re-descargar)
# Por defecto va a ~/.cache/huggingface — cambiarlo si el disco del sistema es pequeño
HF_HOME=/ruta/a/tu/disco/externo/.cache/huggingface

# Evita warnings de tokenizers con múltiples workers
TOKENIZERS_PARALLELISM=false

# Controla cuánta RAM GPU (memoria unificada) usa MPS antes de volcar a RAM del sistema
# 0.0 = sin límite (recomendado en M4 con 16GB+)
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
```

Cargar las variables al activar el entorno (una sola vez):

```bash
echo 'export $(cat .env | xargs)' >> venv/bin/activate
```

> `.env` ya está en `.gitignore` — nunca se sube al repositorio.

---

## 6. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## 7. Ajustar el device en el config

El archivo `configs/disaster.yaml` tiene `device: "cuda"` por defecto.
El código detecta automáticamente MPS si no hay CUDA disponible, pero para
ser explícito puedes cambiarlo:

```yaml
device: "mps"
```

---

## 8. Uso diario

```bash
# Activar (incluye las variables de entorno)
source venv/bin/activate

# Entrenar
python train.py --config configs/disaster.yaml

# Desactivar
deactivate
```

---

## 9. Si falla alguna librería

| Error | Solución |
|---|---|
| `sentencepiece` no compila | `brew install cmake` y reintentar |
| `protobuf` versión incompatible | `pip install protobuf==3.20.*` |
| MPS out of memory | Reducir `batch_size` en el YAML o bajar `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5` |
| Modelos no descargan | Verificar `HF_TOKEN` en `.env` |

---

## 10. Estructura del proyecto

```
damageVLMs-experiment/
├── venv/               ← NO subir a git
├── .env                ← NO subir a git (tokens)
├── requirements.txt
├── train.py
├── configs/
├── engine/
├── models/
├── dataset/
└── outputs/
```

---

## Resumen

| | Mac (M4) | Windows (NVIDIA) |
|---|---|---|
| Activar venv | `source venv/bin/activate` | `venv\Scripts\activate` |
| GPU backend | MPS | CUDA |
| PyTorch install | `pip install torch` | `pip install torch --index-url .../cu118` |
| Verificar GPU | `torch.backends.mps.is_available()` | `torch.cuda.is_available()` |
| `triton` | No usar | No usar |
