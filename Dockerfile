FROM python:3.11

USER root

# Directorio de trabajo
WORKDIR /workspaces/benchmarkingMultimodal

# Dependencias básicas
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    bash \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements.txt .

# Instalar dependencias Python
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Instalar PyTorch con CUDA 11.8
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Instalar OpenCLIP
RUN pip install open_clip_torch

# Instalar CLIP original (OpenAI)
RUN pip install git+https://github.com/openai/CLIP.git

# Instalar Multilingual-CLIP con pesos
RUN git clone https://github.com/FreddeFrallan/Multilingual-CLIP.git /tmp/Multilingual-CLIP
RUN bash /tmp/Multilingual-CLIP/legacy_get-weights.sh
RUN pip install -e /tmp/Multilingual-CLIP

# Comando por defecto
CMD ["/bin/bash"]
