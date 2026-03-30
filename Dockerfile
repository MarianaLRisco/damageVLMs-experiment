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

# Instalar CLIP original (OpenAI)
RUN pip install git+https://github.com/openai/CLIP.git

# NOTA: NO instalamos open_clip_torch porque está en conflicto con FuseLIP
# FuseLIP incluye su propia versión de open_clip en fuselip_repo/src/open_clip/

# Instalar Multilingual-CLIP con pesos
RUN git clone https://github.com/FreddeFrallan/Multilingual-CLIP.git /tmp/Multilingual-CLIP
RUN bash /tmp/Multilingual-CLIP/legacy_get-weights.sh
RUN pip install -e /tmp/Multilingual-CLIP

# Clonar FuseLIP repo (necesario para modelos fuselip_mlp)
RUN git clone https://github.com/chs20/fuselip.git /workspaces/benchmarkingMultimodal/fuselip_repo

# Instalar dependencias específicas de FuseLIP
RUN pip install omegaconf datasets einops timm k-diffusion torchvision

# Comando por defecto
CMD ["/bin/bash"]
