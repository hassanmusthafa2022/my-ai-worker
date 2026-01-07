# 1. Base Image: Python 3.10 with CUDA 11.8 (GPU Support)
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# 2. System Dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    wget \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# 3. Work Directory
WORKDIR /app

# 4. Install ALL Python Libraries (for all models)
# LatentSync has specific requirements, so we will install them from its own file
RUN pip install --no-cache-dir \
    runpod 'numpy<2' scipy librosa soundfile munch pyyaml huggingface-hub \
    einops transformers==4.33.0 torchcrepe vector-quantize-pytorch \
    descript-audio-codec opencv-python tqdm imageio moviepy

# 5. --- CLONE OFFICIAL REPOS DIRECTLY ---
RUN git clone https://github.com/Plachta/Seed-VC.git seed-vc
RUN git clone https://github.com/ByteDance/LatentSync.git latent-sync

# Install LatentSync specific requirements
RUN cd latent-sync && pip install -r requirements.txt && cd ..

# 6. --- PRE-DOWNLOAD ALL MODEL WEIGHTS ---
# Seed-VC Weights
RUN wget https://huggingface.co/Plachta/Seed-VC/resolve/main/DiT_uvit_tat_xlsr_ema.pth -P ./seed-vc/
RUN wget https://huggingface.co/Plachta/Seed-VC/resolve/main/config_dit_mel_seed_uvit_xlsr_tiny.yml -P ./seed-vc/
RUN wget https://github.com/Plachta/Seed-VC/releases/download/v0.0.1/campplus_cn_common.bin -P ./seed-vc/

# Official LatentSync Weights (These are large!)
RUN wget https://huggingface.co/ByteDance/LatentSync-1.6/resolve/main/stable_syncnet.pt -P ./latent-sync/ckpts/
RUN wget https://huggingface.co/ByteDance/LatentSync-1.6/resolve/main/latentsync_unet.pt -P ./latent-sync/ckpts/

# 7. Copy your handler.py file
COPY handler.py .

# 8. Command to start the server
CMD [ "python", "-u", "handler.py" ]
