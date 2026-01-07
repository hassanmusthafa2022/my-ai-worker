# 1. Base Image: Python 3.10 with CUDA 11.8 (GPU Support)
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# 2. System Dependencies (Linux Tools)
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
RUN pip install --no-cache-dir \
    runpod 'numpy<2' scipy librosa soundfile munch pyyaml huggingface-hub \
    einops transformers==4.33.0 torchcrepe vector-quantize-pytorch \
    descript-audio-codec opencv-python tqdm imageio moviepy gdown

# 5. --- SHORTCUT: CLONE CODE DIRECTLY FROM GITHUB ---
# This downloads the code inside the Docker container, not on your PC.
RUN git clone https://github.com/Plachta/Seed-VC.git seed-vc
RUN git clone https://github.com/another-ai-artist/latent-sync.git latent-sync

# 6. --- PRE-DOWNLOAD MODEL WEIGHTS ---
# Seed-VC Weights
RUN wget https://huggingface.co/Plachta/Seed-VC/resolve/main/DiT_uvit_tat_xlsr_ema.pth -P ./seed-vc/
RUN wget https://huggingface.co/Plachta/Seed-VC/resolve/main/config_dit_mel_seed_uvit_xlsr_tiny.yml -P ./seed-vc/
RUN wget https://github.com/Plachta/Seed-VC/releases/download/v0.0.1/campplus_cn_common.bin -P ./seed-vc/

# Latent Sync Weights
RUN wget https://huggingface.co/anotheraiguy/latent-sync/resolve/main/visual_model.pth -P ./latent-sync/models/
RUN wget https://huggingface.co/anotheraiguy/latent-sync/resolve/main/audio_model.pth -P ./latent-sync/models/

# 7. Copy ONLY your handler.py file from your local machine
COPY handler.py .

# 8. Command to start the server
CMD [ "python", "-u", "handler.py" ]
