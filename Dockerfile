
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Install system tools
RUN apt-get update && apt-get install -y git ffmpeg wget

# Set working directory
WORKDIR /app

# Install all necessary Python libraries at once
RUN pip install --no-cache-dir runpod 'numpy<2' scipy librosa soundfile munch pyyaml huggingface-hub einops transformers==4.33.0 torchcrepe vector-quantize-pytorch descript-audio-codec

# Clone the Seed-VC code into the current directory
RUN git clone https://github.com/Plachta/Seed-VC.git .

# Pre-download all model weights during the build
RUN wget https://huggingface.co/Plachta/Seed-VC/resolve/main/DiT_uvit_tat_xlsr_ema.pth
RUN wget https://huggingface.co/Plachta/Seed-VC/resolve/main/config_dit_mel_seed_uvit_xlsr_tiny.yml
RUN wget https://github.com/Plachta/Seed-VC/releases/download/v0.0.1/campplus_cn_common.bin

# Copy our handler script into the image
COPY handler.py .

# Command to start the server when the container runs
CMD [ "python", "-u", "handler.py" ]
