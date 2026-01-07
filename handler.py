import runpod
import os
import subprocess
import base64
from urllib.parse import urlparse

def download_file(url, output_path):
    """Downloads a file from a URL."""
    print(f"Downloading {url} to {output_path}...")
    subprocess.run(["wget", "-O", output_path, url], check=True)

def handler(job):
    job_input = job["input"]
    
    # 1. Get URLs from input
    source_url = job_input.get("source_audio")
    ref_url = job_input.get("reference_audio")
    
    if not source_url or not ref_url:
        return {"error": "Missing source_audio or reference_audio URL."}

    # 2. Define file paths
    source_path = "source_audio.wav"
    ref_path = "reference_voice.wav"
    output_path = "/tmp/output.wav"
    
    try:
        # 3. Download the audio files
        download_file(source_url, source_path)
        download_file(ref_url, ref_path)

        # 4. Run the actual Seed-VC inference command
        cmd = f'python inference.py --source "{source_path}" --target "{ref_path}" --output "{output_path}" --checkpoint DiT_uvit_tat_xlsr_ema.pth --config config_dit_mel_seed_uvit_xlsr_tiny.yml'
        
        print("Running Seed-VC inference...")
        subprocess.run(cmd, shell=True, check=True)
        
        # 5. Read the output file and return it as a data URI
        if not os.path.exists(output_path):
            return {"error": "Inference finished but output file was not created."}
            
        with open(output_path, "rb") as f:
            audio_data = f.read()
        
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        data_uri = f"data:audio/wav;base64,{audio_base64}"
        
        return {"audio_output": data_uri}

    except Exception as e:
        return {"error": str(e)}
    finally:
        # Clean up downloaded files
        if os.path.exists(source_path): os.remove(source_path)
        if os.path.exists(ref_path): os.remove(ref_path)

# Start the serverless worker
runpod.serverless.start({"handler": handler})
