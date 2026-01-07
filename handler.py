import runpod
import os
import subprocess

def handler(job):
    job_input = job["input"]
    task = job_input.get("task")

    if task == "voice_clone":
        source_url = job_input.get("source_audio")
        ref_url = job_input.get("reference_audio")
        
        # Download files
        os.system(f"wget -O source.wav {source_url}")
        os.system(f"wget -O ref.wav {ref_url}")
        
        output_path = "/tmp/cloned_voice.wav"
        
        # Run Seed-VC
        # We need to be inside the seed-vc directory to run it
        cmd = f'cd seed-vc && python inference.py --source "../source.wav" --target "../ref.wav" --output "{output_path}" --checkpoint DiT_uvit_tat_xlsr_ema.pth --config config_dit_mel_seed_uvit_xlsr_tiny.yml'
        subprocess.run(cmd, shell=True, check=True)
        
        # TODO: Upload and return S3 URL
        return {"audio_url": "s3://path/to/cloned_voice.wav"}

    elif task == "lip_sync":
        video_url = job_input.get("video_url")
        audio_url = job_input.get("audio_url")
        
        # Download files
        os.system(f"wget -O input.mp4 {video_url}")
        os.system(f"wget -O input.wav {audio_url}")
        
        output_dir = "/tmp/synced_output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Run Official LatentSync
        # The command is different. It uses test_sync.py
        cmd = f'cd latent-sync && python test_sync.py --wav "input.wav" --video "input.mp4" --output_dir "{output_dir}"'
        subprocess.run(cmd, shell=True, check=True)
        
        # Find the output video file (it will be inside the output_dir)
        # TODO: Find the actual output file name and upload to S3
        
        return {"video_url": "s3://path/to/synced_video.mp4"}
        
    else:
        return {"error": "Invalid task specified. Choose 'voice_clone' or 'lip_sync'."}

runpod.serverless.start({"handler": handler})
