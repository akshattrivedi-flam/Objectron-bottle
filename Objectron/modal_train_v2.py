import modal
import os
import subprocess

# 1. Define the Modal Image with all dependencies
# We use a standard Debian image and install the specialized CV/Deep Learning libraries
image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "torchvision",
        "opencv-python-headless",
        "numpy",
        "protobuf==3.20.3",
        "google-api-python-client",
        "google-cloud-storage",
        "tqdm",
        "matplotlib",
        "python-docx"
    )
    .apt_install("git")
)

app = modal.App("objectron-bottle-training", image=image)

# 2. Define persistent storage for datasets and checkpoints
# This ensures that even if the container restarts, your frames and models stay safe
volume = modal.NetworkFileSystem.from_name("objectron-data", create_if_missing=True)

@app.function(
    gpu="A100",  # Using the powerful A100 for fast training
    timeout=86400, # 24 hour timeout
    network_file_systems={"/data": volume}
)
def run_training():
    print("🚀 Starting Objectron Modal Cloud Training Pipeline")
    
    # Step 1: Clone the latest code
    repo_url = "https://github.com/akshattrivedi-flam/Objectron-bottle"
    if not os.path.exists("/data/Objectron-bottle"):
        print("Cloning repository...")
        subprocess.run(["git", "clone", repo_url, "/data/Objectron-bottle"])
    else:
        print("Updating repository...")
        subprocess.run(["git", "-C", "/data/Objectron-bottle", "pull"])

    os.chdir("/data/Objectron-bottle/Objectron")
    
    # Step 2: Data Preparation (Stride-8 Extraction)
    # We run the optimized training script which handles:
    # - Downloading the 1543-sample bottle dataset
    # - Extracting frames with Stride-8
    # - Running 20 epochs of fine-tuning (No Early Stopping)
    # - Using AMP (Mixed Precision) and TF32 for A100 speed
    print("🎬 Beginning Data Extraction & Training...")
    
    # We use the 'train_modal_cloud.py' logic which is already optimized for this environment
    subprocess.run(["python3", "train_modal_cloud.py"])

    print("✅ Training Complete. Checkpoints saved to /data/Objectron-bottle/Objectron/checkpoints")

@app.local_entrypoint()
def main():
    run_training.remote()
