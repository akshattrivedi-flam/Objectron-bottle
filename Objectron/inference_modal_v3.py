import modal
import os
import subprocess

# 1. Define the Modal Image with ALL dependencies
image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "torchvision",
        "opencv-python-headless",
        "numpy",
        "tqdm",
        "matplotlib",
        "scipy",
        "pillow",
        "protobuf==3.20.3",
        "requests",
        "gdown"
    )
    .apt_install("git", "tar", "curl")
    # Add the local Objectron code to the image
    .add_local_dir(
        os.path.dirname(os.path.abspath(__file__)),
        remote_path="/root/Objectron",
        ignore=[
            ".git", ".venv", "__pycache__", "*.ipynb", "*.docx", 
            "index", "docs", "notebooks", "*.mp4", "*.mov", "*.MOV", "*.log",
            "dataset_cache", "dataset_frames", "inference_results", 
            "checkpoints", "results", "manifests", "data", "images", "Objectron-bottle"
        ]
    )
)

app = modal.App("objectron-flam-v1", image=image)
# Use the volume for checkpoints and data
volume = modal.NetworkFileSystem.from_name("objectron-flam-data", create_if_missing=True)

# 2. Define the Model Architecture (must match training)
def create_model(num_keypoints=21):
    import torch.nn as nn
    from torchvision import models
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.last_channel, num_keypoints * 2)
    )
    return model

@app.function(
    gpu="A100",
    timeout=86400, # 24 hours
    network_file_systems={"/data": volume}
)
def run_training():
    print("🚀 Starting High-Performance Training on A100...")
    
    # Path to the training script in the image
    train_script_path = "/root/Objectron/train_modal_cloud.py"
    
    # Setup directory structure for outputs
    base_path = "/data/training_results"
    os.makedirs(base_path, exist_ok=True)
    os.chdir(base_path)
    
    # Execute training
    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/Objectron"
    
    subprocess.run(["python3", train_script_path], env=env, check=True)
    print("✅ Training Complete!")
    return f"{base_path}/checkpoints/best_model.pth"

@app.function(
    gpu="A100",
    network_file_systems={"/data": volume}
)
def run_video_inference(video_url_or_path, output_name="output_inference.mp4"):
    import torch
    import cv2
    import numpy as np
    from torchvision import transforms
    from tqdm import tqdm
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Paths - Check if model exists in the volume
    print(f"🔍 Checking for model in /data...")
    for root, dirs, files in os.walk("/data"):
        for file in files:
            if "best_model.pth" in file:
                print(f"Found: {os.path.join(root, file)}")
    
    model_path = "/data/training_results/checkpoints/best_model.pth"
    if not os.path.exists(model_path):
        # Check if it exists with a slightly different name or in the same directory
        possible_paths = [
            "/data/training_results/checkpoints/best_model.pth",
            "/data/training_results/checkpoints/training_results/checkpoints/best_model.pth", # nested due to previous volume put
            "/data/training_results/checkpoints/bottle_mobilenet_v2.pth",
            "/data/training_results/checkpoints/training_results/checkpoints/bottle_mobilenet_v2.pth"
        ]
        
        found = False
        for p in possible_paths:
            if os.path.exists(p):
                model_path = p
                found = True
                print(f"✅ Found model at: {p}")
                break
        
        if not found:
            # Fallback to older path if exists
            old_path = "/data/Objectron-bottle/Objectron/checkpoints/best_model.pth"
            if os.path.exists(old_path):
                model_path = old_path
                found = True
            else:
                return f"❌ Model not found. Searched in: {possible_paths}. Please run training first."

    # Load Model
    print("🤖 Loading model...")
    model = create_model(num_keypoints=21)
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # Input Video
    print(f"📹 Opening video: {video_url_or_path}")
    cap = cv2.VideoCapture(video_url_or_path)
    if not cap.isOpened():
        return f"❌ Could not open video: {video_url_or_path}"

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output Video
    output_path = f"/data/{output_name}"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Preprocessing
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print(f"🚀 Running inference on {total_frames} frames...")
    
    with torch.no_grad():
        for _ in tqdm(range(total_frames)):
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = transform(rgb_frame).unsqueeze(0).to(device)

            outputs = model(input_tensor)
            keypoints = outputs.cpu().numpy()[0].reshape(-1, 2)

            for i, (x, y) in enumerate(keypoints):
                px = int(x * width)
                py = int(y * height)
                cv2.circle(frame, (px, py), 5, (0, 255, 0), -1)
                
                if i > 0 and i < 7: # Body
                    prev_x, prev_y = keypoints[i-1]
                    cv2.line(frame, (int(prev_x*width), int(prev_y*height)), (px, py), (255, 0, 0), 2)
                elif i > 7 and i < 14: # Neck
                    prev_x, prev_y = keypoints[i-1]
                    cv2.line(frame, (int(prev_x*width), int(prev_y*height)), (px, py), (255, 0, 0), 2)
                elif i > 14 and i < 21: # Top
                    prev_x, prev_y = keypoints[i-1]
                    cv2.line(frame, (int(prev_x*width), int(prev_y*height)), (px, py), (255, 0, 0), 2)

            out.write(frame)

    cap.release()
    out.release()
    
    print(f"✅ Inference complete! Saved to {output_path}")
    return output_path

@app.local_entrypoint()
def main(action: str = "inference", video_path: str = None, video_url: str = "https://storage.googleapis.com/objectron/videos/bottle/batch-1/0/video.MOV"):
    if action == "train":
        print("🎬 Starting high-performance training action...")
        run_training.remote()
        print("✨ Training finished!")
    elif action == "inference":
        if video_path and os.path.exists(video_path):
            print(f"📤 Uploading local video {video_path} to Modal volume...")
            remote_video_path = f"/data/input_{os.path.basename(video_path)}"
            subprocess.run(["modal", "volume", "put", "objectron-flam-data", video_path, remote_video_path])
            video_to_process = remote_video_path
        else:
            video_to_process = video_url
            
        print(f"🎬 Starting inference on {video_to_process}")
        result_path = run_video_inference.remote(video_to_process)
        
        if result_path.startswith("❌"):
            print(result_path)
            return

        local_result = "inference_result.mp4"
        print(f"📥 Downloading result to {local_result}...")
        subprocess.run(["modal", "volume", "get", "objectron-flam-data", result_path, local_result])
        print(f"✨ Done! You can view the results in {local_result}")
    else:
        print(f"❌ Unknown action: {action}. Use 'train' or 'inference'.")
