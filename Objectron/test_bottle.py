import os
import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
import urllib.request
from PIL import Image

# Config
MODEL_PATH = "bottle_mobilenet_v2.pth"
INDEX_FILE = "index/bottle_annotations_test"
CACHE_DIR = "dataset_cache"
NUM_KEYPOINTS = 9
INPUT_SIZE = 224
OUTPUT_DIR = "inference_results"

# Box edges (1-based indices into 9 keypoints array, 0 is center)
# From objectron/dataset/box.py
EDGES = (
    [1, 5], [2, 6], [3, 7], [4, 8],  # lines along x-axis
    [1, 3], [5, 7], [2, 4], [6, 8],  # lines along y-axis
    [1, 2], [3, 4], [5, 6], [7, 8]   # lines along z-axis
)

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def build_model():
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(model.last_channel, NUM_KEYPOINTS * 3)
    
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}")
        sd = torch.load(MODEL_PATH, map_location="cpu")
        model.load_state_dict(sd)
    else:
        print(f"Model file {MODEL_PATH} not found! Please make sure the model is trained.")
        return None
        
    model.eval()
    return model

def get_test_videos(limit=5):
    samples = []
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, 'r') as f:
            lines = f.readlines()
            # simple shuffle or pick first 5 that are valid
            for line in lines:
                if len(samples) >= limit:
                    break
                samples.append(line.strip())
    else:
        print(f"Index file {INDEX_FILE} not found.")
        return []

    video_paths = []
    for sample_path in samples:
        parts = sample_path.split('/')
        if len(parts) < 3:
            continue
            
        category, batch, item = parts[0], parts[1], parts[2]
        video_url = f"https://storage.googleapis.com/objectron/videos/{category}/{batch}/{item}/video.MOV"
        local_video_path = os.path.join(CACHE_DIR, f"{category}_{batch}_{item}.MOV")
        
        if not os.path.exists(local_video_path):
            try:
                print(f"Downloading {video_url}...")
                urllib.request.urlretrieve(video_url, local_video_path)
            except Exception as e:
                print(f"Error downloading {video_url}: {e}")
                continue
        
        video_paths.append(local_video_path)
        
    return video_paths

def process_video_frame(video_path, model, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video {video_path}")
        return
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame_idx = total_frames // 2
    idxs = [max(0, min(total_frames - 1, middle_frame_idx + d)) for d in [-2, -1, 0, 1, 2]]
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, f = cap.read()
        if ret:
            frames.append(f)
    cap.release()
    if len(frames) == 0:
        print(f"Could not read frames from {video_path}")
        return
    width = frames[0].shape[1]
    height = frames[0].shape[0]
    
    transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    model = model.to(device)

    print(f"Processing {video_path} -> {output_path}")
    preds_acc = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)
    count = 0
    with torch.no_grad():
        for frame in frames:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            input_tensor = transform(pil_img).unsqueeze(0).to(device)
            outputs = model(input_tensor)
            out = outputs.view(1, NUM_KEYPOINTS, 3)
            out[:, :, 0:2] = torch.sigmoid(out[:, :, 0:2])
            preds = out.view(NUM_KEYPOINTS, 3).cpu().numpy()
            preds_acc += preds
            count += 1
    keypoints = preds_acc / max(count, 1)
    base_frame = frames[len(frames) // 2].copy()
    
    # Draw lines for edges
    for start_idx, end_idx in EDGES:
        if start_idx < NUM_KEYPOINTS and end_idx < NUM_KEYPOINTS:
            pt1 = keypoints[start_idx]
            pt2 = keypoints[end_idx]
            
            x1 = int(pt1[0] * width)
            y1 = int(pt1[1] * height)
            x2 = int(pt2[0] * width)
            y2 = int(pt2[1] * height)
            
            cv2.line(base_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    for i in range(NUM_KEYPOINTS):
        x_norm = keypoints[i, 0]
        y_norm = keypoints[i, 1]
        
        x_px = int(x_norm * width)
        y_px = int(y_norm * height)
        
        color = (0, 255, 0) if i == 0 else (0, 0, 255) # Center green, corners red
        cv2.circle(base_frame, (x_px, y_px), 5, color, -1)
        
    cv2.imwrite(output_path, base_frame)
    print(f"Saved {output_path}")

def process_video_overlay(video_path, model, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    model = model.to(device)
    buf = None
    alpha = 0.6
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            input_tensor = transform(pil_img).unsqueeze(0).to(device)
            outputs = model(input_tensor)
            outk = outputs.view(1, NUM_KEYPOINTS, 3)
            outk[:, :, 0:2] = torch.sigmoid(outk[:, :, 0:2])
            kp = outk.view(NUM_KEYPOINTS, 3).cpu().numpy()
            if buf is None:
                buf = kp
            else:
                buf = alpha * kp + (1 - alpha) * buf
            for s, e in EDGES:
                if s < NUM_KEYPOINTS and e < NUM_KEYPOINTS:
                    p1 = buf[s]
                    p2 = buf[e]
                    x1 = int(p1[0] * width); y1 = int(p1[1] * height)
                    x2 = int(p2[0] * width); y2 = int(p2[1] * height)
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            for i in range(NUM_KEYPOINTS):
                x_px = int(buf[i, 0] * width)
                y_px = int(buf[i, 1] * height)
                color = (0, 255, 0) if i == 0 else (0, 0, 255)
                cv2.circle(frame, (x_px, y_px), 4, color, -1)
            out.write(frame)
    cap.release()
    out.release()

def main():
    model = build_model()
    if model is None:
        return

    print("Reading videos from inference_results...")
    basenames = []
    for f in os.listdir(OUTPUT_DIR):
        if f.startswith("result_") and (f.endswith(".jpg") or f.endswith(".mp4")):
            name = f.replace("result_", "")
            name = name.replace(".jpg", "").replace(".mp4", "")
            basenames.append(name)
    basenames = sorted(list(set(basenames)))
    video_paths = []
    for name in basenames:
        local_video_path = os.path.join(CACHE_DIR, f"{name}.MOV")
        if os.path.exists(local_video_path):
            video_paths.append(local_video_path)

    if not video_paths:
        print("No matching cached videos found. Falling back to index.")
        video_paths = get_test_videos(limit=5)

    for i, video_path in enumerate(video_paths):
        video_name = os.path.basename(video_path)
        img_name = f"result_{video_name.replace('.MOV', '.jpg')}"
        img_path = os.path.join(OUTPUT_DIR, img_name)
        process_video_frame(video_path, model, img_path)
    mp4s = []
    for f in os.listdir(OUTPUT_DIR):
        if f.endswith(".mp4"):
            mp4s.append(os.path.join(OUTPUT_DIR, f))
    for v in sorted(mp4s):
        base = os.path.basename(v)
        out_name = base.replace(".mp4", "_overlay.mp4")
        out_path = os.path.join(OUTPUT_DIR, out_name)
        process_video_overlay(v, model, out_path)

if __name__ == "__main__":
    main()
