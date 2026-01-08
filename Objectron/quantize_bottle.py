import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torch.ao.quantization import get_default_qconfig
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from PIL import Image
import io
import numpy as np
import urllib.request
from objectron.schema import annotation_data_pb2

# Config
MODEL_FP32_PATH = "bottle_mobilenet_v2.pth"
MODEL_INT8_PATH = "bottle_mobilenet_v2_int8.pth"
INDEX_FILE = "index/bottle_annotations_train"
CACHE_DIR = "dataset_cache"
NUM_KEYPOINTS = 9
INPUT_SIZE = 224
CALIBRATION_SAMPLES = 64
BATCH_SIZE = 8

os.makedirs(CACHE_DIR, exist_ok=True)

class ObjectronBottleDataset(torch.utils.data.Dataset):
    def __init__(self, index_file, transform=None, limit=64):
        self.samples = []
        if os.path.exists(index_file):
            with open(index_file, 'r') as f:
                lines = f.readlines()
                for line in lines[:limit]:
                    self.samples.append(line.strip())
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path = self.samples[idx]
        parts = sample_path.split('/')
        if len(parts) < 3:
            img = Image.new("RGB", (480, 640))
            return self.transform(img) if self.transform else transforms.ToTensor()(img)

        category, batch, item = parts[0], parts[1], parts[2]
        video_url = f"https://storage.googleapis.com/objectron/videos/{category}/{batch}/{item}/video.MOV"
        annotation_url = f"https://storage.googleapis.com/objectron/annotations/{category}/{batch}/{item}.pbdata"
        local_video_path = os.path.join(CACHE_DIR, f"{category}_{batch}_{item}.MOV")
        local_ann_path = os.path.join(CACHE_DIR, f"{category}_{batch}_{item}.pbdata")

        if not os.path.exists(local_video_path):
            try:
                urllib.request.urlretrieve(video_url, local_video_path)
            except Exception:
                img = Image.new("RGB", (480, 640))
                return self.transform(img) if self.transform else transforms.ToTensor()(img)
        if not os.path.exists(local_ann_path):
            try:
                urllib.request.urlretrieve(annotation_url, local_ann_path)
            except Exception:
                img = Image.new("RGB", (480, 640))
                return self.transform(img) if self.transform else transforms.ToTensor()(img)

        import cv2
        cap = cv2.VideoCapture(local_video_path)
        # Use middle frame for calibration variety
        frame_id = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) // 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            img = Image.new("RGB", (480, 640))
            return self.transform(img) if self.transform else transforms.ToTensor()(img)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        return self.transform(img) if self.transform else transforms.ToTensor()(img)

def build_fp32_model():
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(model.last_channel, NUM_KEYPOINTS * 3)
    sd = torch.load(MODEL_FP32_PATH, map_location="cpu")
    model.load_state_dict(sd)
    model.eval()
    return model

def main():
    print("Loading FP32 model...")
    model = build_fp32_model()

    print("Setting quantization backend...")
    # Prefer qnnpack on ARM/macOS; fallback to fbgemm otherwise
    engine = "qnnpack"
    try:
        torch.backends.quantized.engine = engine
    except Exception:
        engine = "fbgemm"
        torch.backends.quantized.engine = engine
    print(f"Using quantized engine: {engine}")

    print("Preparing FX graph for quantization...")
    qconfig = get_default_qconfig(engine)
    qconfig_dict = {"": qconfig}
    example_inputs = (torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE),)
    prepared = prepare_fx(model, qconfig_dict, example_inputs)

    print("Calibrating with real samples...")
    transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = ObjectronBottleDataset(INDEX_FILE, transform=transform, limit=CALIBRATION_SAMPLES)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    with torch.no_grad():
        for inputs in loader:
            _ = prepared(inputs)

    print("Converting to int8...")
    quantized_model = convert_fx(prepared)
    quantized_model.eval()

    print(f"Saving quantized model to {MODEL_INT8_PATH} ...")
    torch.save(quantized_model, MODEL_INT8_PATH)
    size_bytes = os.path.getsize(MODEL_INT8_PATH)
    print("Quantized model size (MB):", round(size_bytes / (1024 * 1024), 2))

if __name__ == "__main__":
    main()
