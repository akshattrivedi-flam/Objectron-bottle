import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import cv2
import numpy as np
import urllib.request
from objectron.schema import annotation_data_pb2
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR

# Configuration
CATEGORY = "bottle"
INDEX_FILE = "index/bottle_annotations_train"
CACHE_DIR = "dataset_cache"
BATCH_SIZE = 8
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
NUM_KEYPOINTS = 9
INPUT_SIZE = 224

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

class ObjectronBottleDataset(Dataset):
    def __init__(self, index_file, transform=None, limit=20):
        self.samples = []
        if os.path.exists(index_file):
            with open(index_file, 'r') as f:
                lines = f.readlines()
                # Shuffle or just take first N
                for line in lines[:limit]:
                    self.samples.append(line.strip())
        else:
            print(f"Warning: Index file {index_file} not found.")
            
        self.transform = transform
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_path = self.samples[idx] # e.g. bottle/batch-29/17
        parts = sample_path.split('/')
        if len(parts) >= 3:
            category, batch, item = parts[0], parts[1], parts[2]
        else:
             return torch.zeros(3, INPUT_SIZE, INPUT_SIZE), torch.zeros(NUM_KEYPOINTS * 3)

        # Paths
        video_url = f"https://storage.googleapis.com/objectron/videos/{category}/{batch}/{item}/video.MOV"
        annotation_url = f"https://storage.googleapis.com/objectron/annotations/{category}/{batch}/{item}.pbdata"
        
        local_video_path = os.path.join(CACHE_DIR, f"{category}_{batch}_{item}.MOV")
        local_ann_path = os.path.join(CACHE_DIR, f"{category}_{batch}_{item}.pbdata")
        
        # Download if not exists
        if not os.path.exists(local_video_path):
            try:
                # print(f"Downloading {video_url}...")
                urllib.request.urlretrieve(video_url, local_video_path)
            except Exception as e:
                print(f"Error downloading {video_url}: {e}")
                return torch.zeros(3, INPUT_SIZE, INPUT_SIZE), torch.zeros(NUM_KEYPOINTS * 3)
                
        if not os.path.exists(local_ann_path):
            try:
                # print(f"Downloading {annotation_url}...")
                urllib.request.urlretrieve(annotation_url, local_ann_path)
            except Exception as e:
                print(f"Error downloading {annotation_url}: {e}")
                return torch.zeros(3, INPUT_SIZE, INPUT_SIZE), torch.zeros(NUM_KEYPOINTS * 3)
        
        # Parse Annotation
        try:
            with open(local_ann_path, 'rb') as f:
                annotation_data = f.read()
            sequence = annotation_data_pb2.Sequence()
            sequence.ParseFromString(annotation_data)
        except Exception:
             return torch.zeros(3, INPUT_SIZE, INPUT_SIZE), torch.zeros(NUM_KEYPOINTS * 3)
            
        # Pick a frame with annotations
        valid_frames = [fa for fa in sequence.frame_annotations if len(fa.annotations) > 0]
        if not valid_frames:
             return torch.zeros(3, INPUT_SIZE, INPUT_SIZE), torch.zeros(NUM_KEYPOINTS * 3)
        
        # Randomly select one frame
        frame_annotation = valid_frames[np.random.randint(len(valid_frames))]
        frame_id = frame_annotation.frame_id
        
        # Load Frame
        cap = cv2.VideoCapture(local_video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
             return torch.zeros(3, INPUT_SIZE, INPUT_SIZE), torch.zeros(NUM_KEYPOINTS * 3)
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame)
        
        # Get Keypoints (first object)
        obj_ann = frame_annotation.annotations[0]
        keypoints = []
        # Ensure we have 9 keypoints
        if len(obj_ann.keypoints) != NUM_KEYPOINTS:
             # Handle cases with fewer keypoints if necessary, or skip
             # For now, just pad or truncate (simple hack for robustness)
             pass
             
        for kp in obj_ann.keypoints:
            keypoints.extend([kp.point_2d.x, kp.point_2d.y, kp.point_2d.depth])
        
        # Pad if missing
        expected_len = NUM_KEYPOINTS * 3
        if len(keypoints) < expected_len:
            keypoints.extend([0.0] * (expected_len - len(keypoints)))
        elif len(keypoints) > expected_len:
            keypoints = keypoints[:expected_len]
            
        keypoints = torch.tensor(keypoints, dtype=torch.float32)
        
        # Transform Image
        if self.transform:
            image_tensor = self.transform(frame_pil)
        else:
            image_tensor = transforms.ToTensor()(frame_pil)
            
        return image_tensor, keypoints

class SampleListBottleDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        sample_path = self.samples[idx]
        parts = sample_path.split('/')
        if len(parts) >= 3:
            category, batch, item = parts[0], parts[1], parts[2]
        else:
            return torch.zeros(3, INPUT_SIZE, INPUT_SIZE), torch.zeros(NUM_KEYPOINTS * 3)
        video_url = f"https://storage.googleapis.com/objectron/videos/{category}/{batch}/{item}/video.MOV"
        annotation_url = f"https://storage.googleapis.com/objectron/annotations/{category}/{batch}/{item}.pbdata"
        local_video_path = os.path.join(CACHE_DIR, f"{category}_{batch}_{item}.MOV")
        local_ann_path = os.path.join(CACHE_DIR, f"{category}_{batch}_{item}.pbdata")
        if not os.path.exists(local_video_path):
            try:
                urllib.request.urlretrieve(video_url, local_video_path)
            except Exception:
                return torch.zeros(3, INPUT_SIZE, INPUT_SIZE), torch.zeros(NUM_KEYPOINTS * 3)
        if not os.path.exists(local_ann_path):
            try:
                urllib.request.urlretrieve(annotation_url, local_ann_path)
            except Exception:
                return torch.zeros(3, INPUT_SIZE, INPUT_SIZE), torch.zeros(NUM_KEYPOINTS * 3)
        try:
            with open(local_ann_path, 'rb') as f:
                annotation_data = f.read()
            sequence = annotation_data_pb2.Sequence()
            sequence.ParseFromString(annotation_data)
        except Exception:
            return torch.zeros(3, INPUT_SIZE, INPUT_SIZE), torch.zeros(NUM_KEYPOINTS * 3)
        valid_frames = [fa for fa in sequence.frame_annotations if len(fa.annotations) > 0]
        if not valid_frames:
            return torch.zeros(3, INPUT_SIZE, INPUT_SIZE), torch.zeros(NUM_KEYPOINTS * 3)
        frame_annotation = valid_frames[np.random.randint(len(valid_frames))]
        frame_id = frame_annotation.frame_id
        cap = cv2.VideoCapture(local_video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return torch.zeros(3, INPUT_SIZE, INPUT_SIZE), torch.zeros(NUM_KEYPOINTS * 3)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame)
        obj_ann = frame_annotation.annotations[0]
        keypoints = []
        for kp in obj_ann.keypoints:
            keypoints.extend([kp.point_2d.x, kp.point_2d.y, kp.point_2d.depth])
        expected_len = NUM_KEYPOINTS * 3
        if len(keypoints) < expected_len:
            keypoints.extend([0.0] * (expected_len - len(keypoints)))
        elif len(keypoints) > expected_len:
            keypoints = keypoints[:expected_len]
        keypoints = torch.tensor(keypoints, dtype=torch.float32)
        if self.transform:
            image_tensor = self.transform(frame_pil)
        else:
            image_tensor = transforms.ToTensor()(frame_pil)
        return image_tensor, keypoints

def train():
    # Transforms
    data_transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Dataset
    print("Initializing dataset...")
    index_path = os.path.join("index", "bottle_annotations_train")
    train_samples = []
    if os.path.exists(index_path):
        with open(index_path, 'r') as f:
            lines = [l.strip() for l in f.readlines()]
            lines = lines[:1500]
            split_idx = int(0.9 * len(lines))
            train_samples = lines[:split_idx]
            val_samples = lines[split_idx:]
    else:
        train_samples = []
        val_samples = []
    dataset_train = SampleListBottleDataset(train_samples, transform=data_transform)
    dataset_val = SampleListBottleDataset(val_samples, transform=data_transform)
    if len(dataset_train) == 0:
        print("Dataset is empty. Check index file path.")
        return

    dataloader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model
    print("Creating model (MobileNetV2)...")
    try:
        model = models.mobilenet_v2(pretrained=True)
    except Exception as e:
        print(f"Failed to load pretrained model: {e}")
        print("Using random init.")
        model = models.mobilenet_v2(pretrained=False)
        
    # Modify classifier for regression (9 keypoints * 3 values = 27 outputs)
    model.classifier[1] = nn.Linear(model.last_channel, NUM_KEYPOINTS * 3)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # On Mac M1/M2, use mps if available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        
    print(f"Using device: {device}")
    model = model.to(device)
    
    # Loss and Optimizer
    weights = torch.ones(NUM_KEYPOINTS * 3, dtype=torch.float32, device=device)
    weights[:3] = 2.0
    def weighted_mse(pred, target):
        diff = pred - target
        loss = (diff * diff) * weights
        return loss.mean()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=LEARNING_RATE * 0.1)
    
    # Training Loop
    print("Starting training...")
    model.train()
    best_val = float('inf')
    patience = 3
    no_improve = 0
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            valid_mask = (targets.abs().sum(dim=1) > 0)
            if valid_mask.sum() == 0:
                continue
            outputs_v = outputs[valid_mask]
            targets_v = targets[valid_mask]
            outv = outputs_v.view(outputs_v.size(0), NUM_KEYPOINTS, 3)
            outv[:, :, 0:2] = torch.sigmoid(outv[:, :, 0:2])
            predv = outv.view(outputs_v.size(0), -1)
            loss = weighted_mse(predv, targets_v)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()
            
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
        scheduler.step()
        model.eval()
        val_loss = 0.0
        count = 0
        with torch.no_grad():
            for vi, (vinputs, vtargets) in enumerate(val_loader):
                vinputs = vinputs.to(device)
                vtargets = vtargets.to(device)
                voutputs = model(vinputs)
                vout = voutputs.view(voutputs.size(0), NUM_KEYPOINTS, 3)
                vout[:, :, 0:2] = torch.sigmoid(vout[:, :, 0:2])
                vpred = vout.view(voutputs.size(0), -1)
                vmask = (vtargets.abs().sum(dim=1) > 0)
                if vmask.sum() == 0:
                    continue
                vpred = vpred[vmask]
                vtargets = vtargets[vmask]
                vloss = weighted_mse(vpred, vtargets)
                val_loss += vloss.item()
                count += 1
        val_loss = val_loss / max(count, 1)
        print(f"Val Loss: {val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "bottle_mobilenet_v2.pth")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping.")
                break
        model.train()
                
    print("Training complete.")
    
    # Save Model
    save_path = "bottle_mobilenet_v2.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train()
