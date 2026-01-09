import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
import urllib.request
import os
from PIL import Image
from objectron.schema import annotation_data_pb2
import json
import sys

# Configuration
MODEL_PATH = "bottle_mobilenet_v2.pth"
TEST_INDEX_FILE = "index/bottle_annotations_test"
RESULTS_DIR = "inference_results"
NUM_KEYPOINTS = 9
INPUT_SIZE = 224
DETECT_INTERVAL = 10
SMOOTH_ALPHA = 0.7

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Edges for 3D bounding box
EDGES = [
  [1, 5], [2, 6], [3, 7], [4, 8],  # lines along x-axis
  [1, 3], [5, 7], [2, 4], [6, 8],  # lines along y-axis
  [1, 2], [3, 4], [5, 6], [7, 8]   # lines along z-axis
]
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (128, 128, 0), (128, 0, 128), 
          (0, 128, 128), (255, 255, 255), (0, 0, 0)]

def canonical_box_keypoints():
    return np.array([
        [0.0, 0.0, 0.0],
        [-0.5, -0.5, -0.5],
        [-0.5, -0.5,  0.5],
        [-0.5,  0.5, -0.5],
        [-0.5,  0.5,  0.5],
        [ 0.5, -0.5, -0.5],
        [ 0.5, -0.5,  0.5],
        [ 0.5,  0.5, -0.5],
        [ 0.5,  0.5,  0.5],
    ], dtype=np.float32)

def smooth_rotation(R_new, R_prev, alpha):
    if R_prev is None: return R_new
    M = alpha * R_new + (1.0 - alpha) * R_prev
    U, _, Vt = np.linalg.svd(M)
    return U @ Vt

def smooth_vector(v_new, v_prev, alpha):
    if v_prev is None: return v_new
    return alpha * v_new + (1.0 - alpha) * v_prev

def estimate_intrinsics_from_pbdata(sample_path):
    parts = sample_path.split('/')
    category, batch, item = parts[0], parts[1], parts[2]
    ann_url = f"https://storage.googleapis.com/objectron/annotations/{category}/{batch}/{item}.pbdata"
    ann_file = os.path.join(RESULTS_DIR, f"{category}_{batch}_{item}.pbdata")
    
    if not os.path.exists(ann_file):
        try:
            urllib.request.urlretrieve(ann_url, ann_file)
        except Exception:
            return None
            
    try:
        with open(ann_file, 'rb') as f:
            data = f.read()
        seq = annotation_data_pb2.Sequence()
        seq.ParseFromString(data)
        for fa in seq.frame_annotations:
            cam = fa.camera
            width = cam.image_resolution_width if cam.image_resolution_width else None
            height = cam.image_resolution_height if cam.image_resolution_height else None
            intr = cam.intrinsics
            if intr and len(intr) >= 9 and width and height:
                K = np.array(intr[:9], dtype=np.float32).reshape(3, 3)
                return K, width, height
        return None
    except Exception:
        return None

def load_model():
    print("Loading model...")
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(model.last_channel, NUM_KEYPOINTS * 3)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device

def download_video(sample_path):
    parts = sample_path.split('/')
    category, batch, item = parts[0], parts[1], parts[2]
    url = f"https://storage.googleapis.com/objectron/videos/{category}/{batch}/{item}/video.MOV"
    filename = os.path.join(RESULTS_DIR, f"{category}_{batch}_{item}.MOV")
    
    if not os.path.exists(filename):
        # print(f"Downloading {url}...")
        try:
            urllib.request.urlretrieve(url, filename)
        except Exception as e:
            print(f"Failed to download {url}: {e}")
            return None
    return filename

def to_pixels(norm_kps, width, height):
    kps_px = norm_kps.copy()
    kps_px[:, 0] = kps_px[:, 0] * width
    kps_px[:, 1] = kps_px[:, 1] * height
    return kps_px

def lift_to_camera(kps_px, depths, K):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    pts_cam = []
    for i in range(kps_px.shape[0]):
        x, y = kps_px[i, 0], kps_px[i, 1]
        z = depths[i]
        X = (x - cx) / fx * z
        Y = (y - cy) / fy * z
        pts_cam.append([X, Y, z])
    return np.array(pts_cam, dtype=np.float32)

def fit_affine_RT_scale(X_obj, X_cam):
    n = X_obj.shape[0]
    M = np.zeros((n * 3, 12), dtype=np.float32)
    b = np.zeros((n * 3,), dtype=np.float32)
    for i in range(n):
        ox, oy, oz = X_obj[i]
        cx, cy, cz = X_cam[i]
        M[3*i + 0, 0:3] = [ox, oy, oz]; M[3*i + 0, 9:12] = [1.0, 0.0, 0.0]; b[3*i + 0] = cx
        M[3*i + 1, 3:6] = [ox, oy, oz]; M[3*i + 1, 9:12] = [0.0, 1.0, 0.0]; b[3*i + 1] = cy
        M[3*i + 2, 6:9] = [ox, oy, oz]; M[3*i + 2, 9:12] = [0.0, 0.0, 1.0]; b[3*i + 2] = cz
    u, *_ = np.linalg.lstsq(M, b, rcond=None)
    A = u[:9].reshape(3, 3)
    T = u[9:12]
    U, S, Vt = np.linalg.svd(A)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    X_obj_est = (R.T @ (X_cam - T).T).T
    corners = X_obj_est[1:]
    scale = np.array([2.0*np.max(np.abs(corners[:, 0])), 2.0*np.max(np.abs(corners[:, 1])), 2.0*np.max(np.abs(corners[:, 2]))], dtype=np.float32)
    return R, T, scale

def process_single_sample(model, device, sample_path):
    parts = sample_path.split('/')
    if len(parts) < 3: return
    category, batch, item = parts[0], parts[1], parts[2]
    sample_id = f"{category}_{batch}_{item}"
    
    video_path = download_video(sample_path)
    if not video_path: return
    
    output_video_path = os.path.join(RESULTS_DIR, f"{sample_id}_inference.avi")
    output_json_path = os.path.join(RESULTS_DIR, f"{sample_id}_pose.json")
    
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    intrinsics_info = estimate_intrinsics_from_pbdata(sample_path)
    if intrinsics_info:
        K, _, _ = intrinsics_info
        K[0, 2] = width / 2.0
        K[1, 2] = height / 2.0
    else:
        K = np.array([[0.8*width, 0, width/2.0], [0, 0.8*height, height/2.0], [0, 0, 1]], dtype=np.float32)
        
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    frame_count = 0
    prev_gray = None
    prev_pts = None
    R_prev, T_prev, scale_prev = None, None, None
    last_detect_keypoints = None
    lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    pose_stream = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if frame_count % DETECT_INTERVAL == 0 or prev_pts is None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            input_tensor = transform(pil_img).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_tensor)
            keypoints = output.cpu().numpy().reshape(NUM_KEYPOINTS, 3)
            last_detect_keypoints = keypoints.copy()
            kps_px = to_pixels(keypoints, width, height)
            track_pts = kps_px[1:, :2].astype(np.float32)
            prev_pts = track_pts.reshape(-1, 1, 2)
        else:
            next_pts, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, prev_pts, None, **lk_params)
            tracked_px = next_pts.reshape(-1, 2)
            keypoints = last_detect_keypoints.copy()
            kps_px_all = to_pixels(keypoints, width, height)
            kps_px_all[1:, :2] = tracked_px
            keypoints[:, 0] = kps_px_all[:, 0] / width
            keypoints[:, 1] = kps_px_all[:, 1] / height
            prev_pts = next_pts
            
        prev_gray = frame_gray
        
        # Draw 2D keypoints
        for i in range(NUM_KEYPOINTS):
            cv2.circle(frame, (int(keypoints[i, 0]*width), int(keypoints[i, 1]*height)), 5, COLORS[0], -1)
        for edge in EDGES:
            p1 = (int(keypoints[edge[0]][0]*width), int(keypoints[edge[0]][1]*height))
            p2 = (int(keypoints[edge[1]][0]*width), int(keypoints[edge[1]][1]*height))
            cv2.line(frame, p1, p2, COLORS[2], 2)
            
        # Pose
        kps_px_all = to_pixels(keypoints, width, height)
        depths = keypoints[:, 2]
        pts_cam = lift_to_camera(kps_px_all[1:, :2], depths[1:], K)
        X_obj = canonical_box_keypoints()[1:]
        R_est, T_est, scale_est = fit_affine_RT_scale(X_obj, pts_cam)
        
        R_sm = smooth_rotation(R_est, R_prev, SMOOTH_ALPHA)
        T_sm = smooth_vector(T_est, T_prev, SMOOTH_ALPHA)
        scale_sm = smooth_vector(scale_est, scale_prev, SMOOTH_ALPHA)
        R_prev, T_prev, scale_prev = R_sm, T_sm, scale_sm
        
        pose_stream.append({
            "frame_index": int(frame_count),
            "rotation": R_sm.reshape(-1).tolist(),
            "translation": T_sm.reshape(-1).tolist(),
            "scale": scale_sm.reshape(-1).tolist(),
            "landmarks_2d_norm": keypoints[:, :2].tolist(),
            "landmarks_depth": depths.tolist()
        })
        
        out.write(frame)
        frame_count += 1
        
    cap.release()
    out.release()
    
    with open(output_json_path, "w") as f:
        json.dump({
            "sample": sample_path,
            "camera_intrinsics": K.reshape(-1).tolist(),
            "fps": float(fps),
            "width": int(width),
            "height": int(height),
            "frames": pose_stream
        }, f)
    # print(f"Finished {sample_id}")

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Model {MODEL_PATH} not found.")
        return
        
    model, device = load_model()
    
    with open(TEST_INDEX_FILE, 'r') as f:
        test_samples = [line.strip() for line in f.readlines()]
        
    print(f"Found {len(test_samples)} test samples.")
    
    # Process all samples
    for i, sample in enumerate(test_samples):
        print(f"Processing [{i+1}/{len(test_samples)}]: {sample}")
        try:
            process_single_sample(model, device, sample)
        except Exception as e:
            print(f"Error processing {sample}: {e}")
            
    print("Batch inference complete. Results saved to 'inference_results' directory.")

if __name__ == "__main__":
    main()
