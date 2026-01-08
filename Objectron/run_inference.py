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

# Configuration
MODEL_PATH = "bottle_mobilenet_v2.pth"
TEST_SAMPLE = "bottle/batch-3/14" # From index/bottle_annotations_test
OUTPUT_VIDEO = "bottle_inference.avi"
POSE_OUTPUT_JSON = "bottle_pose.json"
NUM_KEYPOINTS = 9
INPUT_SIZE = 224
DETECT_INTERVAL = 10
SMOOTH_ALPHA = 0.7

# Edges for 3D bounding box
EDGES = [
  [1, 5], [2, 6], [3, 7], [4, 8],  # lines along x-axis
  [1, 3], [5, 7], [2, 4], [6, 8],  # lines along y-axis
  [1, 2], [3, 4], [5, 6], [7, 8]   # lines along z-axis
]
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (128, 128, 0), (128, 0, 128), 
          (0, 128, 128), (255, 255, 255), (0, 0, 0)]

def canonical_box_keypoints():
    # Object keypoint order per schema: center, then 8 corners
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
    if R_prev is None:
        return R_new
    M = alpha * R_new + (1.0 - alpha) * R_prev
    U, _, Vt = np.linalg.svd(M)
    return U @ Vt

def smooth_vector(v_new, v_prev, alpha):
    if v_prev is None:
        return v_new
    return alpha * v_new + (1.0 - alpha) * v_prev

def estimate_intrinsics_from_pbdata(sample_path):
    parts = sample_path.split('/')
    category, batch, item = parts[0], parts[1], parts[2]
    ann_url = f"https://storage.googleapis.com/objectron/annotations/{category}/{batch}/{item}.pbdata"
    ann_file = "test_annotation.pbdata"
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
        # Use the first frame that has camera info
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
    # Recreate model structure
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(model.last_channel, NUM_KEYPOINTS * 3)
    
    # Load weights
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
    filename = "test_video.MOV"
    
    if not os.path.exists(filename):
        print(f"Downloading test video from {url}...")
        urllib.request.urlretrieve(url, filename)
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
    # Solve A(3x3) and T(3) s.t. A X_obj + T ~ X_cam
    n = X_obj.shape[0]
    M = np.zeros((n * 3, 12), dtype=np.float32)
    b = np.zeros((n * 3,), dtype=np.float32)
    for i in range(n):
        ox, oy, oz = X_obj[i]
        cx, cy, cz = X_cam[i]
        # x row
        M[3*i + 0, 0:3] = [ox, oy, oz]
        M[3*i + 0, 9:12] = [1.0, 0.0, 0.0]
        b[3*i + 0] = cx
        # y row
        M[3*i + 1, 3:6] = [ox, oy, oz]
        M[3*i + 1, 9:12] = [0.0, 1.0, 0.0]
        b[3*i + 1] = cy
        # z row
        M[3*i + 2, 6:9] = [ox, oy, oz]
        M[3*i + 2, 9:12] = [0.0, 0.0, 1.0]
        b[3*i + 2] = cz
    # Least squares
    u, *_ = np.linalg.lstsq(M, b, rcond=None)
    A = u[:9].reshape(3, 3)
    T = u[9:12]
    # Polar decomposition to get rotation R
    U, S, Vt = np.linalg.svd(A)
    R = U @ Vt
    # Ensure proper rotation (determinant +1)
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    # Estimate per-axis scale by transforming camera points into object frame
    # X_obj_est = R^T (X_cam - T)
    X_obj_est = (R.T @ (X_cam - T).T).T
    # Scale is 2 * max absolute of coordinates across corners (skip center)
    corners = X_obj_est[1:]
    sx = 2.0 * np.max(np.abs(corners[:, 0]))
    sy = 2.0 * np.max(np.abs(corners[:, 1]))
    sz = 2.0 * np.max(np.abs(corners[:, 2]))
    scale = np.array([sx, sy, sz], dtype=np.float32)
    return R, T, scale

def process_video(model, device, video_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Camera intrinsics from pbdata if available, else approximate
    intrinsics_info = estimate_intrinsics_from_pbdata(TEST_SAMPLE)
    if intrinsics_info is not None:
        K, w_pb, h_pb = intrinsics_info
        # If pbdata dims differ, adjust principal point to current frame size
        K_adj = K.copy()
        K_adj[0, 2] = width / 2.0
        K_adj[1, 2] = height / 2.0
        K = K_adj
    else:
        fx = 0.8 * width
        fy = 0.8 * height
        cx = width / 2.0
        cy = height / 2.0
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    
    # Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))
    
    transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    print("Running inference...")
    frame_count = 0
    prev_gray = None
    prev_pts = None
    R_prev = None
    T_prev = None
    scale_prev = None
    last_detect_keypoints = None
    lk_params = dict(winSize=(21, 21), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    pose_stream = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect or track
        if frame_count % DETECT_INTERVAL == 0 or prev_pts is None:
            # Run model detection
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            input_tensor = transform(pil_img).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_tensor)
            keypoints = output.cpu().numpy().reshape(NUM_KEYPOINTS, 3)
            last_detect_keypoints = keypoints.copy()
            # Create tracking points from 2D pixels (skip center for stability)
            kps_px = to_pixels(keypoints, width, height)
            track_pts = kps_px[1:, :2].astype(np.float32)
            prev_pts = track_pts.reshape(-1, 1, 2)
        else:
            # Track with optical flow (skip center)
            next_pts, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, prev_pts, None, **lk_params)
            # Update last 2D positions; keep depths from last detect
            tracked_px = next_pts.reshape(-1, 2)
            keypoints = last_detect_keypoints.copy()
            kps_px_all = to_pixels(keypoints, width, height)
            kps_px_all[1:, :2] = tracked_px
            # Convert back to normalized for drawing only
            keypoints[:, 0] = kps_px_all[:, 0] / width
            keypoints[:, 1] = kps_px_all[:, 1] / height
            prev_pts = next_pts
        
        prev_gray = frame_gray
        
        # Draw
        for i in range(NUM_KEYPOINTS):
            x = int(keypoints[i, 0] * width)
            y = int(keypoints[i, 1] * height)
            # depth = keypoints[i, 2] # Not used for 2D drawing directly
            
            cv2.circle(frame, (x, y), 5, COLORS[0], -1)
            
        for edge in EDGES:
            pt1 = keypoints[edge[0]]
            pt2 = keypoints[edge[1]]
            
            x1 = int(pt1[0] * width)
            y1 = int(pt1[1] * height)
            x2 = int(pt2[0] * width)
            y2 = int(pt2[1] * height)
            
            cv2.line(frame, (x1, y1), (x2, y2), COLORS[2], 2)
        
        # Pose estimation via lifting + affine fit (EPnP alternative with depths)
        # Use corners only for pose (indices 1..8)
        kps_px_all = to_pixels(keypoints, width, height)
        depths = keypoints[:, 2]
        corners_px = kps_px_all[1:, :2]
        corners_depths = depths[1:]
        # Build KPS with pixels + depths to 3D camera points
        pts_cam = lift_to_camera(corners_px, corners_depths, K)
        X_obj = canonical_box_keypoints()[1:]
        R_est, T_est, scale_est = fit_affine_RT_scale(X_obj, pts_cam)
        # Smoothing
        R_sm = smooth_rotation(R_est, R_prev, SMOOTH_ALPHA)
        T_sm = smooth_vector(T_est, T_prev, SMOOTH_ALPHA)
        scale_sm = smooth_vector(scale_est, scale_prev, SMOOTH_ALPHA)
        R_prev, T_prev, scale_prev = R_sm, T_sm, scale_sm
        # Print pose every few frames
        if frame_count % DETECT_INTERVAL == 0:
            print("Rotation:\n", R_sm)
            print("Translation:", T_sm)
            print("Scale:", scale_sm)
        # Record stream entry
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
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames")
            
    cap.release()
    out.release()
    with open(POSE_OUTPUT_JSON, "w") as f:
        json.dump({
            "sample": TEST_SAMPLE,
            "camera_intrinsics": K.reshape(-1).tolist(),
            "fps": float(fps),
            "width": int(width),
            "height": int(height),
            "frames": pose_stream
        }, f)
    print(f"Inference complete. Video saved to {OUTPUT_VIDEO}")
    print(f"Pose stream saved to {POSE_OUTPUT_JSON}")

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"Model file {MODEL_PATH} not found. Train the model first.")
    else:
        model, device = load_model()
        video_path = download_video(TEST_SAMPLE)
        process_video(model, device, video_path)
