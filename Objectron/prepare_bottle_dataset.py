import os
import cv2
import urllib.request
from torchvision import transforms
from PIL import Image

INDEX_ALL = "index/bottle_annotations"
CACHE_DIR = "dataset_cache"
FRAMES_DIR = "dataset_frames"
STRIDE = 6

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)

def download_video(category, batch, item):
    url = f"https://storage.googleapis.com/objectron/videos/{category}/{batch}/{item}/video.MOV"
    local_path = os.path.join(CACHE_DIR, f"{category}_{batch}_{item}.MOV")
    if not os.path.exists(local_path):
        try:
            urllib.request.urlretrieve(url, local_path)
        except Exception:
            return None
    return local_path

def extract_frames(video_path, out_root, stride=STRIDE):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    saved = 0
    name = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = os.path.join(out_root, name)
    os.makedirs(out_dir, exist_ok=True)
    i = 0
    while i < total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        out_path = os.path.join(out_dir, f"frame_{i:06d}.jpg")
        cv2.imwrite(out_path, frame)
        saved += 1
        i += stride
    cap.release()
    return saved

def main():
    if not os.path.exists(INDEX_ALL):
        print("Missing index/bottle_annotations")
        return
    with open(INDEX_ALL, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    print(f"Total listed videos: {len(lines)}")
    total_saved = 0
    for line in lines:
        parts = line.split("/")
        if len(parts) < 3:
            continue
        category, batch, item = parts[0], parts[1], parts[2]
        vp = download_video(category, batch, item)
        if vp is None:
            continue
        saved = extract_frames(vp, FRAMES_DIR, STRIDE)
        total_saved += saved
        print(f"{os.path.basename(vp)} -> {saved} frames")
    print(f"Total frames saved: {total_saved}")

if __name__ == "__main__":
    main()
