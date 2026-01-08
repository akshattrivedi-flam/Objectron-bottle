import urllib.request
import os
import cv2
import numpy as np
import sys

# Ensure we can import objectron
sys.path.append(os.getcwd())

from objectron.schema import annotation_data_pb2
from objectron.dataset import graphics

def download_file(url, filename):
    print(f"Downloading {url} to {filename}...")
    try:
        urllib.request.urlretrieve(url, filename)
        print("Download complete.")
        return True
    except Exception as e:
        print(f"Failed to download. Error: {e}")
        return False

def main():
    # Define the sample to download
    # Using the first entry from index/bottle_annotations: bottle/batch-1/0
    category = "bottle"
    batch = "batch-1"
    item = "0"
    
    video_url = f"https://storage.googleapis.com/objectron/videos/{category}/{batch}/{item}/video.MOV"
    annotation_url = f"https://storage.googleapis.com/objectron/annotations/{category}/{batch}/{item}.pbdata"
    
    video_filename = "bottle_sample.MOV"
    annotation_filename = "bottle_annotation.pbdata"
    
    if not os.path.exists(video_filename):
        if not download_file(video_url, video_filename):
            return
            
    if not os.path.exists(annotation_filename):
        if not download_file(annotation_url, annotation_filename):
            return

    # Parse annotation
    print("Parsing annotation...")
    with open(annotation_filename, 'rb') as f:
        annotation_data = f.read()
        
    sequence = annotation_data_pb2.Sequence()
    sequence.ParseFromString(annotation_data)
    
    print(f"Sequence has {len(sequence.frame_annotations)} frames.")
    
    # Process a few frames
    cap = cv2.VideoCapture(video_filename)
    if not cap.isOpened():
        print("Error opening video file")
        return

    # Let's pick a frame in the middle where the object is likely visible
    target_frame_idx = len(sequence.frame_annotations) // 2
    
    # Find the annotation for this frame
    frame_annotation = None
    for fa in sequence.frame_annotations:
        if fa.frame_id == target_frame_idx:
            frame_annotation = fa
            break
            
    if frame_annotation is None:
        print(f"No annotation found for frame {target_frame_idx}")
        # Try finding the first frame with annotation
        for fa in sequence.frame_annotations:
            if len(fa.annotations) > 0:
                target_frame_idx = fa.frame_id
                frame_annotation = fa
                print(f"Using frame {target_frame_idx} instead.")
                break

    if frame_annotation is None:
        print("No frames with annotations found.")
        return

    # Seek to the frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
    ret, frame = cap.read()
    
    if not ret:
        print("Could not read frame.")
        return
        
    # Prepare keypoints for visualization
    # graphics.draw_annotation_on_image expects a flattened list of keypoints [x, y, d]
    # corresponding to the objects.
    
    object_annotations = []
    num_keypoints = []
    
    for obj_ann in frame_annotation.annotations:
        # obj_ann.keypoints is repeated AnnotatedKeyPoint
        # AnnotatedKeyPoint has point_2d (NormalizedPoint2D) and point_3d (Point3D)
        
        # graphics.py expects [x, y, d] for each keypoint.
        # Check objectron/dataset/graphics.py:
        # keypoints = np.split(object_annotations, np.array(np.cumsum(num_keypoints)))
        # keypoints = [points.reshape(-1, 3) for points in keypoints]
        
        points_list = []
        for kp in obj_ann.keypoints:
            points_list.append([kp.point_2d.x, kp.point_2d.y, kp.point_2d.depth])
        
        object_annotations.extend(points_list)
        num_keypoints.append(len(obj_ann.keypoints))
        
    if not object_annotations:
        print("No object annotations in this frame.")
    else:
        print(f"Found {len(num_keypoints)} objects in frame.")
        
        # Convert to numpy array
        object_annotations = np.array(object_annotations)
        
        # Draw
        image_with_box = graphics.draw_annotation_on_image(frame, object_annotations, num_keypoints)
        
        output_filename = "bottle_result.jpg"
        cv2.imwrite(output_filename, image_with_box)
        print(f"Result saved to {output_filename}")

    cap.release()

if __name__ == "__main__":
    main()
