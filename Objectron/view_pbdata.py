import os
import sys
import json
from google.protobuf.json_format import MessageToDict

# Add the directory containing the schema to sys.path
sys.path.append(os.path.join(os.getcwd(), 'Objectron'))

try:
    from objectron.schema import annotation_data_pb2
except ImportError as e:
    print(f"Error importing schema: {e}")
    print("Trying alternative path...")
    sys.path.append(os.getcwd())
    from objectron.schema import annotation_data_pb2

import numpy as np

def extract_rotation_and_translation(file_path, frame_indices):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    sequence = annotation_data_pb2.Sequence()
    with open(file_path, 'rb') as f:
        sequence.ParseFromString(f.read())
    
    data = MessageToDict(sequence)
    num_total_frames = len(data.get("frameAnnotations", []))
    
    print("\n" + "="*50)
    print("ROTATION MATRIX EXTRACTION (3x3)")
    print("="*50)

    for idx in frame_indices:
        if idx < num_total_frames:
            frame = data.get("frameAnnotations", [])[idx]
            transform = np.array(frame['camera']['transform']).reshape(4, 4)
            
            # The upper-left 3x3 of the 4x4 transform is the Rotation Matrix
            rotation_matrix = transform[:3, :3]
            translation_vector = transform[:3, 3]
            
            print(f"\n[ FRAME {idx} ]")
            print("Rotation Matrix (3x3):")
            for row in rotation_matrix:
                print(f"  [ {row[0]:.6f}, {row[1]:.6f}, {row[2]:.6f} ]")
            
            print(f"Translation Vector (t):")
            print(f"  [ {translation_vector[0]:.6f}, {translation_vector[1]:.6f}, {translation_vector[2]:.6f} ]")
        else:
            print(f"\n[ FRAME {idx} ] - OUT OF RANGE")

if __name__ == "__main__":
    pb_file = "Objectron/bottle_annotation.pbdata"
    extract_rotation_and_translation(pb_file, [216, 234])
