import cv2
import numpy as np
import os
import glob
import math
import shutil
import pickle
import supervision as sv
from ultralytics import YOLO
import argparse
import torch
import gc

# --- CONFIGURATION ---
TARGET_FPS = 30
TARGET_FRAME_COUNT = 300 # User requested fixed 300 frames
NORMALIZED_HEIGHT = 64
NORMALIZED_WIDTH  = 44
MODEL_NAME = "yolov8m-seg.pt"

# Output Mapping
# User data: has folders like "hasta", "saglikli".
# We need to preserve this info in the folder structure for OpenGait loader or Partition builder.
# Goal Structure: DATASET_ROOT / SUBJECT_ID / STATUS / VIEW / SEQ_ID / ...

def get_sampling_indices(total_frames, original_fps, target_fps):
    if total_frames == 0: return []
    duration = total_frames / original_fps
    num_output_frames = int(duration * target_fps)
    if num_output_frames == 0:
        return []
    return np.linspace(0, total_frames - 1, num_output_frames).astype(int)

def preprocess_silhouette_final(mask_crop, target_h, target_w):
    h, w = mask_crop.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((target_h, target_w), dtype=np.uint8)

    scale = target_h / float(h)
    new_w = int(w * scale)
    new_h = target_h

    if new_w > target_w:
        scale = target_w / float(new_w)
        new_w = target_w
        new_h = int(new_h * scale)

    resized = cv2.resize(mask_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    final_img = np.zeros((target_h, target_w), dtype=np.uint8)

    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2

    final_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return final_img

def save_as_pkl(frames, file_path):
    # OpenGait expects the sequence itself (T, H, W) or List of (H, W).
    # collate_fn calls len() on it to get sequence length.
    # So we save the numpy array directly.
    data = np.array(frames, dtype=np.uint8)
    
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)



def get_view_code(video_path):
    # User requested to IGNORE 'yandan' (side) views and only use 'onden' (front).
    # But if we are processing, we mark 'onden' as '000'.
    # Filtering happens in the main loop.
    return '000' # Always front for this specific request


def get_subject_id(video_path, existing_ids):
    # Try to extract a unique subject name.
    # Example: /.../Patient1/video.mp4 -> Patient1
    # You might need to adjust this logic based on your exact folder structure!
    parts = video_path.split(os.sep)
    # Assumes structure like .../Split/Class/Subject/Video.mp4 OR .../Split/Subject_Class_etc.mp4
    # Let's use the filename base as part of ID if subject folders aren't clear
    base = os.path.splitext(os.path.basename(video_path))[0]
    return base # Simple unique ID for now

def process_video(video_path, output_root, model, tracker):
    print(f"Processing: {video_path}")
    
    # Identify Metadata
    path_lower = video_path.lower()
    if 'hasta' in path_lower or 'scoliosis' in path_lower or 'patient' in path_lower:
        status = 'patient'
    elif 'saglik' in path_lower or 'healthy' in path_lower or 'negativ' in path_lower:
        status = 'healthy'
    elif 'neutral' in path_lower or 'notr' in path_lower:
        status = 'neutral' # Will be mapped to 0/Healthy by model, but keep label distinct here
    else:
        status = 'unknown'

    view = get_view_code(video_path)
    
    # Generate Subject ID
    # User Note: "video isimlerini ... kişi ayrımında kullanabilirsin"
    # User Note: "train/test/val ... birinde olan kişi diğerinde yok"
    # Strategy: Use Filename as Subject ID.
    # This ensures that each video is treated as a distinct sequence/subject in the output structure.
    # The partition builder will then map this ID back to Train/Test based on original location.
    
    subject_id = os.path.splitext(os.path.basename(video_path))[0]
    subject_id = subject_id.replace(" ", "_") # Safety


    # Final Output Directory: Root / Subject / Status / View / Seq00
    # Note: OpenGait usually expects Root / Subject / Type / View / Files...
    # We are using "Status" as "Type".
    
    # Final Output File: Root / Subject / {Status}_{Seq} / View / data.pkl
    # We delay directory creation until we know how many chunks we have.
    # But we need basic info.
    
    # NOTE: OpenGait merges files in View folder as "Modalities". 
    # To have multiple temporal chunks, we must distinguish them at the STATUS (Type) level.
    # e.g. "patient_00", "patient_01".



    tracker.reset()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_indices = set(get_sampling_indices(total_frames, orig_fps, TARGET_FPS))

    frame_idx = 0
    silhouette_frames = []
    
    # Variables for simple tracking (keep biggest person)
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        if frame_idx in target_indices:
            # Predict
            results = model.predict(frame, classes=[0], retina_masks=True, verbose=False)[0]
            
            # Extract Mask
            final_mask = np.zeros((NORMALIZED_HEIGHT, NORMALIZED_WIDTH), dtype=np.uint8)
            
            if results.masks is not None:
                # Get largest mask
                masks = results.masks.data.cpu().numpy() # HW
                boxes = results.boxes.data.cpu().numpy()
                
                if len(masks) > 0:
                    # Find biggest area
                    areas = np.sum(masks, axis=(1, 2))
                    best_idx = np.argmax(areas)
                    
                    mask = (masks[best_idx] * 255).astype(np.uint8)
                    
                    # Cut to box
                    x1, y1, x2, y2 = map(int, boxes[best_idx][:4])
                    # Crop
                    mask_crop = mask[y1:y2, x1:x2]
                    
                    final_mask = preprocess_silhouette_final(mask_crop, NORMALIZED_HEIGHT, NORMALIZED_WIDTH)
                    _, final_mask = cv2.threshold(final_mask, 127, 255, cv2.THRESH_BINARY)
            
            silhouette_frames.append(final_mask)
        
        frame_idx += 1
    
    cap.release()

    # --- CHUNKING LOGIC ---
    # Rule 1: < 300 -> Pad to 300
    # Rule 2: 300 <= N < 450 -> Truncate to 300
    # Rule 3: >= 450 -> Pad to multiple of 300 and split
    
    N = len(silhouette_frames)
    if N == 0:
        return

    processed_chunks = []
    
    if N < 300:
        # Pad to 300
        padding = 300 - N
        # Pad with zeros or loop? Loop is better for gait.
        chunk = list(silhouette_frames)
        while len(chunk) < 300:
            chunk.extend(silhouette_frames) # extend with original content
        processed_chunks.append(chunk[:300]) # trim exact
        
    elif 300 <= N < 450:
        # Truncate to 300
        processed_chunks.append(silhouette_frames[:300])
        
    else: # N >= 450
        # Pad to next multiple of 300
        num_chunks = math.ceil(N / 300)
        target_len = num_chunks * 300
        
        chunk = list(silhouette_frames)
        while len(chunk) < target_len:
            # Pad with original frames looping
            chunk.extend(silhouette_frames)
        chunk = chunk[:target_len] # Ensure exact size if loop overshot slightly
        
        # Split
        for i in range(num_chunks):
            start = i * 300
            end = start + 300
            processed_chunks.append(chunk[start:end])

    # Save
    
    # Save
    
    # We need to determine the start Seq ID for this specific (Subject, Status, View) combination?
    # Actually, we are modifying the 'Status' folder name to include Seq ID.
    # e.g. .../Subject/patient_00/View/data.pkl
    
    # We need to find the next available Status_Seq index for this Subject?
    # Let's count how many "patient_XX" folders exist for this SUBJECT?
    # This is getting complicated because we are inside a loop over videos.
    # Simpler approach: 
    # Just use an incrementing counter based on existing folders in Subject dir?
    
    subject_dir = os.path.join(output_root, subject_id)
    
    for i, chunk in enumerate(processed_chunks):
        # Find next available suffix for this status
        # We want to create folder: output_root / subject_id / {status}_{k} / view / data.pkl
        
        k = 0
        while True:
            type_folder = f"{status}_{k:02d}"
            # verification path
            # But wait, if we process multiple videos of same person, we want to append.
            # If "patient_00" exists, check if it has "view" present? 
            # OpenGait allows missing views. 
            # But here we are adding a NEW sequence (new video execution).
            # So generally we should pick a NEW k for every NEW video chunk.
            # Unless we want to merge views (which we don't have, we only have 000).
            
            # Check if this type_folder matches an EXISTING processed sequence that we should append to?
            # No, user says inputs are distinct videos. Treat as distinct sequences.
            
            check_path = os.path.join(subject_dir, type_folder)
            if not os.path.exists(check_path):
                break
            # If exists, we skip to next k to avoid overwriting or merging with previous video's chunks
            k += 1
            
        final_type_folder = f"{status}_{k:02d}"
        final_file_path = os.path.join(subject_dir, final_type_folder, view, "data.pkl")
            
        save_as_pkl(chunk, final_file_path)
        print(f"Saved chunk {i+1}/{len(processed_chunks)} ({len(chunk)} frames) to {final_file_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help="Root path of user dataset (containing train/test/val)")
    parser.add_argument('--output', type=str, required=True, help="Output root for formatted silhouettes")
    args = parser.parse_args()

    model = YOLO(MODEL_NAME)
    tracker = sv.ByteTrack(track_activation_threshold=0.25, lost_track_buffer=30, frame_rate=TARGET_FPS)

    # Walk through all videos
    # Supported extensions
    exts = ('*.mp4', '*.avi', '*.mov', '*.mkv')
    video_files = []
    for ext in exts:
        video_files.extend(glob.glob(os.path.join(args.input, "**", ext), recursive=True))
    
    print(f"Found {len(video_files)} videos.")
    
    for vid in video_files:
        # SKIP 'yandan' as requested
        # Check full path for 'yandan' folder component
        parts = vid.lower().replace('\\', '/').split('/')
        if 'yandan' in parts:
            print(f"Skipping Side View: {vid}")
            continue
            
        process_video(vid, args.output, model, tracker)

if __name__ == "__main__":
    main()
