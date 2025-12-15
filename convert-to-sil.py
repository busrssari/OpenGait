# ===============================
# SPLIT-AWARE SILHOUETTE PIPELINE
# ===============================

import cv2
import numpy as np
import os
import glob
import math
import pandas as pd
import supervision as sv
from ultralytics import YOLO
import argparse
import torch
import gc

# --- SPLITS ---
SPLITS = ["train-data", "val-data", "test-data"]

# --- ROOT PATHS ---
DEFAULT_INPUT_ROOT = "/mnt/c/users/yusuf/OneDrive/Masaüstü/data"
DEFAULT_OUTPUT_ROOT = "silhouettes"

# --- STANDARTLAR (ScoNet / GaitSet) ---
TARGET_FPS = 15
TARGET_FRAME_COUNT = 300
NORMALIZED_HEIGHT = 64
NORMALIZED_WIDTH  = 44

MODEL_NAME = "yolov8m-seg.pt"

CLASS_LABELS = {'saglikli': 0, 'hasta': 1}
VIEW_LABELS = {'onden': 0, 'yandan': 1}

# ===============================
# YARDIMCI FONKSİYONLAR (AYNI)
# ===============================

def get_sampling_indices(total_frames, original_fps, target_fps):
    duration = total_frames / original_fps
    num_output_frames = int(duration * target_fps)
    if num_output_frames == 0:
        return set()
    return set(np.linspace(0, total_frames - 1, num_output_frames).astype(int))


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


def process_chunk_rules(chunk):
    if len(chunk) < 50:
        return None
    while len(chunk) < TARGET_FRAME_COUNT:
        chunk.extend(chunk)
    return chunk[:TARGET_FRAME_COUNT]


def save_as_images(frames, folder_path):
    os.makedirs(folder_path, exist_ok=True)
    for i, frame in enumerate(frames):
        cv2.imwrite(os.path.join(folder_path, f"{i:03d}.png"), frame)


# ===============================
# ANA İŞLEM (SPLIT-BAZLI)
# ===============================

def process_split(split_name, input_root, output_root, model):
    print(f"\n🚀 SPLIT BAŞLIYOR: {split_name.upper()}")

    tracker = sv.ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=30,
        frame_rate=TARGET_FPS
    )

    morph_kernel = np.ones((5, 5), np.uint8)

    split_input = os.path.join(input_root, split_name)
    split_output = os.path.join(output_root, split_name)

    video_files = []
    for ext in ['*.mp4', '*.mov', '*.avi']:
        video_files.extend(
            glob.glob(os.path.join(split_input, '**', ext), recursive=True)
        )

    print(f"🎥 {len(video_files)} video bulundu.")

    for idx, video_path in enumerate(video_files):
        print(f"[{idx+1}/{len(video_files)}] {video_path}")

        tracker.reset()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            continue

        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        target_indices = get_sampling_indices(total_frames, orig_fps, TARGET_FPS)
        sampled_frames = []

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in target_indices:
                sampled_frames.append(frame)
            frame_idx += 1
        cap.release()

        if not sampled_frames:
            continue

        silhouette_frames = []
        target_track_id = None

        for frame in sampled_frames:
            results = model.predict(frame, classes=[0], retina_masks=True, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            tracked = tracker.update_with_detections(detections)

            final_crop = None
            if target_track_id is None and len(tracked) > 0:
                areas = (tracked.xyxy[:, 2] - tracked.xyxy[:, 0]) * (tracked.xyxy[:, 3] - tracked.xyxy[:, 1])
                best_idx = np.argmax(areas)
                target_track_id = tracked.tracker_id[best_idx]

            if target_track_id is not None:
                mask_idx = tracked.tracker_id == target_track_id
                if np.any(mask_idx):
                    mask = (tracked.mask[mask_idx][0] * 255).astype(np.uint8)
                    bbox = tracked.xyxy[mask_idx][0]
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, morph_kernel)
                    x1, y1, x2, y2 = map(int, bbox)
                    final_crop = mask[y1:y2, x1:x2]

            if final_crop is not None:
                norm_img = preprocess_silhouette_final(final_crop, NORMALIZED_HEIGHT, NORMALIZED_WIDTH)
            else:
                norm_img = np.zeros((NORMALIZED_HEIGHT, NORMALIZED_WIDTH), dtype=np.uint8)

            _, norm_img = cv2.threshold(norm_img, 127, 255, cv2.THRESH_BINARY)
            silhouette_frames.append(norm_img)

        rel_path = os.path.relpath(video_path, split_input)
        base_name = os.path.splitext(os.path.basename(rel_path))[0]
        sub_dir = os.path.dirname(rel_path)

        for i in range(math.ceil(len(silhouette_frames) / TARGET_FRAME_COUNT)):
            chunk = silhouette_frames[i*TARGET_FRAME_COUNT:(i+1)*TARGET_FRAME_COUNT]
            final_chunk = process_chunk_rules(chunk)
            if final_chunk:
                out_dir = os.path.join(split_output, sub_dir, f"{base_name}_part{i+1}")
                save_as_images(final_chunk, out_dir)

        del sampled_frames, silhouette_frames
        gc.collect()


# ===============================
# MAIN
# ===============================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=DEFAULT_INPUT_ROOT)
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    model = YOLO(MODEL_NAME)

    for split in SPLITS:
        process_split(split, args.input, args.output, model)

    print("\n✅ TÜM SPLIT’LER BAŞARIYLA İŞLENDİ.")
