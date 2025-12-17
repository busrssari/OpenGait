
# ===============================
# STREAM PIPELINE: VIDEO -> PKL
# (No Intermediate PNGs)
# ===============================

import cv2
import numpy as np
import os
import glob
import math
import pickle
import supervision as sv
from ultralytics import YOLO
import argparse
import torch
import gc
from tqdm import tqdm

# --- AYARLAR ---
SPLITS = ["train-data", "test-data"]
INPUT_ROOT = "/mnt/c/Users/yusuf/OneDrive/Masaüstü/transfer-data"

# --- MODEL VE GÖRÜNTÜ AYARLARI ---
MODEL_NAME = "yolov8m-seg.pt"
TARGET_FPS = 15
TARGET_FRAME_COUNT = 300
NORMALIZED_HEIGHT = 64
NORMALIZED_WIDTH = 44
SEQ_LEN = 30
STRIDE = 10

# ===============================
# YARDIMCI FONKSİYONLAR
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
    if len(chunk) < 150:
        return None
    while len(chunk) < TARGET_FRAME_COUNT:
        chunk.extend(chunk)
    return chunk[:TARGET_FRAME_COUNT]

# ===============================
# ANA İŞLEM (SPLIT-BAZLI)
# ===============================

def process_split(split_name, input_root, model):
    print(f"\n🚀 SPLIT BAŞLIYOR: {split_name.upper()}")

    tracker = sv.ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=30,
        frame_rate=TARGET_FPS
    )
    morph_kernel = np.ones((5, 5), np.uint8)

    split_input = os.path.join(input_root, split_name)
    
    video_files = []
    for ext in ['*.mp4', '*.mov', '*.avi']:
        video_files.extend(
            glob.glob(os.path.join(split_input, '**', ext), recursive=True)
        )
    
    print(f"🎥 {len(video_files)} video bulundu.")
    
    dataset = []
    person_id_map = {}
    next_person_id = 0

    for idx, video_path in enumerate(video_files):
        print(f"[{idx+1}/{len(video_files)}] İşleniyor: {os.path.basename(video_path)}")

        # 1. LABEL BELİRLEME
        # Klasör yolunda "hasta" veya "scoliosis" geçiyorsa 1, yoksa 0
        folder_path = os.path.dirname(video_path)
        label = 1 if ("hasta" in folder_path.lower() or "scoliosis" in folder_path.lower()) else 0

        # 2. PERSON ID BELİRLEME
        # Dosya adını baz alıyoruz (part bilgisi zaten dosya sonunda yok, videonun kendisi bütün)
        video_key = os.path.splitext(os.path.basename(video_path))[0]
        
        if video_key not in person_id_map:
            person_id_map[video_key] = next_person_id
            next_person_id += 1
        
        pid = person_id_map[video_key]

        # 3. VİDEO OKUMA VE SİLÜET ÇIKARMA
        tracker.reset()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"⚠️ Video açılamadı: {video_path}")
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
                    
                    # Sınır kontrolü
                    h_img, w_img = mask.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w_img, x2), min(h_img, y2)
                    
                    if x2 > x1 and y2 > y1:
                        final_crop = mask[y1:y2, x1:x2]

            if final_crop is not None:
                norm_img = preprocess_silhouette_final(final_crop, NORMALIZED_HEIGHT, NORMALIZED_WIDTH)
            else:
                norm_img = np.zeros((NORMALIZED_HEIGHT, NORMALIZED_WIDTH), dtype=np.uint8)

            _, norm_img = cv2.threshold(norm_img, 127, 255, cv2.THRESH_BINARY)
            silhouette_frames.append(norm_img)

        # 4. CHUNKING & DATASET'E EKLEME
        # Uzun videoları 300 karelik parçalara böl
        num_parts = math.ceil(len(silhouette_frames) / TARGET_FRAME_COUNT)
        
        for i in range(num_parts):
            chunk = silhouette_frames[i*TARGET_FRAME_COUNT : (i+1)*TARGET_FRAME_COUNT]
            final_chunk = process_chunk_rules(chunk) # Padding yapar veya kısa ise None döner
            
            if final_chunk:
                # final_chunk bir list of numpy arrays.
                # create-pkl-file mantığına göre bunu doğrudan gait sequence olarak ekleyeceğiz.
                # Ancak create-pkl-file'da bir de SEQ_LEN (30) ile kesme işlemi vardı.
                # Burada OpenGait standartlarında genellikle 300 karelik bir "sequence"
                # PKL içine tek bir "set" olarak değil, alt klipler halinde girer mi?
                # create-pkl-file.py, 300 karelik klasörü okuyup, içinden 30'luk strideli klipler çıkarıyordu.
                # Biz de aynısını yapacağız.
                
                # final_chunk: [300, 64, 44] boyutunda resim listesi
                
                # 30'luk klipler oluştur
                for start in range(0, len(final_chunk) - SEQ_LEN + 1, STRIDE):
                    clip = final_chunk[start:start + SEQ_LEN]
                    
                    if len(clip) != SEQ_LEN:
                        continue
                        
                    sample = {
                        "silhouette": np.array(clip, dtype=np.uint8), # (30, 64, 44)
                        "label": label,
                        "id": pid
                    }
                    dataset.append(sample)

        # Bellek Temizliği
        del sampled_frames, silhouette_frames
        gc.collect()

    # 5. PKL KAYDETME
    clean_split = split_name.replace("-data", "")
    save_name = f"transfer_data_{clean_split}.pkl"
    save_path = os.path.join(".", save_name)
    
    print(f"\n💾 PKL Kaydediliyor: {save_path}")
    print(f"   • Toplam Sample: {len(dataset)}")
    print(f"   • Toplam Kişi: {len(person_id_map)}")
    
    with open(save_path, "wb") as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("✅ PKL Yazıldı.")

# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    
    print(f"Kullanılan Model: {MODEL_NAME}")
    model = YOLO(MODEL_NAME)
    
    for split in SPLITS:
        process_split(split, INPUT_ROOT, model)
        
    print("\n🎉 TÜM İŞLEMLER TAMAMLANDI. (Intermediate dosyalar oluşturulmadı)")
