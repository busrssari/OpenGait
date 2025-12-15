import os
import numpy as np
import cv2
import pickle
from tqdm import tqdm

# =====================
# AYARLAR
# =====================
DATA_ROOT = "silhouettes"   # convert-to-sil çıktısı
SPLITS = ["train", "val", "test"]

SEQ_LEN = 30
STRIDE = 10
IMG_SIZE = (64, 44)   # (H, W)
DEBUG = True


# =====================
# FRAME OKUMA
# =====================
def read_frame(path):
    img = cv2.imread(path, 0)
    if img is None:
        img = np.zeros(IMG_SIZE, dtype=np.uint8)
    if img.shape != IMG_SIZE:
        img = cv2.resize(img, (IMG_SIZE[1], IMG_SIZE[0]))
    return img


# =====================
# DATASET OLUŞTURMA
# =====================
def build_dataset(video_folders, seq_len, stride):
    dataset = []

    person_id_map = {}   # video_name → id
    next_person_id = 0

    for folder_path in tqdm(video_folders, desc="Sekanslar işleniyor"):

        frame_files = sorted([
            f.name for f in os.scandir(folder_path)
            if f.name.endswith(".png")
        ])

        if len(frame_files) < seq_len:
            continue

        # -----------------
        # LABEL (2 sınıf)
        # -----------------
        label = 1 if (
            "hasta" in folder_path.lower()
            or "scoliosis" in folder_path.lower()
        ) else 0

        # -----------------
        # PERSON ID (video bazlı)
        # -----------------
        video_key = os.path.basename(folder_path).split("_part")[0]

        if video_key not in person_id_map:
            person_id_map[video_key] = next_person_id
            next_person_id += 1

        pid = person_id_map[video_key]

        # -----------------
        # CLIP OLUŞTURMA
        # -----------------
        for start in range(0, len(frame_files) - seq_len + 1, stride):
            clip_paths = frame_files[start:start + seq_len]

            clip = [
                read_frame(os.path.join(folder_path, f))
                for f in clip_paths
            ]

            if len(clip) != seq_len:
                continue

            sample = {
                "silhouette": np.array(clip, dtype=np.uint8),  # (30, 64, 44)
                "label": label,
                "id": pid
            }

            dataset.append(sample)

    return dataset, person_id_map


# =====================
# MAIN
# =====================
if __name__ == "__main__":

    for split in SPLITS:
        split_root = os.path.join(DATA_ROOT, split)
        print(f"\n📂 SPLIT: {split.upper()}")

        video_folders = []
        for root, dirs, files in os.walk(split_root):
            if any(f.endswith(".png") for f in files):
                video_folders.append(root)

        print(f"🔍 Toplam sekans: {len(video_folders)}")

        dataset, pid_map = build_dataset(
            video_folders,
            SEQ_LEN,
            STRIDE
        )

        save_name = f"sconet_mt_{split}.pkl"
        save_path = os.path.join(".", save_name)

        print(f"💾 Kaydediliyor: {save_path}")

        with open(save_path, "wb") as f:
            pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"✅ TAMAMLANDI")
        print(f"   • Sample sayısı: {len(dataset)}")
        print(f"   • Kişi (ID): {len(pid_map)}")
