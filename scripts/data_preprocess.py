import os
import cv2
import numpy as np
from tqdm import tqdm

RAW_DATA_DIR = "../data/raw/BdSLW60"
PROC_DATA_DIR = "../data/processed"

TARGET_FRAMES = 16
TARGET_SIZE = (224, 224)  # (width, height)

def preprocess_video(video_path, save_dir, class_name, sample_idx):
    cap = cv2.VideoCapture(video_path)
    frames = []
    success, frame = cap.read()
    while success:
        frame = cv2.resize(frame, TARGET_SIZE)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        success, frame = cap.read()
    cap.release()

    if len(frames) < TARGET_FRAMES:
        while len(frames) < TARGET_FRAMES:
            frames.append(frames[-1].copy())
    elif len(frames) > TARGET_FRAMES:
        idxs = np.linspace(0, len(frames) - 1, TARGET_FRAMES, dtype=int)
        frames = [frames[i] for i in idxs]

    video_arr = np.stack(frames, axis=0).astype(np.float32) / 255.0  # shape (T, H, W, 3)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{class_name}__{sample_idx}.npy")
    np.save(save_path, video_arr)



def main():
    os.makedirs(PROC_DATA_DIR, exist_ok=True)

    for class_name in sorted(os.listdir(RAW_DATA_DIR)):
        class_folder = os.path.join(RAW_DATA_DIR, class_name)
        if not os.path.isdir(class_folder):
            continue
        out_class_folder = os.path.join(PROC_DATA_DIR, class_name)
        os.makedirs(out_class_folder, exist_ok=True)

        video_files = [
            f for f in os.listdir(class_folder)
            if f.endswith(".mp4") or f.endswith(".avi")
        ]
        for idx, vid in enumerate(tqdm(video_files, desc=f"Processing {class_name}")):
            video_path = os.path.join(class_folder, vid)
            preprocess_video(video_path, out_class_folder, class_name, idx)



if __name__ == "__main__":
    main()
