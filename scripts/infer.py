# scripts/infer.py

import os
import argparse
import torch
import numpy as np
import cv2
from src.backend.model import Simple3DCNN

def preprocess_video_for_inference(video_path, target_frames=16, target_size=(224, 224)):
    frames = []
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    while success:
        frame = cv2.resize(frame, target_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        success, frame = cap.read()
    cap.release()

    if len(frames) < target_frames:
        while len(frames) < target_frames:
            frames.append(frames[-1].copy())
    elif len(frames) > target_frames:
        idxs = np.linspace(0, len(frames) - 1, target_frames, dtype=int)
        frames = [frames[i] for i in idxs]

    video_arr = np.stack(frames, axis=0).astype(np.float32) / 255.0  # (T, H, W, 3)
    video_tensor = torch.from_numpy(video_arr).permute(3, 0, 1, 2).unsqueeze(0)  # (1, 3, T, H, W)
    return video_tensor

def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    classes = checkpoint["classes"]
    num_classes = len(classes)
    model = Simple3DCNN(num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, classes

def main():
    parser = argparse.ArgumentParser(description="Infer SignBD-Word videos to Bangla text")
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to models/best_model.pth"
    )
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Folder containing input videos (mp4/avi)"
    )
    parser.add_argument(
        "--output_file", type=str, required=True,
        help="File to write predictions (e.g., results.txt)"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes = load_model(args.model_path, device)

    video_files = [
        f for f in os.listdir(args.input_dir)
        if f.endswith(".mp4") or f.endswith(".avi")
    ]
    video_files = sorted(video_files)

    with open(args.output_file, "w", encoding="utf-8") as fout:
        for vf in video_files:
            vf_path = os.path.join(args.input_dir, vf)
            tensor = preprocess_video_for_inference(vf_path).to(device)
            with torch.no_grad():
                outputs = model(tensor)
                predicted_idx = outputs.argmax(dim=1).item()
                predicted_word = classes[predicted_idx]
            fout.write(f"{vf}\t{predicted_word}\n")
            print(f"{vf} → {predicted_word}")

    print(f"\nInference complete. Results saved to: {args.output_file}")

if __name__ == "__main__":
    main()
