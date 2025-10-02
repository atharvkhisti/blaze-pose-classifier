import argparse
import csv
import glob
import os
from typing import List

import cv2
import mediapipe as mp
from tqdm import tqdm


mp_pose = mp.solutions.pose


def extract_landmarks_from_video(video_path: str, out_csv: str, sample_rate: int = 2) -> None:
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    cap = cv2.VideoCapture(video_path)
    frame_i = 0
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            header = ["video", "frame", "timestamp", "width", "height"]
            for i in range(33):
                header += [f"lm{i}_x", f"lm{i}_y", f"lm{i}_z", f"lm{i}_vis"]
            writer.writerow(header)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_i % sample_rate == 0:
                h, w = frame.shape[:2]
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(img)
                row = [os.path.basename(video_path), frame_i, cap.get(cv2.CAP_PROP_POS_MSEC), w, h]
                if res.pose_landmarks:
                    for lm in res.pose_landmarks.landmark:
                        row += [lm.x, lm.y, lm.z, lm.visibility]
                else:
                    row += [None] * (33 * 4)
                writer.writerow(row)
            frame_i += 1
    cap.release()


def find_videos_in_folders(folders: List[str]) -> List[str]:
    videos: List[str] = []
    for folder in folders:
        if not os.path.isdir(folder):
            continue
        videos.extend(glob.glob(os.path.join(folder, "**", "*.mp4"), recursive=True))
        videos.extend(glob.glob(os.path.join(folder, "**", "*.mov"), recursive=True))
        videos.extend(glob.glob(os.path.join(folder, "**", "*.avi"), recursive=True))
    return sorted(videos)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract MediaPipe BlazePose landmarks from videos")
    parser.add_argument("--input", nargs="+", required=True, help="One or more input folders with videos")
    parser.add_argument("--output", required=True, help="Output folder for landmarks CSV")
    parser.add_argument("--name", default="mmfit_videos_landmarks.csv", help="Output CSV filename")
    parser.add_argument("--sample-rate", type=int, default=2, help="Process every Nth frame (default: 2)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    out_csv = os.path.join(args.output, args.name)

    videos = find_videos_in_folders(args.input)
    print(f"Found {len(videos)} videos")
    for v in tqdm(videos):
        extract_landmarks_from_video(v, out_csv, sample_rate=args.sample_rate)
    print(f"Saved landmarks to {out_csv}")


if __name__ == "__main__":
    main()
