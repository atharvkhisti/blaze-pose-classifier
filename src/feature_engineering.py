import argparse
import pickle
from pathlib import Path
import os

import numpy as np
import pandas as pd

from utils.pose_constants import POSE_IDX as IDX
from utils.geometry import angle_between


def get_landmarks(row: pd.Series) -> np.ndarray:
    pts = []
    for i in range(33):
        pts.append((row.get(f"lm{i}_x", np.nan), row.get(f"lm{i}_y", np.nan), row.get(f"lm{i}_z", np.nan)))
    return np.array(pts, dtype=float)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        lm = get_landmarks(row)
        w, h = float(row["width"]), float(row["height"])
        pts = np.copy(lm)
        # convert to pixel space
        pts[:, 0] *= w
        pts[:, 1] *= h

        l_elbow = angle_between(pts[IDX['left_shoulder']], pts[IDX['left_elbow']], pts[IDX['left_wrist']])
        r_elbow = angle_between(pts[IDX['right_shoulder']], pts[IDX['right_elbow']], pts[IDX['right_wrist']])
        l_knee = angle_between(pts[IDX['left_hip']], pts[IDX['left_knee']], pts[IDX['left_ankle']])
        r_knee = angle_between(pts[IDX['right_hip']], pts[IDX['right_knee']], pts[IDX['right_ankle']])
        l_hip = angle_between(pts[IDX['left_shoulder']], pts[IDX['left_hip']], pts[IDX['left_knee']])
        r_hip = angle_between(pts[IDX['right_shoulder']], pts[IDX['right_hip']], pts[IDX['right_knee']])

        mid_hip = (pts[IDX['left_hip']] + pts[IDX['right_hip']]) / 2.0
        nose = pts[IDX['nose']]
        vec = nose - mid_hip
        if np.any(np.isnan(vec)):
            body_tilt = float('nan')
        else:
            body_tilt = float(np.degrees(np.arctan2(vec[1], vec[0])))

        # ratios
        def dist(a, b):
            if np.any(np.isnan(a[:2])) or np.any(np.isnan(b[:2])):
                return float('nan')
            return float(np.linalg.norm(a[:2] - b[:2]))

        shoulder_width = dist(pts[IDX['left_shoulder']], pts[IDX['right_shoulder']])
        hip_width = dist(pts[IDX['left_hip']], pts[IDX['right_hip']])
        torso_len = dist((pts[IDX['left_shoulder']] + pts[IDX['right_shoulder']]) / 2.0, mid_hip)
        arm_len = dist(pts[IDX['left_shoulder']], pts[IDX['left_wrist']])

        ratio_sh_hip = shoulder_width / (hip_width + 1e-6) if not np.isnan(shoulder_width) and not np.isnan(hip_width) else float('nan')
        ratio_arm_torso = arm_len / (torso_len + 1e-6) if not np.isnan(arm_len) and not np.isnan(torso_len) else float('nan')

        rows.append({
            'video': row['video'], 'frame': int(row['frame']), 'timestamp': float(row['timestamp']),
            'l_elbow': l_elbow, 'r_elbow': r_elbow,
            'l_knee': l_knee, 'r_knee': r_knee,
            'l_hip': l_hip, 'r_hip': r_hip,
            'body_tilt': body_tilt,
            'ratio_sh_hip': ratio_sh_hip,
            'ratio_arm_torso': ratio_arm_torso,
            'shoulder_width': shoulder_width,
            'hip_width': hip_width,
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Compute per-frame features from landmarks CSV")
    ap.add_argument("--landmarks", required=True, help="Path to landmarks CSV")
    ap.add_argument("--out", required=True, help="Output features pkl path")
    args = ap.parse_args()

    df = pd.read_csv(args.landmarks)
    # Drop rows where no landmarks (lm0_x is None)
    if 'lm0_x' in df.columns:
        df = df.dropna(subset=['lm0_x']).reset_index(drop=True)

    feat_df = compute_features(df)
    Path(os.path.dirname(args.out)).mkdir(parents=True, exist_ok=True)
    feat_df.to_pickle(args.out)
    print(f"Saved per-frame features to {args.out} with shape {feat_df.shape}")


if __name__ == "__main__":
    main()
