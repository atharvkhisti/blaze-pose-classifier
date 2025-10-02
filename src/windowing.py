import argparse
import os
from typing import List

import numpy as np
import pandas as pd


def build_windows(feat_df: pd.DataFrame, fps: float = 30.0, window_sec: float = 1.5, step_ratio: float = 0.5) -> pd.DataFrame:
    window_size = int(window_sec * fps / 2)  # divide by 2 due to sample_rate=2 default
    step = max(1, int(window_size * step_ratio))
    rows: List[dict] = []

    numeric_cols = [
        'l_elbow','r_elbow','l_knee','r_knee','l_hip','r_hip','body_tilt','ratio_sh_hip','ratio_arm_torso'
    ]

    for video, vdf in feat_df.groupby('video'):
        vdf = vdf.sort_values('frame').reset_index(drop=True)
        n = len(vdf)
        for start in range(0, n - window_size + 1, step):
            window = vdf.iloc[start:start+window_size]
            agg = {"video": video, "start_frame": int(window.iloc[0]['frame'])}
            for c in numeric_cols:
                agg[f"{c}_mean"] = float(window[c].mean())
                agg[f"{c}_std"] = float(window[c].std())
                agg[f"{c}_min"] = float(window[c].min())
                agg[f"{c}_max"] = float(window[c].max())
            rows.append(agg)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Aggregate per-frame features into temporal windows and merge labels")
    ap.add_argument("--features", required=True, help="Input per-frame features pkl")
    ap.add_argument("--labels", required=True, help="CSV mapping: video,exercise,form")
    ap.add_argument("--out", required=True, help="Output windows pkl path")
    ap.add_argument("--fps", type=float, default=30.0, help="Assumed FPS for window sizing")
    ap.add_argument("--window-sec", type=float, default=1.5)
    ap.add_argument("--step-ratio", type=float, default=0.5)
    args = ap.parse_args()

    feat_df = pd.read_pickle(args.features)
    labels_df = pd.read_csv(args.labels)

    win_df = build_windows(feat_df, fps=args.fps, window_sec=args.window_sec, step_ratio=args.step_ratio)
    # merge labels
    win_df = win_df.merge(labels_df, on='video', how='left')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    win_df.to_pickle(args.out)
    print(f"Saved windows to {args.out} with shape {win_df.shape}")


if __name__ == "__main__":
    main()
