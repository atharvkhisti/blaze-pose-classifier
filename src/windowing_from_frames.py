import argparse
import os
from collections import Counter
from typing import List

import numpy as np
import pandas as pd


BASE_NUMERIC_COLS = [
    'l_elbow','r_elbow','l_knee','r_knee','l_hip','r_hip','body_tilt','ratio_sh_hip','ratio_arm_torso'
]


def mode_ignore_nan(values: List[object]) -> str | None:
    cleaned: List[str] = []
    for v in values:
        if v is None:
            continue
        s = str(v).strip()
        if s == '' or s.lower() in ('none', 'nan', 'null'):
            continue
        cleaned.append(s)
    if not cleaned:
        return None
    return Counter(cleaned).most_common(1)[0][0]


def build_windows_from_frames(feat_df: pd.DataFrame, window_frames: int = 60, step_ratio: float = 0.5) -> pd.DataFrame:
    step = max(1, int(window_frames * step_ratio))
    rows: List[dict] = []

    # Dynamically extend numeric cols with any *_vel, *_acc, *_sm columns
    dynamic_cols = [c for c in feat_df.columns if any(c.endswith(suf) for suf in ('_vel','_acc','_sm'))]
    numeric_cols = BASE_NUMERIC_COLS + dynamic_cols

    for video, vdf in feat_df.groupby('video'):
        vdf = vdf.sort_values('frame').reset_index(drop=True)
        n = len(vdf)
        for start in range(0, n - window_frames + 1, step):
            window = vdf.iloc[start:start+window_frames]
            agg = {"video": video, "start_frame": int(window.iloc[0]['frame'])}
            for c in numeric_cols:
                agg[f"{c}_mean"] = float(window[c].mean())
                agg[f"{c}_std"] = float(window[c].std())
                agg[f"{c}_min"] = float(window[c].min())
                agg[f"{c}_max"] = float(window[c].max())
            agg['exercise'] = mode_ignore_nan(window['exercise'].tolist())
            rows.append(agg)
    df = pd.DataFrame(rows)
    # Drop windows without a label
    df = df.dropna(subset=['exercise']).reset_index(drop=True)
    return df


def main():
    ap = argparse.ArgumentParser(description='Build labeled windows from per-frame labeled features')
    ap.add_argument('--features', required=True, help='Pickle or CSV with per-frame labeled features (columns include video, frame, exercise, numeric features)')
    ap.add_argument('--out', required=True, help='Output windows pkl path')
    ap.add_argument('--window-frames', type=int, default=45, help='Number of frames per window (e.g., 45 ~ 1.5s at 30fps)')
    ap.add_argument('--step-ratio', type=float, default=0.5, help='Step size as a ratio of window size')
    args = ap.parse_args()

    # Load features (pkl is preferred for speed)
    if args.features.endswith('.pkl'):
        feat_df = pd.read_pickle(args.features)
    else:
        feat_df = pd.read_csv(args.features)

    win_df = build_windows_from_frames(feat_df, window_frames=args.window_frames, step_ratio=args.step_ratio)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    win_df.to_pickle(args.out)
    print(f"Saved labeled windows to {args.out} with shape {win_df.shape}")


if __name__ == '__main__':
    main()
