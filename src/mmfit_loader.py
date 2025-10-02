import os
import glob
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.geometry import angle_between


# OpenPose-like 19 joints used in MM-Fit 2D arrays observed as (2, T, K=19)
# Index map based on BODY_25 subset ordering used earlier
OP_IDX = {
    'nose': 0,
    'neck': 1,
    'right_shoulder': 2,
    'right_elbow': 3,
    'right_wrist': 4,
    'left_shoulder': 5,
    'left_elbow': 6,
    'left_wrist': 7,
    'mid_hip': 8,
    'right_hip': 9,
    'right_knee': 10,
    'right_ankle': 11,
    'left_hip': 12,
    'left_knee': 13,
    'left_ankle': 14,
    'right_eye': 15,
    'left_eye': 16,
    'right_ear': 17,
    'left_ear': 18,
}


def _load_pose2d(path: str) -> np.ndarray:
    """Load 2D pose array and normalize to shape (T, K, 2)."""
    arr = np.load(path)
    if arr.ndim == 3:
        if arr.shape[0] in (2, 3):
            # (C, T, K) -> (T, K, C)
            arr = np.transpose(arr, (1, 2, 0))
        # ensure only x,y
        if arr.shape[2] > 2:
            arr = arr[:, :, :2]
        return arr
    elif arr.ndim == 2:
        T, twoK = arr.shape
        if twoK % 2 != 0:
            raise ValueError(f"Unexpected 2D pose shape {arr.shape} for {path}")
        K = twoK // 2
        return arr.reshape(T, K, 2)
    else:
        raise ValueError(f"Unsupported pose array shape {arr.shape} for {path}")


def _load_labels_csv(path: str) -> pd.DataFrame:
    """MM-Fit labels are headerless CSV with columns: start_frame,end_frame,rep_count,exercise."""
    df = pd.read_csv(path, header=None, names=['start', 'end', 'reps', 'exercise'])
    # ensure ints
    df['start'] = df['start'].astype(int)
    df['end'] = df['end'].astype(int)
    df['exercise'] = df['exercise'].astype(str)
    return df


def _compute_frame_labels(T: int, seg_df: pd.DataFrame) -> np.ndarray:
    """Return an array of length T with exercise label per frame (or 'unknown')."""
    labels = np.array(['unknown'] * T, dtype=object)
    for _, r in seg_df.iterrows():
        s, e, ex = int(r['start']), int(r['end']), r['exercise']
        s = max(0, s)
        e = min(T - 1, e)
        if e >= s:
            labels[s:e + 1] = ex
    return labels


def _compute_angles_2d(coords: np.ndarray) -> Dict[str, float]:
    """Compute joint angles and simple posture ratios from a (K,2) coord array."""
    K = coords.shape[0]
    def g(name: str) -> Optional[np.ndarray]:
        idx = OP_IDX.get(name, None)
        if idx is None or idx >= K:
            return None
        return coords[idx]

    def ang(a, b, c) -> float:
        if a is None or b is None or c is None:
            return float('nan')
        # pad to 3D for geometry helper
        pa = np.array([a[0], a[1], 0.0])
        pb = np.array([b[0], b[1], 0.0])
        pc = np.array([c[0], c[1], 0.0])
        return angle_between(pa, pb, pc)

    l_elbow = ang(g('left_shoulder'), g('left_elbow'), g('left_wrist'))
    r_elbow = ang(g('right_shoulder'), g('right_elbow'), g('right_wrist'))
    l_knee = ang(g('left_hip'), g('left_knee'), g('left_ankle'))
    r_knee = ang(g('right_hip'), g('right_knee'), g('right_ankle'))
    l_hip = ang(g('left_shoulder'), g('left_hip'), g('left_knee'))
    r_hip = ang(g('right_shoulder'), g('right_hip'), g('right_knee'))

    # body tilt: vector nose -> mid_hip
    nose = g('nose')
    lh, rh = g('left_hip'), g('right_hip')
    if nose is None or lh is None or rh is None:
        body_tilt = float('nan')
    else:
        mid_hip = (lh + rh) / 2.0
        vec = nose - mid_hip
        body_tilt = float(np.degrees(np.arctan2(vec[1], vec[0])))

    def dist(a, b) -> float:
        if a is None or b is None:
            return float('nan')
        return float(np.linalg.norm(a - b))

    shoulder_width = dist(g('left_shoulder'), g('right_shoulder'))
    hip_width = dist(g('left_hip'), g('right_hip'))
    l_sh, r_sh = g('left_shoulder'), g('right_shoulder')
    if l_sh is None or r_sh is None or lh is None or rh is None:
        torso_len = float('nan')
    else:
        mid_sh = (l_sh + r_sh) / 2.0
        mid_hip = (lh + rh) / 2.0
        torso_len = dist(mid_sh, mid_hip)
    arm_len = dist(g('left_shoulder'), g('left_wrist'))

    ratio_sh_hip = shoulder_width / (hip_width + 1e-6) if not np.isnan(shoulder_width) and not np.isnan(hip_width) else float('nan')
    ratio_arm_torso = arm_len / (torso_len + 1e-6) if not np.isnan(arm_len) and not np.isnan(torso_len) else float('nan')

    return {
        'l_elbow': l_elbow,
        'r_elbow': r_elbow,
        'l_knee': l_knee,
        'r_knee': r_knee,
        'l_hip': l_hip,
        'r_hip': r_hip,
        'body_tilt': body_tilt,
        'ratio_sh_hip': ratio_sh_hip,
        'ratio_arm_torso': ratio_arm_torso,
        'shoulder_width': shoulder_width,
        'hip_width': hip_width,
    }


def load_all_subjects(base_path: str, out_dir: Optional[str] = 'features/per_subject', compute_features: bool = True) -> pd.DataFrame:
    """Loop subjects in base_path (w00..), align per-frame exercise labels, compute optional features, and save per-subject CSVs.

    Returns a concatenated DataFrame across subjects with columns:
      ['video','frame','exercise', ... feature columns ...]
    """
    subjects = sorted([p for p in glob.glob(os.path.join(base_path, 'w*')) if os.path.isdir(p)])
    all_rows: List[pd.DataFrame] = []
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    for spath in subjects:
        vid = os.path.basename(spath)
        pose2d_files = glob.glob(os.path.join(spath, '*_pose_2d.npy'))
        label_files = glob.glob(os.path.join(spath, '*_labels.csv'))
        if not pose2d_files or not label_files:
            continue
        pose_path = pose2d_files[0]
        lab_path = label_files[0]

        coords = _load_pose2d(pose_path)  # (T,K,2)
        T, K, _ = coords.shape
        segs = _load_labels_csv(lab_path)
        frame_labels = _compute_frame_labels(T, segs)

        rows: List[dict] = []
        for t in range(T):
            row = {'video': vid, 'frame': t, 'exercise': frame_labels[t]}
            if compute_features:
                row.update(_compute_angles_2d(coords[t]))
            rows.append(row)
        df = pd.DataFrame(rows)

        if out_dir:
            out_csv = os.path.join(out_dir, f'{vid}_features.csv')
            df.to_csv(out_csv, index=False)

        all_rows.append(df)

    if all_rows:
        return pd.concat(all_rows, ignore_index=True)
    return pd.DataFrame(columns=['video','frame','exercise'])


def make_windows_from_labeled_frames(frames_df: pd.DataFrame, fps: float = 30.0, window_sec: float = 1.5, step_ratio: float = 0.5) -> pd.DataFrame:
    """Build windows from per-frame labeled features and assign window label by majority class in the window."""
    window_size = int(window_sec * fps / 2)  # if data was subsampled later, adjust this accordingly
    step = max(1, int(window_size * step_ratio))

    numeric_cols = [c for c in frames_df.columns if c not in ('video','frame','exercise')]
    rows: List[dict] = []
    for vid, vdf in frames_df.groupby('video'):
        vdf = vdf.sort_values('frame').reset_index(drop=True)
        n = len(vdf)
        for start in range(0, n - window_size + 1, step):
            window = vdf.iloc[start:start+window_size]
            agg = {"video": vid, "start_frame": int(window.iloc[0]['frame'])}
            # label by mode of exercise in window (excluding 'unknown' if possible)
            ex_counts = window['exercise'].value_counts()
            if 'unknown' in ex_counts and len(ex_counts) > 1:
                ex_counts = ex_counts.drop('unknown')
            agg['exercise'] = ex_counts.idxmax() if not ex_counts.empty else 'unknown'
            for c in numeric_cols:
                agg[f"{c}_mean"] = float(window[c].mean())
                agg[f"{c}_std"] = float(window[c].std())
                agg[f"{c}_min"] = float(window[c].min())
                agg[f"{c}_max"] = float(window[c].max())
            rows.append(agg)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Load MM-Fit per-subject pose+labels and export features and windows')
    ap.add_argument('--base', required=True, help='Base path containing wXX folders')
    ap.add_argument('--out-frames', default='features/mmfit_labeled_frames.pkl')
    ap.add_argument('--out-windows', default='features/mmfit_windows_labeled.pkl')
    ap.add_argument('--save-per-subject', action='store_true', help='Save per-subject CSVs under features/per_subject')
    ap.add_argument('--fps', type=float, default=30.0)
    ap.add_argument('--window-sec', type=float, default=1.5)
    ap.add_argument('--step-ratio', type=float, default=0.5)
    args = ap.parse_args()

    frames = load_all_subjects(args.base, out_dir=('features/per_subject' if args.save_per_subject else None), compute_features=True)
    os.makedirs(os.path.dirname(args.out_frames), exist_ok=True)
    frames.to_pickle(args.out_frames)
    print(f'Saved frames with labels to {args.out_frames} shape={frames.shape}')

    windows = make_windows_from_labeled_frames(frames, fps=args.fps, window_sec=args.window_sec, step_ratio=args.step_ratio)
    os.makedirs(os.path.dirname(args.out_windows), exist_ok=True)
    windows.to_pickle(args.out_windows)
    print(f'Saved labeled windows to {args.out_windows} shape={windows.shape}')
