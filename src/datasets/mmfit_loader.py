from __future__ import annotations

import os
import glob
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.geometry import angle_between


# Approx OpenPose BODY_25 subset indices matching what we saw (K≈19)
OPENPOSE_MAP = {
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
    # eyes/ears indices likely 15-18 in 19-keypoint exports
}


def _to_TKC(arr: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """Normalize array to shape (T, K, C) where C>=2 and return (coords, T, K).
    Supports (T,K,C), (C,T,K) with C in {2,3}, and (T,2K).
    """
    if arr.ndim == 3:
        if arr.shape[0] in (2, 3):
            C, T, K = arr.shape
            coords = np.transpose(arr, (1, 2, 0))
        else:
            T, K, C = arr.shape
            coords = arr
        if coords.shape[2] > 2:
            coords = coords[:, :, :2]
        return coords, coords.shape[0], coords.shape[1]
    if arr.ndim == 2:
        T, twoK = arr.shape
        if twoK % 2 != 0:
            raise ValueError(f"Unexpected shape {arr.shape}; cannot reshape to (T,K,2)")
        K = twoK // 2
        coords = arr.reshape(T, K, 2)
        return coords, T, K
    raise ValueError(f"Unsupported pose array shape {arr.shape}")


def _safe_point(coords: np.ndarray, t: int, k: int) -> np.ndarray:
    if 0 <= k < coords.shape[1]:
        x, y = coords[t, k, 0], coords[t, k, 1]
        return np.array([x, y, 0.0], dtype=float)
    return np.array([np.nan, np.nan, 0.0], dtype=float)


def _compute_row_features(coords: np.ndarray, t: int, idx: Dict[str, int]) -> Dict[str, float]:
    p = lambda name: _safe_point(coords, t, idx.get(name, -1))

    l_elbow = angle_between(p('left_shoulder'), p('left_elbow'), p('left_wrist'))
    r_elbow = angle_between(p('right_shoulder'), p('right_elbow'), p('right_wrist'))
    l_knee = angle_between(p('left_hip'), p('left_knee'), p('left_ankle'))
    r_knee = angle_between(p('right_hip'), p('right_knee'), p('right_ankle'))
    l_hip = angle_between(p('left_shoulder'), p('left_hip'), p('left_knee'))
    r_hip = angle_between(p('right_shoulder'), p('right_hip'), p('right_knee'))

    left_hip = p('left_hip')
    right_hip = p('right_hip')
    mid_hip = (left_hip + right_hip) / 2.0
    nose = p('nose')
    vec = nose - mid_hip
    if np.any(np.isnan(vec[:2])):
        body_tilt = float('nan')
    else:
        body_tilt = float(np.degrees(np.arctan2(vec[1], vec[0])))

    def dist(a: np.ndarray, b: np.ndarray) -> float:
        if np.any(np.isnan(a[:2])) or np.any(np.isnan(b[:2])):
            return float('nan')
        return float(np.linalg.norm(a[:2] - b[:2]))

    shoulder_width = dist(p('left_shoulder'), p('right_shoulder'))
    hip_width = dist(left_hip, right_hip)
    torso_len = dist((p('left_shoulder') + p('right_shoulder')) / 2.0, mid_hip)
    arm_len = dist(p('left_shoulder'), p('left_wrist'))

    ratio_sh_hip = shoulder_width / (hip_width + 1e-6) if not np.isnan(shoulder_width) and not np.isnan(hip_width) else float('nan')
    ratio_arm_torso = arm_len / (torso_len + 1e-6) if not np.isnan(arm_len) and not np.isnan(torso_len) else float('nan')

    return {
        'l_elbow': l_elbow, 'r_elbow': r_elbow,
        'l_knee': l_knee, 'r_knee': r_knee,
        'l_hip': l_hip, 'r_hip': r_hip,
        'body_tilt': body_tilt,
        'ratio_sh_hip': ratio_sh_hip,
        'ratio_arm_torso': ratio_arm_torso,
        'shoulder_width': shoulder_width,
        'hip_width': hip_width,
    }


def _load_subject(subject_dir: str, idx_map: Dict[str, int] = OPENPOSE_MAP) -> pd.DataFrame:
    subj = os.path.basename(subject_dir.rstrip('/\\'))
    pose2d = glob.glob(os.path.join(subject_dir, '*_pose_2d.npy'))
    label_csv = glob.glob(os.path.join(subject_dir, '*_labels.csv'))
    if not pose2d or not label_csv:
        return pd.DataFrame()

    coords_raw = np.load(pose2d[0])
    coords, T, K = _to_TKC(coords_raw)

    # labels file seems headerless: start,end,count,exercise
    lab = pd.read_csv(label_csv[0], header=None, names=['start', 'end', 'count', 'exercise'])
    # Build per-frame exercise assignment
    exercise = np.array([None] * T, dtype=object)
    for _, r in lab.iterrows():
        s = int(r['start'])
        e = int(r['end'])
        ex = str(r['exercise'])
        s = max(0, s)
        e = min(T - 1, e)
        if s <= e:
            exercise[s:e+1] = ex

    rows: List[dict] = []
    for t in range(T):
        feats = _compute_row_features(coords, t, idx_map)
        rows.append({'video': subj, 'frame': t, 'exercise': exercise[t], **feats})

    df = pd.DataFrame(rows)
    out_dir = os.path.join('features', 'per_subject')
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, f'{subj}_features.csv'), index=False)
    return df


def load_all_subjects(base_path: str, subjects_filter: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Loop over wXX folders under base_path, read pose_2d and *labels.csv, align per-frame exercise,
    compute simple angle/ratio features, and save per-subject CSVs under features/per_subject/.

    Returns a concatenated DataFrame for all subjects with columns:
      ['video','frame','exercise','l_elbow','r_elbow','l_knee','r_knee','l_hip','r_hip',
       'body_tilt','ratio_sh_hip','ratio_arm_torso','shoulder_width','hip_width']
    """
    subjects = sorted(glob.glob(os.path.join(base_path, 'w*')))
    if subjects_filter:
        subjects = [s for s in subjects if os.path.basename(s) in set(subjects_filter)]
    dfs = []
    for sdir in subjects:
        df = _load_subject(sdir)
        if not df.empty:
            dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    all_df = pd.concat(dfs, ignore_index=True)
    # Also persist a combined pkl for convenience
    os.makedirs('features', exist_ok=True)
    all_df.to_pickle(os.path.join('features', 'mmfit_features_labeled.pkl'))
    return all_df


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='MM-Fit loader: per-frame features with exercise labels')
    ap.add_argument('--base', required=True, help='Path with wXX folders (e.g., data/mmfit_videos/mm-fit/mm-fit)')
    ap.add_argument('--subjects', help='Comma-separated list like w20,w01 to restrict processing')
    args = ap.parse_args()
    subjects = [s.strip() for s in args.subjects.split(',')] if args.subjects else None
    df = load_all_subjects(args.base, subjects_filter=subjects)
    print('Loaded frames:', len(df))
    print('Videos:', df['video'].nunique() if not df.empty else 0)
    print('Exercises:', df['exercise'].dropna().unique()[:10] if not df.empty else [])
