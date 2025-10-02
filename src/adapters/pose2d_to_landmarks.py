import argparse
import csv
import glob
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

# Target is a BlazePose-like CSV schema with 33 landmarks (x,y,z,visibility)
# We'll fill only the subset we can map from COCO-17 or OpenPose-25; others become None.

BLAZEPOSE_SIZE = 33

# Minimal joints we use in downstream features
TARGET_JOINTS = {
    'nose': 0,
    'left_shoulder': 11,
    'right_shoulder': 12,
    'left_elbow': 13,
    'right_elbow': 14,
    'left_wrist': 15,
    'right_wrist': 16,
    'left_hip': 23,
    'right_hip': 24,
    'left_knee': 25,
    'right_knee': 26,
    'left_ankle': 27,
    'right_ankle': 28,
}

# COCO-17 indices
COCO17 = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16,
}

# OpenPose-25 BODY_25 indices (approx)
OPENPOSE25 = {
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
    # feet and toes omitted for our minimal mapping
}

SCHEMAS = {
    'coco17': COCO17,
    'openpose25': OPENPOSE25,
}


def infer_schema(n_kpts: int) -> Optional[str]:
    # Heuristics: many MM-Fit exports use an OpenPose-like 18/19 joint set (with neck)
    if n_kpts == 17:
        return 'coco17'
    if n_kpts in (18, 19, 25):
        return 'openpose25'
    return None


def build_mapping(schema_name: str) -> Dict[int, int]:
    src = SCHEMAS[schema_name]
    mapping: Dict[int, int] = {}
    # map only the keys we need
    pairs = [
        ('nose', 'nose'),
        ('left_shoulder', 'left_shoulder'),
        ('right_shoulder', 'right_shoulder'),
        ('left_elbow', 'left_elbow'),
        ('right_elbow', 'right_elbow'),
        ('left_wrist', 'left_wrist'),
        ('right_wrist', 'right_wrist'),
        ('left_hip', 'left_hip'),
        ('right_hip', 'right_hip'),
        ('left_knee', 'left_knee'),
        ('right_knee', 'right_knee'),
        ('left_ankle', 'left_ankle'),
        ('right_ankle', 'right_ankle'),
    ]
    for name_src, name_tgt in pairs:
        if name_src in src and name_tgt in TARGET_JOINTS:
            mapping[src[name_src]] = TARGET_JOINTS[name_tgt]
    return mapping


def write_csv_header(writer: csv.writer) -> None:
    header = ['video', 'frame', 'timestamp', 'width', 'height']
    for i in range(BLAZEPOSE_SIZE):
        header += [f'lm{i}_x', f'lm{i}_y', f'lm{i}_z', f'lm{i}_vis']
    writer.writerow(header)


essential_joint_names = list(TARGET_JOINTS.keys())


def convert_pose2d_npy(npy_path: str, writer: csv.writer, video_id: str, mapping: Dict[int, int], stride: int = 1) -> None:
    arr = np.load(npy_path)  # supports (T,K,2), (T,2K), or (C,T,K) with C in {2,3}
    if arr.ndim == 3:
        if arr.shape[0] in (2, 3):
            # shape (C, T, K) -> transpose to (T, K, C)
            C, T, K = arr.shape
            coords = np.transpose(arr, (1, 2, 0))
        else:
            # shape (T, K, C)
            T, K, C = arr.shape
            coords = arr
        # if we have more than 2 channels, keep only x,y
        if coords.shape[2] > 2:
            coords = coords[:, :, :2]
    elif arr.ndim == 2:
        # try to reshape (T, 2K)
        T, twoK = arr.shape
        if twoK % 2 != 0:
            raise ValueError(f"Unexpected pose array shape {arr.shape} in {npy_path}")
        K = twoK // 2
        coords = arr.reshape(T, K, 2)
    else:
        raise ValueError(f"Unsupported pose array shape {arr.shape} in {npy_path}")

    timestamp_ms = 0.0
    for t in range(0, T, stride):
        row = [video_id, t, timestamp_ms, 1, 1]  # width=1,height=1; positions assumed normalized/pixels unknown
        # initialize all landmarks to None
        lm_vals: List[Optional[float]] = [None] * (BLAZEPOSE_SIZE * 4)
        # fill mapped joints
        for src_idx, tgt_idx in mapping.items():
            if src_idx < K:
                x, y = float(coords[t, src_idx, 0]), float(coords[t, src_idx, 1])
                off = tgt_idx * 4
                lm_vals[off + 0] = x
                lm_vals[off + 1] = y
                lm_vals[off + 2] = 0.0  # z unknown
                lm_vals[off + 3] = 1.0  # visibility default
        row += lm_vals
        writer.writerow(row)
        timestamp_ms += 33.33  # ~30 fps placeholder


def main():
    ap = argparse.ArgumentParser(description='Convert pose_2d.npy sequences to BlazePose-like landmarks CSV')
    ap.add_argument('--input', required=True, help='Root folder containing wXX subfolders with *_pose_2d.npy')
    ap.add_argument('--output', required=True, help='Output folder for CSV')
    ap.add_argument('--name', default='mmfit_pose2d_landmarks.csv', help='Output CSV filename')
    ap.add_argument('--schema', choices=['auto','coco17','openpose25'], default='auto', help='Input keypoint schema')
    ap.add_argument('--stride', type=int, default=1, help='Take every Nth frame')
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    out_csv = os.path.join(args.output, args.name)

    npy_files = sorted(glob.glob(os.path.join(args.input, 'w*', '*_pose_2d.npy')))
    if not npy_files:
        print('No *_pose_2d.npy files found under', args.input)
        return

    # Try to infer schema if requested
    schema_name = args.schema
    if schema_name == 'auto':
        sample = np.load(npy_files[0])
        if sample.ndim == 3:
            if sample.shape[0] in (2, 3):
                # (C, T, K)
                K = sample.shape[2]
            else:
                # (T, K, C)
                K = sample.shape[1]
        elif sample.ndim == 2:
            K = sample.shape[1] // 2
        else:
            K = 0
        inferred = infer_schema(K)
        if inferred is None:
            raise SystemExit(f"Could not infer schema from K={K}. Specify --schema and/or implement a custom mapping.")
        schema_name = inferred
        print(f"Inferred schema: {schema_name} (K={K})")

    mapping = build_mapping(schema_name)

    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        write_csv_header(writer)
        for npy_path in npy_files:
            # derive a video_id (e.g., w00) from folder name
            parent = os.path.basename(os.path.dirname(npy_path))
            video_id = parent  # e.g., 'w00'
            convert_pose2d_npy(npy_path, writer, video_id, mapping, stride=args.stride)

    print('Saved landmarks to', out_csv)


if __name__ == '__main__':
    main()
