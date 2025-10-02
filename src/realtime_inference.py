import argparse
import collections
import json
import os
import time
from datetime import datetime
from typing import Deque, Dict, List, Tuple

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd


WINDOW_SIZE = 60  # frames (~2s at 30 fps)

ANGLE_JOINT_SETS = {
    'l_elbow': ('left_shoulder', 'left_elbow', 'left_wrist'),
    'r_elbow': ('right_shoulder', 'right_elbow', 'right_wrist'),
    'l_knee': ('left_hip', 'left_knee', 'left_ankle'),
    'r_knee': ('right_hip', 'right_knee', 'right_ankle'),
    'l_hip': ('left_shoulder', 'left_hip', 'left_knee'),
    'r_hip': ('right_shoulder', 'right_hip', 'right_knee'),
}

REP_EXERCISE_PRIMARY = {
    'bicep_curls': 'l_elbow',
    'pushups': 'l_elbow',
    'tricep_extensions': 'l_elbow',
    'squats': 'l_knee',
    'lunges': 'l_knee',
    'situps': 'l_hip',
    'dumbbell_rows': 'l_elbow',
    'dumbbell_shoulder_press': 'l_elbow',
    'lateral_shoulder_raises': 'l_elbow',
    'jumping_jacks': 'l_hip',
}

# Mediapipe indices (BlazePose full body 33 landmarks) we rely on for angles
LM_IDX = {
    'nose': 0,
    'left_shoulder': 11, 'right_shoulder': 12,
    'left_elbow': 13, 'right_elbow': 14,
    'left_wrist': 15, 'right_wrist': 16,
    'left_hip': 23, 'right_hip': 24,
    'left_knee': 25, 'right_knee': 26,
    'left_ankle': 27, 'right_ankle': 28,
}


def angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    if any(np.isnan(x).any() for x in (a, b, c)):
        return np.nan
    v1 = a - b
    v2 = c - b
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return np.nan
    cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


class RepCounter:
    def __init__(self, down_thresh: float, up_thresh: float):
        self.down_thresh = down_thresh
        self.up_thresh = up_thresh
        self.state = 'up'
        self.count = 0

    def update(self, val: float):
        if np.isnan(val):
            return self.count
        if self.state == 'up' and val < self.down_thresh:
            self.state = 'down'
        elif self.state == 'down' and val > self.up_thresh:
            self.state = 'up'
            self.count += 1
        return self.count

    def reset(self):
        self.state = 'up'
        self.count = 0


class AdvancedRepCounter:
    """Stricter rep counter with amplitude, velocity, dwell, and separation constraints."""
    def __init__(self, down_thresh: float, up_thresh: float, min_amplitude: float = 25.0,
                 min_separation_frames: int = 15, min_down_hold: int = 3,
                 velocity_thresh: float = 1.5):
        self.down_thresh = down_thresh
        self.up_thresh = up_thresh
        self.min_amplitude = min_amplitude
        self.min_separation_frames = min_separation_frames
        self.min_down_hold = min_down_hold
        self.velocity_thresh = velocity_thresh
        self.count = 0
        self.phase = 'idle'  # idle -> descending -> down_hold -> ascending
        self.last_angle = np.nan
        self.last_rep_frame = -10_000
        self.down_hold_frames = 0
        self.down_min_angle = None
        self.up_max_angle = None
        self.last_quality = '---'

    def update(self, angle: float, frame_idx: int) -> int:
        if np.isnan(angle):
            return self.count
        if np.isnan(self.last_angle):
            self.last_angle = angle
            return self.count
        vel = angle - self.last_angle
        self.last_angle = angle

        # State machine
        if self.phase == 'idle':
            if angle > self.up_thresh:  # start from top
                self.up_max_angle = angle
                self.phase = 'descending'
        elif self.phase == 'descending':
            if self.up_max_angle is None or angle > self.up_max_angle:
                self.up_max_angle = angle
            if vel < -self.velocity_thresh and angle < self.down_thresh:
                self.phase = 'down_hold'
                self.down_hold_frames = 0
                self.down_min_angle = angle
        elif self.phase == 'down_hold':
            if angle < self.down_min_angle:
                self.down_min_angle = angle
            self.down_hold_frames += 1
            if self.down_hold_frames >= self.min_down_hold and vel > self.velocity_thresh:
                self.phase = 'ascending'
        elif self.phase == 'ascending':
            if self.up_max_angle is None or angle > self.up_max_angle:
                self.up_max_angle = angle
            if angle > self.up_thresh and vel < self.velocity_thresh * 0.2:  # reached top & slowing
                valid_sep = (frame_idx - self.last_rep_frame) >= self.min_separation_frames
                have_extremes = (self.down_min_angle is not None and self.up_max_angle is not None)
                if valid_sep and have_extremes:
                    amplitude = self.up_max_angle - self.down_min_angle
                    if amplitude >= self.min_amplitude:
                        self.count += 1
                        self.last_rep_frame = frame_idx
                        self.last_quality = f'OK amp={amplitude:.1f}'
                    else:
                        self.last_quality = f'LOW amp={amplitude:.1f}'
                else:
                    self.last_quality = 'REJECT sep' if not valid_sep else 'REJECT data'
                self.phase = 'descending'
                self.down_hold_frames = 0
                self.down_min_angle = None
                self.up_max_angle = angle
        return self.count

    def reset(self):
        self.__init__(self.down_thresh, self.up_thresh, self.min_amplitude,
                      self.min_separation_frames, self.min_down_hold, self.velocity_thresh)


REP_THRESHOLDS = {
    'l_elbow': (55, 160),
    'l_knee': (70, 165),
    'l_hip': (30, 150),  # situps / hip flexion heuristic
}

def draw_progress_bar(frame, fraction: float, y: int = 50, color=(0,255,0)):
    """Draw a simple horizontal progress bar (0..1)."""
    h, w, _ = frame.shape
    x0, x1 = 60, w - 40
    bar_w = x1 - x0
    frac = max(0.0, min(1.0, fraction))
    cv2.rectangle(frame, (x0, y-8), (x1, y+8), (255,255,255), 1)
    cv2.rectangle(frame, (x0, y-8), (x0 + int(bar_w * frac), y+8), color, -1)
    cv2.putText(frame, f'{frac*100:4.1f}%', (x0, y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def compute_frame_features(landmarks_xy: np.ndarray) -> Dict[str, float]:
    """Compute per-frame angles & ratios; NaNs allowed for invisible joints."""
    feats: Dict[str, float] = {}
    for name, (a, b, c) in ANGLE_JOINT_SETS.items():
        if a in LM_IDX and b in LM_IDX and c in LM_IDX:
            A = landmarks_xy[LM_IDX[a]]
            B = landmarks_xy[LM_IDX[b]]
            C = landmarks_xy[LM_IDX[c]]
            feats[name] = angle(A, B, C)
    nose = landmarks_xy[LM_IDX['nose']]
    lh = landmarks_xy[LM_IDX['left_hip']]
    rh = landmarks_xy[LM_IDX['right_hip']]
    mid = (lh + rh) / 2.0
    vec = nose - mid
    feats['body_tilt'] = float(np.degrees(np.arctan2(vec[1], vec[0]))) if not np.isnan(vec).any() else np.nan
    def dist(i, j):
        p1, p2 = landmarks_xy[LM_IDX[i]], landmarks_xy[LM_IDX[j]]
        if np.isnan(p1).any() or np.isnan(p2).any():
            return np.nan
        return float(np.linalg.norm(p1 - p2))
    shoulder_width = dist('left_shoulder', 'right_shoulder')
    hip_width = dist('left_hip', 'right_hip')
    ls = landmarks_xy[LM_IDX['left_shoulder']]
    rs = landmarks_xy[LM_IDX['right_shoulder']]
    torso_len = np.nan
    if not (np.isnan(ls).any() or np.isnan(rs).any() or np.isnan(mid).any()):
        torso_len = float(np.linalg.norm(((ls + rs) / 2.0) - mid))
    lw = landmarks_xy[LM_IDX['left_wrist']]
    arm_len = np.nan
    if not (np.isnan(ls).any() or np.isnan(lw).any()):
        arm_len = float(np.linalg.norm(ls - lw))
    feats['ratio_sh_hip'] = shoulder_width / (hip_width + 1e-6) if not np.isnan(shoulder_width) and not np.isnan(hip_width) else np.nan
    feats['ratio_arm_torso'] = arm_len / (torso_len + 1e-6) if not np.isnan(arm_len) and not np.isnan(torso_len) else np.nan
    feats['shoulder_width'] = shoulder_width
    feats['hip_width'] = hip_width
    return feats


def aggregate_window(window_frames: List[Dict[str, float]]) -> Dict[str, float]:
    df = pd.DataFrame(window_frames)
    agg: Dict[str, float] = {}
    dynamic_bases = ['l_elbow','r_elbow','l_knee','r_knee','l_hip','r_hip','body_tilt']
    for col in df.columns:
        series = df[col].astype(float)
        agg[f'{col}_mean'] = float(series.mean())
        agg[f'{col}_std'] = float(series.std())
        agg[f'{col}_min'] = float(series.min())
        agg[f'{col}_max'] = float(series.max())
        if col in dynamic_bases:
            vel = series.diff().fillna(0.0)
            acc = vel.diff().fillna(0.0)
            sm = series.rolling(5, min_periods=1, center=True).mean()
            agg[f'{col}_vel_mean'] = float(vel.mean())
            agg[f'{col}_vel_std'] = float(vel.std())
            agg[f'{col}_acc_mean'] = float(acc.mean())
            agg[f'{col}_acc_std'] = float(acc.std())
            agg[f'{col}_sm_mean'] = float(sm.mean())
            agg[f'{col}_sm_std'] = float(sm.std())
    return agg


def load_models(models_dir: str):
    with open(os.path.join(models_dir, 'feature_metadata.json'), 'r', encoding='utf-8') as f:
        meta = json.load(f)
    le = joblib.load(os.path.join(models_dir, 'label_encoder.pkl'))
    models = []
    paths = [
        ('rf_enhanced', os.path.join(models_dir, 'rf_enhanced.pkl'), meta['enhanced_feature_columns']),
        ('xgb_enhanced', os.path.join(models_dir, 'xgb_enhanced.pkl'), meta['enhanced_feature_columns']),
        ('rf_baseline', os.path.join(models_dir, 'rf_baseline.pkl'), meta['baseline_feature_columns']),
    ]
    for name, path, cols in paths:
        if os.path.exists(path):
            models.append((name, joblib.load(path), cols))
    return le, models


def main():
    ap = argparse.ArgumentParser(description='Realtime webcam exercise classification & rep counting')
    ap.add_argument('--models', default='models', help='Models directory with trained pickles and metadata')
    ap.add_argument('--device', type=int, default=0, help='Webcam device index')
    ap.add_argument('--min-prob', type=float, default=0.35, help='Minimum probability to accept prediction')
    ap.add_argument('--snapshot-dir', default='results/snapshots', help='Directory to save snapshots')
    ap.add_argument('--results-dir', default='results', help='Directory for session logs')
    ap.add_argument('--width', type=int, default=960, help='Capture width request (lower for speed)')
    ap.add_argument('--height', type=int, default=540, help='Capture height request (lower for speed)')
    ap.add_argument('--list-devices', action='store_true', help='Probe indices 0..5 and exit')
    ap.add_argument('--raw-preview', action='store_true', help='Show raw camera frames only (no pose/model) to debug camera feed')
    ap.add_argument('--skip-n', dest='skip_n', type=int, default=0, help='Process pose on every Nth frame only (0 = every frame)')
    ap.add_argument('--diagnostic', action='store_true', help='Print extra diagnostic info (frame stats, processing times)')
    ap.add_argument('--window-size', type=int, default=WINDOW_SIZE, help='Override window size (frames)')
    ap.add_argument('--auto-recover', action='store_true', help='Attempt automatic capture backend recovery if feed is black or landmarks stall')
    ap.add_argument('--model-complexity', type=int, choices=[0,1,2], default=1, help='Mediapipe pose model complexity (0 fastest)')
    ap.add_argument('--smooth', type=int, default=0, help='Prediction smoothing window (majority vote over last N accepted predictions)')
    ap.add_argument('--force-exercise', choices=list(REP_EXERCISE_PRIMARY.keys()), help='Force rep counting for a specific exercise (ignores predicted label for counting)')
    ap.add_argument('--show-angle', action='store_true', help='Overlay primary joint angle and progress bar for reps')
    ap.add_argument('--elbow-down-thresh', type=float, help='Override down (flexed) threshold for elbow-based reps')
    ap.add_argument('--elbow-up-thresh', type=float, help='Override up (extended) threshold for elbow-based reps')
    ap.add_argument('--vis-thresh', type=float, default=0.5, help='Landmark visibility minimum (set higher in low light)')
    ap.add_argument('--adaptive-reps', action='store_true', help='Enable adaptive thresholds (learn from your min/max angles)')
    ap.add_argument('--angle-smooth-alpha', type=float, default=0.3, help='EMA smoothing factor for primary angle')
    ap.add_argument('--strict-reps', action='store_true', help='Use advanced rep counting with amplitude & velocity gating')
    ap.add_argument('--min-rep-amplitude', type=float, default=25.0, help='Minimum angle excursion (deg) for a valid rep in strict mode')
    ap.add_argument('--min-rep-separation', type=int, default=15, help='Minimum frames between counted reps in strict mode')
    ap.add_argument('--min-down-hold', type=int, default=3, help='Frames to hold bottom position before ascent (strict mode)')
    ap.add_argument('--velocity-thresh', type=float, default=1.5, help='Min absolute deg/frame velocity to consider direction change (strict mode)')
    ap.add_argument('--landmark-smooth-alpha', type=float, default=0.4, help='EMA smoothing alpha for landmark coordinates (reduces jitter)')
    ap.add_argument('--rep-mode', choices=['simple','strict'], default='simple', help='Rep counting mode: simple (earlier stable) or strict (advanced gating)')
    args = ap.parse_args()

    os.makedirs(args.snapshot_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    if args.list_devices:
        print('Scanning video devices (0..5)...')
        for idx in range(0, 6):
            cap_test = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            opened = cap_test.isOpened()
            if opened:
                ret, _ = cap_test.read()
                print(f'  Device {idx}: opened={opened} frame_read={ret}')
            else:
                print(f'  Device {idx}: opened={opened}')
            cap_test.release()
        print('Done device scan. Use --device <index> to select.')
        return

    le, model_list = load_models(args.models)
    if not model_list:
        raise SystemExit('No models found. Train models first.')
    model_idx = 0

    mp_pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=args.model_complexity, enable_segmentation=False)
    mp_draw = mp.solutions.drawing_utils
    mp_style = mp.solutions.drawing_styles

    # Try backends in order for Windows
    backend_attempts = [
        ('CAP_DSHOW', cv2.CAP_DSHOW),
        ('CAP_MSMF', cv2.CAP_MSMF),
        ('CAP_ANY', 0)
    ]
    current_backend_index = 0
    cap = None
    for i, (name, backend) in enumerate(backend_attempts):
        cap = cv2.VideoCapture(args.device, backend)
        if cap.isOpened():
            print(f'[INFO] Opened device {args.device} using backend {name}')
            current_backend_index = i
            break
        cap.release()
        cap = None
    if cap is None or not cap.isOpened():
        raise SystemExit(f'Cannot open webcam (device={args.device}). Try --list-devices, close other apps, or adjust privacy settings.')

    # Set resolution (might not succeed on all cameras)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    window_size = args.window_size
    frame_buffer: Deque[Dict[str, float]] = collections.deque(maxlen=window_size)
    pred_history: Deque[int] = collections.deque(maxlen=15)
    rep_counters: Dict[str, RepCounter] = {}
    session_rows: List[Dict[str, str]] = []
    last_pred_label = ''
    last_probs: List[float] = []
    fps_time = time.time()
    frame_count = 0
    last_primary_angle = np.nan
    smooth_angle = np.nan
    adaptive_stats = {ex: {'min': None, 'max': None} for ex in REP_EXERCISE_PRIMARY.keys()}

    pose_process_time_sum = 0.0
    pose_frames = 0
    black_frame_count = 0
    last_landmark_time = time.time()
    last_recovery_time = 0.0

    def attempt_recovery(reason: str):
        nonlocal cap, current_backend_index, last_recovery_time, black_frame_count
        if time.time() - last_recovery_time < 5:
            return
        last_recovery_time = time.time()
        print(f'[RECOVER] Attempting capture re-init due to: {reason}')
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        # cycle to next backend
        current_backend_index = (current_backend_index + 1) % len(backend_attempts)
        for k in range(len(backend_attempts)):
            idx = (current_backend_index + k) % len(backend_attempts)
            name, backend = backend_attempts[idx]
            new_cap = cv2.VideoCapture(args.device, backend)
            if new_cap.isOpened():
                cap = new_cap
                current_backend_index = idx
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
                print(f'[RECOVER] Re-opened device {args.device} with backend {name}')
                black_frame_count = 0
                return
            new_cap.release()
        print('[RECOVER] Failed to reopen camera on all backends.')
    smoothed_coords = None  # (33,2) smoothed
    frame_index_global = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame is None:
            continue

        if args.raw_preview:
            cv2.imshow('Realtime Exercise Classification', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            continue

        # Basic check for black / stalled feed
        if frame.mean() < 1:
            black_frame_count += 1
            if args.diagnostic and black_frame_count % 30 == 0:
                print(f'[WARN] Still receiving black frames ({black_frame_count}).')
            if args.auto_recover and black_frame_count >= 90:  # ~3 seconds at 30 fps
                attempt_recovery('persistent black frames')
                continue
        else:
            black_frame_count = 0
        do_process = True
        if args.skip_n > 0:
            # process only every (skip-n + 1)th frame
            if frame_count % (args.skip_n + 1) != 0:
                do_process = False

        if do_process:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t0 = time.time()
            try:
                res = mp_pose.process(frame_rgb)
            except KeyboardInterrupt:
                print('\n[INFO] Pose processing interrupted by user.')
                break
            except Exception as e:
                if args.diagnostic:
                    print('[ERROR] Pose processing failed:', e)
                res = None
            t1 = time.time()
            pose_process_time_sum += (t1 - t0)
            pose_frames += 1
            if args.diagnostic and pose_frames % 30 == 0:
                avg_pose = pose_process_time_sum / max(1, pose_frames)
                print(f'[DIAG] Avg pose time per processed frame: {avg_pose*1000:.1f} ms')
        else:
            res = None

        if res and res.pose_landmarks:
            last_landmark_time = time.time()
            h, w, _ = frame.shape
            coords = np.full((33,2), np.nan, dtype=float)
            for idx, lm in enumerate(res.pose_landmarks.landmark):
                if lm.visibility >= args.vis_thresh:
                    x = lm.x * w
                    y = lm.y * h
                    if smoothed_coords is None:
                        coords[idx] = (x, y)
                    else:
                        # EMA smoothing per visible joint
                        if not np.isnan(smoothed_coords[idx,0]):
                            coords[idx,0] = args.landmark_smooth_alpha * x + (1-args.landmark_smooth_alpha) * smoothed_coords[idx,0]
                            coords[idx,1] = args.landmark_smooth_alpha * y + (1-args.landmark_smooth_alpha) * smoothed_coords[idx,1]
                        else:
                            coords[idx] = (x, y)
            smoothed_coords = coords if smoothed_coords is None else np.where(np.isnan(coords), smoothed_coords, coords)
            feats = compute_frame_features(coords)
            frame_buffer.append(feats)
            last_frame_features = feats
            # Simple-mode immediate rep counting (decoupled from classification/window)
            if args.rep_mode == 'simple':
                rep_label_simple = args.force_exercise if args.force_exercise else None
                if rep_label_simple in REP_EXERCISE_PRIMARY:
                    joint = REP_EXERCISE_PRIMARY[rep_label_simple]
                    if joint.startswith('l_'):
                        alt = 'r_' + joint[2:]
                        ang_primary = feats.get(joint, np.nan)
                        ang_alt = feats.get(alt, np.nan)
                        if np.isnan(ang_primary) and not np.isnan(ang_alt):
                            joint = alt
                            ang_primary = ang_alt
                    else:
                        ang_primary = feats.get(joint, np.nan)
                    base_key = joint.replace('r_','l_')
                    down, up = REP_THRESHOLDS.get(base_key, (60,160))
                    if base_key == 'l_elbow':
                        if args.elbow_down_thresh is not None:
                            down = args.elbow_down_thresh
                        if args.elbow_up_thresh is not None:
                            up = args.elbow_up_thresh
                    if rep_label_simple not in rep_counters:
                        rep_counters[rep_label_simple] = RepCounter(down, up)
                    rep_counters[rep_label_simple].update(ang_primary)
        else:
            # If we have not seen landmarks for a while, attempt recovery (maybe driver glitch)
            if args.auto_recover and (time.time() - last_landmark_time) > 12:
                attempt_recovery('no landmarks for 12s')
                last_landmark_time = time.time()

        pred_label = '...'
        prob = 0.0
        probs_vector = []
        if len(frame_buffer) == window_size:
            agg = aggregate_window(list(frame_buffer))
            name, model, cols = model_list[model_idx]
            # Ensure all expected cols exist
            row = {c: agg.get(c, 0.0) for c in cols}
            X = pd.DataFrame([row])[cols].fillna(0)
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)[0]
                # Down-weight lower body classes when hips/knees not visible
                lower_body = {'squats','lunges','jumping_jacks','situps'}
                if 'last_frame_features' in locals():
                    knees_missing = np.isnan(last_frame_features.get('l_knee', np.nan)) and np.isnan(last_frame_features.get('r_knee', np.nan))
                    hips_missing = np.isnan(last_frame_features.get('l_hip', np.nan)) and np.isnan(last_frame_features.get('r_hip', np.nan))
                    if knees_missing or hips_missing:
                        for i_cls, cls in enumerate(le.classes_):
                            if cls in lower_body:
                                proba[i_cls] *= 0.05
                        s = proba.sum()
                        if s > 0:
                            proba = proba / s
                idx = int(np.argmax(proba))
                pred_label = le.classes_[idx]
                prob = float(proba[idx])
                probs_vector = proba.tolist()
            else:
                idx = int(model.predict(X)[0])
                pred_label = le.classes_[idx]
                prob = 0.0
            # Probability threshold
            if prob < args.min_prob:
                pred_label = 'unknown'
            else:
                pred_history.append(idx)

            # Rep counting (window-based) skipped in simple mode (already done per frame above)
            # Decide which label to use for rep counting
            rep_label = args.force_exercise if args.force_exercise else pred_label
            if args.rep_mode == 'simple':
                # already handled; just maintain last_primary_angle for overlay if forced exercise
                if rep_label in REP_EXERCISE_PRIMARY and 'last_frame_features' in locals():
                    joint_tmp = REP_EXERCISE_PRIMARY[rep_label]
                    last_primary_angle = last_frame_features.get(joint_tmp, np.nan)
                count = rep_counters.get(rep_label, RepCounter(0,0)).count if rep_label in rep_counters else 0
            elif rep_label in REP_EXERCISE_PRIMARY:
                joint = REP_EXERCISE_PRIMARY[rep_label]
                # side adaptation: if using left joint but right has higher motion (std) or left NaN, switch
                if joint.startswith('l_'):
                    alt = 'r_' + joint[2:]
                    primary_angle = (last_frame_features.get(joint) if 'last_frame_features' in locals() else agg.get(f'{joint}_mean', np.nan))
                    alt_angle = agg.get(f'{alt}_mean', np.nan)
                    primary_std = agg.get(f'{joint}_std', 0.0)
                    alt_std = agg.get(f'{alt}_std', -1.0)
                    if (np.isnan(primary_angle) and not np.isnan(alt_angle)) or (alt_std > primary_std * 1.15):
                        joint = alt
                        primary_angle = alt_angle
                    else:
                        primary_angle = primary_angle
                else:
                    primary_angle = (last_frame_features.get(joint) if 'last_frame_features' in locals() else agg.get(f'{joint}_mean', np.nan))
                # Threshold overrides for elbow-based reps
                base_key = joint.replace('r_','l_')
                down, up = REP_THRESHOLDS.get(base_key, (60, 160))
                if base_key == 'l_elbow':
                    if args.elbow_down_thresh is not None:
                        down = args.elbow_down_thresh
                    if args.elbow_up_thresh is not None:
                        up = args.elbow_up_thresh
                if args.adaptive_reps and not np.isnan(primary_angle):
                    st = adaptive_stats[rep_label]
                    st['min'] = primary_angle if st['min'] is None else min(st['min'], primary_angle)
                    st['max'] = primary_angle if st['max'] is None else max(st['max'], primary_angle)
                    if st['min'] is not None and st['max'] is not None and st['max'] - st['min'] > 25:
                        span = st['max'] - st['min']
                        down = st['min'] + span * 0.25
                        up = st['max'] - span * 0.25
                if rep_label not in rep_counters:
                    if args.strict_reps or args.rep_mode == 'strict':
                        rep_counters[rep_label] = AdvancedRepCounter(down, up,
                                                                     min_amplitude=args.min_rep_amplitude,
                                                                     min_separation_frames=args.min_rep_separation,
                                                                     min_down_hold=args.min_down_hold,
                                                                     velocity_thresh=args.velocity_thresh)
                    else:
                        rep_counters[rep_label] = RepCounter(down, up)
                count = rep_counters[rep_label].update(primary_angle, frame_index_global) if (args.strict_reps or args.rep_mode=='strict') else rep_counters[rep_label].update(primary_angle)
                last_primary_angle = primary_angle
                if not np.isnan(primary_angle):
                    if np.isnan(smooth_angle):
                        smooth_angle = primary_angle
                    else:
                        smooth_angle = args.angle_smooth_alpha * primary_angle + (1 - args.angle_smooth_alpha) * smooth_angle
            else:
                count = 0
                last_primary_angle = np.nan
        else:
            count = 0

        # Draw landmarks
        if res.pose_landmarks:
            mp_draw.draw_landmarks(
                frame,
                res.pose_landmarks,
                mp.solutions.pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_style.get_default_pose_landmarks_style())

        # Overlay info
        fps_now = time.time()
        if fps_now - fps_time >= 1.0:
            fps = frame_count / (fps_now - fps_time)
            fps_time = fps_now
            frame_count = 0
        else:
            fps = None
        frame_count += 1

        # Prediction smoothing (majority vote) if enabled
        if args.smooth and len(pred_history) >= args.smooth:
            import collections as _c
            last_labels = [le.classes_[i] for i in list(pred_history)[-args.smooth:]]
            most = _c.Counter(last_labels).most_common(1)[0][0]
            # Only override if not unknown and majority present
            if most != 'unknown':
                pred_label = most

        rep_display_label = args.force_exercise if args.force_exercise else pred_label
        overlay_lines = [
            f'Model: {model_list[model_idx][0]}',
            f'Pred: {pred_label} ({prob:.2f})',
            f'Reps[{rep_display_label}]: {rep_counters.get(rep_display_label, RepCounter(0,0)).count if rep_display_label in rep_counters else 0}',
            f'Window: {len(frame_buffer)}/{window_size}',
            f'FPS: {fps:.1f}' if fps else ''
        ]
        y0 = 20
        for ln in overlay_lines:
            if not ln:
                continue
            cv2.putText(frame, ln, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 1, cv2.LINE_AA)
            y0 += 18

        # Angle overlay / progress bar for rep feedback
        if args.show_angle and not np.isnan(last_primary_angle):
            # Determine thresholds shown (for forced exercise if any)
            target_label = args.force_exercise if args.force_exercise else pred_label
            if target_label in REP_EXERCISE_PRIMARY:
                joint_key = REP_EXERCISE_PRIMARY[target_label].replace('r_','l_')
                dflt_down, dflt_up = REP_THRESHOLDS.get(joint_key, (60,160))
                if joint_key == 'l_elbow':
                    if args.elbow_down_thresh is not None:
                        dflt_down = args.elbow_down_thresh
                    if args.elbow_up_thresh is not None:
                        dflt_up = args.elbow_up_thresh
                if args.adaptive_reps:
                    st = adaptive_stats.get(target_label)
                    if st and st['min'] is not None and st['max'] is not None and st['max'] - st['min'] > 25:
                        span = st['max'] - st['min']
                        dflt_down = st['min'] + span * 0.25
                        dflt_up = st['max'] - span * 0.25
                rng = (dflt_up - dflt_down) if dflt_up > dflt_down else 1.0
                display_angle = smooth_angle if not np.isnan(smooth_angle) else last_primary_angle
                frac = (display_angle - dflt_down) / rng
                draw_progress_bar(frame, frac, y=frame.shape[0]-40)
                quality_txt = ''
                if args.strict_reps and rep_display_label in rep_counters:
                    rc = rep_counters[rep_display_label]
                    if isinstance(rc, AdvancedRepCounter):
                        quality_txt = f' | {rc.last_quality}'
                cv2.putText(frame, f'Angle: {display_angle:.1f}  (down {dflt_down:.1f} / up {dflt_up:.1f}){quality_txt}', (10, frame.shape[0]-60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 1, cv2.LINE_AA)

        cv2.imshow('Realtime Exercise Classification', frame)
        frame_index_global += 1

        # Log row
        if probs_vector:
            session_rows.append({
                'ts': time.time(),
                'model': model_list[model_idx][0],
                'prediction': pred_label,
                'prob': prob,
            })

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            model_idx = (model_idx + 1) % len(model_list)
        elif key == ord('r'):
            for rc in rep_counters.values():
                rc.reset()
        elif key == ord('s'):
            snap_name = datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f') + '.png'
            cv2.imwrite(os.path.join(args.snapshot_dir, snap_name), frame)
            print('Saved snapshot', snap_name)

    cap.release()
    cv2.destroyAllWindows()

    if session_rows:
        log_path = os.path.join(args.results_dir, f'realtime_session_{int(time.time())}.csv')
        pd.DataFrame(session_rows).to_csv(log_path, index=False)
        print('Saved session log to', log_path)


if __name__ == '__main__':
    main()
