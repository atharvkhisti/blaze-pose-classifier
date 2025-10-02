import argparse
import os
import pandas as pd

ANGLE_COLS = ['l_elbow','r_elbow','l_knee','r_knee','l_hip','r_hip','body_tilt']

def augment(df: pd.DataFrame, smooth_window: int = 5) -> pd.DataFrame:
    df = df.sort_values(['video','frame']).reset_index(drop=True)
    parts = []
    for video, vdf in df.groupby('video'):
        vdf = vdf.copy()
        for col in ANGLE_COLS:
            if col not in vdf.columns: continue
            v = vdf[col].astype(float)
            vdf[f'{col}_vel'] = v.diff().fillna(0.0)
            vdf[f'{col}_acc'] = vdf[f'{col}_vel'].diff().fillna(0.0)
            vdf[f'{col}_sm'] = v.rolling(smooth_window, min_periods=1, center=True).mean()
        parts.append(vdf)
    out = pd.concat(parts, ignore_index=True)
    return out

def main():
    ap = argparse.ArgumentParser(description='Add dynamic features (velocity, acceleration, smoothed angles)')
    ap.add_argument('--in-features', required=True, help='Input per-frame labeled features pkl')
    ap.add_argument('--out-features', required=True, help='Output augmented per-frame features pkl')
    ap.add_argument('--smooth-window', type=int, default=5)
    args = ap.parse_args()

    df = pd.read_pickle(args.in_features)
    aug = augment(df, smooth_window=args.smooth_window)
    os.makedirs(os.path.dirname(args.out_features), exist_ok=True)
    aug.to_pickle(args.out_features)
    print('Saved augmented features to', args.out_features, 'shape', aug.shape)

if __name__ == '__main__':
    main()
