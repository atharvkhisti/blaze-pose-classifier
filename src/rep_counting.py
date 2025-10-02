import argparse
import pandas as pd


def count_reps_for_series(angles, down_thresh: float, up_thresh: float) -> int:
    state = 'up'
    count = 0
    for a in angles:
        if state == 'up' and a < down_thresh:
            state = 'down'
        elif state == 'down' and a > up_thresh:
            state = 'up'
            count += 1
    return count


def main():
    ap = argparse.ArgumentParser(description="Simple rule-based rep counter using elbow/knee angles")
    ap.add_argument("--features", required=True, help="Per-frame features pkl path")
    ap.add_argument("--video", required=True, help="Video filename to analyze")
    ap.add_argument("--exercise", required=True, choices=['biceps','squat','pushup'])
    args = ap.parse_args()

    df = pd.read_pickle(args.features)
    vdf = df[df['video'] == args.video].sort_values('frame')
    if vdf.empty:
        print("Video not found in features")
        return

    if args.exercise == 'biceps':
        ang = vdf['l_elbow'].fillna(method='ffill').fillna(method='bfill').values
        reps = count_reps_for_series(ang, down_thresh=50, up_thresh=160)
    elif args.exercise == 'squat':
        ang = vdf['l_knee'].fillna(method='ffill').fillna(method='bfill').values
        reps = count_reps_for_series(ang, down_thresh=70, up_thresh=160)
    else:  # pushup
        ang = vdf['l_elbow'].fillna(method='ffill').fillna(method='bfill').values
        reps = count_reps_for_series(ang, down_thresh=60, up_thresh=160)

    print(f"Estimated reps for {args.video} ({args.exercise}): {reps}")


if __name__ == "__main__":
    main()
