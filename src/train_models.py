import argparse
import json
import os
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


BASELINE_COLS = ['l_elbow_mean','r_elbow_mean','l_knee_mean','r_knee_mean','l_hip_mean','r_hip_mean']


def train_and_eval(df: pd.DataFrame, results_dir: str, models_dir: str, random_state: int = 42, save_reports: bool = True):
    df = df.dropna(subset=['exercise']).reset_index(drop=True)
    le = LabelEncoder()
    df['exercise_lbl'] = le.fit_transform(df['exercise'])

    videos = df['video'].unique()
    train_vids, test_vids = train_test_split(videos, test_size=0.2, random_state=random_state)
    train_df = df[df['video'].isin(train_vids)]
    test_df = df[df['video'].isin(test_vids)]

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # Persist split and label encoder metadata for reuse (e.g. XGBoost, realtime)
    meta = {
        'train_videos': train_vids.tolist(),
        'test_videos': test_vids.tolist(),
        'classes': le.classes_.tolist(),
        'timestamp': datetime.utcnow().isoformat(),
        'random_state': random_state,
    }
    with open(os.path.join(models_dir, 'train_test_split.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    joblib.dump(le, os.path.join(models_dir, 'label_encoder.pkl'))

    # Baseline feature set (means only of joint angles) expected already aggregated in windows dataset
    X_train_b = train_df[BASELINE_COLS].fillna(0)
    X_test_b = test_df[BASELINE_COLS].fillna(0)
    y_train = train_df['exercise_lbl']
    y_test = test_df['exercise_lbl']

    rf_b = RandomForestClassifier(n_estimators=220, max_depth=18, random_state=random_state, n_jobs=-1, class_weight='balanced')
    rf_b.fit(X_train_b, y_train)
    y_pred_b = rf_b.predict(X_test_b)

    report_b = classification_report(y_test, y_pred_b, target_names=le.classes_, zero_division=0, output_dict=True)
    print("Baseline classification report")
    print(classification_report(y_test, y_pred_b, target_names=le.classes_, zero_division=0))
    joblib.dump(rf_b, os.path.join(models_dir, 'rf_baseline.pkl'))

    # Enhanced features: include posture metrics, ratios, and all *_std plus any *_mean for dynamic features automatically present
    std_cols = [c for c in df.columns if c.endswith('_std')]
    dynamic_means = [c for c in df.columns if any(s in c for s in ('_vel_mean', '_acc_mean', '_sm_mean'))]
    enhanced_cols = BASELINE_COLS + ['body_tilt_mean','ratio_sh_hip_mean','ratio_arm_torso_mean'] + std_cols + dynamic_means
    # Deduplicate while preserving order
    seen=set(); enhanced_cols=[c for c in enhanced_cols if not (c in seen or seen.add(c))]
    X_train_e = train_df[enhanced_cols].fillna(0)
    X_test_e = test_df[enhanced_cols].fillna(0)

    rf_e = RandomForestClassifier(n_estimators=400, max_depth=24, random_state=random_state, n_jobs=-1, class_weight='balanced_subsample')
    rf_e.fit(X_train_e, y_train)
    y_pred_e = rf_e.predict(X_test_e)

    report_e = classification_report(y_test, y_pred_e, target_names=le.classes_, zero_division=0, output_dict=True)
    print("Enhanced classification report")
    print(classification_report(y_test, y_pred_e, target_names=le.classes_, zero_division=0))
    joblib.dump(rf_e, os.path.join(models_dir, 'rf_enhanced.pkl'))

    # Save feature metadata for realtime inference & comparison scripts
    feature_meta = {
        'baseline_feature_columns': BASELINE_COLS,
        'enhanced_feature_columns': enhanced_cols,
    }
    with open(os.path.join(models_dir, 'feature_metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(feature_meta, f, indent=2)

    # Confusion matrices
    cm_b = confusion_matrix(y_test, y_pred_b)
    cm_e = confusion_matrix(y_test, y_pred_e)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.heatmap(cm_b, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Baseline RF')
    plt.subplot(1, 2, 2)
    sns.heatmap(cm_e, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Enhanced RF')
    plt.tight_layout()
    out_png = os.path.join(results_dir, 'confusion_comparison.png')
    plt.savefig(out_png, dpi=200)
    print(f"Saved confusion matrices to {out_png}")

    if save_reports:
        # Persist classification reports to CSV for paper
        rb_df = pd.DataFrame(report_b).transpose()
        re_df = pd.DataFrame(report_e).transpose()
        rb_df.to_csv(os.path.join(results_dir, 'report_rf_baseline.csv'))
        re_df.to_csv(os.path.join(results_dir, 'report_rf_enhanced.csv'))
        # Summary comparison
        summary = pd.DataFrame([
            {'model':'rf_baseline','accuracy':report_b['accuracy'],'macro_f1':np.mean([report_b[c]['f1-score'] for c in le.classes_])},
            {'model':'rf_enhanced','accuracy':report_e['accuracy'],'macro_f1':np.mean([report_e[c]['f1-score'] for c in le.classes_])},
        ])
        summary.to_csv(os.path.join(results_dir, 'rf_summary.csv'), index=False)
        print('Saved RF classification reports and summary CSVs.')


def main():
    ap = argparse.ArgumentParser(description="Train RandomForest baseline and enhanced models with metadata export")
    ap.add_argument("--windows", required=True, help="Input windows pkl")
    ap.add_argument("--results", required=True, help="Results directory for figures")
    ap.add_argument("--models", required=True, help="Models directory for .pkl files")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_pickle(args.windows)
    train_and_eval(df, args.results, args.models, random_state=args.seed)


if __name__ == "__main__":
    main()
