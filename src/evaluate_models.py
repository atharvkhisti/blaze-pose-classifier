import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score


def load_models(models_dir: str, meta: dict):
    models = []
    candidates = [
        ("rf_baseline", meta.get("baseline_feature_columns", [])),
        ("rf_enhanced", meta.get("enhanced_feature_columns", [])),
        ("xgb_enhanced", meta.get("enhanced_feature_columns", [])),
    ]
    for name, cols in candidates:
        path = Path(models_dir) / f"{name}.pkl"
        if path.exists():
            models.append((name, joblib.load(path), cols))
    return models


def evaluate(model_name, model, cols, df, y_true, test_indices, label_encoder, out_dir):
    X = df[cols].fillna(0)
    X_test = X.iloc[test_indices]
    y_test = y_true.iloc[test_indices]

    if hasattr(model, "predict_proba"):
        y_pred = model.predict(X_test)
        _ = model.predict_proba(X_test)  # probability matrix not currently persisted
    else:
        y_pred = model.predict(X_test)

    # If model outputs integer labels, map back to original class names
    if np.issubdtype(getattr(y_pred, 'dtype', type(y_pred[0])), np.integer):
        try:
            y_pred_decoded = label_encoder.inverse_transform(y_pred)
            y_pred = y_pred_decoded
        except Exception as e:
            print(f"[WARN] Failed to inverse transform predictions for {model_name}: {e}")

    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).T
    report_path = Path(out_dir) / f"report_{model_name}_test.csv"
    report_df.to_csv(report_path, index=True)

    cm = confusion_matrix(y_test, y_pred, labels=label_encoder.classes_)
    cm_df = pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_)
    cm_path = Path(out_dir) / f"confusion_{model_name}.csv"
    cm_df.to_csv(cm_path)

    # Plot confusion matrix (normalized)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=False, cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title(f"Confusion Matrix (Normalized) - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    fig_path = Path(out_dir) / f"confusion_{model_name}.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()

    summary = {
        "model": model_name,
        "n_features": len(cols),
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "macro_precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
    }
    return summary, report_df, cm_df


def write_latex_tables(perf_rows, reports, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    perf_df = pd.DataFrame(perf_rows)
    perf_cols = ["model", "accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall", "n_features"]
    perf_df = perf_df[perf_cols]
    perf_df.sort_values("accuracy", ascending=False, inplace=True)
    perf_tex = perf_df.to_latex(index=False, float_format=lambda x: f"{x:.3f}")
    (out_dir / "performance_table.tex").write_text(perf_tex)

    # Per-class table for best model (first row after sort)
    best_model = perf_df.iloc[0]["model"]
    best_report = reports[best_model]
    class_rows = best_report.loc[~best_report.index.str.contains("accuracy|macro avg|weighted avg", case=False)].copy()
    keep_cols = [c for c in ["precision", "recall", "f1-score", "support"] if c in class_rows.columns]
    class_tex = class_rows[keep_cols].to_latex(float_format=lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else x)
    (out_dir / f"per_class_{best_model}.tex").write_text(class_tex)

    # Macro note file
    # Generate README note (LaTeX comment lines). Use literal % for LaTeX comments.
    note = (
        "% Automatically generated tables. Include with:\n"
        "% \\input{results/latex/performance_table.tex}\n"
        f"% \\input{{results/latex/per_class_{best_model}.tex}}\n"
    )
    (out_dir / "README.txt").write_text(note)


def main():
    ap = argparse.ArgumentParser(description="Evaluate trained models, produce metrics, confusion matrices, and LaTeX tables")
    ap.add_argument('--windows', required=True, help='Path to window-level dataset (pickle or CSV) used for training')
    ap.add_argument('--models-dir', default='models', help='Directory containing model pickles and metadata JSON')
    ap.add_argument('--results-dir', default='results', help='Directory to write evaluation artifacts')
    args = ap.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    latex_dir = Path(args.results_dir) / 'latex'
    latex_dir.mkdir(parents=True, exist_ok=True)

    # Load windows
    if args.windows.endswith('.pkl'):
        win_df = pd.read_pickle(args.windows)
    else:
        win_df = pd.read_csv(args.windows)

    if 'exercise' not in win_df.columns:
        raise SystemExit("Dataset must contain 'exercise' column as target.")

    # Load metadata & splits
    meta_path = Path(args.models_dir) / 'feature_metadata.json'
    split_path = Path(args.models_dir) / 'train_test_split.json'
    if not meta_path.exists():
        raise SystemExit('feature_metadata.json not found in models dir.')
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    with open(split_path, 'r', encoding='utf-8') as f:
        split = json.load(f)

    # Load label encoder
    le = joblib.load(Path(args.models_dir) / 'label_encoder.pkl')

    models = load_models(args.models_dir, meta)
    if not models:
        raise SystemExit('No model pickles found to evaluate.')

    # Ensure ordering and index alignment
    test_indices = split.get('test_indices')
    if test_indices is None:
        # Attempt to infer from video column and provided test_videos list
        test_videos = set(split.get('test_videos', []))
        if 'video' in win_df.columns and test_videos:
            inferred = [i for i, v in enumerate(win_df['video']) if v in test_videos]
            if not inferred:
                raise SystemExit('train_test_split.json missing test_indices and could not infer any indices from test_videos.')
            test_indices = inferred
            # Persist augmentation back to JSON for future runs (best-effort)
            try:
                split['test_indices'] = test_indices
                with open(split_path, 'w', encoding='utf-8') as f:
                    json.dump(split, f, indent=2)
            except Exception as e:
                print(f"[WARN] Failed to persist inferred test_indices: {e}")
            print(f"[INFO] Inferred {len(test_indices)} test indices from video column.")
        else:
            raise SystemExit('train_test_split.json missing test_indices and cannot infer (need video column and test_videos).')
    if max(test_indices) >= len(win_df):
        raise SystemExit('Test indices out of range for provided window dataset.')

    y_true = win_df['exercise']

    perf_rows = []
    reports = {}
    for name, model, cols in models:
        missing = [c for c in cols if c not in win_df.columns]
        if missing:
            print(f"[WARN] Skipping {name}; missing feature columns: {len(missing)}")
            continue
        summary, report_df, cm_df = evaluate(name, model, cols, win_df, y_true, test_indices, le, args.results_dir)
        perf_rows.append(summary)
        reports[name] = report_df

    if not perf_rows:
        raise SystemExit('No models successfully evaluated.')

    perf_df = pd.DataFrame(perf_rows)
    perf_df.to_csv(Path(args.results_dir) / 'performance_summary.csv', index=False)

    write_latex_tables(perf_rows, reports, latex_dir)
    print('Evaluation complete.')
    print(perf_df.sort_values('accuracy', ascending=False))
    print(f"LaTeX tables written to: {latex_dir}")


if __name__ == '__main__':
    main()
