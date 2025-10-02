"""Generate a suite of additional diagnostic plots to better understand model performance.

Outputs written to results/analysis/ :
  - per_class_metrics_rf_enhanced.png
  - per_class_f1_delta.png (enhanced - baseline)
  - feature_importance_rf_enhanced_top20.png
  - feature_importance_rf_enhanced_cumulative.png
  - precision_recall_scatter_rf_enhanced.png
  - support_vs_f1_bubble_rf_enhanced.png
  - top_confusions_rf_enhanced.png (top off-diagonal confusions)

Requirements: performance_summary.csv, report_*_test.csv, confusion_rf_enhanced.csv, models/*.pkl, models/feature_metadata.json
"""
from __future__ import annotations
from pathlib import Path
import json
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

RESULTS = Path('results')
ANALYSIS = RESULTS / 'analysis'
MODELS = Path('models')

sns.set_context('talk')
sns.set_style('whitegrid')


def load_report(name: str) -> pd.DataFrame:
    path = RESULTS / f'report_{name}_test.csv'
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, index_col=0)
    # Ensure numeric columns
    for c in ['precision','recall','f1-score','support']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def plot_per_class_metrics(report_df: pd.DataFrame, out: Path):
    per_class = report_df.loc[~report_df.index.str.contains('accuracy|macro avg|weighted avg', case=False)].copy()
    melt = per_class[['precision','recall','f1-score']].reset_index().melt(id_vars='index', value_name='value', var_name='metric')
    plt.figure(figsize=(12,6))
    sns.barplot(data=melt, x='index', y='value', hue='metric')
    plt.xticks(rotation=35, ha='right')
    plt.ylim(0,1)
    plt.ylabel('Score')
    plt.xlabel('Class')
    plt.title('Per-Class Precision / Recall / F1 (rf_enhanced)')
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def plot_f1_delta(report_base: pd.DataFrame, report_enh: pd.DataFrame, out: Path):
    def filt(df):
        return df.loc[~df.index.str.contains('accuracy|macro avg|weighted avg', case=False)]
    b = filt(report_base)['f1-score']
    e = filt(report_enh)['f1-score']
    common = sorted(set(b.index) & set(e.index))
    delta = (e.loc[common] - b.loc[common]).sort_values(ascending=False)
    plt.figure(figsize=(10,5))
    sns.barplot(x=delta.index, y=delta.values, palette=['#2ca02c' if v>0 else '#d62728' for v in delta.values])
    plt.axhline(0,color='black',linewidth=1)
    plt.ylabel('Δ F1 (Enhanced - Baseline)')
    plt.xlabel('Class')
    plt.title('Per-Class F1 Improvement')
    for i,v in enumerate(delta.values):
        plt.text(i, v + (0.01 if v>=0 else -0.02), f"{v:+.2f}", ha='center', va='bottom' if v>=0 else 'top', fontsize=9)
    plt.xticks(rotation=35, ha='right')
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def plot_feature_importance(model_path: Path, feature_meta: Path, out_bar: Path, out_cum: Path):
    model = joblib.load(model_path)
    with open(feature_meta,'r',encoding='utf-8') as f:
        meta = json.load(f)
    feat_cols = meta.get('enhanced_feature_columns')
    if not hasattr(model,'feature_importances_'):
        print('[WARN] Model has no feature_importances_; skipping feature importance plots.')
        return
    importances = pd.Series(model.feature_importances_, index=feat_cols).sort_values(ascending=False)
    top20 = importances.head(20)
    plt.figure(figsize=(8,8))
    top20.sort_values().plot(kind='barh', color='#1f77b4')
    plt.xlabel('Importance')
    plt.title('Top 20 Feature Importances (rf_enhanced)')
    plt.tight_layout()
    plt.savefig(out_bar, dpi=200)
    plt.close()

    # cumulative
    cum = importances.cumsum()
    plt.figure(figsize=(8,5))
    plt.plot(range(1,len(cum)+1), cum.values, marker='o')
    plt.axhline(0.8, color='red', linestyle='--', label='80%')
    plt.axhline(0.9, color='orange', linestyle='--', label='90%')
    # mark k for 80%
    k80 = next((i+1 for i,v in enumerate(cum.values) if v>=0.8), None)
    if k80:
        plt.text(k80, cum.values[k80-1]+0.02, f'{k80} feats -> 80%', ha='center')
    plt.ylabel('Cumulative Importance')
    plt.xlabel('Feature Rank')
    plt.ylim(0,1.05)
    plt.title('Cumulative Feature Importance (rf_enhanced)')
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_cum, dpi=200)
    plt.close()


def plot_precision_recall_scatter(report_enh: pd.DataFrame, out: Path):
    per = report_enh.loc[~report_enh.index.str.contains('accuracy|macro avg|weighted avg', case=False)].copy()
    plt.figure(figsize=(7,6))
    sizes = (per['support'] / per['support'].max()) * 800 + 50
    plt.scatter(per['precision'], per['recall'], s=sizes, alpha=0.6, c=per['f1-score'], cmap='viridis')
    for cls, row in per.iterrows():
        plt.text(row['precision'], row['recall']+0.01, cls, ha='center', fontsize=8)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('Precision vs Recall (Bubble size ~ Support, color = F1)')
    cbar = plt.colorbar()
    cbar.set_label('F1-score')
    plt.xlim(0,1)
    plt.ylim(0,1.05)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def plot_support_vs_f1(report_enh: pd.DataFrame, out: Path):
    per = report_enh.loc[~report_enh.index.str.contains('accuracy|macro avg|weighted avg', case=False)].copy()
    plt.figure(figsize=(8,5))
    sns.scatterplot(x='support', y='f1-score', data=per, s=120, hue='precision', palette='coolwarm', edgecolor='black')
    for cls, row in per.iterrows():
        plt.text(row['support'], row['f1-score']+0.01, cls, ha='center', fontsize=8)
    plt.xlabel('Support (test samples)')
    plt.ylabel('F1-score')
    plt.title('Support vs F1 (Color = Precision)')
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def plot_top_confusions(confusion_csv: Path, out: Path, top_n: int = 15):
    if not confusion_csv.exists():
        print('[WARN] Missing confusion matrix file; skipping top confusions plot.')
        return
    cm = pd.read_csv(confusion_csv, index_col=0)
    records = []
    for true_cls in cm.index:
        for pred_cls in cm.columns:
            if true_cls == pred_cls:
                continue
            val = cm.loc[true_cls, pred_cls]
            if val>0:
                records.append({'true': true_cls, 'pred': pred_cls, 'count': val})
    if not records:
        return
    df = pd.DataFrame(records).sort_values('count', ascending=False).head(top_n)
    plt.figure(figsize=(10,6))
    sns.barplot(data=df, x='count', y=df.apply(lambda r: f"{r['true']}→{r['pred']}", axis=1), palette='magma')
    plt.xlabel('Misclassified Count')
    plt.ylabel('True → Predicted')
    plt.title(f'Top {top_n} Off-Diagonal Confusions (rf_enhanced)')
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def main():
    ANALYSIS.mkdir(parents=True, exist_ok=True)

    report_base = load_report('rf_baseline')
    report_enh = load_report('rf_enhanced')

    # Per-class metrics
    plot_per_class_metrics(report_enh, ANALYSIS / 'per_class_metrics_rf_enhanced.png')
    # F1 delta
    plot_f1_delta(report_base, report_enh, ANALYSIS / 'per_class_f1_delta.png')
    # Feature importance & cumulative
    plot_feature_importance(MODELS / 'rf_enhanced.pkl', MODELS / 'feature_metadata.json',
                            ANALYSIS / 'feature_importance_rf_enhanced_top20.png',
                            ANALYSIS / 'feature_importance_rf_enhanced_cumulative.png')
    # Precision/Recall scatter
    plot_precision_recall_scatter(report_enh, ANALYSIS / 'precision_recall_scatter_rf_enhanced.png')
    # Support vs F1 bubble
    plot_support_vs_f1(report_enh, ANALYSIS / 'support_vs_f1_bubble_rf_enhanced.png')
    # Top confusions
    plot_top_confusions(RESULTS / 'confusion_rf_enhanced.csv', ANALYSIS / 'top_confusions_rf_enhanced.png')

    print(f'Written plots to {ANALYSIS}')

if __name__ == '__main__':
    main()
