"""Generate a single visual comparing Accuracy and Macro Precision for baseline vs enhanced models.

Output: results/model_precision_accuracy.png
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_CSV = Path('results') / 'performance_summary.csv'
OUT_PATH = Path('results') / 'model_precision_accuracy.png'


def main():
    if not RESULTS_CSV.exists():
        raise SystemExit(f"Missing performance summary: {RESULTS_CSV}")
    df = pd.read_csv(RESULTS_CSV)

    # Focus on baseline vs enhanced variants (keep both enhanced models for completeness)
    keep = df[df['model'].isin(['rf_baseline', 'rf_enhanced', 'xgb_enhanced'])].copy()

    # Order for readability
    order = ['rf_baseline', 'rf_enhanced', 'xgb_enhanced']
    keep['model'] = pd.Categorical(keep['model'], categories=order, ordered=True)
    keep.sort_values('model', inplace=True)

    plt.figure(figsize=(8,5))
    ax = plt.gca()

    width = 0.35
    x = range(len(keep))

    # Bars for accuracy
    ax.bar([i - width/2 for i in x], keep['accuracy'], width=width, label='Accuracy', color='#1f77b4')
    # Bars for macro precision
    ax.bar([i + width/2 for i in x], keep['macro_precision'], width=width, label='Macro Precision', color='#ff7f0e')

    # Add value labels
    for i, (acc, prec) in enumerate(zip(keep['accuracy'], keep['macro_precision'])):
        ax.text(i - width/2, acc + 0.01, f"{acc:.2f}", ha='center', va='bottom', fontsize=9)
        ax.text(i + width/2, prec + 0.01, f"{prec:.2f}", ha='center', va='bottom', fontsize=9)

    ax.set_xticks(list(x))
    ax.set_xticklabels([m.replace('_','\n') for m in keep['model']])
    ax.set_ylim(0, max(keep['accuracy'].max(), keep['macro_precision'].max()) + 0.1)
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison: Accuracy vs Macro Precision')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=200)
    print(f"Saved plot to {OUT_PATH}")

if __name__ == '__main__':
    main()
