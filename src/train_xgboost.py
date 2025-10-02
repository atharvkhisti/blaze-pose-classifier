import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix


def main():
    ap = argparse.ArgumentParser(description='Train XGBoost model using existing RF split and feature metadata')
    ap.add_argument('--windows', required=True, help='Windows pkl (augmented)')
    ap.add_argument('--models', required=True, help='Models directory (must contain train_test_split.json & feature_metadata.json)')
    ap.add_argument('--results', required=True, help='Results directory for reports')
    ap.add_argument('--variant', choices=['baseline','enhanced'], default='enhanced')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.results, exist_ok=True)
    os.makedirs(args.models, exist_ok=True)

    df = pd.read_pickle(args.windows)
    # Load metadata
    split_path = os.path.join(args.models, 'train_test_split.json')
    feature_meta_path = os.path.join(args.models, 'feature_metadata.json')
    le_path = os.path.join(args.models, 'label_encoder.pkl')
    if not (os.path.exists(split_path) and os.path.exists(feature_meta_path) and os.path.exists(le_path)):
        raise SystemExit('Required metadata files missing. Run train_models.py first.')

    with open(split_path, 'r', encoding='utf-8') as f:
        split = json.load(f)
    with open(feature_meta_path, 'r', encoding='utf-8') as f:
        feat_meta = json.load(f)
    le = joblib.load(le_path)

    df = df.dropna(subset=['exercise']).reset_index(drop=True)
    df['exercise_lbl'] = le.transform(df['exercise'])

    train_vids = set(split['train_videos'])
    test_vids = set(split['test_videos'])
    train_df = df[df['video'].isin(train_vids)]
    test_df = df[df['video'].isin(test_vids)]

    if args.variant == 'baseline':
        cols = feat_meta['baseline_feature_columns']
        model_name = 'xgb_baseline'
    else:
        cols = feat_meta['enhanced_feature_columns']
        model_name = 'xgb_enhanced'

    X_train = train_df[cols].fillna(0)
    X_test = test_df[cols].fillna(0)
    y_train = train_df['exercise_lbl']
    y_test = test_df['exercise_lbl']

    xgb = XGBClassifier(
        n_estimators=600,
        max_depth=8,
        learning_rate=0.045,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.2,
        objective='multi:softprob',
        eval_metric='mlogloss',
        random_state=args.seed,
        n_jobs=-1,
        tree_method='hist'
    )
    xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = xgb.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0, output_dict=True)
    print(f"XGBoost ({args.variant}) classification report")
    print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

    joblib.dump(xgb, os.path.join(args.models, f'{model_name}.pkl'))
    # Save report
    pd.DataFrame(report).transpose().to_csv(os.path.join(args.results, f'report_{model_name}.csv'))

    # Append/merge summary table
    summary_path = os.path.join(args.results, 'model_comparison.csv')
    macro_f1 = np.mean([report[c]['f1-score'] for c in le.classes_])
    new_row = pd.DataFrame([{'model': model_name, 'accuracy': report['accuracy'], 'macro_f1': macro_f1}])
    if os.path.exists(summary_path):
        existing = pd.read_csv(summary_path)
        existing = existing[existing['model'] != model_name]
        comp = pd.concat([existing, new_row], ignore_index=True)
    else:
        comp = new_row
    comp.to_csv(summary_path, index=False)
    print('Updated model comparison at', summary_path)


if __name__ == '__main__':
    main()
