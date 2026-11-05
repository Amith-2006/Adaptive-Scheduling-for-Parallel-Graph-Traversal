
#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='results/all_levels.csv')
    ap.add_argument('--out', default='results/analysis')
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    keys = ['graph', 'level']

    timing = df.groupby(
        keys + ['scheduler'], as_index=False
    ).agg(
        median_seconds=('level_seconds', 'median')
    )

    timing_wide = timing.pivot(
        index=keys,
        columns='scheduler',
        values='median_seconds'
    ).reset_index()

    timing_wide.columns.name = None

    feature_cols = [
        'frontier_size',
        'frontier_edges',
        'degree_mean',
        'degree_variance',
        'degree_skew',
        'degree_cv'
    ]

    feats = df.groupby(keys, as_index=False).agg(
        **{col: (col, 'first') for col in feature_cols}
    )

    data = feats.merge(timing_wide, on=keys)

    scheduler_cols = [
        'static',
        'dynamic',
        'guided',
        'edge_balanced'
    ]

    if not set(scheduler_cols).issubset(data.columns):
        raise ValueError('One or more scheduler timing columns are missing.')

    data['best_scheduler'] = data[scheduler_cols].idxmin(axis=1)

    data.to_csv(out / 'level_dataset.csv', index=False)

    if data.graph.nunique() < 5:
        raise SystemExit(
            'At least five distinct graph instances are required '
            'for 5-fold grouped cross-validation.'
        )

    feature_sets = {
        'mean_only': [
            'frontier_size',
            'frontier_edges',
            'degree_mean'
        ],
        'mean_plus_variance': feature_cols
    }

    cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=29
    )

    all_metrics = []
    all_predictions = []

    for feature_name, cols in feature_sets.items():

        model = make_pipeline(
            SimpleImputer(strategy='median'),
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=29,
                n_jobs=-1
            )
        )

        fold_results = []
        oof_true = []
        oof_pred = []
        for fold, (train_idx, test_idx) in enumerate(
            cv.split(data[cols], data['best_scheduler'], groups=data['graph']),
            start=1
        ):
            train = data.iloc[train_idx]
            test = data.iloc[test_idx]

            model.fit(train[cols], train['best_scheduler'])
            predictions = model.predict(test[cols])
            oof_true.extend(test['best_scheduler'].tolist())
            oof_pred.extend(predictions.tolist())

            y_true = test['best_scheduler']

            fold_pred = test[
                ['graph', 'level', 'best_scheduler'] + scheduler_cols
            ].copy()

            fold_pred['predicted_scheduler'] = predictions
            fold_pred['fold'] = fold
            fold_pred['features'] = feature_name

            selected_times = [
                fold_pred.iloc[i][predictions[i]]
                for i in range(len(fold_pred))
            ]

            fold_pred['selected_seconds'] = selected_times
            fold_pred['oracle_seconds'] = fold_pred[scheduler_cols].min(axis=1)
            fold_pred['static_seconds'] = fold_pred['static']

            fold_pred['regret'] = (
                fold_pred['selected_seconds'] /
                fold_pred['oracle_seconds'] - 1
            ) * 100

            fold_pred['static_regret'] = (
                fold_pred['static_seconds'] /
                fold_pred['oracle_seconds'] - 1
            ) * 100

            all_predictions.append(fold_pred)

            graph_results = fold_pred.groupby('graph').agg(
                selected_total=('selected_seconds', 'sum'),
                oracle_total=('oracle_seconds', 'sum'),
                static_total=('static_seconds', 'sum')
            )

            graph_results['model_regret'] = (
                graph_results['selected_total'] /
                graph_results['oracle_total'] - 1
            ) * 100

            graph_results['static_regret'] = (
                graph_results['static_total'] /
                graph_results['oracle_total'] - 1
            ) * 100

            fold_results.append({
                'features': feature_name,
                'fold': fold,
                'train_graphs': train.graph.nunique(),
                'test_graphs': test.graph.nunique(),
                'accuracy': accuracy_score(y_true, predictions),
                'balanced_accuracy': balanced_accuracy_score(
                    y_true, predictions
                ),
                'model_regret_percent': (
                    fold_pred.selected_seconds.sum() /
                    fold_pred.oracle_seconds.sum() - 1
                ) * 100,
                'static_regret_percent': (
                    fold_pred.static_seconds.sum() /
                    fold_pred.oracle_seconds.sum() - 1
                ) * 100,
                'mean_graph_regret_percent': graph_results.model_regret.mean(),
                'mean_graph_static_regret_percent': graph_results.static_regret.mean()
            })

        fold_df = pd.DataFrame(fold_results)
        all_metrics.append(fold_df)

        (out / f'{feature_name}_report.txt').write_text(
            classification_report(
                oof_true,
                oof_pred,
                labels=[
                    'static',
                    'dynamic',
                    'guided',
                    'edge_balanced'
                ],
                zero_division=0
            )
        )

    metrics = pd.concat(all_metrics, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)

    metrics.to_csv(out / 'grouped_cv_metrics.csv', index=False)
    predictions.to_csv(out / 'grouped_cv_predictions.csv', index=False)

    summary = metrics.groupby('features').agg(
        accuracy_mean=('accuracy', 'mean'),
        accuracy_std=('accuracy', 'std'),
        balanced_accuracy_mean=('balanced_accuracy', 'mean'),
        balanced_accuracy_std=('balanced_accuracy', 'std'),
        model_regret_mean=('model_regret_percent', 'mean'),
        model_regret_std=('model_regret_percent', 'std'),
        static_regret_mean=('static_regret_percent', 'mean'),
        mean_graph_regret=('mean_graph_regret_percent', 'mean')
    )

    summary.to_csv(out / 'grouped_cv_summary.csv')

    print('\nGrouped cross-validation summary:')
    print(summary.to_string())

    print(f'\nSaved analysis to {out}')


if __name__ == '__main__':
    main()
