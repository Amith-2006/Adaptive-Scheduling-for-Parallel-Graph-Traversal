
import argparse
from pathlib import Path

import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--input',
        default='results/corrected/analysis/level_dataset.csv'
    )
    ap.add_argument(
        '--out',
        default='results/corrected/analysis/lofo'
    )
    ap.add_argument(
        '--balance-graphs',
        action='store_true'
    )
    args = ap.parse_args()

    data = pd.read_csv(args.input)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    feature_sets = {
        'mean_only': [
            'frontier_size',
            'frontier_edges',
            'degree_mean'
        ],
        'mean_plus_variance': [
            'frontier_size',
            'frontier_edges',
            'degree_mean',
            'degree_variance',
            'degree_skew',
            'degree_cv'
        ]
    }

    scheduler_cols = [
        'static',
        'dynamic',
        'guided',
        'edge_balanced'
    ]

    data['family'] = (
        data['graph'].str.split('_').str[0].str.lower()
    )

    families = ['er', 'grid', 'rmat']

    if set(data['family'].unique()) != set(families):
        raise ValueError(
            f'Expected families {families}, '
            f'found {sorted(data.family.unique())}'
        )

    if data[scheduler_cols].isna().any().any():
        raise ValueError('Missing scheduler timing values detected.')

    all_predictions = []
    all_graph_metrics = []
    all_summary = []

    for feature_name, cols in feature_sets.items():

        for heldout_family in families:

            train = data[data['family'] != heldout_family].copy()
            test = data[data['family'] == heldout_family].copy()

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

            if args.balance_graphs:
                graph_counts = (
                    train.groupby('graph')['graph']
                    .transform('size')
                )
                sample_weights = (
                    len(train) /
                    (train.graph.nunique() * graph_counts)
                )
                model.fit(
                    train[cols],
                    train['best_scheduler'],
                    randomforestclassifier__sample_weight=sample_weights
                )
            else:
                model.fit(
                    train[cols],
                    train['best_scheduler']
                )
            predictions = model.predict(test[cols])

            classifier = model.named_steps['randomforestclassifier']

            importance = pd.DataFrame({
                'feature': cols,
                'importance': classifier.feature_importances_
            }).sort_values(
                'importance',
                ascending=False
            )

            print(
                f'\n{feature_name} | heldout={heldout_family}'
            )
            print(importance.to_string(index=False))

            result = test[
                ['graph', 'level', 'family', 'best_scheduler']
                + scheduler_cols
            ].copy()

            result['predicted_scheduler'] = predictions
            result['features'] = feature_name
            result['heldout_family'] = heldout_family

            result['selected_seconds'] = [
                result.iloc[i][predictions[i]]
                for i in range(len(result))
            ]

            result['oracle_seconds'] = (
                result[scheduler_cols].min(axis=1)
            )

            result['static_seconds'] = result['static']

            result['regret_percent'] = (
                result['selected_seconds'] /
                result['oracle_seconds'] - 1
            ) * 100

            result['static_regret_percent'] = (
                result['static_seconds'] /
                result['oracle_seconds'] - 1
            ) * 100

            all_predictions.append(result)

            graph_metrics = result.groupby('graph').agg(
                selected_total=('selected_seconds', 'sum'),
                oracle_total=('oracle_seconds', 'sum'),
                static_total=('static_seconds', 'sum')
            ).reset_index()

            graph_metrics['model_regret_percent'] = (
                graph_metrics['selected_total'] /
                graph_metrics['oracle_total'] - 1
            ) * 100

            graph_metrics['static_regret_percent'] = (
                graph_metrics['static_total'] /
                graph_metrics['oracle_total'] - 1
            ) * 100

            graph_metrics['features'] = feature_name
            graph_metrics['heldout_family'] = heldout_family

            all_graph_metrics.append(graph_metrics)

            all_summary.append({
                'features': feature_name,
                'heldout_family': heldout_family,
                'train_graphs': train.graph.nunique(),
                'test_graphs': test.graph.nunique(),
                'accuracy': accuracy_score(
                    test['best_scheduler'], predictions
                ),
                'balanced_accuracy': balanced_accuracy_score(
                    test['best_scheduler'], predictions
                ),
                'pooled_model_regret_percent': (
                    result['selected_seconds'].sum() /
                    result['oracle_seconds'].sum() - 1
                ) * 100,
                'mean_graph_regret_percent': (
                    graph_metrics['model_regret_percent'].mean()
                ),
                'median_graph_regret_percent': (
                    graph_metrics['model_regret_percent'].median()
                ),
                'static_mean_graph_regret_percent': (
                    graph_metrics['static_regret_percent'].mean()
                )
            })

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    graph_df = pd.concat(all_graph_metrics, ignore_index=True)
    summary_df = pd.DataFrame(all_summary)

    predictions_df.to_csv(
        out / 'lofo_predictions.csv', index=False
    )

    graph_df.to_csv(
        out / 'lofo_graph_metrics.csv', index=False
    )

    summary_df.to_csv(
        out / 'lofo_summary.csv', index=False
    )

    print('\nLeave-One-Family-Out Results:')
    print(summary_df.round(4).to_string(index=False))

    print(f'\nSaved LOFO analysis to {out}')


if __name__ == '__main__':
    main()
