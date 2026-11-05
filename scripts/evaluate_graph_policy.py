import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report
)


def load_graph_data(path):
    df = pd.read_csv(path)

    required = {
        'graph',
        'trial',
        'scheduler',
        'level',
        'frontier_edges',
        'level_seconds'
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')

    schedulers = ['guided', 'edge_balanced']

    df = df[df['scheduler'].isin(schedulers)].copy()

    features = (
        df[df['scheduler'] == 'guided']
        .loc[df['level'] == 0]
        .groupby('graph', as_index=False)
        .agg(frontier_edges=('frontier_edges', 'median'))
    )

    trial_times = (
        df.groupby(['graph', 'trial', 'scheduler'], as_index=False)
        .agg(run_seconds=('level_seconds', 'sum'))
    )

    graph_times = (
        trial_times.groupby(['graph', 'scheduler'], as_index=False)
        .agg(median_seconds=('run_seconds', 'median'))
    )

    timing_wide = graph_times.pivot(
        index='graph',
        columns='scheduler',
        values='median_seconds'
    )

    timing_wide.columns.name = None

    if not set(schedulers).issubset(timing_wide.columns):
        raise ValueError('Missing scheduler timing results.')

    data = features.merge(
        timing_wide.reset_index(),
        on='graph',
        validate='one_to_one'
    )

    data['winner'] = np.where(
        data['guided'] <= data['edge_balanced'],
        'guided',
        'edge_balanced'
    )

    return data.sort_values('graph').reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--train',
        default='results/corrected/all_levels.csv'
    )

    parser.add_argument(
        '--test',
        default='/tmp/project29_scale_500k_results/all_levels.csv'
    )

    parser.add_argument(
        '--out',
        default='results/graph_policy'
    )

    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train = load_graph_data(args.train)
    test = load_graph_data(args.test)

    if train['graph'].nunique() != 15:
        raise ValueError('Expected 15 training graph instances.')

    if test['graph'].nunique() != 15:
        raise ValueError('Expected 15 testing graph instances.')

    feature = ['frontier_edges']

    model = DecisionTreeClassifier(
        max_depth=2,
        min_samples_leaf=2,
        random_state=29
    )

    model.fit(
        train[feature],
        train['winner']
    )

    test['predicted_scheduler'] = model.predict(
        test[feature]
    )

    test['policy_seconds'] = [
        test.iloc[i][test.iloc[i]['predicted_scheduler']]
        for i in range(len(test))
    ]

    test['guided_seconds'] = test['guided']
    test['edge_balanced_seconds'] = test['edge_balanced']

    test['oracle_seconds'] = test[
        ['guided', 'edge_balanced']
    ].min(axis=1)

    test['policy_regret_pct'] = (
        test['policy_seconds'] / test['oracle_seconds'] - 1
    ) * 100

    test['guided_regret_pct'] = (
        test['guided_seconds'] / test['oracle_seconds'] - 1
    ) * 100

    test['edge_balanced_regret_pct'] = (
        test['edge_balanced_seconds'] / test['oracle_seconds'] - 1
    ) * 100

    accuracy = accuracy_score(
        test['winner'],
        test['predicted_scheduler']
    )

    balanced_accuracy = balanced_accuracy_score(
        test['winner'],
        test['predicted_scheduler']
    )

    labels = ['guided', 'edge_balanced']

    matrix = confusion_matrix(
        test['winner'],
        test['predicted_scheduler'],
        labels=labels
    )

    summary_rows = []

    for name, column in [
        ('adaptive_policy', 'policy_seconds'),
        ('always_guided', 'guided_seconds'),
        ('always_edge_balanced', 'edge_balanced_seconds'),
        ('oracle', 'oracle_seconds')
    ]:
        summary_rows.append({
            'strategy': name,
            'total_graph_median_seconds': test[column].sum(),
            'mean_graph_median_seconds': test[column].mean(),
            'mean_graph_regret_pct': test[
                column.replace('_seconds', '_regret_pct')
            ].mean() if name in [
                'adaptive_policy',
                'always_guided',
                'always_edge_balanced'
            ] else 0.0
        })

    summary = pd.DataFrame(summary_rows)

    test.to_csv(
        out / 'cross_scale_predictions.csv',
        index=False
    )

    summary.to_csv(
        out / 'cross_scale_summary.csv',
        index=False
    )

    train.to_csv(
        out / 'training_graph_data.csv',
        index=False
    )

    report = classification_report(
        test['winner'],
        test['predicted_scheduler'],
        labels=labels,
        zero_division=0
    )

    (out / 'classification_report.txt').write_text(report)

    (out / 'decision_tree.txt').write_text(
        export_text(model, feature_names=feature)
    )

    print('\nTraining graph count:', len(train))
    print('Testing graph count:', len(test))

    print('\nTraining winner distribution:')
    print(train['winner'].value_counts().to_string())

    print('\nTesting winner distribution:')
    print(test['winner'].value_counts().to_string())

    print('\nDecision tree:')
    print(export_text(model, feature_names=feature))

    print('\nCross-scale accuracy:', round(accuracy, 4))
    print('Cross-scale balanced accuracy:', round(balanced_accuracy, 4))

    print('\nConfusion matrix (guided, edge_balanced):')
    print(matrix)

    print('\nClassification report:')
    print(report)

    print('\nRuntime comparison:')
    print(summary.to_string(index=False))

    print('\nSaved results to:', out)


if __name__ == '__main__':
    main()