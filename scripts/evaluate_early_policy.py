
import argparse
import re
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


SCHEDULERS = ['guided', 'edge_balanced']
WINDOWS = {
    'W1_level_0': 0,
    'W2_levels_0_1': 1,
    'W3_levels_0_4': 4
}


def extract_graph_size(graph):
    match = re.search(r'_n(\d+)_', graph)

    if not match:
        raise ValueError(f'Cannot extract graph size from: {graph}')

    return int(match.group(1))


def load_graph_data(path):
    df = pd.read_csv(path)

    required = {
        'graph',
        'trial',
        'scheduler',
        'level',
        'frontier_size',
        'frontier_edges',
        'degree_mean',
        'degree_cv',
        'level_seconds'
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')

    df = df[df['scheduler'].isin(SCHEDULERS)].copy()

    if df.empty:
        raise ValueError('No guided or edge_balanced records found.')

    graphs = sorted(df['graph'].unique())

    if len(graphs) != 15:
        raise ValueError(f'Expected 15 graphs, found {len(graphs)}.')

    profile = df[
        (df['scheduler'] == 'guided') &
        (df['trial'] == 0) &
        (df['level'] <= 4)
    ].copy()

    counts = profile.groupby('graph')['level'].nunique()

    if len(counts) != 15 or not (counts == 5).all():
        raise ValueError('Every graph must have guided trial-0 levels 0 to 4.')

    if profile.duplicated(['graph', 'level']).any():
        raise ValueError('Duplicate profiling records found.')

    profile['graph_size'] = profile['graph'].apply(extract_graph_size)

    feature_rows = []

    for graph, group in profile.groupby('graph'):
        group = group.sort_values('level').reset_index(drop=True)

        root_degree = float(group.loc[0, 'frontier_edges'])
        graph_size = int(group.loc[0, 'graph_size'])

        row = {
            'graph': graph,
            'graph_size': graph_size,
            'root_degree': root_degree,
            'normalized_root_degree': root_degree / (graph_size - 1)
        }

        for window_name, max_level in WINDOWS.items():
            observed = group[group['level'] <= max_level]

            row[f'{window_name}_mean_degree'] = observed[
                'degree_mean'
            ].mean()

            row[f'{window_name}_mean_cv'] = observed[
                'degree_cv'
            ].mean()

            if max_level >= 1:
                initial_size = float(observed.iloc[0]['frontier_size'])
                final_size = float(observed.iloc[-1]['frontier_size'])

                if initial_size == 0:
                    raise ValueError(f'Zero initial frontier size: {graph}')

                row[f'{window_name}_frontier_growth'] = (
                    final_size / initial_size
                )

        feature_rows.append(row)

    features = pd.DataFrame(feature_rows)

    trial_times = (
        df.groupby(
            ['graph', 'trial', 'scheduler'],
            as_index=False
        )
        .agg(run_seconds=('level_seconds', 'sum'))
    )

    trial_counts = trial_times.groupby(
        ['graph', 'scheduler']
    )['trial'].nunique()

    if not (trial_counts == 30).all():
        raise ValueError(
            'Expected exactly 30 trials for every graph and scheduler.'
        )

    graph_times = (
        trial_times.groupby(
            ['graph', 'scheduler'],
            as_index=False
        )
        .agg(median_seconds=('run_seconds', 'median'))
    )

    timing_wide = graph_times.pivot(
        index='graph',
        columns='scheduler',
        values='median_seconds'
    )

    timing_wide.columns.name = None

    if not set(SCHEDULERS).issubset(timing_wide.columns):
        raise ValueError('Missing scheduler timing results.')

    if timing_wide[SCHEDULERS].isna().any().any():
        raise ValueError('Missing graph-level scheduler timings.')

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


def get_experiments(window_name):
    window_prefix = window_name

    experiments = {
        'F1_root_degree': [
            'root_degree'
        ],
        'F2_normalized_root_degree': [
            'normalized_root_degree'
        ]
    }

    structural_features = [
        'root_degree',
        f'{window_prefix}_mean_degree',
        f'{window_prefix}_mean_cv'
    ]

    if window_name != 'W1_level_0':
        structural_features.append(
            f'{window_prefix}_frontier_growth'
        )

    experiments['F3_early_structural_profile'] = structural_features

    return experiments


def evaluate_configuration(train, test, feature, config_name, out):
    model = DecisionTreeClassifier(
        max_depth=2,
        min_samples_leaf=2,
        random_state=29
    )

    model.fit(
        train[feature],
        train['winner']
    )

    result = test.copy()

    result['predicted_scheduler'] = model.predict(
        result[feature]
    )

    result['policy_seconds'] = np.where(
        result['predicted_scheduler'] == 'guided',
        result['guided'],
        result['edge_balanced']
    )

    result['guided_seconds'] = result['guided']
    result['edge_balanced_seconds'] = result['edge_balanced']

    result['oracle_seconds'] = result[
        ['guided', 'edge_balanced']
    ].min(axis=1)

    for strategy in [
        'policy',
        'guided',
        'edge_balanced'
    ]:
        result[f'{strategy}_regret_pct'] = (
            result[f'{strategy}_seconds'] /
            result['oracle_seconds'] - 1
        ) * 100

    labels = SCHEDULERS

    accuracy = accuracy_score(
        result['winner'],
        result['predicted_scheduler']
    )

    balanced_accuracy = balanced_accuracy_score(
        result['winner'],
        result['predicted_scheduler']
    )

    matrix = confusion_matrix(
        result['winner'],
        result['predicted_scheduler'],
        labels=labels
    )

    report = classification_report(
        result['winner'],
        result['predicted_scheduler'],
        labels=labels,
        zero_division=0
    )

    summary_rows = []

    for name, column in [
        ('adaptive_policy', 'policy_seconds'),
        ('always_guided', 'guided_seconds'),
        ('always_edge_balanced', 'edge_balanced_seconds'),
        ('oracle', 'oracle_seconds')
    ]:
        regret_column = column.replace('_seconds', '_regret_pct')

        summary_rows.append({
            'configuration': config_name,
            'strategy': name,
            'total_graph_median_seconds': result[column].sum(),
            'mean_graph_median_seconds': result[column].mean(),
            'mean_graph_regret_pct': (
                result[regret_column].mean()
                if regret_column in result.columns
                else 0.0
            )
        })

    config_dir = out / config_name
    config_dir.mkdir(parents=True, exist_ok=True)

    result.to_csv(
        config_dir / 'predictions.csv',
        index=False
    )

    (config_dir / 'decision_tree.txt').write_text(
        export_text(model, feature_names=feature)
    )

    (config_dir / 'classification_report.txt').write_text(report)

    (config_dir / 'confusion_matrix.txt').write_text(
        f'Labels: {labels}\n{matrix}'
    )

    print(f'\n--- {config_name} ---')
    print('Features:', feature)
    print('Accuracy:', round(accuracy, 4))
    print('Balanced accuracy:', round(balanced_accuracy, 4))
    print('Confusion matrix (guided, edge_balanced):')
    print(matrix)
    print('Decision tree:')
    print(export_text(model, feature_names=feature))

    return summary_rows, accuracy, balanced_accuracy


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
        default='results/early_policy_forward'
    )

    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print('Loading training data:', args.train)
    train = load_graph_data(args.train)

    print('Loading testing data:', args.test)
    test = load_graph_data(args.test)

    if train['graph'].nunique() != 15:
        raise ValueError('Expected 15 training graphs.')

    if test['graph'].nunique() != 15:
        raise ValueError('Expected 15 testing graphs.')

    print('\nTraining graphs:', len(train))
    print('Testing graphs:', len(test))

    print('\nTraining winner distribution:')
    print(train['winner'].value_counts().to_string())

    print('\nTesting winner distribution:')
    print(test['winner'].value_counts().to_string())

    all_summaries = []
    all_metrics = []

    for window_name in WINDOWS:
        experiments = get_experiments(window_name)

        for feature_name, feature in experiments.items():
            config_name = f'{window_name}_{feature_name}'

            summary_rows, accuracy, balanced_accuracy = (
                evaluate_configuration(
                    train,
                    test,
                    feature,
                    config_name,
                    out
                )
            )

            all_summaries.extend(summary_rows)

            all_metrics.append({
                'configuration': config_name,
                'accuracy': accuracy,
                'balanced_accuracy': balanced_accuracy
            })

    summary = pd.DataFrame(all_summaries)
    metrics = pd.DataFrame(all_metrics)

    summary.to_csv(
        out / 'all_runtime_summaries.csv',
        index=False
    )

    metrics.to_csv(
        out / 'all_classification_metrics.csv',
        index=False
    )

    train.to_csv(
        out / 'training_graph_data.csv',
        index=False
    )

    test.to_csv(
        out / 'testing_graph_data.csv',
        index=False
    )

    print('\n========== ALL CONFIGURATIONS ==========')
    print(metrics.to_string(index=False))

    print('\n========== RUNTIME COMPARISON ==========')
    print(summary.to_string(index=False))

    print('\nSaved results to:', out)


if __name__ == '__main__':
    main()
