#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--binary', default='./build/bfs_bench')
    ap.add_argument('--graphs', default='graphs')
    ap.add_argument('--out', default='results')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--repeats', type=int, default=10)
    args = ap.parse_args()

    if args.threads <= 0:
        ap.error('--threads must be positive')

    if args.repeats <= 0:
        ap.error('--repeats must be positive')

    binary = Path(args.binary)
    graph_dir = Path(args.graphs)
    out = Path(args.out)

    if not binary.is_file():
        raise SystemExit(f'Benchmark binary not found: {binary}')

    if not graph_dir.is_dir():
        raise SystemExit(f'Graph directory not found: {graph_dir}')

    out.mkdir(parents=True, exist_ok=True)

    files = sorted(graph_dir.glob('*.edgelist'))

    if not files:
        raise SystemExit(f'No .edgelist files found in {graph_dir}')

    frames = []

    for graph in files:
        dest = out / f'{graph.stem}.csv'

        print(f'Running {graph} -> {dest}', flush=True)

        subprocess.run(
            [
                str(binary),
                str(graph),
                str(args.threads),
                str(dest),
                str(args.repeats)
            ],
            check=True
        )

        df = pd.read_csv(dest)
        df.insert(0, 'graph', graph.stem)
        df['threads'] = args.threads
        df['repeats'] = args.repeats
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(out / 'all_levels.csv', index=False)

    print(f'Wrote {out / "all_levels.csv"}')
    print(f'Graphs processed: {len(files)}')
    print(f'Threads: {args.threads}')
    print(f'Repetitions: {args.repeats}')


if __name__ == '__main__':
    main()
