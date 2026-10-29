#!/usr/bin/env python3
import argparse, subprocess
from pathlib import Path
import pandas as pd

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--binary',default='./build/bfs_bench'); ap.add_argument('--graphs',default='graphs'); ap.add_argument('--out',default='results'); ap.add_argument('--threads',type=int,default=4); ap.add_argument('--repeats',type=int,default=5); args=ap.parse_args()
    graph_dir=Path(args.graphs); out=Path(args.out); out.mkdir(parents=True,exist_ok=True); frames=[]
    files=sorted(graph_dir.glob('*.edgelist'))
    if not files: raise SystemExit(f'No .edgelist files found in {graph_dir}')
    for graph in files:
        dest=out/f'{graph.stem}.csv'
        print(f'Running {graph} -> {dest}',flush=True)
        subprocess.run([args.binary,str(graph),str(args.threads),str(dest),str(args.repeats)],check=True)
        df=pd.read_csv(dest); df.insert(0,'graph',graph.stem); df['threads']=args.threads; frames.append(df)
    pd.concat(frames,ignore_index=True).to_csv(out/'all_levels.csv',index=False)
    print(f'Wrote {out / "all_levels.csv"}')
if __name__=='__main__': main()
