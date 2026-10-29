#!/usr/bin/env python3
import argparse, random
from pathlib import Path

def write_graph(path, n, edges):
    clean=set()
    for u,v in edges:
        if u != v:
            clean.add((min(u,v),max(u,v)))
    with path.open('w') as f:
        f.write(f'{n} {len(clean)}\n')
        for u,v in sorted(clean): f.write(f'{u} {v}\n')

def grid(n):
    w=max(1,int(n**0.5)); h=(n+w-1)//w
    edges=[]
    for u in range(n):
        r,c=divmod(u,w)
        if c+1<w and u+1<n: edges.append((u,u+1))
        if u+w<n: edges.append((u,u+w))
    return edges

def erdos_renyi(n, p, rng):
    edges=[]
    for u in range(n):
        for v in range(u+1,n):
            if rng.random()<p: edges.append((u,v))
    return edges

def rmat(n, m_factor, rng):
    m=max(n, n*m_factor); edges=set(); bits=max(1,(n-1).bit_length())
    while len(edges)<m and len(edges)<n*(n-1)//2:
        u=v=0
        for bit in range(bits):
            x=rng.random()
            if x<0.57: pass
            elif x<0.76: v |= 1<<bit
            elif x<0.95: u |= 1<<bit
            else: u |= 1<<bit; v |= 1<<bit
        u%=n; v%=n
        if u!=v: edges.add((min(u,v),max(u,v)))
    return list(edges)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',default='graphs'); ap.add_argument('--sizes',nargs='+',type=int,default=[1000,3000]); ap.add_argument('--seed',type=int,default=29); args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    for n in args.sizes:
        write_graph(out/f'grid_n{n}.edgelist',n,grid(n))
        rng=random.Random(args.seed+n)
        p=min(0.02, 8/max(1,n-1))
        write_graph(out/f'er_n{n}.edgelist',n,erdos_renyi(n,p,rng))
        write_graph(out/f'rmat_n{n}.edgelist',n,rmat(n,4,rng))
        print(f'generated 3 graph families, n={n}')
if __name__=='__main__': main()
