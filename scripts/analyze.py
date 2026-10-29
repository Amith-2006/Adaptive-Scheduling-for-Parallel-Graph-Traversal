#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',default='results/all_levels.csv'); ap.add_argument('--out',default='results/analysis'); args=ap.parse_args()
    df=pd.read_csv(args.input); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    # Aggregate timing over repeated trials. A scheduler label is the fastest median scheduler for a graph/level.
    keys=['graph','level']
    timing=df.groupby(keys+['scheduler'],as_index=False).agg(median_seconds=('level_seconds','median'))
    winners=timing.loc[timing.groupby(keys).median_seconds.idxmin(),keys+['scheduler']].rename(columns={'scheduler':'best_scheduler'})
    feats=df.groupby(keys,as_index=False).agg(frontier_size=('frontier_size','first'),frontier_edges=('frontier_edges','first'),degree_mean=('degree_mean','first'),degree_variance=('degree_variance','first'),degree_skew=('degree_skew','first'),degree_cv=('degree_cv','first'))
    data=feats.merge(winners,on=keys); data.to_csv(out/'level_dataset.csv',index=False)
    # Need enough distinct graphs for meaningful group-held-out evaluation.
    if data.graph.nunique()<3:
        print(f"Saved descriptive dataset; classifier evaluation skipped because only {data.graph.nunique()} distinct graph(s) are available. Generate at least 6 graphs for a more credible split.")
        return
    feature_sets={'mean_only':['frontier_size','frontier_edges','degree_mean'],'mean_plus_variance':['frontier_size','frontier_edges','degree_mean','degree_variance','degree_skew','degree_cv']}
    split=GroupShuffleSplit(n_splits=1,test_size=0.33,random_state=29); train_idx,test_idx=next(split.split(data,groups=data.graph))
    rows=[]
    for name,cols in feature_sets.items():
        model=make_pipeline(SimpleImputer(strategy='median'),RandomForestClassifier(n_estimators=300,min_samples_leaf=2,class_weight='balanced',random_state=29))
        model.fit(data.iloc[train_idx][cols],data.iloc[train_idx].best_scheduler)
        pred=model.predict(data.iloc[test_idx][cols]); y=data.iloc[test_idx].best_scheduler
        rows.append({'features':name,'train_graphs':';'.join(sorted(data.iloc[train_idx].graph.unique())),'test_graphs':';'.join(sorted(data.iloc[test_idx].graph.unique())),'accuracy':accuracy_score(y,pred),'balanced_accuracy':balanced_accuracy_score(y,pred)})
        (out/f'{name}_report.txt').write_text(classification_report(y,pred,zero_division=0))
    pd.DataFrame(rows).to_csv(out/'heldout_metrics.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False)); print(f'Wrote analysis to {out}')
if __name__=='__main__': main()
