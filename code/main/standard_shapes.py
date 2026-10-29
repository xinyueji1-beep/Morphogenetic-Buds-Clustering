"""Public labelled SIPU shape benchmarks, using the frozen ARFF observations.

The observations are fixed; seeds affect the algorithms only. No fitting uses y.
"""
from pathlib import Path
import json
import time
import warnings
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
from exp_driver import METHODS, run_baseline

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/exp'

def main():
    rows=[]; plot_data={}
    for name in ('aggregation','compound','spiral'):
        data,meta=arff.loadarff(ROOT/'data/raw/sipu'/f'{name}.arff')
        frame=pd.DataFrame(data)
        x=StandardScaler().fit_transform(frame.iloc[:,:-1].to_numpy(float))
        y=pd.factorize(frame.iloc[:,-1])[0]; k=len(np.unique(y))
        plot_data[name]={'x':x.tolist(),'target':y.tolist(),'predictions':{}}
        for seed in (3,7,11):
            for method in METHODS:
                start=time.perf_counter()
                with warnings.catch_warnings(record=True) as seen:
                    warnings.simplefilter('always')
                    labels,units=run_baseline(method,x,k,seed)
                row=dict(dataset=name,seed=seed,method=method,n=len(x),d=x.shape[1],K=k,
                         ari=adjusted_rand_score(y,labels),seconds=time.perf_counter()-start,
                         reported_units=units,actual_clusters=len(set(labels)-{-1}),
                         warnings=' | '.join(str(w.message) for w in seen))
                rows.append(row)
                if seed==3: plot_data[name]['predictions'][method]=np.asarray(labels).tolist()
                OUT.mkdir(parents=True,exist_ok=True)
                pd.DataFrame(rows).to_csv(OUT/'standard_shapes_in_progress.csv',index=False)
                print(row,flush=True)
    pd.DataFrame(rows).to_csv(OUT/'standard_shapes_raw.csv',index=False)
    (OUT/'standard_shapes_plot_data.json').write_text(json.dumps(plot_data),encoding='utf-8')

if __name__=='__main__': main()
