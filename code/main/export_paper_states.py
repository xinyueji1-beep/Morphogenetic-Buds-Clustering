"""Compute and preserve numeric figure inputs; plotting is a separate stage.

The legacy analysis driver also makes exploratory previews. The submission
figures are all rebuilt from the JSON written here, without fitting in plot/.
"""
from pathlib import Path
import inspect
import json
import sys
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.io import arff
from scipy.spatial.distance import cdist
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from exp_driver import make_mbc, load
from mbc import MorphogeneticBudsClustering, squared_distances
import manuscript_pipeline as legacy

ROOT=Path(__file__).resolve().parents[2]
PAPER=ROOT/'results/paper_results'
OUT=ROOT/'results/figure_data'

def write(name,obj):
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/(name+'.json')).write_text(json.dumps(obj,allow_nan=False),encoding='utf-8')

def trace_wine(x):
    source,start=inspect.getsourcelines(MorphogeneticBudsClustering.fit)
    line_map={start+i:line.strip() for i,line in enumerate(source)}
    records=[]; current={}; migrated=None
    def trace(frame,event,arg):
        nonlocal current,migrated
        if frame.f_code is not MorphogeneticBudsClustering.fit.__code__: return None
        if event!='line': return trace
        s=line_map.get(frame.f_lineno,'');v=frame.f_locals
        if s=='pull = targets - state.centers':
            state=v['state'];mass=v['mass']
            current=dict(iteration=v['_']+1,buds_before=len(state.centers),
                         support_min=float(mass.min()),support_max=float(mass.max()),
                         first_bud=state.centers[0].tolist(),first_target=v['targets'][0].tolist(),
                         first_support=float(mass[0]))
        elif s=='state = self._apply_lateral_inhibition(state, inhibit_radius)':
            migrated=v['state'].centers.copy()
            current['mean_velocity_norm']=float(np.linalg.norm(v['state'].velocities,axis=1).mean())
        elif s=='vigor = mass / (mass.mean() + 1e-12)':
            current['mean_inhibition_displacement']=float(np.linalg.norm(v['state'].centers-migrated,axis=1).mean())
        elif s=='state = self._maybe_birth_bud(x, state, sigma, rng)':
            current['buds_after_deletion']=len(v['state'].centers)
        elif s.startswith('state = self._merge_overgrown_buds'):
            current['buds_after_birth']=len(v['state'].centers)
        elif s=='self.history_.append(state.centers.copy())':
            state=v['state'];current['buds_after_merge']=len(state.centers)
            current['coverage_inertia']=float(squared_distances(x,state.centers).min(axis=1).sum())
            records.append(current.copy())
        return trace
    model=make_mbc(x,3,3)
    sys.settrace(trace)
    try: model.fit(x)
    finally: sys.settrace(None)
    write('wine_equation_walkthrough',dict(seed=3,K=3,d=x.shape[1],sigma=model.sigma_,
          records=records,final_ari=adjusted_rand_score(load_wine().target,model.labels_)))

def main():
    geometry={}
    for name in ('two_moons','circles','noisy_moons','noisy_circles','varied_density','anisotropic_blobs'):
        x,y=load(name,3);geometry[name]=dict(x=x.tolist(),target=y.tolist())
    write('primary_geometry',geometry)
    x,y=load('two_moons',3);model=make_mbc(x,2,3).fit(x)
    write('overview',dict(x=x.tolist(),target=y.tolist(),centers=model.centers_.tolist(),labels=model.labels_.tolist()))
    write('six_structures',{name:dict(x=x.tolist(),target=y.tolist()) for name,x,y in legacy.reviewer2_make_nonconvex(3)})
    for fname in ('paper_ablation_raw','reviewer3_dynamics_trajectory','reviewer3r2_refinement_trajectory'):
        df=pd.read_csv(PAPER/(fname+'.csv')).replace({np.nan:None})
        write(fname,df.to_dict('records'))
    x,_=load('wine',3);trace_wine(x)
    stability=[]
    for steps in (100,120,140,160,180,200):
        model=make_mbc(x,3,3);model.steps=steps;model.fit(x)
        stability.append(dict(steps=steps,buds=len(model.centers_),ari=adjusted_rand_score(load_wine().target,model.labels_)))
    write('wine_stopping_steps',stability)
    morphology=[]
    def gap_ratio(name,seed,x,y):
        distances=cdist(x,x);same=y[:,None]==y[None,:]
        gap=float(distances[~same].min());local=[]
        for label in np.unique(y):
            ids=np.flatnonzero(y==label);dd=distances[np.ix_(ids,ids)].copy()
            np.fill_diagonal(dd,np.inf)
            neighbor=min(6,len(ids)-1)
            if neighbor>0: local.extend(np.partition(dd,neighbor-1,axis=1)[:,neighbor-1])
        h=float(np.quantile(local,.9))
        morphology.append(dict(dataset=name,seed=seed,minimum_between_class_gap=gap,
                               within_class_local_scale=h,gap_ratio=gap/h))
    from exp_driver import DATASETS,SEEDS
    for name in DATASETS:
        for seed in SEEDS:
            x,y=load(name,seed);gap_ratio(name,seed,x,y)
    for name in ('aggregation','compound','spiral'):
        arr,_=arff.loadarff(ROOT/'data/raw/sipu'/f'{name}.arff');df=pd.DataFrame(arr)
        x=StandardScaler().fit_transform(df.iloc[:,:-1].to_numpy(float));y=pd.factorize(df.iloc[:,-1])[0]
        gap_ratio(name,0,x,y)
    pd.DataFrame(morphology).to_csv(ROOT/'results/exp/morphology_diagnostics.csv',index=False)
    cube=loadmat(ROOT/'data/raw/hsi/Indian_pines.mat')['indian_pines'].astype(float)
    gt=loadmat(ROOT/'data/raw/hsi/Indian_pines_gt.mat')['indian_pines_gt'].astype(int)
    mask=gt>0;r,c=np.nonzero(mask);y=gt[mask]-1
    pca=PCA(n_components=10,random_state=0).fit_transform(StandardScaler().fit_transform(cube[mask]))
    x=np.column_stack([pca,.35*r/(gt.shape[0]-1),.35*c/(gt.shape[1]-1)])
    labels={'Ground truth':y.tolist()};checks=[]
    raw=pd.read_csv(PAPER/'reviewer3r2_indian_pines_raw.csv')
    for method in ('MBC','MeanShift','Spectral','KMeans'):
        lab=(legacy.main_benchmark_run_algorithm(method,x,16,7) if method!='MeanShift'
             else legacy.reviewer2_run_related(method,x,16,7))
        ari=adjusted_rand_score(y,lab)
        expected=raw[(raw.features=='pca10_xy')&(raw.seed==7)&(raw.algorithm==method)].ari.item()
        assert abs(ari-expected)<1e-10,(method,ari,expected)
        labels[method]=np.asarray(lab).tolist();checks.append(dict(method=method,ari=ari,raw_ari=expected))
        print('Indian Pines saved map',method,ari,flush=True)
    write('indian_maps',dict(shape=list(gt.shape),mask=mask.ravel().tolist(),labels=labels,seed=7,checks=checks))
    for name in ('primary_ari','complexity','standard_shapes_raw'):
        df=pd.read_csv(ROOT/'results/exp'/(name+'.csv')).replace({np.nan:None})
        write(name,df.to_dict('records'))
    print('All submission figure data saved as JSON.',flush=True)

if __name__=='__main__': main()
