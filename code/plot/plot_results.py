"""Draw every submission figure from saved numeric JSON; never fit a model."""
import json
import hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'results/figure_data'
OUT=ROOT/'paper/figures/paper_figures'
PREVIEW=ROOT/'results/generated_figures'
outputs=[]

def read(name): return json.loads((DATA/(name+'.json')).read_text(encoding='utf-8'))
def save(fig,name):
    OUT.mkdir(parents=True,exist_ok=True);PREVIEW.mkdir(parents=True,exist_ok=True)
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(PREVIEW/(name+'.png'),dpi=160,bbox_inches='tight')
    plt.close(fig);outputs.append('paper/figures/paper_figures/'+name+'.pdf')
def scatter(ax,record,title):
    x=np.asarray(record['x']);ax.scatter(x[:,0],x[:,1],c=record['target'],cmap='tab10',s=7,linewidths=0)
    ax.set_title(title.replace('_',' '));ax.set_xlabel('feature 1');ax.set_ylabel('feature 2')
    ax.set_aspect('equal',adjustable='box')

def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    geometry=read('primary_geometry');fig,axes=plt.subplots(2,3,figsize=(10.5,6))
    for ax,(name,record) in zip(axes.ravel(),geometry.items()):scatter(ax,record,name)
    fig.tight_layout();save(fig,'synthetic_geometries')
    obj=read('overview');x=np.asarray(obj['x']);centers=np.asarray(obj['centers'])
    fig,axes=plt.subplots(1,3,figsize=(10.5,3.1))
    axes[0].scatter(*x.T,c='#6b7280',s=7);axes[0].set_title('Input observations')
    axes[1].scatter(*x.T,c='#d1d5db',s=7);axes[1].scatter(*centers.T,marker='x',c='black',s=25);axes[1].set_title('Final local buds')
    axes[2].scatter(*x.T,c=obj['labels'],cmap='tab10',s=7);axes[2].set_title('Continuity-derived components')
    for ax in axes:ax.set_xlabel('feature 1');ax.set_ylabel('feature 2');ax.set_aspect('equal',adjustable='box')
    fig.tight_layout();save(fig,'publication_morphogenesis_diagram')
    df=pd.DataFrame(read('paper_ablation_raw'));means=df.groupby('algorithm').ari.mean().sort_values()
    fig,ax=plt.subplots(figsize=(7.4,3));ax.hlines(range(len(means)),0,means,color='#a1a1aa');ax.scatter(means,range(len(means)),color='#2563eb')
    ax.set_yticks(range(len(means)),means.index.str.replace('MBC_',''));ax.set_xlabel('Mean ARI, 14 datasets x 5 seeds')
    for i,value in enumerate(means):ax.text(value+.012,i,f'{value:.4f}',va='center')
    ax.set_xlim(0,.75);fig.tight_layout();save(fig,'paper_fig_ablation_lollipop')
    six=read('six_structures');fig,axes=plt.subplots(2,3,figsize=(10.5,6))
    for ax,(name,record) in zip(axes.ravel(),six.items()):scatter(ax,record,name)
    fig.tight_layout();save(fig,'reviewer3r2_six_structures')
    df=pd.DataFrame(read('reviewer3_dynamics_trajectory'));fig,axes=plt.subplots(2,1,figsize=(8,5.3),sharex=True)
    for name,group in df[df.iteration<=140].groupby('representation'):
        axes[0].plot(group.iteration,group.coverage_inertia,label=name.replace('_',' '))
        axes[1].plot(group.iteration,group.buds)
    axes[0].set_yscale('log');axes[0].set_ylabel('Nearest-bud coverage inertia');axes[0].legend(fontsize=8)
    axes[1].set_ylabel('Bud count');axes[1].set_xlabel('Iteration (seed 3)')
    for ax in axes:ax.grid(alpha=.2)
    fig.tight_layout();save(fig,'reviewer3_dynamics_trajectory')
    df=pd.DataFrame(read('reviewer3r2_refinement_trajectory'));fig,axes=plt.subplots(1,2,figsize=(9.5,3.5))
    for ax,(name,group) in zip(axes,df.groupby('representation',sort=False)):
        for candidate,g in group.groupby('candidate'):
            component=g.candidate_kind.iloc[0]=='component_derived'
            ax.plot(g.refinement_step,g.inertia,color='#d95f02' if component else '#94a3b8',lw=2 if component else .8,
                    label='Component-derived' if component else None)
        ax.set_title(name.replace('_',' '));ax.set_xlabel('Refinement update / final record');ax.set_ylabel('Inertia');ax.grid(alpha=.2)
    axes[0].plot([],[],color='#94a3b8',label='Nine k-means++ candidates');axes[0].legend(fontsize=8)
    fig.tight_layout();save(fig,'reviewer3r2_refinement_trajectory')
    df=pd.DataFrame(read('complexity'));fig,axes=plt.subplots(1,3,figsize=(10.5,3.3))
    for ax,axis in zip(axes,('n','K','d')):
        s=df[df.axis==axis].groupby('value').time_s.agg(['mean','std'])
        ax.errorbar(s.index,s['mean'],yerr=s['std'],marker='o',capsize=3)
        ax.set_xscale('log');ax.set_yscale('log');ax.set_xlabel(axis);ax.set_ylabel('CPU wall time (s)');ax.grid(alpha=.2,which='both')
    fig.tight_layout();save(fig,'reviewer3_runtime_axes')
    obj=read('indian_maps');truth=np.asarray(obj['labels']['Ground truth']);mask=np.asarray(obj['mask'])
    colors=list(plt.get_cmap('tab20').colors[:16])+[(.65,.65,.65)]
    cmap=ListedColormap(colors);norm=BoundaryNorm(np.arange(-.5,17.5),17)
    fig,axes=plt.subplots(2,3,figsize=(9.5,6.7))
    for ax,(name,lab) in zip(axes.ravel(),obj['labels'].items()):
        lab=np.asarray(lab);unique=np.unique(lab[lab>=0]);counts=np.zeros((len(unique),16),int)
        for i,c in enumerate(unique):counts[i]=np.bincount(truth[lab==c],minlength=16)
        ii,jj=linear_sum_assignment(-counts);mapping={unique[i]:j for i,j in zip(ii,jj)}
        display=np.asarray([mapping.get(c,16) for c in lab]);grid=np.full(len(mask),np.nan);grid[mask]=display
        ax.imshow(grid.reshape(obj['shape']),cmap=cmap,norm=norm,interpolation='nearest')
        ax.set_title(name);ax.set_xticks([]);ax.set_yticks([])
    axes.ravel()[-1].axis('off');axes.ravel()[-1].legend(handles=[Patch(color=colors[i],label=f'Class {i+1}') for i in range(16)]+[Patch(color=colors[16],label='Unmatched / noise')],loc='center',ncol=2,fontsize=8,frameon=False)
    fig.suptitle('Indian Pines, PCA + XY, seed 7; colors matched only for display')
    fig.tight_layout();save(fig,'reviewer3r2_indian_pines_maps')
    # Public shape observations and seed-3 predictions are also archived as JSON.
    public=json.loads((ROOT/'results/exp/standard_shapes_plot_data.json').read_text())
    fig,axes=plt.subplots(3,3,figsize=(9.5,9))
    for row,(name,obj) in enumerate(public.items()):
        for col,key in enumerate(('target','MBC','HDBSCAN')):
            record=dict(x=obj['x'],target=obj['target'] if key=='target' else obj['predictions'][key])
            scatter(axes[row,col],record,name+' / '+('reference' if key=='target' else key))
    fig.tight_layout();save(fig,'public_shape_benchmarks')
    manifest=dict(status='complete',model_fits=0,outputs=outputs,
                  inputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in DATA.glob('*.json')})
    (PREVIEW/'plot_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
