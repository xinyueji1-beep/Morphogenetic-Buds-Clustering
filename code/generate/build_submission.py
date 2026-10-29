"""Rebuild submission text and tables from the completed local result ledgers."""
from pathlib import Path
import json
import re
import sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/main'))
from exp_driver import DATASETS,METHODS
P=ROOT/'results/paper_results';E=ROOT/'results/exp';T=ROOT/'code/templates'
G=ROOT/'paper/generated';G.mkdir(parents=True,exist_ok=True)
specs={}
names={'KMeans':'k-means','MeanShift':'Mean shift','AffinityPropagation':'Affinity propagation','CompactRefineOnly':'Control','GNG':'GNG'}
ds_names={'two_moons':'Two Moons','noisy_moons':'Noisy Moons','circles':'Circles','noisy_circles':'Noisy Circles','varied_density':'Varied Density','anisotropic_blobs':'Anisotropic Blobs','iris':'Iris','wine':'Wine','breast_cancer':'Breast Cancer','digits_0_4':'Digits 0--4','ecoli':'Ecoli','glass':'Glass','vehicle':'Vehicle','cmc':'CMC'}
def n(x,d=4):return f'{float(x):.{d}f}'
def rev(s):return '\\Rev{'+s+'}'
def esc(s):return str(s).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')
def display(m):return names.get(m,m)
def table(key,caption,headers,rows,source='',rank=False):
    rendered=[];plain=[]
    for row in rows:
        plain.append([float(v) if isinstance(v,(float,np.floating)) else str(v) for v in row])
        rendered.append(' & '.join(rev(n(v) if isinstance(v,(float,np.floating)) else str(v)) for v in row)+r'\\')
    body='\n'.join([r'\begin{table}[!htbp]',r'\centering\scriptsize',r'\caption{'+rev(caption)+'}',r'\label{tab:'+key+'}',
         r'\resizebox{\linewidth}{!}{\begin{tabular}{'+'l'+'r'*(len(headers)-1)+'}',r'\toprule',' & '.join(headers)+r'\\',r'\midrule',*rendered,r'\bottomrule',r'\end{tabular}}',r'\end{table}'])
    (G/(key+'.tex')).write_text(body,encoding='utf-8');specs[key]=dict(source=source,headers=headers,rows=plain)
    return '\\input{generated/'+key+'.tex}\n'
def ranked(values):
    # Rank displayed values: equal four-decimal entries share dense ranks.
    a=np.round(np.asarray(values,float),4);unique=sorted(set(a),reverse=True)
    return [({'0':'','1':''}.get('', '') if False else (r'\best{' if unique.index(v)==0 else r'\second{' if unique.index(v)==1 else r'\third{' if unique.index(v)==2 else ''))+n(v)+('}' if unique.index(v)<3 else '') for v in a]
def figure(name,caption,label,width='.92'):
    return '\n'.join([r'\begin{figure}[!htbp]',r'\centering',r'\includegraphics[width='+width+r'\linewidth]{'+name+'.pdf}',r'\caption{'+rev(caption)+'}',r'\label{fig:'+label+'}',r'\end{figure}',''])
def quote(s):return r'\begin{quote}\small '+s+r'\end{quote}'

def main():
    raw=pd.read_csv(E/'primary_ari.csv')
    primary=raw.groupby(['dataset','method']).ari.mean().unstack().reindex(list(DATASETS))
    overall=primary.mean();stats=json.loads((E/'primary_block_statistics.json').read_text())
    blocks=pd.read_csv(E/'primary_block_means.csv',index_col=0)
    blockw=pd.read_csv(E/'primary_block_wilcoxon.csv')
    dis=pd.read_csv(P/'reviewer3r2_disjoint_seeds_raw.csv')
    assert set(dis.dataset)==set(DATASETS) and len(dis)==490
    dismean=dis.groupby(['dataset','algorithm']).ari.mean().unstack().mean()
    sixraw=pd.read_csv(P/'reviewer2_related_methods_raw.csv')
    six=sixraw.groupby(['dataset','method']).ari.mean().unstack()
    publicraw=pd.read_csv(E/'standard_shapes_raw.csv');public=publicraw.groupby(['dataset','method']).ari.mean().unstack()
    scraw=pd.read_csv(P/'reviewer3_scale_all_methods_raw.csv')
    sc=scraw.groupby(['case','method'])[['ari','seconds']].mean()
    sc.reset_index().to_csv(P/'reviewer3_scale_all_methods_summary.csv',index=False)
    nc=pd.read_csv(E/'scale.csv');nc=nc[nc['case'].str.startswith('moons_')].copy();nc['n']=nc['case'].str.extract(r'(\d+)').astype(int)
    nc=pd.concat([nc,pd.read_csv(E/'true_gng_nonconvex_scale.csv')],ignore_index=True)
    ncmean=nc.groupby(['n','method'])[['ari','time_s']].mean()
    nc.to_csv(E/'nonconvex_scale_final_raw.csv',index=False)
    indianraw=pd.read_csv(P/'reviewer3r2_indian_pines_raw.csv')
    indian=indianraw.groupby(['features','algorithm']).agg(ari=('ari','mean'),sd=('ari','std'),seconds=('seconds','mean'))
    comp=pd.read_csv(E/'complexity.csv');fit=json.loads((E/'complexity_loglog_fit.json').read_text())
    timing=comp.groupby(['axis','value']).agg(seconds=('time_s','mean'),sd=('time_s','std'),k=('realized_k','mean'),ksd=('realized_k','std'),mean_k=('mean_k','mean'),mean_k2=('mean_k_squared','mean'))
    abl=pd.read_csv(P/'paper_ablation_raw.csv').groupby('algorithm').ari.mean()
    winners=pd.read_csv(P/'reviewer2_initialization_winners.csv')
    strict=int((winners.mbc_inertia>winners.winning_inertia).sum());ties=int((winners.mbc_inertia==winners.winning_inertia).sum())
    assert strict+ties==len(winners)
    trigger=pd.read_csv(P/'reviewer3_trigger_sensitivity_raw.csv')
    trigcol='trigger' if 'trigger' in trigger else 'dim_trigger' if 'dim_trigger' in trigger else 'threshold'
    if trigcol not in trigger:trigcol=[c for c in trigger if 'trigger' in c][0]
    trig=trigger.groupby([trigcol,'dataset']).ari.mean().unstack();trigmeans=trig.mean(axis=1)
    ks=pd.read_csv(P/'reviewer1_k_estimation_summary.csv');correct=int((ks.true_k==ks.estimated_k_by_silhouette).sum())
    externalraw=pd.read_csv(P/'external_real_benchmark_raw.csv');ext=externalraw.groupby(['dataset','algorithm']).ari.mean().unstack()
    dyn=pd.read_csv(P/'reviewer3_dynamics_trajectory.csv')
    walkthrough=json.loads((ROOT/'results/figure_data/wine_equation_walkthrough.json').read_text());steps=walkthrough['records']
    stopping=json.loads((ROOT/'results/figure_data/wine_stopping_steps.json').read_text())
    seedtext=r'$\{3,7,11,19,23\}$';newseed=r'$\{101,103,107,109,113\}$'
    quotes={}
    quotes['selection']=r'The primary settings were frozen before this local reproduction. The historical selection of all seven default-to-primary changes is not fully documented. Sensitivity results use ARI and are diagnostic, not preregistered or independent tuning evidence. We therefore withdraw the claim that no score informed parameter development; no dataset-specific search is performed in this rerun.'
    quotes['disjoint']='The disjoint set '+newseed+' covers all fourteen primary datasets and the original seven methods. MBC has mean ARI '+n(dismean['MBC'])+', compared with '+n(overall['MBC'])+' on '+seedtext+'. The added HDBSCAN, mean shift, affinity propagation and GNG methods were not rerun on these disjoint seeds; the seed check is limited to the original seven methods.'
    quotes['refine']=f'Across {len(winners)} eligible primary fits, k-means++ has strictly lower inertia in {strict}; the other {ties} are exact ties retained by the first-candidate rule. The component-derived candidate never wins strictly. The control uses ten k-means++ candidates and the same eighteen-update budget, whereas MBC uses one component-derived candidate plus nine k-means++ candidates.'
    w0=steps[0];w20=steps[19];w140=steps[139]
    wine=dyn[(dyn.representation=='wine_pca_10d')&(dyn.iteration<=140)]
    quotes['walk']=r'For Wine PCA-10, seed 3, $K=3$, iteration 1 evaluates attractions/responses/support and targets (Eqs.~\ref{eq:attraction}--\ref{eq:response}); the support range is '+n(w0['support_min'],3)+'--'+n(w0['support_max'],3)+r'. Momentum (Eqs.~\ref{eq:velocity-update}--\ref{eq:momentum-update}) has mean velocity norm '+n(w0['mean_velocity_norm'],3)+r'; inhibition (Eq.~\ref{eq:lateral-inhibition}) adds mean displacement '+n(w0['mean_inhibition_displacement'],3)+r'. Leakage (Eq.~\ref{eq:strength-leakage}) precedes deletion, generation and redundancy merging (Eq.~\ref{eq:redundant-merge}). The counts in iteration 1 are '+str(w0['buds_before'])+' before deletion, '+str(w0['buds_after_deletion'])+' after deletion, '+str(w0['buds_after_birth'])+' after generation, and '+str(w0['buds_after_merge'])+' after merging. Iterations 20 and 140 end with '+str(w20['buds_after_merge'])+' and '+str(w140['buds_after_merge'])+r' buds. The final 40 iterations range from '+str(int(wine[wine.iteration>100].buds.min()))+' to '+str(int(wine[wine.iteration>100].buds.max()))+r' buds: the count does not stabilize. Fusion then applies Eqs.~\ref{eq:local-geometry}--\ref{eq:fusion-predicate}, followed by supplied-$K$ merging (Eq.~\ref{eq:component-affinity}) and refinement (Eqs.~\ref{eq:compact-refinement}--\ref{eq:refinement-objective}). These records describe the sequence, not a causal attribution of class separation to inhibition.'
    quotes['public']='On fixed public Aggregation, Compound and two-spiral observations, MBC mean ARIs are '+', '.join(n(public.loc[d,'MBC']) for d in public.index)+'. Seeds affect algorithm initialization only. These records differ from the independently generated three-arm spiral and analogous shapes; they are reported separately.'
    # Explicit order for the public statement.
    quotes['public']='On public Aggregation (788 observations, seven labels), Compound (399, six labels) and the Handl--Knowles two-spiral data (1000, two labels), MBC mean ARIs are '+', '.join(n(public.loc[d,'MBC']) for d in ['aggregation','compound','spiral'])+'. Seeds affect algorithm initialization only. These fixed records differ from the independently generated three-arm spiral and analogous shapes; they are reported separately.'
    abstract='We study clustering with local prototype dynamics and explicit continuity tests. Morphogenetic Buds Clustering (MBC) updates, deletes and generates prototypes, joins compatible fragments, and optionally selects a compact refinement. The evaluated configuration requires a supplied target count before initialization. On fourteen primary datasets with seeds '+seedtext+', MBC mean ARI is '+n(overall.MBC)+', below spectral clustering ('+n(overall.Spectral)+'). On four moon/ring cases, MBC obtains '+n(primary.loc[['two_moons','noisy_moons','circles','noisy_circles'],'MBC'].mean())+' versus '+n(primary.loc[['two_moons','noisy_moons','circles','noisy_circles'],'HDBSCAN'].mean())+' for HDBSCAN. Eleven methods are compared over twelve blocks after merging correlated variants; the Friedman test gives $p='+n(stats['block_friedman_p'])+'$, without establishing MBC superiority. The original seven methods are also rerun on disjoint seeds '+newseed+'; MBC mean ARI is '+n(dismean.MBC)+'. Public labelled shape tests yield '+n(public.loc['aggregation','MBC'])+' on Aggregation, '+n(public.loc['compound','MBC'])+' on Compound and '+n(public.loc['spiral','MBC'])+' on two spirals. Generated three-arm spirals and disconnected classes expose contrasting failures. Scale tests cover all eleven methods on two compact cases and ten methods on increasing two-moon samples. Indian Pines and runtime measurements further delimit applicability. On refinement-eligible primary fits, the component-derived candidate never wins strictly against k-means++ candidates. The evidence supports an inspectable procedure for selected continuous morphologies, with sensitivity to geometry, supplied count and refinement.'
    setup=r'''\section{Experimental Setup}\label{sec:setup}
\Rev{The primary suite contains fourteen datasets: Two/Noisy Moons, Circles/Noisy Circles, Varied Density, Anisotropic Blobs, Iris, Wine, Breast Cancer, Digits 0--4, Ecoli, Glass, Vehicle and CMC. Seeds are $\{3,7,11,19,23\}$. Synthetic observations change with seed; the eight empirical datasets retain the same observations across algorithm seeds. Inputs are standardized; dimensions above ten are reduced by PCA (random state 0) to ten. All methods receive the same retained matrix. Reference classes are used for ARI and, for count-requiring methods, to supply $K$; class identities are not passed to fitting.}
\Rev{Public labelled shape observations are distributed through the clustering-benchmark mirror of SIPU (Aggregation and Compound) and the Handl--Knowles two-spiral benchmark. The archived ARFF headers identify their source publications~\cite{gionis2007aggregation,zahn1971,handl2004spiral}. They are artificial shape benchmarks, not empirical scenes. Source bytes and SHA-256 digests are frozen. The primary source papers and the Indian Pines distribution are listed in S1~\cite{fisher1936,aeberhard1992,street1993,kaynak1995,horton1996,evett1987,siebert1987,lim2000,zenodo2026}.}
'''
    inventory=[]
    for ds,(k,family) in DATASETS.items():
        d=pd.read_csv(ROOT/'data/processed/main_benchmark'/f'{ds}_seed3_processed.csv')
        inventory.append([ds_names[ds],family,str(len(d)),str(d.shape[1]-1),str(k)])
    setup+=table('inventory','Primary matrices supplied to every method. Wine is assigned to Low-d.', ['Dataset','Family','$n$','$d$','$K$'],inventory,'data/processed/main_benchmark/*seed3_processed.csv')
    setup+=figure('synthetic_geometries','Primary synthetic datasets, seed 3, standardized retained features. Colors show reference labels.','synthetic')
    setup+=rev(quotes['selection'])+'\n'
    setup+=r'''\Rev{The primary MBC parameters are $k_0=\max(12,6K)$, $T=140$, growth rate 0.15, inhibition 0.045, deletion threshold $\max(2.5,0.0055n)$, birth quantile 0.88, organ radius $3.6\sigma$, bridge ratio 0.18, alignment 0.35 and maximum tube gap 0.46. The refinement trigger is $d\ge3$, with eighteen updates and ten candidates. S1 lists the structural constants and the seven constructor-default changes. CPU array computations ran in Python 3.10.18 on Windows 10.0.19045, Intel i9-10885H, 31.73 GiB installed RAM. OMP, OpenBLAS, MKL and NumExpr thread limits were one; the host was not exclusively reserved and no memory cap was imposed. Runtime is elapsed fitting time, excluding preprocessing unless explicitly stated. Experiment subprocesses blocked external Python socket connections; preparation-time metadata retrieval is logged separately.}
'''
    rules=[['k-means',r'30 initializations; Lloyd algorithm; random state = seed','supplied'],['GMM',r'Full covariance; one k-means initialization; 100 maximum iterations; tolerance $10^{-3}$','supplied'],['Agglomerative','Ward linkage, Euclidean distance, no connectivity constraint','supplied'],['Spectral',r'Nearest-neighbour affinity; $\min(12,n-1)$ neighbours; k-means assignment, 10 initializations','supplied'],['DBSCAN',r'$\varepsilon$: 72nd percentile of sixth-neighbour distance, query includes self; $\min\_samples=\max(4,d+1)$','inferred'],['HDBSCAN',r'$\min\_cluster\_size=\max(10,\lfloor n/40\rfloor)$; $\min\_samples=5$; EOM selection','inferred'],['Mean shift',r'Bandwidth $2\times$ median eighth-neighbour distance (includes self); bin seeding','inferred'],['Affinity propagation',r'Euclidean affinity; median-similarity preference; damping 0.75; max 500, convergence 15 iterations','inferred'],['Control','Ten k-means++ candidates; eighteen finite Lloyd updates; lowest inertia','supplied'],['GNG',r'12 epochs; $\lambda=\max(50,\min(100,\lfloor n/4\rfloor))$; max nodes $\max(32,\min(120,12K))$','supplied']]
    setup+=table('comparator-rules','Fixed primary baseline rules. Remaining library defaults are frozen by versions and exported in baseline\_parameters.json. Count column indicates access to reference K.', ['Method','Rule','Count'],rules,'code/main/exp_driver.py')
    setup+=r'''\Rev{GNG uses two initial nodes, competitive Hebbian edges, age limit 50, winner/neighbour rates 0.20/0.006, insertion error factor 0.5 and global factor 0.995. Node vectors are grouped by k-means with 20 initializations; the grouping count is $\min(K,\text{node count})$, so the budget can undershoot $K$. Public shapes use this primary protocol. The separate generator-defined analogous shapes use twelve epochs and $\lambda=\max(50,\lfloor n/4\rfloor)$; compact scale and moon-scale GNG use three epochs and the same uncapped insertion interval. No universal GNG performance claim follows from these budgets. Noise label $-1$ remains one predicted label when computing ARI.}
'''
    results=r'\section{Results}\label{sec:results}'+'\n'+r'\subsection{Primary comparison and mechanism checks}'+'\n'
    quotes['statistics']='Across fourteen dataset means, Friedman $\\chi^2='+n(stats['raw_friedman_chi2'],3)+'$, $p='+n(stats['raw_friedman_p'])+'$. Collapsing clean/noisy Moons into one block and clean/noisy Circles into one gives twelve blocks, $\\chi^2='+n(stats['block_friedman_chi2'],3)+'$, $p='+n(stats['block_friedman_p'])+'$. The standard Iman--Davenport correction gives $F(10,110)='+n(stats['iman_davenport_f'],3)+'$, $p='+n(stats['iman_davenport_p'])+'$. These are omnibus differences, not proof that MBC is superior.'
    results+=rev(quotes['statistics'])+'\n'
    fams=['Non-convex','Gaussian','Density','Image','Low-d','Tabular']
    aggregate=pd.DataFrame({'Overall':overall,**{f:primary.loc[[ds for ds in DATASETS if DATASETS[ds][1]==f]].mean() for f in fams}}).reindex(METHODS)
    mark={c:ranked(aggregate[c]) for c in aggregate}
    results+=table('primary-short','Mean ARI over 14 datasets and five seeds. Bold, underline and italic denote the three highest distinct displayed values per column, with shared marks for ties.', ['Method',*aggregate.columns],[[display(m),*[mark[c][i] for c in aggregate]] for i,m in enumerate(METHODS)],'results/exp/primary_ari.csv')
    results+=rev('MBC ranks '+str(int(overall.rank(ascending=False,method='min')['MBC']))+' overall. CompactRefineOnly has mean ARI '+n(overall.CompactRefineOnly)+'. Removing refinement gives '+n(abl['MBC_no_compact'])+', micro-bud-only labels '+n(abl['MBC_micro_only'])+', removing polarity '+n(abl['MBC_no_polarity'])+', and removing channel continuity '+n(abl['MBC_no_capillary'])+'. The latter is slightly above full MBC ('+n(abl.MBC)+'), so this ablation provides no positive aggregate benefit for the channel predicate.')+'\n'
    results+=rev(quotes['refine'])+'\n'+figure('paper_fig_ablation_lollipop','Ablation means, 14 datasets and five seeds. Small changes are retained; removing channel continuity is not an exactly unchanged score.','ablation','.75')
    per=[]
    for ds,row in primary.iterrows():
        best='/'.join(display(m) for m in METHODS if round(row[m],4)==round(row.max(),4))
        per.append([ds_names[ds],DATASETS[ds][1],row.MBC,row.HDBSCAN,row.MeanShift,row.AffinityPropagation,best,float(row.max()),n(raw[(raw.dataset==ds)&(raw.method=='MBC')].time_s.mean(),2)])
    results+=table('primary-per-dataset','Per-dataset mean ARI, added comparators, best methods (ties at four decimals), and MBC fitting seconds. All eleven comparator rows are in the raw ledger.', ['Dataset','Family','MBC','HDBSCAN','Mean shift','Affinity','Best','ARI','MBC s'],per,'results/exp/primary_ari.csv')
    results+=rev('On Noisy Moons HDBSCAN ('+n(primary.loc['noisy_moons','HDBSCAN'])+') and DBSCAN ('+n(primary.loc['noisy_moons','DBSCAN'])+') exceed MBC ('+n(primary.loc['noisy_moons','MBC'])+'). On CMC the displayed MBC and k-means values are '+n(primary.loc['cmc','MBC'])+' and '+n(primary.loc['cmc','KMeans'])+'. The per-dataset table states observed winners; the method does not lead universally.')+'\n'
    wrows=[]
    for _,r in blockw.iterrows():wrows.append([r.comparison.replace('CompactRefineOnly','Control'),r.other_mean,r.mean_difference,str(int(r.n_nonzero_differences)),n(r.wilcoxon_statistic,1),r.p_value])
    results+=table('wilcoxon-short','Exploratory two-sided paired Wilcoxon tests on twelve collapsed blocks; exact zero differences are omitted. Values are nominal; none passes Holm correction over the ten comparisons.', ['Comparison','Baseline mean','MBC minus baseline','Effective $n$','$W$','$p$'],wrows,'results/exp/primary_block_wilcoxon.csv')
    results+=rev('The common MBC block mean is '+n(blocks.MBC.mean())+'. The raw 14-dataset mean '+n(overall.MBC)+' weights moons and circles twice; it is a different estimand. S1 lists block definitions and effective counts. Seeds are repetitions within a dataset, not independent dataset blocks.')+'\n'
    results+=r'\subsection{Sensitivity and independent seeds}'+'\n'+rev(quotes['disjoint'])+'\n'
    results+=rev(r'The dimension-trigger diagnostic uses Two Moons, Circles, Iris, Wine, Ecoli, Glass and Digits 0--4 with the five primary seeds. For thresholds $'+','.join(str(int(v)) for v in trigmeans.index)+'$, means are '+', '.join(n(v) for v in trigmeans)+'. It switches refinement when it crosses a retained dimension. At threshold 11, Wine and Digits 0--4 give '+n(trig.loc[11,'wine'])+' and '+n(trig.loc[11,'digits_0_4'])+' without refinement.')+'\n'
    sens=pd.read_csv(P/'reviewer3r2_sensitivity_reconcile.csv')
    results+=rev(r'The radius-sensitivity eleven-dataset set is Two Moons, Noisy Moons, Circles, Noisy Circles, Varied Density, Anisotropic Blobs, Iris, Wine, Breast Cancer, Digits 0--4 and Vehicle. The four-case subset contains the four moon/ring cases. At radii 2.8, 3.6 and 4.4, the eleven-case means are '+', '.join(n(v) for v in sens.grid_all_11)+', and four-case means are '+', '.join(n(v) for v in sens.grid_nonconvex_4)+'. Both decrease; the subset is not flat. The broader grid is in S1.')+'\n'
    results+=r'\subsection{Labelled shapes and scope}'+'\n'+rev(quotes['public'])+'\n'
    results+=table('public-shapes','Public labelled shapes, seeds 3, 7 and 11. Observations are fixed across seeds. Results use primary comparator rules and remain separate from the primary suite.', ['Method','Aggregation','Compound','Two spirals'],[[display(m),*[public.loc[d,m] for d in ['aggregation','compound','spiral']]] for m in METHODS],'results/exp/standard_shapes_raw.csv')
    results+=rev('MBC leads on Aggregation, while DBSCAN leads on Compound. On two spirals MBC, spectral clustering, DBSCAN and HDBSCAN all reach ARI 1.0000. This success does not transfer to the noisy generated three-arm spiral: MBC gives '+n(six.loc['three_spirals','MBC'])+'. On disconnected classes it gives '+n(six.loc['disconnected_manifolds','MBC'])+'. Thus performance depends on the exact sampled geometry and class definition, not the name of a shape family.')+'\n'
    results+=table('six-short','Six generated stress structures, three seeds; eight methods. Additional analogous-shape generators are listed separately in S1.', ['Structure',*map(display,six.columns)],[[esc(ds),*six.loc[ds].values] for ds in six.index],'results/paper_results/reviewer2_related_methods_raw.csv')
    results+=figure('reviewer3r2_six_structures','Six generated structures, seed 3. Reference labels define evaluation classes, including the deliberately disconnected limit case.','six')
    results+=r'\subsection{Scale and labelled spatial checks}'+'\n'
    cases=['high_d_6000_K20_imbalanced_pca10','large_n_imbalanced_largeK_d30']
    results+=table('scale-short','All eleven methods on two bounded compact scale cases, three seeds. Entries are mean ARI and fitting seconds; preprocessing is excluded. Warning details and realized GNG counts are in S1.', ['Method','High-$d$ ARI','s','Large-$n$ ARI','s'],[[display(m),sc.loc[(cases[0],m),'ari'],n(sc.loc[(cases[0],m),'seconds'],2),sc.loc[(cases[1],m),'ari'],n(sc.loc[(cases[1],m),'seconds'],2)] for m in METHODS],'results/paper_results/reviewer3_scale_all_methods_raw.csv')
    ncmethods=['MBC','Spectral','HDBSCAN','DBSCAN','MeanShift','KMeans','Agglomerative','GMM','CompactRefineOnly','GNG']
    results+=table('scale-nonconv','True two-moon scale generator: make\_moons noise 0.08, standardization, seeds 3, 7 and 11. Ten methods; affinity propagation was not run in this panel.', ['$n$',*map(display,ncmethods)],[[str(size),*[ncmean.loc[(size,m),'ari'] for m in ncmethods]] for size in [1000,2000,4000,8000]],'results/exp/nonconvex_scale_final_raw.csv')
    quotes['scale']='The large-sample case has 10,446 observations, 30 dimensions and fifteen imbalanced groups. The high-dimensional case has 1,011 observations, 6,000 features and twenty groups, standardized and reduced to ten PCA components. All eleven methods completed both panels. On true two moons, MBC mean ARI is '+n(ncmean.loc[(1000,'MBC'),'ari'])+' at $n=1000$ and '+n(ncmean.loc[(8000,'MBC'),'ari'])+' at $n=8000$, with fitting times '+n(ncmean.loc[(1000,'MBC'),'time_s'],2)+' and '+n(ncmean.loc[(8000,'MBC'),'time_s'],2)+' s. Affinity propagation is absent from this non-convex panel; no unrun value is imputed.'
    results+=rev(quotes['scale'])+'\n'
    indianmethods=list(indian.loc['pca10'].sort_values('ari',ascending=False).index)
    results+=table('indian-short','Indian Pines, 10,249 labelled pixels and sixteen classes; three seeds. Mean ARI and fitting seconds. PCA+XY has twelve retained dimensions.', ['Method','PCA ARI','s','PCA+XY ARI','s'],[[display(m),indian.loc[('pca10',m),'ari'],n(indian.loc[('pca10',m),'seconds'],2),indian.loc[('pca10_xy',m),'ari'],n(indian.loc[('pca10_xy',m),'seconds'],2)] for m in indianmethods],'results/paper_results/reviewer3r2_indian_pines_raw.csv')
    results+=rev('Indian Pines provides a real labelled spatial check. MBC gives '+n(indian.loc[('pca10','MBC'),'ari'])+'/'+n(indian.loc[('pca10_xy','MBC'),'ari'])+' for PCA/PCA+XY; mean shift gives '+n(indian.loc[('pca10','MeanShift'),'ari'])+'/'+n(indian.loc[('pca10_xy','MeanShift'),'ari'])+'. Affinity propagation and GNG were not evaluated on this scene. This is a nine-method panel, not a claim that all eleven methods were run. Affinity propagation was omitted from this scene panel as a computational-budget choice; its separately reported compact-scale results are unaffected.')+'\n'
    results+=figure('reviewer3r2_indian_pines_maps','Indian Pines, PCA+XY, seed 7. Ground truth and four methods include the best mean-ARI baseline, mean shift. Colors are matched to reference classes by maximum-overlap assignment only after fitting, for display. Gray denotes unmatched clusters/noise; this mapping does not alter ARI.','indian')
    results+=r'\subsection{Controlled runtime and process audit}'+'\n'
    quotes['complexity']=r'The timing generator puts centres on a circle of radius $3\sqrt K$ in the first two coordinates, adds independent Gaussian noise SD 0.3, allocates $\max(20,\lfloor n/K\rfloor)$ observations per centre, replicates with SD 0.05 jitter only to fill a shortfall, and truncates to $n$. The $n$ axis fixes $K=8,d=30$; the $K$ axis fixes $n=2000,d=2$; the $d$ axis fixes $n=2000,K=8$. The zero-offset added coordinates contain the same SD 0.3 noise. Geometry therefore belongs to a fixed family rather than a fixed realization across dimensions or K.'
    results+=rev(quotes['complexity'])+'\n'
    results+=table('runtime-short','Controlled CPU timing: three seeds, mean and sample SD; realized counts and full per-run rows are in S1.', ['Axis','Value','Mean s','SD s'],[[('$'+axis+'$'),str(int(v)),n(row.seconds,3),n(row.sd,3)] for (axis,v),row in timing.iterrows()],'results/exp/complexity.csv')
    coeff=fit['eq20_ols_coefficients'];neg=any(v<0 for v in coeff)
    results+=rev(r'The four-term Eq.~\ref{eq:work-bound} fit uses entering-iteration mean $k_t$, mean $k_t^2$, final $k_f$, and an active-refinement indicator. OLS gives $R^2='+n(fit['eq20_ols_r_squared'])+'$, coefficients '+', '.join(f'${v:.3g}$' for v in coeff)+'. '+('At least one coefficient is negative, so it cannot be interpreted as a physical operation cost. ' if neg else 'Positive fitted coefficients do not establish a physical scaling law. ')+r'Non-negative least squares gives $R^2='+n(fit['eq20_nonnegative_r_squared'])+'. Additional component-regularization and geometry costs remain outside the four-term proxy. The fits are descriptive.')+'\n'
    slopes=[]
    for item in fit['loglog']:slopes.append(item['axis']+': '+n(item['slope'],3)+' ['+n(item['ci95_low'],3)+', '+n(item['ci95_high'],3)+']')
    results+=rev('Log-log slopes and approximate normal 95\% intervals are '+('; '.join(slopes))+'. They are generator-specific, with only four to six mean points per fit. Non-monotone timings co-occur with changing realized bud counts and seed geometry; the measurements do not isolate their causes. For requested $K=32$, final buds are '+n(timing.loc[('K',32),'k'],2)+r'$\pm$'+n(timing.loc[('K',32),'ksd'],2)+' and fitting time '+n(timing.loc[('K',32),'seconds'],3)+r'$\pm$'+n(timing.loc[('K',32),'sd'],3)+' s. No saturation or universal runtime law is claimed.')+'\n'
    results+=rev(quotes['walk'])+'\n'+figure('reviewer3_dynamics_trajectory','Bud coverage and count for Two Moons (2D), Wine (13D) and Wine PCA-10, all seed 3. Only the 140 pre-refinement iterations are plotted. Oscillations are retained.','main-dynamics')
    results+=rev('For Wine PCA-10, seed 3, stopping at '+', '.join(str(r['steps']) for r in stopping)+' iterations gives final ARIs '+', '.join(n(r['ari']) for r in stopping)+'. This checks sensitivity to the stopping point on one run, not convergence.')+'\n'
    results+=figure('reviewer3r2_refinement_trajectory','Wine, seed 3, original 13D and PCA-10. Orange is the component-derived candidate; gray curves are the nine k-means++ candidates. The last record is the final assignment evaluation. No empty-cluster event is inferred from these curves.','refinement')
    results+=r'\subsection{Unknown count and external data}'+'\n'
    quotes['silhouette']=r'The 21 silhouette cases are seven datasets (Two Moons, Circles, Noisy Moons, Noisy Circles, Varied Density, Wine and Breast Cancer) crossed with seeds $\{3,7,11\}$. Rerunning MBC for $K\in\{2,3,4,5,6\}$ and choosing the largest silhouette recovers the reference count in '+str(correct)+' cases. Mean ARI is '+n(ks.estimated_k_ari.mean())+' with selected $K$, versus '+n(ks.true_k_ari.mean())+' with supplied $K$ on the same cases. The trigger-sweep mean '+n(trigmeans.loc[3])+' concerns a different seven-dataset set. Baselines were not rerun under silhouette selection, so it is an MBC diagnostic only.'
    results+=rev(quotes['silhouette'])+'\n'
    extnames={'Leukemia_2class':'Leukemia-2','Leukemia_3class':'Leukemia-3','Lung_5class':'Lung-5','Lymphoma_3class':'Lymphoma-3','MiceProtein_8class':'MiceProtein-8'}
    lung=[d for d in ext.index if 'Lung' in d][0];mice='MiceProtein_8class'
    quotes['external']='External labelled cases are OpenML 40966, 45090, 45091, 45094 and 45093, with a twelve-component PCA cap. Lung-5 gives MBC '+n(ext.loc[lung,'MBC'])+', DBSCAN '+n(ext.loc[lung,'DBSCAN'])+' and spectral '+n(ext.loc[lung,'Spectral'])+'; the largest observed method is '+display(ext.loc[lung].idxmax())+' ('+n(ext.loc[lung].max())+'). MiceProtein-8 gives MBC '+n(ext.loc[mice,'MBC'])+' and spectral '+n(ext.loc[mice,'Spectral'])+'. These results are kept separate from the primary means.'
    results+=rev(quotes['external'])+r'~\cite{higuera2015,golub1999,alizadeh2000,bhattacharjee2001}'+'\n'
    yyc=pd.read_csv(P/'hyperspectral_yyc_mbc_metrics.csv');ym=yyc[(yyc.algorithm=='MBC')&(yyc.feature=='spectral_pca_xy')].iloc[0]
    results+=rev('YYC200 has no reference labels. For eight standardized PCA features plus two coordinates (weight 0.35), MBC with $K=6$, seed 7 gives sampled silhouette '+n(ym.silhouette_sample)+', Davies--Bouldin '+n(ym.davies_bouldin)+', and Calinski--Harabasz '+n(ym.calinski_harabasz,2)+'. S1 gives all internal indices. They do not measure land-cover accuracy.')+'\n'
    discussion=r'''\section{Discussion}\label{sec:discussion}
\Rev{MBC recovers selected continuous structures, including the fixed public two-spiral sample, while failing on the generated noisy three-arm spiral, density gradients and disconnected classes. These cases differ in sampling, noise and topology; a family name is insufficient to predict success. Local separation is a labelled diagnostic rather than a hyperparameter oracle. Public shape success does not establish robustness on arbitrary non-convex data.}
'''+rev(quotes['refine'])+'\n'+r'''\Rev{The compact-refinement control and ablations describe associations under a fixed finite protocol. They do not isolate a universal causal benefit for each biological term. In particular, removing channel continuity slightly increases the aggregate ARI; it is not evidence of a positive aggregate contribution. Reference K remains favourable information. Historical parameter development is not fully reconstructable, although all parameters and actual outputs in the present rerun are recorded.}
\Rev{The runtime experiment measures one CPU host without an exclusive reservation or imposed memory cap. Its four-term fit includes both k-squared terms but is descriptive and omits separate component-regularization overhead. The three-epoch GNG scale budget can leave fewer nodes than K and is therefore reported as a budget-limited comparator. Repeated algorithm seeds characterize initialization/generator variability, not uncertainty over unseen datasets. Future work can test adaptive neighbourhood scales, continuity estimators after PCA and count selection under a common baseline protocol.}
\section{Conclusion}\label{sec:conclusion}
\Rev{MBC provides inspectable local-prototype dynamics and explicit continuity predicates for selected geometries. The primary suite does not show overall superiority to spectral clustering; public shape tests show both competitive outcomes and failures, and the real spatial and scale panels delimit the scope. Refinement commonly selects a k-means++ candidate. The released offline inputs, raw outputs, parameter records and generation scripts support checking these claims without imputing unrun experiments.}
'''
    prefix=(T/'main_prefix.tex').read_text(encoding='utf-8').replace('%%ABSTRACT%%',r'\abstract{'+rev(abstract)+'}')
    bib=(T/'bibliography.tex').read_text(encoding='utf-8')
    maintext=prefix+setup+results+discussion+'\n'+bib
    for kind in ['Clean','Tracked']:
        text=maintext if kind=='Clean' else maintext.replace(r'\newcommand{\Rev}[1]{#1}',r'\newcommand{\Rev}[1]{\textcolor{red}{#1}}')
        (ROOT/'paper'/f'MBC_CMC_revision2_{kind}.tex').write_text(text,encoding='utf-8')
    # Supplement: complete numerical records, without repeated main figures.
    sup=r'\section{Protocol and source records}'+'\n'+rev(quotes['selection'])+'\n'+(T/'core_parameter_tables.tex').read_text(encoding='utf-8')
    sup+=rev(r'The seven constructor-default changes are organ radius $4.2\to3.6$, bridge ratio $0.28\to0.18$, polarity $0.42\to0.35$, channel gap $0.52\to0.46$, growth $0.18\to0.15$, inhibition $0.035\to0.045$, and birth quantile $0.92\to0.88$. They were inherited and frozen for this rerun; the saved grid does not document the historical choice of all seven. Eight candidate observations are checked for a possible birth, and at most one is accepted per step. A count at or below two bypasses deletion, rather than enforcing a hard two-node minimum.')+'\n'
    sup+=rev(r'The main comparator-rules table and \texttt{results/exp/baseline\_parameters.json} enumerate library settings, including defaults and data-derived bandwidths. Primary/public GNG uses the insertion cap 100; analogous-shape and three-epoch scale GNG do not. The latter may return fewer than K groups. Full warning strings are retained per run.')+'\n'
    sup+=r'\subsection{Originating references and exact inputs}'+'\n'+(T/'source_records.tex').read_text(encoding='utf-8')
    sup+=table('s1-inventory','Primary matrix inventory, excluding target column.', ['Dataset','Family','$n$','$d$','$K$'],inventory,'data/processed/main_benchmark/*seed3_processed.csv')
    sup+=r'''\Rev{Primary generators: Two Moons uses make\_moons(n=500, noise=0.07, random state=seed); Noisy Moons uses noise 0.13 and seed+1. Circles uses make\_circles(n=500, factor=0.42, noise=0.045, seed); Noisy Circles uses factor 0.5, noise 0.06, seed+1. Anisotropic Blobs uses make\_blobs(n=500, four centres, SDs 0.55/0.8/1.15/0.45, seed), then right-multiplies by $[(0.6,-0.7);(1.4,0.35)]$. Varied Density concatenates Gaussians with sizes 140/220/140, means $(-1.8,0),(0,0.2),(1.9,-0.1)$, and coordinate SDs $(0.18,0.55),(0.62,0.2),(0.28,0.32)$. Every primary synthetic matrix is standardized after generation.}
'''
    sup+=r'\section{Sensitivity, seeds and count selection}'+'\n'+rev(quotes['disjoint'])+'\n'
    seven=['Spectral','MBC','GMM','Agglomerative','DBSCAN','KMeans','CompactRefineOnly']
    sup+=table('s1-seeds','Seed sets cover the same 14 datasets. Seven-method seed diagnostic only.', ['Seeds',*map(display,seven)],[[seedtext,*[overall[m] for m in seven]],[newseed,*[dismean[m] for m in seven]]],'results/exp/primary_ari.csv; results/paper_results/reviewer3r2_disjoint_seeds_raw.csv')
    sup+=table('s1-trigger','Dimension-trigger diagnostic. Entries are dataset mean ARI over the five primary seeds.', ['Dataset',*[str(int(v)) for v in trig.index]],[[ds_names[ds],*trig[ds]] for ds in trig.columns],'results/paper_results/reviewer3_trigger_sensitivity_raw.csv')
    broad=pd.read_csv(P/'reviewer1_broad_sensitivity_raw.csv').groupby(['parameter','value']).ari.mean()
    sup+=table('s1-grid','Actual eleven-dataset one-factor grid: all levels are shown. The named eleven- and four-dataset sets are defined in the main sensitivity paragraph.', ['Parameter','Level','Mean ARI'],[[esc(p),str(v),a] for (p,v),a in broad.items()],'results/paper_results/reviewer1_broad_sensitivity_raw.csv')
    sup+=rev(quotes['silhouette'])+'\n'
    sup+=table('s1-count','Every silhouette-selected count case.', ['Dataset','Seed','Reference K','Chosen K','Chosen ARI','Supplied ARI'],[[ds_names[r.dataset],str(r.seed),str(r.true_k),str(r.estimated_k_by_silhouette),r.estimated_k_ari,r.true_k_ari] for r in ks.itertuples()],'results/paper_results/reviewer1_k_estimation_summary.csv')
    sup+=r'\section{Process records}'+'\n'+rev('The main process and refinement figures are not repeated here. Numeric histories are saved in results/figure\_data, including a traced Wine PCA-10 run. The trace observes the unmodified algorithm and does not change fitting.')+'\n'
    sup+=table('s1-walk','Wine PCA-10, seed 3: recorded lifecycle counts and coverage inertia.', ['Iteration','Entering buds','After deletion','After birth','After merge','Coverage inertia'],[[str(r['iteration']),str(r['buds_before']),str(r['buds_after_deletion']),str(r['buds_after_birth']),str(r['buds_after_merge']),n(r['coverage_inertia'],3)] for r in [steps[i-1] for i in [1,2,5,20,60,100,120,140]]],'results/figure_data/wine_equation_walkthrough.json')
    sup+=table('s1-stopping','Wine PCA-10 seed 3 stopping-point diagnostic.', ['Iterations','Final buds','Final ARI'],[[str(r['steps']),str(r['buds']),r['ari']] for r in stopping],'results/figure_data/wine_stopping_steps.json')
    sup+=r'\section{Extended shapes and scale protocols}'+'\n'+(T/'extended_generators.tex').read_text(encoding='utf-8')+'\n'
    analogous=pd.read_csv(E/'shape.csv',index_col=0)
    sup+=table('s1-analogues','Public SIPU shape records: full per-method ARI over seeds 3, 7 and 11. These are the fixed observations behind the main Table~8; GNG uses its separately declared twelve-epoch uncapped-insertion shape protocol.', ['Shape',*map(display,analogous.columns)],[[esc(ds),*analogous.loc[ds]] for ds in analogous.index],'results/exp/shape.csv')
    sup+=figure('public_shape_benchmarks','Public fixed shape observations (left) and seed-3 predictions. Panel colors are categorical within each panel; numerical agreement is evaluated with ARI.','s1-public')
    morph=pd.read_csv(E/'morphology_diagnostics.csv').groupby('dataset').gap_ratio.mean()
    sup+=table('s1-separation','Local gap ratio g/h on retained features, averaged across primary generator seeds (fixed public observations evaluated once). Values at most one violate the diagnostic local-separation condition; success is not guaranteed when greater than one.', ['Dataset','Mean $g/h$'],[[esc(ds),v] for ds,v in morph.items()],'results/exp/morphology_diagnostics.csv')
    gng=pd.read_csv(P/'reviewer3_true_gng_scale_raw.csv')
    sup+=table('s1-gng-budget','Three-epoch GNG scale budget: realized nodes and actual predicted groups, with supplied reference K.', ['Case','Seed','K','Nodes','Groups','ARI'],[[('High-d' if r.case.startswith('high_d') else 'Large-n'),str(r.seed),str(r.K),str(r.realized_nodes),str(r.actual_clusters),r.ari] for r in gng.itertuples()],'results/paper_results/reviewer3_true_gng_scale_raw.csv')
    warns=scraw[scraw.warnings.fillna('').str.len()>0]
    sup+=rev('Scale warnings were returned in '+str(len(warns))+' of '+str(len(scraw))+' rows. Warning-bearing method/case/seed records are listed in the machine-readable ledger; the returned labels are included. A convergence warning is not counted as successful convergence. The non-convex scale panel has ten methods, and the two compact panels have eleven.')+'\n'
    sup+=r'\section{External and spatial records}'+'\n'
    exts=[d for d in extnames if d in ext.index]
    # Preserve exact actual names; reject absent target datasets.
    if len(exts)!=5:exts=[d for d in ext.index if any(k in d for k in ['Leukemia','Lymphoma','Lung','MiceProtein_8class']) and not any(k in d for k in ['genotype','behavior','treatment'])]
    exts=list(dict.fromkeys(exts));assert len(exts)==5,exts
    sup+=table('s1-external','External cases, three seeds. MBC, k-means, GMM, Spectral, Ward and Control receive reference K; DBSCAN infers count. The three highest displayed ARIs per row share dense ranks.', ['Dataset',*map(display,seven)],[[extnames.get(ds,esc(ds)),*ranked([ext.loc[ds,m] for m in seven])] for ds in exts],'results/paper_results/external_real_benchmark_raw.csv')
    sup+=table('s1-indian-sd','Indian Pines sample standard deviations across three seeds; means match the main table.', ['Method','PCA mean','SD','PCA+XY mean','SD'],[[display(m),indian.loc[('pca10',m),'ari'],indian.loc[('pca10',m),'sd'],indian.loc[('pca10_xy',m),'ari'],indian.loc[('pca10_xy',m),'sd']] for m in indianmethods],'results/paper_results/reviewer3r2_indian_pines_raw.csv')
    sup+=table('s1-yyc','YYC200, seed 7: internal indices, inferred group counts (noise excluded in the following explanatory note), and fitting seconds. This is an unlabeled scene; no ARI is available.', ['Method','Features','Reported labels','Silhouette','DB','CH','s'],[[r.algorithm,esc(r.feature),str(r.clusters),r.silhouette_sample,r.davies_bouldin,n(r.calinski_harabasz,2),n(r.runtime_seconds,2)] for r in yyc.itertuples()],'results/paper_results/hyperspectral_yyc_mbc_metrics.csv')
    sup+=rev(r'The YYC200 metric ledger counts distinct returned labels, including noise $-1$ for density methods; this count is not necessarily the number of non-noise groups. Silhouette uses a fixed sample of up to 6000 observations. The image is 200 by 200 pixels with 64 bands. Bands must have less than 20\% zeros, SD above 1, and a nonzero 1--99\% range; retained values are clipped at those percentiles, standardized, PCA-reduced to eight components, and standardized again. Coordinates range from -1 to 1 and carry weight 0.35. GMM uses full covariance; YYC200 Ward alone uses a ten-neighbour connectivity graph to keep its scene run bounded. No semantic class accuracy is inferred.')+'\n'
    sup+=r'\section{Timing and statistical records}'+'\n'+rev(quotes['complexity'])+'\n'
    sup+=table('s1-realized-k','All controlled timing points. k is the final bud count; the fit separately uses entering-iteration mean k and mean k squared. SD is across three seeds.', ['Axis','Value','Final k','SD k','Mean $k_t$','Mean $k_t^2$','Mean s','SD s'],[['$'+axis+'$',str(int(v)),n(r.k,2),n(r.ksd,2),n(r.mean_k,2),n(r.mean_k2,2),n(r.seconds,3),n(r.sd,3)] for (axis,v),r in timing.iterrows()],'results/exp/complexity.csv')
    sup+=figure('reviewer3_runtime_axes','Controlled log-log elapsed fitting times; error bars are sample SD over three seeds. This figure uses the same controlled ledger as the table, including eight centres on the d axis.','s1-runtime')
    sup+=rev('The twelve rank-test blocks are Moons (mean of clean/noisy means), Circles (likewise), Varied Density, Anisotropic Blobs, Iris, Wine, Breast Cancer, Digits 0--4, Ecoli, Glass, Vehicle and CMC. All eleven methods are included. The block matrix and nominal two-sided Wilcoxon tests are released; neither seeds nor correlated variants are treated as additional independent datasets.')+'\n'
    sup+=table('s1-wilcoxon-block','Twelve-block Wilcoxon statistics matching the main table; nominal p values. Zero differences are excluded.', ['Comparison','Baseline mean','Difference','Effective n','W','p'],wrows,'results/exp/primary_block_wilcoxon.csv')
    sup+=r'\subsection{Reproduction scope}'+'\n'+rev(r'The local experiment run regenerated processed matrices and computed all result-bearing stages from bundled source inputs. A preliminary attempt was stopped to correct the disjoint-seed loader; its logs are preserved as an interrupted attempt, not claimed as completion. Completion records include actual exit codes, elapsed times, code/input hashes and immutable output snapshots. The experiment used hologpu Python 3.10.18; the bundled Windows-x64 portable runtime is Python 3.10.11 with the same pinned core numerical libraries. Portable import, fitting, plotting and notebook smoke checks are distinct from the complete hologpu experiment. External Python network access is blocked; Windows itself was not physically disconnected. Recompiling PDFs additionally requires an installed LaTeX distribution; precompiled PDFs and source are included.')+'\n'
    supbib=bib.replace(r'\begin{thebibliography}{99}',r'\begin{thebibliography}{99}')
    # Cite shared methods explicitly; the complete bibliography is shared verbatim.
    sup+=r'\Rev{Shared method sources: k-means~\cite{hartigan1979}, GMM~\cite{dempster1977}, Ward~\cite{ward1963}, spectral~\cite{shi2000}, DBSCAN~\cite{ester1996}, HDBSCAN~\cite{mcinnes2017hdbscan}, mean shift~\cite{comaniciu2002}, affinity propagation~\cite{frey2007}, GNG~\cite{fritzke1995}, k-means++~\cite{arthur2007}, and scikit-learn~\cite{pedregosa2011}.}'+'\n'
    suptext=(T/'supplement_prefix.tex').read_text(encoding='utf-8')+sup+'\n'+supbib
    for kind in ['Clean','Tracked']:
        text=suptext if kind=='Clean' else suptext.replace(r'\newcommand{\Rev}[1]{#1}',r'\newcommand{\Rev}[1]{\textcolor{red}{#1}}')
        (ROOT/'paper'/f'MBC_CMC_revision_supplement_{kind}.tex').write_text(text,encoding='utf-8')
    (G/'table_manifest.json').write_text(json.dumps(specs,indent=2),encoding='utf-8')
    summary=dict(primary_means=overall.to_dict(),disjoint_means=dismean.to_dict(),stats=stats,
                 public_shapes=public.to_dict(),eligible_fits=len(winners),strict_kmeanspp_wins=strict,ties=ties,
                 abstract_word_count=len(re.sub(r'\\[A-Za-z]+','',abstract).split()),quotes=quotes)
    (ROOT/'results/submission_numbers.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    # The response generator uses exact paragraphs emitted above as its quotations.
    from build_response import build_response
    build_response(ROOT,quotes,summary)
    print('Submission sources and',len(specs),'tables rebuilt from local ledgers.')

if __name__=='__main__':main()
