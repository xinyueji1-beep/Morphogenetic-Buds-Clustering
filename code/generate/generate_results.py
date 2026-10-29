"""Run all experiments; resume only verified, immutable stage snapshots."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / 'code/main'
CHECKPOINTS = ROOT / 'results/checkpoints'
JSON_RESULTS = ROOT / 'results/generated_json'
LABELS = [
    'Export bundled datasets', 'Main benchmark', 'Sensitivity analysis',
    'External real-data benchmark', 'Runtime and biological visuals',
    'YYC200 case study', 'Cluster figures', 'Advanced figures', 'YYC200 analysis',
    'Reviewer 1 analyses', 'Reviewer 2 analyses', 'Reviewer 3 analyses',
    'Reviewer 3 round-2 analyses', 'Round-2 follow-up tables',
]
STAGES = [dict(name=f'manuscript_{i:02d}', label=label,
               command=['manuscript_pipeline.py', '--only', label],
               sources=['manuscript_pipeline.py', 'mbc.py'])
          for i, label in enumerate(LABELS, 1)]
for name, script, sources in [
    ('r3_exp_driver', 'exp_driver.py', ['exp_driver.py', 'mbc.py', 'gng.py']),
    ('r3_scale_driver', 'scale_driver.py', ['scale_driver.py', 'exp_driver.py', 'mbc.py', 'gng.py']),
    ('r3_scale_all_methods', 'reviewer3_scale_all_methods.py', ['reviewer3_scale_all_methods.py', 'exp_driver.py', 'mbc.py', 'gng.py']),
    ('r3_true_gng', 'reviewer3_true_gng.py', ['reviewer3_true_gng.py', 'reviewer3_scale_all_methods.py', 'exp_driver.py', 'mbc.py', 'gng.py']),
    ('r3_block_statistics', 'reviewer3_statistics.py', ['reviewer3_statistics.py', 'exp_driver.py']),
    ('r3_true_gng_shapes', 'reviewer3_true_gng_shapes.py', ['reviewer3_true_gng_shapes.py', 'gng.py']),
    ('r3_complexity_analysis', 'reviewer3_complexity_analysis.py', ['reviewer3_complexity_analysis.py']),
]:
    STAGES.append(dict(name=name, label=name, command=[script], sources=sources))

# Figure-data export and SIPU shape benchmark are prerequisites for plot_results.py
# but were historically maintained outside the core STAGES. Wire them in so the pure
# code+data package reproduces the submission figures end-to-end.
STAGES.append(dict(name='standard_shapes', label='standard_shapes',
                   command=['standard_shapes.py'], sources=['standard_shapes.py', 'exp_driver.py', 'mbc.py']))
STAGES.append(dict(name='export_figure_data', label='export_figure_data',
                   command=['export_paper_states.py'], sources=['export_paper_states.py', 'mbc.py', 'exp_driver.py', 'manuscript_pipeline.py']))

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def inventory():
    paths = []
    for folder in ['results/paper_results', 'results/exp', 'results/hyperspectral_exploration', 'figures', 'data/processed']:
        d = ROOT / folder
        if d.exists(): paths.extend(p for p in d.rglob('*') if p.is_file())
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}

def run(stage, force, upstream):
    import shutil
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    JSON_RESULTS.mkdir(parents=True, exist_ok=True)
    marker = CHECKPOINTS / (stage['name']+'.json')
    sources = {s: sha(MAIN/s) for s in stage['sources']}
    if stage['name'] in {'r3_scale_driver', 'r3_true_gng'}:
        sources['scale_geometry.py'] = sha(MAIN/'scale_geometry.py')
    key = hashlib.sha256(json.dumps({'sources': sources, 'upstream': upstream}, sort_keys=True).encode()).hexdigest()
    if not force and marker.exists():
        old = json.loads(marker.read_text(encoding='utf-8'))
        if old.get('status') == 'complete' and old.get('fingerprint') == key and all(
            (ROOT/o['snapshot']).is_file() and sha(ROOT/o['snapshot']) == o['sha256'] for o in old['outputs']):
            print('[resume verified snapshot]', stage['name'], flush=True)
            return old, False
    command = [sys.executable, '-u', str(MAIN/stage['command'][0]), *stage['command'][1:]]
    record = dict(stage=stage['name'], label=stage['label'], command=command, status='running',
                  started_unix=time.time(), sources=sources, fingerprint=key, resumed=False)
    marker.write_text(json.dumps(record,indent=2),encoding='utf-8')
    env = os.environ.copy()
    env.update(MBC_OFFLINE='1', MPLBACKEND='Agg', PYTHONUTF8='1',
               PYTHONPATH=str(ROOT/'code/offline_guard'))
    audit_dir = ROOT / 'audit'
    if audit_dir.exists():
        env['MBC_NETWORK_AUDIT'] = str(audit_dir / 'network_attempts.jsonl')
    before = inventory()
    started = time.perf_counter()
    print('==>', stage['name'], stage['label'], flush=True)
    logfile = JSON_RESULTS/(stage['name']+'.log')
    with logfile.open('w', encoding='utf-8') as stream:
        proc = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    record.update(returncode=proc.returncode, completed_unix=time.time(), elapsed_seconds=time.perf_counter()-started)
    if proc.returncode:
        record['status']='failed'; marker.write_text(json.dumps(record,indent=2),encoding='utf-8')
        raise RuntimeError(f"Stage failed: {stage['name']}; see {logfile}")
    after = inventory()
    outputs=[]
    for rel, digest in after.items():
        if before.get(rel) != digest:
            snapshot=Path('results/stage_snapshots')/stage['name']/rel
            (ROOT/snapshot).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/rel, ROOT/snapshot)
            outputs.append(dict(path=rel,snapshot=snapshot.as_posix(),sha256=digest,bytes=(ROOT/rel).stat().st_size))
    record.update(status='complete', outputs=outputs, log=logfile.relative_to(ROOT).as_posix(), log_sha256=sha(logfile))
    marker.write_text(json.dumps(record,indent=2),encoding='utf-8')
    (JSON_RESULTS/(stage['name']+'.json')).write_text(json.dumps(record,indent=2),encoding='utf-8')
    print('<==',stage['name'],f"{record['elapsed_seconds']:.1f}s",'outputs',len(outputs),flush=True)
    return record, True

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--list',action='store_true')
    parser.add_argument('--force',action='store_true')
    parser.add_argument('--stage',help='start at a stage (preceding stages must have verified checkpoints)')
    args=parser.parse_args()
    if args.list:
        for s in STAGES: print(s['name'],s['label'])
        return
    if args.stage and args.stage not in [s['name'] for s in STAGES]: raise ValueError(args.stage)
    initial_path = ROOT / 'audit' / 'initial_inputs.json'
    if initial_path.exists():
        initial = json.loads(initial_path.read_text(encoding='utf-8'))
        upstream = hashlib.sha256(json.dumps(initial['inputs'], sort_keys=True).encode()).hexdigest()
    else:
        # Pure code+data reproduction: the provenance manifest is optional.
        upstream = hashlib.sha256(b'no-initial-inputs-manifest').hexdigest()
    changed=args.force; records=[]
    for stage in STAGES:
        if args.stage == stage['name']: changed=True
        record, ran=run(stage,changed,upstream)
        changed=changed or ran
        upstream=hashlib.sha256(json.dumps({'fingerprint':record['fingerprint'],'outputs':record['outputs']},sort_keys=True).encode()).hexdigest()
        records.append(record)
    (JSON_RESULTS/'run_summary.json').write_text(json.dumps(dict(status='complete',count=len(records),stages=records),indent=2),encoding='utf-8')
    print('All experiment stages completed.',flush=True)

if __name__=='__main__': main()
