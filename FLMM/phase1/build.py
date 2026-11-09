"""Build self-contained private Kaggle CPU-prep and gated GPU notebooks."""
import base64
import json
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'kaggle/results/smoke-v6/smoke-results'

def markdown(text): return {'cell_type':'markdown','metadata':{},'source':text.splitlines(True)}
def code(text): return {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':text.splitlines(True)}
def notebook(cells):
    for i,cell in enumerate(cells): cell['id']=hashlib.sha256((str(i)+''.join(cell['source'])).encode()).hexdigest()[:12]
    return dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python',version='3.13')),nbformat=4,nbformat_minor=5)

def payload():
    files={}
    for p in ROOT.glob('*.py'):
        if p.name not in ['build.py','snapshot_quota.py','test_common.py']: files[p.name]=base64.b64encode(p.read_bytes()).decode()
    files['config.json']=base64.b64encode((ROOT/'config.json').read_bytes()).decode()
    for p in (ROOT/'docs').glob('*.md'): files['docs/'+p.name]=base64.b64encode(p.read_bytes()).decode()
    files['quota_snapshot.json']=base64.b64encode((ROOT/'results/quota_snapshot.json').read_bytes()).decode()
    freeze=(ROOT.parent/'kaggle/results/smoke-v6/setup-report/freeze.log').read_text()
    files['constraints.txt']=base64.b64encode(('\n'.join(x for x in freeze.splitlines() if '==' in x)+'\n').encode()).decode()
    files['regression-manifest.json']=base64.b64encode((OLD/'sample-manifest.json').read_bytes()).decode()
    files['reference/metrics.json']=base64.b64encode((OLD/'metrics.json').read_bytes()).decode()
    files['reference/subset-evaluator.py']=base64.b64encode((ROOT.parent/'kaggle/evaluator/eval_refcoco_subset.py').read_bytes()).decode()
    for p in OLD.glob('sample-*.npz'): files['reference/'+p.name]=base64.b64encode(p.read_bytes()).decode()
    return files

def main():
    target=ROOT/'cpu-prep'; target.mkdir(exist_ok=True)
    extraction=f'''import base64,json,os,subprocess,sys,time
from pathlib import Path
phase=Path('/kaggle/working/phase1'); phase.mkdir(exist_ok=True)
payload={payload()!r}
for name,encoded in payload.items():
    path=phase/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(base64.b64decode(encoded))
'''
    cpu=notebook([markdown('# F-LMM Phase 1 — CPU preparation\nVerified data/model cache and reusable environment. No GPU requested. Original smoke output preserved.'),code(extraction),code("subprocess.run([sys.executable,str(phase/'prepare.py')],check=True,timeout=5400)\n")])
    (target/'prepare.ipynb').write_text(json.dumps(cpu,ensure_ascii=False),encoding='utf-8')
    metadata=dict(id='hoofngocjhuy/flmm-phase1-cache',title='FLMM Phase1 Cache',code_file='prepare.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=False,enable_internet=True,dataset_sources=[],competition_sources=[],kernel_sources=[])
    (target/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
    print(target)
    build_gpu(extraction)

def build_gpu(extraction):
    for resumed in [False,True]:
        folder=ROOT/('gpu-resume' if resumed else 'gpu-gates'); folder.mkdir(exist_ok=True)
        configure=extraction+f'''sys.path.insert(0,str(phase))
configuration=json.loads((phase/'config.json').read_text())
configuration['STOP_AFTER_IMAGES']={None if resumed else 75!r}
configuration['RESUME_INPUT']={('flmm-phase1-gates' if resumed else None)!r}
(phase/'config.json').write_text(json.dumps(configuration,indent=2))
from bootstrap import initialize,stage
gate0_only=configuration.get('STOP_AFTER_GATE')==0
'''
        cells=[markdown('# F-LMM Phase 1 — feasibility-first reproduction\nPrivate, bounded, manifest-driven. Run All obeys integrity/resource gates; it does not run every benchmark. Original 20-image regression is immutable. No research improvement.'),
               markdown('## 1. Configuration\n120-minute walltime budget, 75-minute evaluation budget, 20-minute export reserve; disk floor 3 GiB. Larger evaluations disabled by default. Resume accepts only validated matching fingerprints.'),code(configure),
               markdown('## 2. Environment and checkpoint audit\nRestore the verified CPU-prepared environment once; no reinstall or network access during inference. G0 requires tensor provenance and exact 20-image evaluator equivalence.'),
               code("cache=initialize()\n"+("print('Using prior G0; setup validates resume fingerprint before reusing outputs')\n" if resumed else "stage('audit')\n")),
               markdown('## 3. Dataset validation and caching\nImage/ref/sentence/mask mapping comes from official annotations and is checked against installed loader. 50 probe and 100 stability images are nested in development; holdout is disjoint and untouched.'),
               code("manifest=json.loads((Path('/kaggle/working/results/dataset_manifest.json')).read_text())\nprint(manifest['counts']); print(manifest['distribution'])\n"),
               markdown('## 4. Resource preflight\nReal GPU/RAM/disk, pinned cache hashes, observed quota snapshot and persistence limitations.'),
               code("print(Path('/kaggle/working/results/resource_preflight.json').read_text())\n"),
               markdown('## 5. 50-image feasibility probe\nSynchronized post-warm-up latency, peak allocated/reserved VRAM, RAM, output bytes and data preparation costs measured separately.'),
               code("if gate0_only:\n    print('UNVERIFIED: user requested stop after Gate 0; probe not run')\nelse:\n    "+("print('Reusing completed probe from prior committed output')\n" if resumed else "stage('probe')\n")),
               markdown('## 6. Go/No-Go decision\nNo expansion while G0 is unverified. Stale/unknown quota limits scope; disk, timing and cached-data gates apply.'),code("if not gate0_only:\n    stage('gate')\nelse:\n    print('Not run: stop after Gate 0')\n"),
               markdown('## 7. 100-image stability test\nThe first job stops at 75 validated stability images and commits output; the resume job skips them and completes the remaining 25. Already completed 50 probe images are reused. This deliberately tests actual cross-session persistence. Currently disabled by user request: stop after G0.'),code("if not gate0_only:\n    stage('stability')\nelse:\n    print('UNVERIFIED: stability and resume not run')\n"),
               markdown('## 8. Resumable baseline evaluation\nDevelopment/holdout/full validation are disabled by default. Enabling them also requires G2, a fresh GO decision, sufficient prepared cache and the configured image cap.'),
               code("if not gate0_only and Path('/kaggle/working/results/G2.json').exists():\n    stage('baseline')\nelse:\n    print('Not run: expansion disabled until user audit and later gates')\n"),
               markdown('## 9. Metrics and qualitative inspection\nGlobal sums of intersections/unions; one first sentence per selected annotation. Per-expression CSV and detailed masks remain auditable. No claim of all-expression coverage.'),code("stage('export')\nprint(Path('/kaggle/working/results/baseline_metrics.json').read_text())\n"),
               markdown('## 10. Export artifacts and final audit\nAtomic per-image records and periodic archives survive normal stage errors. Completed Kaggle Output is required for cross-session durability; scratch is not persistent.'),code("print(Path('/kaggle/working/results/run_summary.json').read_text())\n")]
        output=notebook(cells)
        # Compile each code cell to detect packaging syntax errors without GPU execution.
        for i,cell in enumerate(cells):
            if cell['cell_type']=='code': compile(''.join(cell['source']),f'cell-{i}','exec')
        (folder/'phase1.ipynb').write_text(json.dumps(output,ensure_ascii=False),encoding='utf-8')
        sources=['hoofngocjhuy/flmm-phase1-cache']+(['hoofngocjhuy/flmm-phase1-gates'] if resumed else [])
        meta=dict(id='hoofngocjhuy/flmm-phase1-resume' if resumed else 'hoofngocjhuy/flmm-phase1-gates',title='FLMM Phase1 Resume' if resumed else 'FLMM Phase1 Gates',code_file='phase1.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=True,enable_internet=False,dataset_sources=[],competition_sources=[],kernel_sources=sources)
        (folder/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
        print(folder)
if __name__=='__main__': main()
