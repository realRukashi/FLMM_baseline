"""Package full-validation jobs, preserving the G0-approved computation source."""
import base64
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent; PHASE=ROOT.parent/'phase1'

def md(s): return dict(cell_type='markdown',metadata={},source=s.splitlines(True))
def code(s): return dict(cell_type='code',metadata={},source=s.splitlines(True),execution_count=None,outputs=[])
def nb(cells):
    for i,c in enumerate(cells): c['id']=hashlib.sha256((str(i)+''.join(c['source'])).encode()).hexdigest()[:12]
    return dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python')),nbformat=4,nbformat_minor=5)

def write_job(folder,title,slug,cells,gpu,sources):
    folder.mkdir(parents=True,exist_ok=True)
    for c in cells:
        if c['cell_type']=='code': compile(''.join(c['source']),'cell','exec')
    (folder/'notebook.ipynb').write_text(json.dumps(nb(cells),ensure_ascii=False),encoding='utf-8')
    metadata=dict(id='hoofngocjhuy/'+slug,title=title,code_file='notebook.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=gpu,enable_internet=not gpu,dataset_sources=[],competition_sources=[],kernel_sources=sources)
    (folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))

def main():
    cfg=json.loads((PHASE/'config.json').read_text())
    cfg.update(STOP_AFTER_GATE=None,RUN_FULL_VALIDATION=True,MAX_IMAGES_THIS_JOB=500,RESUME_INPUT=None)
    files={p.name:base64.b64encode(p.read_bytes()).decode() for p in [PHASE/'common.py',PHASE/'engine.py',PHASE/'bootstrap.py']}
    for name in ['extend_cache.py','full_worker.py','full_control.py','full_finalize.py']:
        p=ROOT/name
        if p.exists(): files[name]=base64.b64encode(p.read_bytes()).decode()
    files['config.json']=base64.b64encode(json.dumps(cfg,indent=2).encode()).decode()
    files['quota_snapshot.json']=base64.b64encode((PHASE/'results/quota_snapshot.json').read_bytes()).decode()
    contract=json.loads((PHASE/'results/run_contract.json').read_text())
    assert hashlib.sha256((PHASE/'engine.py').read_bytes()).hexdigest()==contract['contract']['engine_sha256'],'Engine changed after G0'
    assert hashlib.sha256((PHASE/'common.py').read_bytes()).hexdigest()==contract['contract']['common_sha256'],'Common changed after G0'
    extraction=f'''import base64,json,os,sys,subprocess
from pathlib import Path
phase=Path('/kaggle/working/phase1'); phase.mkdir(exist_ok=True)
for name,value in {files!r}.items():
    (phase/name).write_bytes(base64.b64decode(value))
sys.path.insert(0,str(phase))
'''
    write_job(ROOT/'cpu-data','FLMM Full Data','flmm-full-data',[
        md('# Full RefCOCO UNC val — CPU data cache\nReuse verified environment/model cache; download and validate only missing images. No GPU.'),
        code(extraction),code("subprocess.run([sys.executable,str(phase/'extend_cache.py')],check=True,timeout=2400)\n")],False,['hoofngocjhuy/flmm-phase1-cache'])
    for chunk,end in enumerate([500,1000,1500],1):
        prior=None if chunk==1 else f'flmm-full-{chunk-1}'
        configure=extraction+f'''cfg=json.loads((phase/'config.json').read_text()); cfg['FULL_END_INDEX']={end}; cfg['FULL_CHUNK']={chunk}; cfg['FULL_PRIOR']={prior!r}
(phase/'config.json').write_text(json.dumps(cfg,indent=2))
'''
        sources=['hoofngocjhuy/flmm-phase1-cache','hoofngocjhuy/flmm-full-data','hoofngocjhuy/flmm-phase1-gates']+(['hoofngocjhuy/'+prior] if prior else [])
        cells=[md(f'# F-LMM Full RefCOCO UNC validation — chunk {chunk}\nTarget prefix {end}/1500 images; 3811 selected masks over complete split. Pinned G0-approved engine, model and math attention. Independent GPU replicas, no model parallelism or method changes.'),
               md('## Configuration and provenance'),code(configure),
               md('## Restore verified cache and completed prior output\nNo network during GPU inference. Each chunk commits atomic records; later chunks skip only checksum/fingerprint-validated DONE samples.'),
               code("subprocess.run([sys.executable,str(phase/'full_control.py')],check=True,timeout=6900)\n"),
               md('## Metrics, per-target IoU, latency and failure cases\nOnly complete 1500-image output is labeled full validation. Partial outputs remain resumable and explicitly partial.'),
               code("print(Path('/kaggle/working/results/full_summary.json').read_text())\n")]
        write_job(ROOT/f'gpu-{chunk}',f'FLMM Full {chunk}',f'flmm-full-{chunk}',cells,True,sources)
    print('Built CPU full-data and 3 bounded full-validation chunks')

if __name__=='__main__': main()
