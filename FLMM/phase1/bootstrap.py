"""Notebook control plane. Restore a verified CPU cache once; bounded GPU stages."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from common import atomic_json,read_json,sha256

HERE=Path(__file__).resolve().parent
OUT=Path('/kaggle/working/results'); OUT.mkdir(exist_ok=True)

def initialize():
    started=time.time()
    atomic_json(OUT/'session.json',dict(started_unix=started))
    if (HERE/'docs').exists(): shutil.copytree(HERE/'docs',OUT.parent/'docs',dirs_exist_ok=True)
    caches=list(Path('/kaggle/input').rglob('cache/environment.json'))
    if len(caches)!=1: raise RuntimeError(f'Expected exactly one verified environment cache, found {caches}')
    cache=caches[0].parent
    envinfo=read_json(cache/'environment.json')
    if sha256(cache/'environment.tar.gz')!=envinfo['sha256']: raise ValueError('Environment cache checksum mismatch')
    Path('/kaggle/temp').mkdir(parents=True,exist_ok=True)
    if not Path('/kaggle/temp/flmm-env/bin/python').exists():
        subprocess.run(['tar','-xzf',str(cache/'environment.tar.gz'),'-C','/kaggle/temp'],check=True,timeout=900)
    os.environ['FLMM_CACHE']=str(cache)
    os.environ['PYTHONPATH']=str(HERE)+':'+str(cache/'F-LMM')
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    os.environ['LD_LIBRARY_PATH']='/kaggle/temp/flmm-env/lib/python3.10/site-packages/torch/lib:'+os.environ.get('LD_LIBRARY_PATH','')
    os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'; os.environ['HF_HUB_DISABLE_XET']='1'
    # An explicit completed prior kernel output is the cross-session checkpoint.
    c=read_json(HERE/'config.json')
    prior=[]
    if c.get('RESUME_INPUT'):
        prior=[p for p in Path('/kaggle/input').rglob('results/run_contract.json') if c['RESUME_INPUT'] in p.parts]
        if len(prior)!=1: raise RuntimeError('Requested resume output not found uniquely')
    for p in prior:
        source=p.parent
        for item in source.iterdir():
            if item.name in ['session.json','resource_preflight.json','go_no_go.json','run_summary.json','evaluation_budget.json']: continue
            dest=OUT/item.name
            if item.is_dir(): shutil.copytree(item,dest,dirs_exist_ok=True)
            elif item.suffix in ['.json','.jsonl','.csv']: shutil.copy2(item,dest)
        docs=source.parent/'docs'
        if docs.is_dir(): shutil.copytree(docs,OUT.parent/'docs',dirs_exist_ok=True)
    session=dict(started_unix=started,restore_s=time.time()-started,cache_source=str(cache),resume_source=[str(p.parent) for p in prior])
    atomic_json(OUT/'session.json',session)
    import psutil
    p=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=30)
    atomic_json(OUT/'resource_preflight.json',dict(status='PASS' if p.returncode==0 else 'UNVERIFIED',
                observed_at_unix=time.time(),gpu_query=p.stdout.strip(),gpu_query_error=p.stderr.strip(),
                ram_total_bytes=psutil.virtual_memory().total,ram_available_bytes=psutil.virtual_memory().available,
                working_disk=dict(zip(['total','used','free'],shutil.disk_usage('/kaggle/working'))),
                scratch_disk=dict(zip(['total','used','free'],shutil.disk_usage('/kaggle/temp'))),
                output_storage='working exported only when Kaggle commits completed job output; temp ephemeral; input read-only prior committed outputs',
                gpu_quota_snapshot=read_json(HERE/'quota_snapshot.json'),operational_budgets=c,
                requested_kernel_timeout_s=7200,external_platform_max_session_limit='UNVERIFIED from dynamic public docs; not assumed for scheduling',
                cache_manifest=read_json(cache.parent/'results/cache-manifest.json'),network_during_evaluation=False,
                restored_cache_sha256=envinfo['sha256'],setup_restored_without_install=True))
    shutil.copy2(cache.parent/'results/dataset_manifest.json',OUT/'dataset_manifest.json')
    return cache

def stage(name):
    cfg=read_json(HERE/'config.json'); session=read_json(OUT/'session.json')
    remaining=(cfg['MAX_TOTAL_WALLTIME_MIN']-cfg['SHUTDOWN_RESERVE_MIN'])*60-(time.time()-session['started_unix'])
    if remaining<60: raise TimeoutError('Insufficient time before export reserve')
    print('STAGE',name,'budget remaining',round(remaining),'seconds',flush=True)
    with (OUT/(name+'-stage.log')).open('w') as f:
        result=subprocess.run(['/kaggle/temp/flmm-env/bin/python',str(HERE/'engine.py'),name],
                              stdout=f,stderr=subprocess.STDOUT,env=os.environ.copy(),timeout=remaining)
    print((OUT/(name+'-stage.log')).read_text(errors='replace')[-2500:],flush=True)
    if result.returncode:
        atomic_json(OUT/(name+'-controller-failure.json'),dict(status='FAIL',returncode=result.returncode))
        raise RuntimeError(f'{name} stage failed; earlier valid artifacts preserved')

