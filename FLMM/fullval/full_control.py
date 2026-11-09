"""Bounded multi-GPU chunks with committed-output resume and scalar aggregation."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from common import atomic_json,digest,read_json,sha256,valid_record
from bootstrap import initialize

HERE=Path(__file__).resolve().parent; C=read_json(HERE/'config.json')
OUT=Path('/kaggle/working/results'); OUT.mkdir(exist_ok=True)
WORKERS=Path('/kaggle/working/workers'); WORKERS.mkdir(exist_ok=True)

def merge_records(manifest,process_count):
    dest=OUT/'records'; dest.mkdir(exist_ok=True); fingerprints=set()
    for rank in range(process_count):
        root=WORKERS/str(rank)
        if not (root/'run_contract.json').exists(): continue
        contract=read_json(root/'run_contract.json'); fingerprints.add(contract['fingerprint'])
        if len(fingerprints)>1: raise ValueError('Worker fingerprints differ')
        atomic_json(OUT/'run_contract.json',contract)
        samples={s['sample_id']:s for s in manifest['samples']}
        for path in (root/'records').glob('*.json'):
            sample=samples[path.stem]
            r=valid_record(path,contract['fingerprint'],sample['sample_hash'])
            if not r: continue
            target=dest/path.name
            if target.exists():
                old=valid_record(target,contract['fingerprint'],sample['sample_hash'])
                if old and (old['intersection']!=r['intersection'] or old['mask_sha256']!=r['mask_sha256']):
                    raise ValueError('Conflicting completed sample '+path.stem)
            if 'physical_gpu_index' not in r:
                r.update(physical_gpu_index=rank,chunk_created=C['FULL_CHUNK'])
                r['record_hash']=digest({k:v for k,v in r.items() if k!='record_hash'})
            shutil.copy2(root/'records'/r['mask_file'],dest/r['mask_file'])
            atomic_json(target,r)
    return next(iter(fingerprints)) if fingerprints else None

def main():
    base=initialize()
    extras=list(Path('/kaggle/input').rglob('full-data/image-cache.json'))
    if len(extras)!=1: raise ValueError('Expected one complete full-data cache')
    extra=extras[0].parent
    full_images=read_json(extra/'image-cache.json')
    if full_images['status']!='PASS' or len(full_images['images'])!=1500: raise ValueError('Full data cache incomplete')
    g0s=[p for p in Path('/kaggle/input').rglob('results/G0.json') if 'flmm-phase1-gates' in p.parts]
    if len(g0s)!=1 or read_json(g0s[0])['status']!='PASS': raise ValueError('G0 evidence missing')
    old_contract=read_json(g0s[0].parent/'run_contract.json')['contract']
    for name,key in [('engine.py','engine_sha256'),('common.py','common_sha256')]:
        if sha256(HERE/name)!=old_contract[key]: raise ValueError('Source changed after G0: '+name)
    audit=Path('/kaggle/working/audit'); audit.mkdir(exist_ok=True)
    for name in ['G0.json','checkpoint_integrity.json','evaluator_equivalence.json','run_contract.json','precision.json','installed_versions.json']:
        shutil.copy2(g0s[0].parent/name,audit/name)
    runtime=Path('/kaggle/temp/full-runtime'); cache=runtime/'cache'; prep=runtime/'results'
    cache.mkdir(parents=True,exist_ok=True); prep.mkdir(exist_ok=True)
    for name in ['F-LMM','models','environment.json','environment.tar.gz']:
        (cache/name).symlink_to(base/name,target_is_directory=(base/name).is_dir())
    data=cache/'data/coco'; data.mkdir(parents=True,exist_ok=True)
    (data/'refcoco').symlink_to(base/'data/coco/refcoco',target_is_directory=True)
    images=data/'train2014'; images.mkdir(exist_ok=True)
    for item in full_images['images']:
        original=base/'data/coco/train2014'/item['file_name']
        image=original if original.exists() else extra/'images'/item['file_name']
        if not image.is_file(): raise FileNotFoundError(image)
        (images/item['file_name']).symlink_to(image)
    for name in ['dataset_manifest.json','cache-manifest.json']:
        shutil.copy2(base.parent/'results'/name,prep/name)
    shutil.copy2(extra/'image-cache.json',prep/'image-cache.json')
    manifest=read_json(prep/'dataset_manifest.json')
    if manifest['counts']['images']!=1500 or manifest['counts']['selected_target_masks']!=3811: raise ValueError('Wrong evaluation unit/count')
    os.environ['FLMM_CACHE']=str(cache)
    os.environ['FLMM_G0_RESULTS']=str(g0s[0].parent)
    # Sequential verified reads populate the FUSE/page cache before safetensors mmap
    # in two replicas. This changes I/O scheduling only, not model computation.
    prefetch_start=time.monotonic(); verified=[]
    for model in read_json(base.parent/'results/cache-manifest.json')['models']:
        actual=sha256(cache/model['path'])
        if actual!=model['sha256']: raise ValueError('Checkpoint changed since G0/cache audit')
        verified.append(dict(path=model['path'],sha256=actual))
    atomic_json(OUT/'checkpoint_prefetch.json',dict(status='PASS',seconds=time.monotonic()-prefetch_start,models=verified))
    shutil.copy2(prep/'dataset_manifest.json',OUT/'dataset_manifest.json')
    shutil.copy2(extra/'preparation.json',OUT/'full-data-preparation.json')
    gpu_names=subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True).strip().splitlines()
    count=min(2,len(gpu_names))
    if not count: raise RuntimeError('No GPU')
    prior=None
    if C.get('FULL_PRIOR'):
        paths=[p for p in Path('/kaggle/input').rglob('results/full_summary.json') if C['FULL_PRIOR'] in p.parts]
        if len(paths)!=1: raise ValueError('Completed prior output unavailable')
        prior=paths[0].parent
        if read_json(prior/'full_summary.json')['status'] not in ['PARTIAL','PASS']: raise ValueError('Prior output invalid')
        if (prior/'run_manifest.jsonl').exists(): shutil.copy2(prior/'run_manifest.jsonl',OUT/'run_manifest.jsonl')
    processes=[]; logs=[]
    start=time.monotonic()
    for rank in range(count):
        root=WORKERS/str(rank); root.mkdir(exist_ok=True); (root/'records').mkdir(exist_ok=True)
        shutil.copy2(OUT/'session.json',root/'session.json')
        if prior:
            shutil.copy2(prior/'run_contract.json',root/'run_contract.json')
            for sample in manifest['samples']:
                if sample['dataset_index']%count!=rank: continue
                path=prior/'records'/(sample['sample_id']+'.json')
                if path.exists():
                    r=read_json(path)
                    shutil.copy2(path,root/'records'/path.name)
                    shutil.copy2(prior/'records'/r['mask_file'],root/'records'/r['mask_file'])
        env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']=str(rank)
        env['OMP_NUM_THREADS']='2'; env['MKL_NUM_THREADS']='2'
        log=(root/'worker.log').open('w'); logs.append(log)
        processes.append(subprocess.Popen(['/kaggle/temp/flmm-env/bin/python',str(HERE/'full_worker.py'),str(rank),str(count)],env=env,stdout=log,stderr=subprocess.STDOUT))
    gate_written=False; last_progress=-50
    try:
        while any(p.poll() is None for p in processes):
            if time.monotonic()-start>5700: raise TimeoutError('Chunk controller budget')
            if any(p.poll() not in [None,0] for p in processes):
                raise RuntimeError('A worker failed; stopping peers and retaining valid outputs')
            ready=all((WORKERS/str(i)/'probe-ready.json').exists() for i in range(count))
            progress=sum(len(list((WORKERS/str(i)/'records').glob('*.json'))) for i in range(count))
            if progress>=last_progress+50:
                print('PROGRESS',progress,'validated-record candidates; target prefix',C['FULL_END_INDEX'],flush=True)
                last_progress=progress
            if ready and not gate_written:
                fp=merge_records(manifest,count)
                probe_ids=set(manifest['subsets']['probe'])
                records=[read_json(p) for p in (OUT/'records').glob('*.json') if p.stem in probe_ids]
                if len(records)!=50: raise ValueError('50-image probe incomplete')
                # Regression semantics are unchanged; verify old 20 masks when produced in full range.
                t=sorted(r['timings']['inference_s']+r['timings']['preprocess_s']+r['timings']['postprocess_s'] for r in records)
                p95=t[min(len(t)-1,int(.95*(len(t)-1)))]
                used=time.time()-read_json(OUT/'session.json')['started_unix']
                requested=manifest['subsets']['full_validation'][:C['FULL_END_INDEX']]
                remaining=sum(not (OUT/'records'/(sid+'.json')).exists() for sid in requested)
                estimate=(remaining/count)*p95*1.5
                free=shutil.disk_usage('/kaggle/working').free
                from datetime import datetime,timezone
                q=read_json(HERE/'quota_snapshot.json'); age=(datetime.now(timezone.utc)-datetime.fromisoformat(q['observed_at_utc'])).total_seconds()
                quota_remaining=q['gpu']['remaining_seconds']-age*count
                max_bytes=max(r['output_bytes']+8192 for r in records)
                expected_bytes=remaining*max_bytes*1.5
                go=estimate<min(4500,6000-used) and free-expected_bytes>3*1024**3 and quota_remaining>(estimate+1200)*count and age<21600
                gate=dict(status='GO' if go else 'BLOCKED',probe_images=50,probe_p95_pipeline_s=p95,gpu_names=gpu_names,
                          worker_count=count,estimated_chunk_inference_margin_s=estimate,free_disk_bytes=free,
                          target_prefix=C['FULL_END_INDEX'],prior=C.get('FULL_PRIOR'),fingerprint=fp,
                          remaining_prefix_images=remaining,estimated_additional_output_bytes=expected_bytes,quota_remaining_s_conservative=quota_remaining,
                          note='First 50 seeded probe images are part of full val and reused; no tuning or method changes')
                atomic_json(OUT/'chunk_gate.json',gate); gate_written=True
                print('RESOURCE_GATE',json.dumps(gate),flush=True)
            time.sleep(2)
        if any(p.returncode for p in processes): raise RuntimeError('Worker exit failure')
    finally:
        for p in processes:
            if p.poll() is None: p.terminate()
        for p in processes:
            try: p.wait(timeout=15)
            except subprocess.TimeoutExpired: p.kill(); p.wait()
        for f in logs: f.close()
        merge_records(manifest,count)
        result=subprocess.run(['/kaggle/temp/flmm-env/bin/python',str(HERE/'full_finalize.py')],env=os.environ.copy(),timeout=900)
        if result.returncode: raise RuntimeError('Finalization/independent metric verification failed')
    print('CHUNK_COMPLETE',C['FULL_CHUNK'],flush=True)

if __name__=='__main__': main()
