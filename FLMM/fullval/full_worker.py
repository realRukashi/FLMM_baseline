"""One independent GPU replica using the exact G0-approved inference engine."""
import json
import os
from pathlib import Path
import sys
import time
import tarfile
from common import atomic_json,read_json

rank=int(sys.argv[1]); count=int(sys.argv[2])
root=Path('/kaggle/working/workers')/str(rank)
root.mkdir(parents=True,exist_ok=True)
import engine as E
E.OUT=root; E.DOC=root/'docs'; E.DOC.mkdir(exist_ok=True)
E.RECORDS=root/'records'; E.RECORDS.mkdir(exist_ok=True)
def private_snapshot():
    target=root/'resume-checkpoint.tar.gz'; temp=root/'resume-checkpoint.tmp'
    with tarfile.open(temp,'w:gz',compresslevel=1) as tar:
        for p in sorted(E.RECORDS.glob('*')): tar.add(p,arcname='records/'+p.name)
        if (root/'run_contract.json').exists(): tar.add(root/'run_contract.json',arcname='run_contract.json')
    os.replace(temp,target)
E.save_snapshot=private_snapshot  # Artifact destination only; inference/metric code unchanged.
model,cfg,dataset,fp,load_s=E.setup_model()
warm_s=E.warmup(model,dataset)
regression=[s for s in E.DATA['subsets']['regression'] if E.BYID[s]['dataset_index']%count==rank]
reg_result,reg_records,_=E.evaluate(model,dataset,fp,regression,'full-regression')
if reg_result['status']!='PASS': raise RuntimeError('Regression failed before full expansion')
import numpy as np
legacy=Path(os.environ['FLMM_G0_RESULTS'])
for record in reg_records:
    old=read_json(legacy/'records'/(record['sample_id']+'.json'))
    with np.load(root/'records'/record['mask_file']) as current,np.load(legacy/'records'/old['mask_file']) as original:
        if not np.array_equal(current['prediction'],original['prediction']) or not np.array_equal(current['ground_truth'],original['ground_truth']):
            raise ValueError('Two-GPU full-validation replica differs from G0 reference')
atomic_json(root/'regression-check.json',dict(status='PASS',images=len(reg_records),bitwise_matches_G0=True))
probe=[s for s in E.DATA['subsets']['probe'] if E.BYID[s]['dataset_index']%count==rank]
result,records,stats=E.evaluate(model,dataset,fp,probe,'full-probe')
atomic_json(root/'probe-ready.json',dict(status=result['status'],fingerprint=fp,cold_model_load_s=load_s,warmup_s=warm_s,
            records=[r['sample_id'] for r in records],new_images=len(stats)))
if result['status']!='PASS': raise RuntimeError('Feasibility probe failed; no expansion')
deadline=time.monotonic()+180
while not Path('/kaggle/working/results/chunk_gate.json').exists():
    if time.monotonic()>deadline: raise TimeoutError('Controller gate timeout')
    time.sleep(1)
gate=read_json('/kaggle/working/results/chunk_gate.json')
if gate['status']!='GO': raise RuntimeError('Chunk resource gate not GO')
ids=[s for s in E.DATA['subsets']['full_validation'][:E.C['FULL_END_INDEX']] if E.BYID[s]['dataset_index']%count==rank]
result,records,stats=E.evaluate(model,dataset,fp,ids,'full-validation')
atomic_json(root/'worker-summary.json',dict(**result,physical_gpu_index=rank,cold_model_load_s=load_s,warmup_s=warm_s))
if result['status']!='PASS': raise RuntimeError('Chunk incomplete; valid per-image records preserved')
print('WORKER_PASS',rank,len(records),flush=True)
