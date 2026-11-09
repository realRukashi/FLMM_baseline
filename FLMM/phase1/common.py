"""Pure-standard-library integrity, budget, and resumability primitives."""
import hashlib
import json
import os
from pathlib import Path
import time

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)

def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def atomic_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    with temp.open('w',encoding='utf-8') as f:
        f.write(json.dumps(value,indent=2,ensure_ascii=False)); f.flush(); os.fsync(f.fileno())
    os.replace(temp,path)

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def event(root,sample_id,status,**extra):
    assert status in {'PENDING','RUNNING','DONE','FAILED','SKIPPED'}
    path=Path(root)/'run_manifest.jsonl'; path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8') as f:
        f.write(canonical(dict(time_unix=time.time(),sample_id=sample_id,status=status,**extra))+'\n')
        f.flush(); os.fsync(f.fileno())

def valid_record(path,fingerprint,sample_hash,check_masks=True):
    try:
        r=read_json(path)
        if r['status']!='DONE' or r['fingerprint']!=fingerprint or r['sample_hash']!=sample_hash:
            return None
        if r.get('record_hash')!=digest({k:v for k,v in r.items() if k!='record_hash'}): return None
        if not (len(r['intersection'])==len(r['union'])==len(r['expressions'])): return None
        if any(i<0 or u<0 or i>u for i,u in zip(r['intersection'],r['union'])): return None
        if check_masks and r.get('mask_file'):
            mask=Path(path).parent/r['mask_file']
            if not mask.is_file() or sha256(mask)!=r['mask_sha256']: return None
        return r
    except (OSError,ValueError,KeyError,TypeError): return None

def aggregate(records):
    seen=set(); intersections=[]; unions=[]
    for r in records:
        if r['sample_id'] in seen: raise ValueError('Duplicate sample counted')
        seen.add(r['sample_id']); intersections.extend(r['intersection']); unions.extend(r['union'])
    total_i=sum(intersections); total_u=sum(unions)
    if not unions: return dict(image_count=0,mask_count=0,cIoU=None,mIoU=None)
    # Match RefSegMetric: empty GT/pred has IoU 1; cIoU uses global sums.
    per=[i/u if u else 1.0 for i,u in zip(intersections,unions)]
    return dict(image_count=len(seen),mask_count=len(unions),intersection=total_i,union=total_u,
                cIoU=100*total_i/total_u if total_u else 100.0,mIoU=100*sum(per)/len(per))

def budget_decision(count,seconds_per_image,remaining_wall_s,eval_remaining_s,free_bytes,
                    output_bytes_per_image,quota_remaining_s,config):
    needed=count*seconds_per_image*config['SAFETY_FACTOR']
    reserve=config['SHUTDOWN_RESERVE_MIN']*60
    floor=config['MIN_FREE_DISK_GB']*1024**3
    disk_cap=max(0,int((free_bytes-floor)/max(1,output_bytes_per_image*config['SAFETY_FACTOR'])))
    time_cap=max(0,int(min(max(0,remaining_wall_s-reserve),eval_remaining_s)/max(0.001,seconds_per_image*config['SAFETY_FACTOR'])))
    quota_cap=time_cap if quota_remaining_s is None else max(0,int(max(0,quota_remaining_s-reserve)/max(.001,seconds_per_image*config['SAFETY_FACTOR'])))
    cap=min(count,disk_cap,time_cap,quota_cap)
    status='GO' if cap==count else ('REDUCE_SCOPE' if cap>0 else 'BLOCKED')
    if quota_remaining_s is None and cap>0: status='UNKNOWN'; cap=min(cap,100)
    return dict(status=status,selected_max_images=cap,target_images=count,estimated_inference_with_margin_s=needed,
                time_capacity_images=time_cap,disk_capacity_images=disk_cap,quota_capacity_images=quota_cap,
                shutdown_reserve_s=reserve,reason='Measured latency/output size with safety factor; quota UNKNOWN caps scope at 100' if quota_remaining_s is None else 'Measured latency/output size and observed quota snapshot')
