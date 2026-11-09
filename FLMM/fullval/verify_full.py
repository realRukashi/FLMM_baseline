"""Verify downloaded complete masks, per-target CSV and latency independently."""
import csv
import json
from pathlib import Path
import re
import tarfile
import numpy as np
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'phase1'))
from common import atomic_json,digest,read_json,sha256,valid_record

def main():
    out=ROOT/'results'; archive=ROOT/'full-records.tar.gz'
    archive_info=read_json(out/'records_archive.json')
    assert sha256(archive)==archive_info['sha256']
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers()
        assert len(members)==3000 and len({m.name for m in members})==3000
        for m in members:
            assert m.isfile() and re.fullmatch(r'records/refcoco-val-\d+\.(json|npz)',m.name),m.name
        tar.extractall(out,members=members)
    manifest=read_json(out/'dataset_manifest.json'); contract=read_json(out/'run_contract.json')
    assert len(manifest['samples'])==1500
    target_rows=list(csv.DictReader((out/'per_target_iou.csv').open(encoding='utf-8')))
    assert len(target_rows)==3811 and len({r['prediction_id'] for r in target_rows})==3811
    csv_by_id={r['prediction_id']:r for r in target_rows}
    intersections=[]; unions=[]; times=[]; chunks={}; nonempty=0
    for sample in manifest['samples']:
        r=valid_record(out/'records'/(sample['sample_id']+'.json'),contract['fingerprint'],sample['sample_hash'])
        assert r is not None
        assert digest(r['expressions'])==digest(sample['expressions'])
        with np.load(out/'records'/r['mask_file']) as data:
            p=data['prediction'].astype(bool); g=data['ground_truth'].astype(bool)
            assert p.shape==g.shape and p.shape[0]==len(sample['expressions'])
            i=np.count_nonzero(p&g,axis=(1,2)); u=np.count_nonzero(p|g,axis=(1,2))
            assert i.tolist()==r['intersection'] and u.tolist()==r['union']
            nonempty+=int(np.count_nonzero(p.reshape(len(p),-1).any(axis=1)))
        for e,a,b in zip(sample['expressions'],i,u):
            row=csv_by_id[e['prediction_id']]
            for field in ['ref_id','sent_id']:
                assert (row[field]=='' if e[field] is None else int(row[field])==e[field])
            assert row['expression']==e['expression'] and int(row['mask_id'])==e['mask_id']
            assert int(row['intersection'])==int(a) and int(row['union'])==int(b)
            assert abs(float(row['iou'])-(a/b if b else 1.))<1e-12
        intersections.extend(i.tolist()); unions.extend(u.tolist()); times.append(r['timings']['inference_s'])
        key=str(r['chunk_created']); chunks[key]=chunks.get(key,0)+1
    i=np.asarray(intersections,dtype=np.float64); u=np.asarray(unions,dtype=np.float64)
    per=np.divide(i,u,out=np.ones_like(i),where=u!=0)
    computed=dict(cIoU=float(100*i.sum()/u.sum()),mIoU=float(100*per.mean()))
    official=read_json(out/'baseline_metrics.json')['official_refseg_metric']
    assert all(abs(computed[k]-official[k])<1e-3 for k in computed)
    latency=read_json(out/'latency.json')
    assert len(latency['records'])==1500 and len({r['sample_id'] for r in latency['records']})==1500
    assert abs(np.median(times)-latency['median_inference_s'])<1e-9
    assert abs(np.percentile(times,95)-latency['p95_inference_s'])<1e-9
    assert len(list((out/'records').glob('*.json')))==1500
    assert read_json(out/'full_summary.json')['full_validation_complete']
    failures=read_json(out/'failure_cases.json')
    assert failures['nonempty_predictions']==nonempty
    expected_cases=sorted(target_rows,key=lambda r:(float(r['iou']),int(r['image_id']),int(r['mask_id'])))[:20]
    assert [r['prediction_id'] for r in expected_cases]==[r['prediction_id'] for r in failures['cases']]
    result=dict(status='PASS',images=1500,masks=len(i),unique_target_ids=len(csv_by_id),
                global_intersection=int(i.sum()),global_union=int(u.sum()),independent_metrics=computed,
                official_metrics=official,nonempty_predictions=nonempty,images_first_computed_per_chunk=chunks,
                archive_sha256=archive_info['sha256'],latency_verified=True,all_record_and_mask_hashes_verified=True)
    atomic_json(out/'local_verification.json',result)
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
