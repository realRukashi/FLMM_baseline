"""Local independent verification of downloaded G0 artifacts; never submits a job."""
import json
from pathlib import Path
import numpy as np
from common import aggregate,atomic_json,digest,read_json,valid_record

ROOT=Path(__file__).resolve().parent
R=ROOT/'runs/gate0-v2/results'
OLD=ROOT.parent/'kaggle/results/smoke-v6/smoke-results'
assert read_json(R/'G0.json')['status']=='PASS'
integrity=read_json(R/'checkpoint_integrity.json')
assert integrity['missing_counts']=={'deepseek_vl':529,'sam_image_encoder':345}
assert len(integrity['missing_key_proofs'])==874
assert all(p['pretrained_disk_verified'] and p['unchanged_after_flmm_load'] for p in integrity['missing_key_proofs'])
assert len(integrity['learned_key_proofs'])==193 and not integrity['unexpected_keys']
equivalence=read_json(R/'evaluator_equivalence.json')
assert len(equivalence['comparisons'])==20
assert all(x['official_vs_subset_mask_exact'] and x['legacy_regression_exact'] for x in equivalence['comparisons'])
manifest=read_json(R/'dataset_manifest.json'); samples={s['sample_id']:s for s in manifest['samples']}
fp=read_json(R/'run_contract.json')['fingerprint']; records=[]
for sid in manifest['subsets']['regression']:
    s=samples[sid]; record=valid_record(R/'records'/f'{sid}.json',fp,s['sample_hash'])
    assert record is not None
    assert digest(record['expressions'])==digest(s['expressions'])
    with np.load(R/'records'/record['mask_file']) as new,np.load(OLD/f"sample-{s['dataset_index']}.npz") as old:
        assert np.array_equal(new['prediction'],old['prediction'])
        assert np.array_equal(new['ground_truth'],old['ground_truth'])
        p=new['prediction'].astype(bool); g=new['ground_truth'].astype(bool)
        assert (p&g).sum(axis=(1,2)).tolist()==record['intersection']
        assert (p|g).sum(axis=(1,2)).tolist()==record['union']
    assert all(e['ref_id'] is not None and e['sent_id'] is not None for e in record['expressions'])
    records.append(record)
metrics=aggregate(records)
assert metrics['image_count']==20 and metrics['mask_count']==47
for key in ['cIoU','mIoU']:
    assert abs(metrics[key]-equivalence['official_metrics'][key])<1e-3
assert len(list((R/'records').glob('*.json')))==20
summary=read_json(R/'run_summary.json')
for stage in ['feasibility_probe','go_no_go','stability','G1','G2']: assert summary['stages'][stage]=='UNVERIFIED'
assert not set(manifest['subsets']['development'])&set(manifest['subsets']['holdout'])
result=dict(status='PASS',missing_pretrained_tensor_proofs=874,trained_checkpoint_tensors=193,
            old_prediction_and_gt_masks_bitwise_equal=True,expression_ids_mapped=True,metrics=metrics,
            later_stages_not_run=True,historical_smoke_backbone_revision='UNVERIFIED')
atomic_json(ROOT/'results/gate0_local_verification.json',result)
print(json.dumps(result,indent=2))
