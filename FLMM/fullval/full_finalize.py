"""Independently validate every saved mask; official RefSegMetric and full reports."""
import csv
import json
import os
from pathlib import Path
import shutil
import sys
import time
import tarfile
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from common import aggregate,atomic_json,read_json,sha256,valid_record

HERE=Path(__file__).resolve().parent; C=read_json(HERE/'config.json')
OUT=Path('/kaggle/working/results'); CACHE=Path(os.environ['FLMM_CACHE'])
manifest=read_json(OUT/'dataset_manifest.json'); samples={s['sample_id']:s for s in manifest['samples']}
sys.path.insert(0,str(CACHE/'F-LMM'))
import torch
from mmdet.evaluation import RefSegMetric
from mmdet.structures.mask import BitmapMasks

def main():
    if not (OUT/'run_contract.json').exists():
        atomic_json(OUT/'full_summary.json',dict(status='FAIL',reason='No valid worker contract/result'))
        return
    contract=read_json(OUT/'run_contract.json'); fp=contract['fingerprint']
    evaluator=RefSegMetric(metric=['cIoU','mIoU']); records=[]; targets=[]; nonempty=0
    for sample in manifest['samples']:
        path=OUT/'records'/(sample['sample_id']+'.json')
        r=valid_record(path,fp,sample['sample_hash'])
        if not r: continue
        with np.load(OUT/'records'/r['mask_file']) as masks:
            p=masks['prediction'].astype(bool); g=masks['ground_truth'].astype(bool)
            i=(p&g).sum(axis=(1,2)); u=(p|g).sum(axis=(1,2))
            assert p.shape==g.shape and len(p)==len(sample['expressions'])
            assert i.tolist()==r['intersection'] and u.tolist()==r['union']
            nonempty+=int(p.reshape(len(p),-1).any(axis=1).sum())
            evaluator.process(data_batch={},data_samples=[dict(pred_instances=dict(masks=torch.from_numpy(p)),
                gt_masks=BitmapMasks(g,height=g.shape[1],width=g.shape[2]))])
        for expression,intersection,union in zip(r['expressions'],r['intersection'],r['union']):
            targets.append(dict(**expression,sample_id=r['sample_id'],intersection=intersection,union=union,
                iou=intersection/union if union else 1.0,inference_image_s=r['timings']['inference_s'],
                amortized_inference_target_s=r['timings']['amortized_inference_s_per_selected_expression'],
                physical_gpu_index=r.get('physical_gpu_index')))
        records.append(r)
    measured=aggregate(records)
    official={k:float(v) for k,v in evaluator.compute_metrics(evaluator.results).items()} if records else {}
    for key,value in official.items(): assert abs(value-measured[key])<1e-3,(key,value,measured[key])
    complete=len(records)==1500 and measured['mask_count']==3811
    fields=['image_id','dataset_index','ref_id','sent_id','expression','mask_id','prediction_id','annotation_id','sample_id','intersection','union','iou','inference_image_s','amortized_inference_target_s','physical_gpu_index']
    with (OUT/'per_target_iou.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader(); w.writerows(targets)
    latency=np.array([r['timings']['inference_s'] for r in records])
    profile=dict(status='PASS' if records else 'UNVERIFIED',images=len(records),definition='CUDA-synchronized image forward after separate warm-up; one independent GPU replica per image; no download included',
        median_inference_s=float(np.median(latency)) if len(latency) else None,
        p95_inference_s=float(np.percentile(latency,95)) if len(latency) else None,
        max_inference_s=float(max(latency)) if len(latency) else None,
        sum_image_inference_s=float(sum(latency)),
        peak_allocated_bytes=max((r['memory']['peak_allocated_bytes'] for r in records),default=0),
        peak_reserved_bytes=max((r['memory']['peak_reserved_bytes'] for r in records),default=0),
        peak_worker_ram_rss_bytes=max((r['memory']['ram_rss_bytes'] for r in records),default=0),
        preprocess_median_s=float(np.median([r['timings']['preprocess_s'] for r in records])) if records else None,
        postprocess_median_s=float(np.median([r['timings']['postprocess_s'] for r in records])) if records else None,
        records=[dict(sample_id=r['sample_id'],timings=r['timings'],memory=r['memory'],physical_gpu_index=r.get('physical_gpu_index')) for r in records])
    atomic_json(OUT/'latency.json',profile)
    metrics=dict(status='PASS' if complete else 'PARTIAL',split='RefCOCO UNC val',unit='first sentence per reference, concatenated by image',
                 expected_images=1500,expected_selected_masks=3811,official_refseg_metric=official,independently_recomputed=measured)
    atomic_json(OUT/'baseline_metrics.json',metrics)
    failures=sorted(targets,key=lambda r:(r['iou'],r['image_id'],r['mask_id']))
    cases=OUT/'failure_cases'; cases.mkdir(exist_ok=True)
    font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font=ImageFont.truetype(font_path,15) if Path(font_path).exists() else ImageFont.load_default()
    for n,case in enumerate(failures[:20]):
        sample=samples[case['sample_id']]
        image=Image.open(CACHE/'data/coco/train2014'/sample['file_name']).convert('RGB')
        record=next(r for r in records if r['sample_id']==case['sample_id'])
        with np.load(OUT/'records'/record['mask_file']) as masks:
            pred=masks['prediction'][case['mask_id']].astype(bool); gt=masks['ground_truth'][case['mask_id']].astype(bool)
        sheet=Image.new('RGB',(990,325),'white'); draw=ImageDraw.Draw(sheet)
        draw.text((10,5),f"Image {case['image_id']} | ref {case['ref_id']} | target IoU {case['iou']:.4f}",font=font,fill='black')
        draw.text((10,27),case['expression'][:105],font=font,fill='black')
        for j,(mask,color,title) in enumerate([(None,None,'Input'),(gt,(40,190,90),'Ground truth'),(pred,(245,130,35),'F-LMM prediction')]):
            array=np.array(image).copy()
            if mask is not None: array[mask]=(array[mask]*.5+np.array(color)*.5).astype('uint8')
            panel=Image.fromarray(array); panel.thumbnail((320,240))
            draw.text((10+330*j,52),title,font=font,fill='black'); sheet.paste(panel,(10+330*j,78))
        sheet.save(cases/f'{n:02d}-{case["image_id"]}-{case["mask_id"]}.jpg',quality=90)
    atomic_json(OUT/'failure_cases.json',dict(selection='20 lowest per-target IoUs, deterministic tie-break image_id/mask_id; descriptive examples, not a causal taxonomy',
                targets_below_0_5=sum(t['iou']<.5 for t in targets),zero_iou_targets=sum(t['iou']==0 for t in targets),
                nonempty_predictions=nonempty,cases=failures[:20]))
    # Preserve append-only worker execution journals alongside cumulative records.
    prior_events=OUT/'run_manifest.jsonl'
    if not prior_events.exists(): prior_events.write_text('')
    with prior_events.open('a') as f:
        for journal in Path('/kaggle/working/workers').glob('*/run_manifest.jsonl'):
            f.write(journal.read_text())
    worker_summaries=[]
    for path in Path('/kaggle/working/workers').glob('*/worker-summary.json'):
        worker_summaries.append(read_json(path))
    history=[]
    if C.get('FULL_PRIOR'):
        previous=[p for p in Path('/kaggle/input').rglob('results/full_summary.json') if C['FULL_PRIOR'] in p.parts]
        if len(previous)==1: history=read_json(previous[0]).get('chunk_history',[])
    history.append(dict(chunk=C['FULL_CHUNK'],completed_images=len(records),session_elapsed_s=time.time()-read_json(OUT/'session.json')['started_unix'],
                        new_image_inferences=sum(w['newly_evaluated'] for w in worker_summaries),worker_summaries=worker_summaries))
    atomic_json(OUT/'full_summary.json',dict(status='PASS' if complete else 'PARTIAL',chunk=C['FULL_CHUNK'],
                requested_prefix=C['FULL_END_INDEX'],completed_images=len(records),selected_masks=measured['mask_count'],
                remaining_images=1500-len(records),metrics=metrics,workers=worker_summaries,
                resumed_from=C.get('FULL_PRIOR'),session_elapsed_s=time.time()-read_json(OUT/'session.json')['started_unix'],
                chunk_history=history,
                full_validation_complete=complete,all_10834_expressions_evaluated=False,
                precision='BF16 DeepSeekVL, FP32 grounding/SAM, math SDPA/manual attention fallback as G0',
                source_hashes={name:sha256(HERE/name) for name in ['engine.py','common.py','full_worker.py','full_control.py','full_finalize.py']},
                resume_note='Only validated matching DONE records skipped; completed chunk outputs are cross-session durable. Hard loss before artifact commit may require rerunning current uncommitted chunk.',
                holdout_note='Full validation includes the former internal holdout; it must not be treated as untouched for future method selection. No improvement/tuning in this run.'))
    if complete:
        archive=OUT.parent/'full-records.tar.gz'; temp=OUT.parent/'full-records.tmp'
        with tarfile.open(temp,'w:gz',compresslevel=1) as tar:
            for path in sorted((OUT/'records').glob('*')): tar.add(path,arcname='records/'+path.name)
        os.replace(temp,archive)
        atomic_json(OUT/'records_archive.json',dict(file='full-records.tar.gz',sha256=sha256(archive),bytes=archive.stat().st_size,images=len(records)))
    print(json.dumps(dict(completed_images=len(records),selected_masks=measured['mask_count'],official=official,complete=complete)),flush=True)

if __name__=='__main__': main()
