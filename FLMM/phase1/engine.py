"""Pinned F-LMM audit, gated profiling, and manifest-driven resumable inference."""
import argparse
import collections
import csv
import difflib
import gc
import inspect
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import traceback
from common import aggregate,atomic_json,digest,event,read_json,sha256,valid_record

HERE=Path(__file__).resolve().parent
C=read_json(HERE/'config.json')
OUT=Path('/kaggle/working/results'); DOC=Path('/kaggle/working/docs')
OUT.mkdir(exist_ok=True); DOC.mkdir(exist_ok=True)
CACHE=Path(os.environ['FLMM_CACHE'])
PREP=CACHE.parent/'results'
DATA=read_json(PREP/'dataset_manifest.json')
BYID={s['sample_id']:s for s in DATA['samples']}
SOURCE=CACHE/'F-LMM'
os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
sys.path.insert(0,str(SOURCE))
RECORDS=OUT/'records'; RECORDS.mkdir(exist_ok=True)
CFG_NAME='configs/deepseek_vl/frozen_deepseek_vl_1_3b_chat_unet_sam_l_refcoco_png.py'

def session_elapsed():
    return time.time()-read_json(OUT/'session.json')['started_unix']

def check_budget():
    if session_elapsed()>(C['MAX_TOTAL_WALLTIME_MIN']-C['SHUTDOWN_RESERVE_MIN'])*60:
        raise TimeoutError('Walltime stop: preserving shutdown/export reserve')
    if shutil.disk_usage('/kaggle/working').free<C['MIN_FREE_DISK_GB']*1024**3:
        raise OSError('Disk safety floor reached')
    if (OUT/'STOP').exists(): raise InterruptedError('Operator STOP file')
    if evaluation_consumed()>C['MAX_EVALUATION_MIN']*60:
        raise TimeoutError('Cumulative evaluation budget reached')

def evaluation_consumed():
    path=OUT/'evaluation_budget.json'
    if not path.exists(): return 0.0
    b=read_json(path)
    return b['consumed_s']+(time.time()-b['active_since_unix'] if b.get('active_since_unix') else 0.0)

def evaluation_start():
    atomic_json(OUT/'evaluation_budget.json',dict(consumed_s=evaluation_consumed(),active_since_unix=time.time()))

def evaluation_end():
    atomic_json(OUT/'evaluation_budget.json',dict(consumed_s=evaluation_consumed(),active_since_unix=None))

def tensor_hash(tensor):
    a=tensor.detach().cpu().contiguous().reshape(-1).view(__import__('torch').uint8)
    return __import__('hashlib').sha256(a.numpy().tobytes()).hexdigest()

def percentile(values,p):
    import numpy as np
    return float(np.percentile(values,p)) if values else None

def setup_model(audit=False):
    import torch
    from mmengine.config import Config
    from xtuner.registry import BUILDER
    from deepseek_vl.models import MultiModalityCausalLM
    from xtuner.model.utils import guess_load_checkpoint
    from safetensors import safe_open
    from flmm.datasets.transforms import PILLoadImageFromFile,RefCOCO2PNG
    from mmdet.datasets import RefCocoDataset
    from mmdet.datasets.transforms import LoadAnnotations
    import psutil
    check_budget()
    torch.manual_seed(C['SEED'])
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    if not torch.cuda.is_available(): raise RuntimeError('GPU unavailable: GPU tests UNVERIFIED')
    t=time.perf_counter()
    cfg=Config.fromfile(SOURCE/CFG_NAME)
    backbone=CACHE/'models/backbone'; sam=CACHE/'models/sam_vit_l_0b3195.pth'
    trained=CACHE/'models/flmm/frozen_deepseek_vl_1_3b_chat_unet_sam_l_refcoco_png.pth'
    expected={str(backbone/'model.safetensors'):C['BACKBONE_SHA256'],str(sam):C['SAM_SHA256'],str(trained):C['FLMM_SHA256']}
    if audit:
        for path,h in expected.items():
            if sha256(path)!=h: raise ValueError('Checkpoint digest mismatch: '+path)
    actual_sha=subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()
    if actual_sha!=C['SOURCE_SHA']: raise ValueError('Source revision mismatch')
    if audit:
        versions={name:importlib.metadata.version(name) for name in ['torch','torchvision','transformers','accelerate','mmengine','mmcv','mmdet','mmsegmentation','xtuner','numpy','timm','deepspeed']}
        atomic_json(OUT/'installed_versions.json',dict(python=sys.version,packages=versions,environment_archive_sha256=read_json(CACHE/'environment.json')['sha256']))
        freeze=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True,timeout=60)
        (OUT/'actual_freeze.txt').write_text(freeze)
        for name,expected_version in [('torch','2.1.2+cu118'),('transformers','4.39.1'),('mmdet','3.3.0'),('mmcv','2.1.0'),('xtuner','0.1.23')]:
            if versions[name]!=expected_version: raise ValueError('Installed dependency drift: '+name)
    cfg.model.model.pretrained_model_name_or_path=str(backbone)
    cfg.tokenizer.pretrained_model_name_or_path=str(backbone)
    cfg.model.tokenizer.pretrained_model_name_or_path=str(backbone)
    cfg.image_processor.pretrained_model_name_or_path=str(backbone)
    cfg.model.sam.checkpoint=str(sam)
    pretrained,loading=MultiModalityCausalLM.from_pretrained(str(backbone),torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,output_loading_info=True,local_files_only=True)
    if loading.get('missing_keys') or loading.get('unexpected_keys') or loading.get('mismatched_keys') or loading.get('error_msgs'):
        raise ValueError('Backbone pretrained initialization incomplete: '+str(loading))
    # Wrapper still constructs the official class, but reuses the audited pretrained instance.
    cfg.model.model=dict(type=lambda:pretrained)
    model=BUILDER.build(cfg.model)
    state=guess_load_checkpoint(str(trained))
    disk_sam=torch.load(sam,map_location='cpu') if audit else None
    before={}; disk_verified=[]
    if audit:
        with safe_open(str(backbone/'model.safetensors'),framework='pt',device='cpu') as file:
            for k,v in model.deepseek_vl.state_dict().items():
                reference=file.get_tensor(k).to(v.dtype)
                if not torch.equal(v.cpu(),reference): raise ValueError('Backbone initialization tensor differs: '+k)
                before['deepseek_vl.'+k]=tensor_hash(v); disk_verified.append('deepseek_vl.'+k)
        for k,v in model.sam.model.image_encoder.state_dict().items():
            ref=disk_sam['image_encoder.'+k].to(v.dtype)
            if not torch.equal(v.cpu(),ref): raise ValueError('SAM initialization tensor differs: '+k)
            before['sam.model.image_encoder.'+k]=tensor_hash(v); disk_verified.append('sam.model.image_encoder.'+k)
        del disk_sam; gc.collect()
    missing,unexpected=model.load_state_dict(state,strict=False)
    legitimate=[k for k in missing if k.startswith('deepseek_vl.') or k.startswith('sam.model.image_encoder.')]
    learned_missing=[k for k in missing if k not in legitimate]
    if unexpected or learned_missing: raise ValueError('Learned checkpoint mismatch')
    if sum(k.startswith('deepseek_vl.') for k in missing)!=529 or sum(k.startswith('sam.model.image_encoder.') for k in missing)!=345:
        raise ValueError('Missing-key count differs from original successful checkpoint contract')
    # nn.Module.state_dict on model includes pretrained deepseek keys; SAM wrapper omits encoder,
    # so resolve every tensor through get_parameter/get_buffer rather than wrapper state_dict.
    def resolve(k):
        try: return model.get_parameter(k)
        except AttributeError: return model.get_buffer(k)
    if audit:
        rows=[]
        for k in missing:
            after=tensor_hash(resolve(k))
            ok=k in before and after==before[k]
            rows.append(dict(key=k,pretrained_disk_verified=k in disk_verified,unchanged_after_flmm_load=ok,sha256=after))
            if not ok: raise ValueError('Missing key not proven initialized: '+k)
        learned=[]
        for k,v in state.items():
            actual=resolve(k)
            if not torch.equal(actual.detach().cpu(),v.to(actual.dtype).cpu()): raise ValueError('Trained tensor did not load: '+k)
            learned.append(dict(key=k,shape=list(actual.shape),dtype=str(actual.dtype),sha256=tensor_hash(actual)))
        integrity=dict(status='PASS',source_sha=actual_sha,backbone_revision=C['BACKBONE_REVISION'],
                       checkpoint_sha256=expected,pretrained_loading_info=loading,
                       missing_counts=dict(collections.Counter('deepseek_vl' if k.startswith('deepseek_vl.') else 'sam_image_encoder' for k in missing)),
                       missing_key_proofs=rows,learned_key_proofs=learned,unexpected_keys=list(unexpected),
                       legacy_backbone_revision='UNVERIFIED: old smoke did not record revision; new run pins revision and tests regression masks')
        atomic_json(OUT/'checkpoint_integrity.json',integrity)
    del state; gc.collect()
    model=model.to('cuda:0').eval()
    torch.cuda.synchronize()
    load_time=time.perf_counter()-t
    modules={}
    for name,module in [('backbone',model.deepseek_vl),('vision',model.deepseek_vl.vision_model),('language',model.deepseek_vl.language_model),('mask_head',model.mask_head),('text_proj',model.text_proj),('sam_image_encoder',model.sam.model.image_encoder),('sam_prompt_encoder',model.sam.model.prompt_encoder),('sam_decoder',model.sam.model.mask_decoder)]:
        modules[name]=dict(collections.Counter(str(p.dtype) for p in module.parameters()))
    precision=dict(status='PASS',module_parameter_dtypes=modules,autocast=False,
                   configured_backbone_dtype='torch.bfloat16',attention_backend='SDPA math; flash and memory-efficient disabled',
                   language_attention=model.deepseek_vl.language_model.model.layers[0].self_attn.__class__.__name__,
                   language_attention_note='output_attentions=True triggers transformers 4.39.1 manual attention fallback',
                   vision_casts='F-LMM mask attention interpolation uses float32 then cast to head dtype; upstream unchanged',
                   native_bf16_tensor_cores=False if 'T4' in torch.cuda.get_device_name(0) else None,
                   cold_load_and_audit_s=load_time,gpu=torch.cuda.get_device_name(0),
                   vram_bytes=torch.cuda.get_device_properties(0).total_memory,cuda_capability=list(torch.cuda.get_device_capability(0)))
    atomic_json(OUT/'precision.json',precision)
    params=dict(type=RefCOCO2PNG,image_processor=cfg.image_processor,tokenizer=cfg.tokenizer,prompt_template=cfg.prompt_template,
                concat=True,image2tensor=cfg.get('image2tensor',True),add_image_token=cfg.get('add_image_token',False),
                image_token=cfg.get('image_token','<image>'))
    if cfg.get('prompt') is not None: params['prompt']=cfg.prompt
    pipeline=[dict(type=PILLoadImageFromFile,backend_args=None),dict(type=LoadAnnotations,with_mask=True,with_bbox=False,with_seg=False,with_label=False),params]
    dataset=RefCocoDataset(data_root=str(CACHE/'data/coco'),ann_file='refcoco/instances.json',split_file='refcoco/refs(unc).p',data_prefix=dict(img_path='train2014/'),split='val',text_mode='select_first',pipeline=pipeline)
    contract=dict(source_sha=actual_sha,config_sha256=sha256(SOURCE/CFG_NAME),backbone_sha256=C['BACKBONE_SHA256'],
                  backbone_revision=C['BACKBONE_REVISION'],sam_sha256=C['SAM_SHA256'],flmm_sha256=C['FLMM_SHA256'],
                  annotation_sha256=C['ANNOTATION_SHA256'],manifest_hash=DATA['manifest_hash'],image_cache_sha256=sha256(PREP/'image-cache.json'),text_mode='select_first',concat=True,
                  attention='math',threshold=.5,seed=C['SEED'],engine_sha256=sha256(HERE/'engine.py'),precision=modules)
    contract.update(common_sha256=sha256(HERE/'common.py'),environment_archive_sha256=read_json(CACHE/'environment.json')['sha256'])
    fp=digest(contract)
    previous=OUT/'run_contract.json'
    if previous.exists() and read_json(previous)['fingerprint']!=fp and not audit:
        raise ValueError('Resume contract changed; G0 must be rerun, outputs cannot be reused')
    atomic_json(previous,dict(fingerprint=fp,contract=contract))
    return model,cfg,dataset,fp,load_time

def record_path(sample): return RECORDS/(sample['sample_id']+'.json')

def save_snapshot():
    # Atomic archive is export-ready; durability across sessions is only claimed after
    # Kaggle commits a completed job output. Controllers use bounded jobs as checkpoints.
    target=OUT.parent/'resume-checkpoint.tar.gz'; temp=target.with_suffix('.tmp')
    with tarfile.open(temp,'w:gz',compresslevel=1) as tar:
        for p in sorted(RECORDS.glob('*')): tar.add(p,arcname='records/'+p.name)
        for name in ['run_contract.json','run_manifest.jsonl']:
            if (OUT/name).exists(): tar.add(OUT/name,arcname=name)
    os.replace(temp,target)

def evaluate(model,dataset,fp,ids,stage,limit_new=None):
    import torch
    import torch.nn.functional as F
    import numpy as np
    import psutil
    stats=[]; records=[]; new_count=0; failures=[]; skipped=0; complete=True
    eval_start=time.monotonic()
    evaluation_start()
    for sid in ids:
        sample=BYID[sid]; path=record_path(sample)
        prior=valid_record(path,fp,sample['sample_hash'])
        if prior:
            records.append(prior); skipped+=1; event(OUT,sid,'SKIPPED',reason='validated DONE output reused',fingerprint=fp); continue
        if limit_new is not None and new_count>=limit_new: complete=False; break
        if time.monotonic()-eval_start>C['MAX_EVALUATION_MIN']*60: complete=False; break
        try:
            check_budget()
        except (TimeoutError,OSError,InterruptedError) as e:
            event(OUT,sid,'PENDING',reason=str(e)); complete=False; break
        event(OUT,sid,'RUNNING',stage=stage,fingerprint=fp)
        try:
            info=dataset.get_data_info(sample['dataset_index'])
            if info['img_id']!=sample['image_id'] or digest(info['instances'])!=sample['gt_instances_sha256']:
                raise ValueError('Manifest/loader identity mismatch')
            expected_image=next(x for x in read_json(PREP/'image-cache.json')['images'] if x['sample_id']==sid)
            if sha256(info['img_path'])!=expected_image['sha256']: raise ValueError('Cached image checksum mismatch')
            t=time.perf_counter(); data=dataset[sample['dataset_index']]; preprocess_s=time.perf_counter()-t
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            t=time.perf_counter()
            with torch.no_grad(): logits=model.predict(data)
            torch.cuda.synchronize(); inference_s=time.perf_counter()-t
            allocated_peak=torch.cuda.max_memory_allocated(); reserved_peak=torch.cuda.max_memory_reserved()
            t=time.perf_counter()
            gt=data['gt_masks'].numpy()>0
            pred=F.interpolate(logits[None].float().sigmoid(),size=gt.shape[-2:],mode='bilinear')[0].cpu()>0.5
            pred=pred.numpy(); postprocess_s=time.perf_counter()-t
            if pred.shape!=gt.shape or len(pred)!=len(sample['expressions']): raise ValueError('Mask alignment mismatch')
            intersection=(pred&gt).sum(axis=(1,2)); union=(pred|gt).sum(axis=(1,2))
            mask_name=sid+'.npz'; mask=RECORDS/mask_name; tmp=mask.with_suffix('.tmp')
            t=time.perf_counter()
            with tmp.open('wb') as f:
                np.savez_compressed(f,prediction=pred,ground_truth=gt); f.flush(); os.fsync(f.fileno())
            os.replace(tmp,mask)
            r=dict(status='DONE',sample_id=sid,sample_hash=sample['sample_hash'],fingerprint=fp,
                   image_id=sample['image_id'],dataset_index=sample['dataset_index'],expressions=sample['expressions'],
                   intersection=intersection.tolist(),union=union.tolist(),mask_file=mask_name,mask_sha256=sha256(mask),
                   nonempty_predictions=int(pred.reshape(len(pred),-1).any(axis=1).sum()),
                   timings=dict(preprocess_s=preprocess_s,inference_s=inference_s,postprocess_s=postprocess_s,
                                amortized_inference_s_per_selected_expression=inference_s/len(pred)),
                   memory=dict(peak_allocated_bytes=allocated_peak,peak_reserved_bytes=reserved_peak,
                               allocated_after_bytes=torch.cuda.memory_allocated(),ram_rss_bytes=psutil.Process().memory_info().rss),
                   output_bytes=mask.stat().st_size,stage_created=stage)
            r['timings']['mask_write_s']=time.perf_counter()-t
            r['record_hash']=digest(r); atomic_json(path,r)
            if not valid_record(path,fp,sample['sample_hash']): raise ValueError('Atomic record validation failed')
            records.append(r); stats.append(r); new_count+=1
            event(OUT,sid,'DONE',stage=stage,fingerprint=fp,record_hash=r['record_hash'])
            print('DONE',stage,len(records),'/',len(ids),sid,f'{inference_s:.3f}s',flush=True)
            del logits,pred,gt,data; gc.collect()
            if new_count%C['EXPORT_EVERY_IMAGES']==0: save_snapshot()
        except Exception as e:
            failures.append(dict(sample_id=sid,error=repr(e)))
            event(OUT,sid,'FAILED',stage=stage,error=repr(e),traceback=traceback.format_exc())
            # No automatic dtype/resolution/batch/threshold changes; stop on first failure.
            complete=False; break
    result=dict(status='PASS' if complete and not failures and len(records)==len(ids) else ('FAIL' if failures else 'BLOCKED'),
                stage=stage,requested_images=len(ids),validated_done=len(records),newly_evaluated=new_count,reused_done=skipped,
                failures=failures,wall_s=time.monotonic()-eval_start,metrics=aggregate(records))
    evaluation_end()
    atomic_json(OUT/(stage+'.json'),result); save_snapshot()
    return result,records,stats

def warmup(model,dataset):
    import torch
    activations={}; handles=[]
    def dtypes(value):
        if torch.is_tensor(value): return {str(value.dtype)}
        if isinstance(value,dict): return set().union(*(dtypes(v) for v in value.values())) if value else set()
        if isinstance(value,(tuple,list)): return set().union(*(dtypes(v) for v in value)) if value else set()
        return set()
    def hook(name):
        def capture(module,inputs,output):
            if name not in activations: activations[name]=dict(input_tensor_dtypes=sorted(dtypes(inputs)),output_tensor_dtypes=sorted(dtypes(output)))
        return capture
    for name,module in [('vision',model.deepseek_vl.vision_model),('language',model.deepseek_vl.language_model),('mask_head',model.mask_head),('text_proj',model.text_proj),('sam_image_encoder',model.sam.model.image_encoder),('sam_prompt_encoder',model.sam.model.prompt_encoder),('sam_decoder',model.sam.model.mask_decoder)]:
        handles.append(module.register_forward_hook(hook(name)))
    sample=BYID[DATA['subsets']['regression'][0]]; d=dataset[sample['dataset_index']]
    torch.cuda.synchronize(); t=time.perf_counter()
    try:
        with torch.no_grad(): prediction=model.predict(d)
    finally:
        for h in handles: h.remove()
    torch.cuda.synchronize(); elapsed=time.perf_counter()-t
    del prediction,d; gc.collect()
    precision=read_json(OUT/'precision.json'); precision['observed_first_warmup_activation_dtypes']=activations
    atomic_json(OUT/'precision.json',precision)
    return elapsed

def control_run(model,cfg,kind):
    import torch
    import numpy as np
    source=SOURCE/'scripts/multiprocess_eval_refcoco.py' if kind=='official' else HERE/'reference/subset-evaluator.py'
    original=source.read_text(); changed=original
    substitutions=[('    cfg = Config.fromfile(args.config)','    cfg = SHARED_CFG'),
                   ('    model = BUILDER.build(cfg.model)','    model = SHARED_MODEL'),
                   ("data_root='data/coco/'",f"data_root={str(CACHE/'data/coco')!r}")]
    if kind=='official':
        substitutions += [('    for name, subset in refcoco_subsets.items():','    refcoco_subsets = {"refcoco_val": refcoco_subsets["refcoco_val"]}\n    for name, subset in refcoco_subsets.items():'),
                          ('        data_ids = list(range(len(dataset)))','        data_ids = REGRESSION_INDICES')]
    else:
        substitutions += [('        data_ids = list(range(len(dataset)))','        data_ids = REGRESSION_INDICES')]
    for old,new in substitutions:
        if changed.count(old)!=1: raise ValueError('Control source drift: '+old)
        changed=changed.replace(old,new)
    (DOC/(kind+'-control.diff')).write_text(''.join(difflib.unified_diff(original.splitlines(True),changed.splitlines(True),fromfile=str(source.name),tofile=kind+'-controlled.py')))
    old_argv=sys.argv[:]
    sys.argv=[str(source),str(SOURCE/CFG_NAME),'--concat']
    if kind=='subset': sys.argv+=['--max-samples','20','--output-dir',str(OUT/'subset-control')]
    scope=dict(__name__='__main__',__file__=str(source),SHARED_MODEL=model,SHARED_CFG=cfg,
               REGRESSION_INDICES=[BYID[s]['dataset_index'] for s in DATA['subsets']['regression']])
    check_budget(); evaluation_start()
    try: exec(compile(changed,str(source),'exec'),scope)
    finally:
        sys.argv=old_argv
        evaluation_end()
    # Upstream masks and GT are retained only for these 20 regression images.
    return scope

def audit():
    import numpy as np
    model,cfg,dataset,fp,load_s=setup_model(audit=True)
    warm_s=warmup(model,dataset)
    official=control_run(model,cfg,'official')
    subset=control_run(model,cfg,'subset')
    comparisons=[]
    for pos,sid in enumerate(DATA['subsets']['regression']):
        idx=BYID[sid]['dataset_index']; a=official['results'][pos]; b=subset['results'][pos]
        pa=a['pred_instances']['masks'].numpy(); pb=b['pred_instances']['masks'].numpy()
        ga=a['gt_masks'].masks; gb=b['gt_masks'].masks
        old=np.load(HERE/'reference'/f'sample-{idx}.npz')
        same=np.array_equal(pa,pb) and np.array_equal(ga,gb)
        legacy=np.array_equal(pa,old['prediction']) and np.array_equal(ga,old['ground_truth'])
        comparisons.append(dict(sample_id=sid,official_vs_subset_mask_exact=same,legacy_regression_exact=legacy))
        if not same or not legacy: raise ValueError('G0 mask regression mismatch '+sid)
    del official['results'],subset['results']; gc.collect()
    result,records,_=evaluate(model,dataset,fp,DATA['subsets']['regression'],'regression')
    if result['status']!='PASS': raise RuntimeError('Regression engine failed')
    for sample,r in zip(DATA['subsets']['regression'],records):
        old=np.load(HERE/'reference'/f"sample-{BYID[sample]['dataset_index']}.npz")
        new=np.load(RECORDS/r['mask_file'])
        if not np.array_equal(old['prediction'],new['prediction']) or not np.array_equal(old['ground_truth'],new['ground_truth']):
            raise ValueError('Resumable evaluator differs from regression')
    control_metrics={k:float(v) for k,v in official['metrics'].items()}
    for key,value in control_metrics.items():
        if abs(value-result['metrics'][key])>1e-3: raise ValueError('Official metric mismatch')
    equivalence=dict(status='PASS',images=20,mask_count=result['metrics']['mask_count'],official_metrics=control_metrics,
                     independently_aggregated_metrics=result['metrics'],comparisons=comparisons,
                     scope='Conditional equivalence on identical 20-image selection and math attention; does not establish paper-wide expression coverage',
                     harness_changes='Selection restricted to UNC val and original IDs; pinned local checkpoint paths; shared audited model, unchanged prediction/threshold/interpolation/RefSegMetric',
                     cold_load_and_audit_s=load_s,warmup_s=warm_s)
    atomic_json(OUT/'evaluator_equivalence.json',equivalence)
    (DOC/'checkpoint_integrity.md').write_text('# Checkpoint integrity — PASS\n\nAll 529 DeepSeekVL and 345 SAM image-encoder missing keys were compared to pinned pretrained disk tensors and verified unchanged after loading F-LMM. Every trained checkpoint tensor was verified equal after load. See results/checkpoint_integrity.json for per-key proof.\n\nOld smoke did not record a backbone revision; historical revision remains UNVERIFIED. New pinned revision reproduces its 20 saved prediction masks exactly.\n')
    (DOC/'evaluator_equivalence.md').write_text('# Evaluator equivalence — PASS on 20 regression images\n\nOfficial upstream control, original modified subset evaluator, and resumable engine produced identical binary masks/GT on the original 20 images. Metrics match within 0.001 percentage point. Controls share one audited model and math backend to avoid needless reload. Only selection/local paths are harnessed; exact diffs are retained. This is not an unmodified eight-split official benchmark run.\n\n'+json.dumps(DATA['counts'],indent=2)+'\n\nUnit: first sentence per retained annotation/reference, grouped and concatenated by image. Other sentences are not evaluated. ref_id/sent_id are derived from annotation and checked against loader text and segmentation. See dataset_manifest.json.\n')
    atomic_json(OUT/'G0.json',dict(status='PASS',checkpoint_integrity='PASS',evaluator_equivalence='PASS'))

def probe():
    if read_json(OUT/'G0.json')['status']!='PASS': raise RuntimeError('G0 required')
    model,cfg,dataset,fp,load_s=setup_model()
    warm_s=warmup(model,dataset)
    result,records,stats=evaluate(model,dataset,fp,DATA['subsets']['probe'],'probe')
    if result['status']!='PASS': raise RuntimeError('Probe failed')
    lat=[r['timings']['inference_s'] for r in stats]
    report=dict(status='PASS',images=len(records),new_images=len(stats),cold_model_load_s=load_s,warmup_s=warm_s,
                inference_latency_s=dict(median=percentile(lat,50),p95=percentile(lat,95),maximum=max(lat)),
                amortized_latency_per_selected_expression_s=dict(median=percentile([r['timings']['amortized_inference_s_per_selected_expression'] for r in stats],50),definition='image inference divided by selected first-sentence mask count; not isolated per-expression inference'),
                peak_allocated_bytes=max(r['memory']['peak_allocated_bytes'] for r in records),
                peak_reserved_bytes=max(r['memory']['peak_reserved_bytes'] for r in records),
                peak_ram_rss_bytes=max(r['memory']['ram_rss_bytes'] for r in records),
                output_bytes_per_image=dict(mean=sum(r['output_bytes']+record_path(BYID[r['sample_id']]).stat().st_size for r in records)/len(records),maximum=max(r['output_bytes']+record_path(BYID[r['sample_id']]).stat().st_size for r in records),definition='compressed mask NPZ plus atomic record JSON; shared report/archive overhead budgeted separately'),
                preprocess_median_s=percentile([r['timings']['preprocess_s'] for r in stats],50),
                postprocess_median_s=percentile([r['timings']['postprocess_s'] for r in stats],50),
                writing_total_s=sum(r['timings']['mask_write_s'] for r in stats),failed_samples=result['failures'],
                writing_p95_s=percentile([r['timings']['mask_write_s'] for r in stats],95),
                data_preparation=read_json(PREP/'cache-manifest.json'),no_network_inference=True)
    atomic_json(OUT/'feasibility_probe.json',report)
    (DOC/'feasibility_report.md').write_text('# Feasibility probe — PASS\n\n50 seeded images outside original regression, using cached inputs and pinned model. Actual synchronized measurements below; per-expression latency is amortized, not isolated. Go/no-go is a separate gate.\n\n```json\n'+json.dumps({k:v for k,v in report.items() if k!='data_preparation'},indent=2)+'\n```\n')

def gate():
    from common import budget_decision
    from datetime import datetime,timezone
    if read_json(OUT/'G0.json')['status']!='PASS': raise RuntimeError('G0 required')
    probe=read_json(OUT/'feasibility_probe.json')
    if probe['status']!='PASS': raise RuntimeError('Probe required')
    quota=read_json(HERE/'quota_snapshot.json'); observed=datetime.fromisoformat(quota['observed_at_utc'])
    age=(datetime.now(timezone.utc)-observed).total_seconds()
    remaining=quota['gpu']['remaining_seconds']-session_elapsed()*2 if age<6*3600 else None
    p95=probe['inference_latency_s']['p95']+probe['preprocess_median_s']+probe['postprocess_median_s']+probe['writing_p95_s']
    decisions={}
    cached_ids={x['sample_id'] for x in read_json(PREP/'image-cache.json')['images']}
    for name in ['stability','development','holdout','full_validation']:
        decisions[name]=budget_decision(len(DATA['subsets'][name]),p95,C['MAX_TOTAL_WALLTIME_MIN']*60-session_elapsed(),
                    max(0,C['MAX_EVALUATION_MIN']*60-evaluation_consumed()),shutil.disk_usage('/kaggle/working').free,
                    probe['output_bytes_per_image']['maximum']+8192,remaining,C)
        if not set(DATA['subsets'][name])<=cached_ids:
            decisions[name].update(status='BLOCKED',reason='Required images not fully cached; GPU evaluation must not download')
    report=dict(status=decisions['stability']['status'],G0='PASS',probe='PASS',remaining_wall_s=C['MAX_TOTAL_WALLTIME_MIN']*60-session_elapsed(),
                quota_snapshot_age_s=age,quota_remaining_s_conservative=remaining,measured_p95_plus_prepost_s=p95,
                data_preparation_s=probe['data_preparation']['data_preparation_seconds'],
                setup_and_loading_s=read_json(OUT/'session.json').get('restore_s',0)+probe['cold_model_load_s'],
                budgets=C,decisions=decisions,selected_max_samples=min(C['MAX_IMAGES_THIS_JOB'],decisions['stability']['selected_max_images']),
                rationale='Only stability enabled by default. Larger stages require PASS G0/probe, current budget gate, prepared cache, and explicit configuration. Holdout untouched.')
    atomic_json(OUT/'go_no_go.json',report)
    atomic_json(OUT/'G1.json',dict(status='PASS' if report['selected_max_samples']>=C['STABILITY_IMAGES'] and report['status'] in ['GO','REDUCE_SCOPE'] else 'BLOCKED'))

def stability():
    if read_json(OUT/'G1.json')['status']!='PASS': raise RuntimeError('G1 not passed')
    model,cfg,dataset,fp,load_s=setup_model(); warm_s=warmup(model,dataset)
    ids=DATA['subsets']['stability']
    existing=sum(valid_record(record_path(BYID[s]),fp,BYID[s]['sample_hash']) is not None for s in ids)
    stop=C.get('STOP_AFTER_IMAGES'); limit=None if stop is None else max(0,stop-existing)
    result,records,stats=evaluate(model,dataset,fp,ids,'stability',limit_new=limit)
    result.update(existing_valid_at_start=existing,cold_model_load_s=load_s,warmup_s=warm_s)
    if records:
        allocated=[r['memory']['allocated_after_bytes'] for r in stats]
        result['memory_growth_bytes']=allocated[-1]-allocated[0] if allocated else 0
        result['memory_growth_likely_oom']=result['memory_growth_bytes']>512*1024**2
        if result['memory_growth_likely_oom']: result['status']='FAIL'
    atomic_json(OUT/'stability.json',result)
    if result['status']=='PASS': atomic_json(OUT/'G2.json',dict(status='PASS',images=len(records),resume_reused=existing))

def baseline():
    if not (OUT/'G2.json').exists() or read_json(OUT/'G2.json')['status']!='PASS':
        raise RuntimeError('G2 stability required before baseline expansion')
    model=None
    for name,enabled in [('development',C['RUN_DEVELOPMENT']),('holdout',C['RUN_HOLDOUT']),('full_validation',C['RUN_FULL_VALIDATION'])]:
        if not enabled:
            atomic_json(OUT/(name+'.json'),dict(status='UNVERIFIED',reason='Stage disabled; no unnecessary evaluation'))
            continue
        gate(); decision=read_json(OUT/'go_no_go.json')['decisions'][name]
        if decision['status']!='GO' or len(DATA['subsets'][name])>C['MAX_IMAGES_THIS_JOB']:
            atomic_json(OUT/(name+'.json'),dict(status='BLOCKED',reason='Go/no-go or configured job image cap prevents expansion',decision=decision))
            continue
        if model is None:
            model,cfg,dataset,fp,load_s=setup_model(); warmup(model,dataset)
        result,records,stats=evaluate(model,dataset,fp,DATA['subsets'][name],name)
        if result['status']!='PASS': break

def export():
    import numpy as np
    records=[]
    contract=read_json(OUT/'run_contract.json') if (OUT/'run_contract.json').exists() else None
    if contract:
        for sample in DATA['samples']:
            r=valid_record(record_path(sample),contract['fingerprint'],sample['sample_hash'])
            if r:
                with np.load(RECORDS/r['mask_file']) as masks:
                    p=masks['prediction'].astype(bool); g=masks['ground_truth'].astype(bool)
                    if p.shape!=g.shape or (p&g).sum(axis=(1,2)).tolist()!=r['intersection'] or (p|g).sum(axis=(1,2)).tolist()!=r['union']:
                        raise ValueError('Independent saved-mask count verification failed: '+sample['sample_id'])
                records.append(r)
    fields=['image_id','dataset_index','ref_id','sent_id','expression','mask_id','prediction_id','annotation_id','intersection','union','iou','sample_id']
    with (OUT/'per_expression.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for r in records:
            for e,i,u in zip(r['expressions'],r['intersection'],r['union']):
                row={k:e.get(k) for k in fields}; row.update(intersection=i,union=u,iou=i/u if u else 1.0,sample_id=r['sample_id']); w.writerow(row)
    by_subset={}
    for name,ids in DATA['subsets'].items():
        group=[r for r in records if r['sample_id'] in set(ids)]
        by_subset[name]=dict(status='PASS' if len(group)==len(ids) else 'UNVERIFIED',expected_images=len(ids),**aggregate(group))
    atomic_json(OUT/'baseline_metrics.json',dict(status='PASS' if records else 'UNVERIFIED',scope='selected first-sentence masks, not all expressions',subsets=by_subset))
    atomic_json(OUT/'profiling.json',dict(status='PASS' if records else 'UNVERIFIED',samples=[dict(sample_id=r['sample_id'],timings=r['timings'],memory=r['memory'],output_bytes=r['output_bytes']) for r in records]))
    (DOC/'observation_report.md').write_text('# Phase 1 observations\n\n'+json.dumps({k:v for k,v in by_subset.items() if k in ['regression','probe','stability']},indent=2)+'\n\nMetrics were independently recomputed from saved binary masks. Probe and stability are nested development data, not independent estimates. Holdout is untouched unless explicitly enabled and passed. No method improvement or confidence gate was selected. Memory, timing and output sizes are in profiling.json; do not equate amortized expression time with independent per-expression inference.\n')
    latest={}
    if (OUT/'run_manifest.jsonl').exists():
        for line in (OUT/'run_manifest.jsonl').read_text().splitlines():
            try:
                e=json.loads(line); latest[e['sample_id']]=e['status']
            except (ValueError,KeyError): continue
    done={r['sample_id'] for r in records}
    atomic_json(OUT/'sample_status.json',{s['sample_id']:('DONE' if s['sample_id'] in done else latest.get(s['sample_id'],'PENDING')) for s in DATA['samples']})
    stages={}
    for name in ['checkpoint_integrity','evaluator_equivalence','feasibility_probe','go_no_go','stability','G0','G1','G2']:
        stages[name]=read_json(OUT/(name+'.json')).get('status','UNVERIFIED') if (OUT/(name+'.json')).exists() else 'UNVERIFIED'
    atomic_json(OUT/'run_summary.json',dict(stages=stages,validated_images=len(records),metrics=by_subset,
                session_elapsed_s=session_elapsed(),full_official_reproduction=False,
                failure_records=[p.name for p in OUT.glob('*-failure.json')],
                durable_resume_note='Archives in working are export-ready; only completed Kaggle outputs persist cross-session. This job is deliberately bounded; next job mounts previous committed output.'))
    shutil.copy2(PREP/'dataset_manifest.json',OUT/'dataset_manifest.json')
    save_snapshot()

def main():
    stage=sys.argv[1]
    try:
        if C.get('STOP_AFTER_GATE')==0 and stage not in ['audit','export']:
            raise RuntimeError('BLOCKED by user instruction: stop after Gate 0 for audit')
        if stage=='audit': audit()
        elif stage=='probe': probe()
        elif stage=='gate': gate()
        elif stage=='stability': stability()
        elif stage=='baseline': baseline()
        elif stage=='export': export()
        else: raise ValueError(stage)
    except Exception as e:
        atomic_json(OUT/(stage+'-failure.json'),dict(status='FAIL',error=repr(e),traceback=traceback.format_exc()))
        if stage=='audit': atomic_json(OUT/'G0.json',dict(status='FAIL',reason=repr(e)))
        raise
    finally:
        export()

if __name__=='__main__': main()
