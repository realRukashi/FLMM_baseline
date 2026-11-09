"""CPU-only preparation: verified cache, official loader manifest, reusable environment."""
import concurrent.futures
import inspect
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from common import atomic_json,digest,read_json,sha256

HERE=Path(__file__).resolve().parent
C=read_json(HERE/'config.json')
WORK=Path('/kaggle/working'); CACHE=WORK/'cache'; RESULTS=WORK/'results'

def download(url,path,expected=None):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if path.is_file() and (expected is None or sha256(path)==expected): return
    for attempt in range(C['MAX_RETRIES_PER_DOWNLOAD']):
        temp=path.with_suffix(path.suffix+'.part')
        try:
            with urllib.request.urlopen(url,timeout=C['DOWNLOAD_TIMEOUT_S']) as r,temp.open('wb') as f:
                start=time.monotonic()
                while True:
                    b=r.read(1024*1024)
                    if not b: break
                    f.write(b)
                    if time.monotonic()-start>900: raise TimeoutError('Download total deadline')
            if expected and sha256(temp)!=expected: raise ValueError('Download checksum mismatch')
            os.replace(temp,path); return
        except Exception:
            if attempt==C['MAX_RETRIES_PER_DOWNLOAD']-1: raise
            time.sleep(2**attempt)

def worker():
    from huggingface_hub import snapshot_download,hf_hub_download
    from mmdet.datasets import RefCocoDataset
    from PIL import Image
    import mmengine
    RESULTS.mkdir(exist_ok=True); CACHE.mkdir(exist_ok=True)
    started=time.monotonic()
    annotation=CACHE/'refcoco.zip'
    download('https://web.archive.org/web/20220413011718id_/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip',annotation,C['ANNOTATION_SHA256'])
    data=CACHE/'data/coco'
    with zipfile.ZipFile(annotation) as z:
        for name in ['refcoco/instances.json','refcoco/refs(unc).p']:
            target=data/name; target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(z.read(name))
    dataset=RefCocoDataset(data_root=str(data),ann_file='refcoco/instances.json',split_file='refcoco/refs(unc).p',
                          data_prefix=dict(img_path='train2014/'),split='val',text_mode='select_first',pipeline=[])
    refs=mmengine.load(data/'refcoco/refs(unc).p',file_format='pkl')
    instances=read_json(data/'refcoco/instances.json')
    ann_by_id={a['id']:a for a in instances['annotations']}
    selected={}
    for ref in refs:
        if ref['split']=='val': selected[ref['ann_id']]=ref
    grouped={}
    for ref in selected.values(): grouped.setdefault(ref['image_id'],[]).append(ref)
    records=[]
    for idx in range(len(dataset)):
        info=dataset.get_data_info(idx); expressions=[]
        matching=grouped[info['img_id']]
        if len(matching)!=len(info['text']): raise ValueError('Reference mapping count mismatch')
        for mask_id,(ref,text,inst) in enumerate(zip(matching,info['text'],info['instances'])):
            sentence=ref['sentences'][0]
            if sentence['raw'].lower()!=text or digest(inst['mask'])!=digest(ann_by_id[ref['ann_id']]['segmentation']):
                raise ValueError('Reference mapping differs from official loader')
            expressions.append(dict(image_id=info['img_id'],dataset_index=idx,ref_id=ref.get('ref_id'),
                                    sent_id=sentence.get('sent_id'),expression=text,mask_id=mask_id,
                                    annotation_id=ref['ann_id'],prediction_id=f"refcoco-val-{info['img_id']}:{mask_id}",
                                    gt_segmentation_sha256=digest(inst['mask']),category_id=ref.get('category_id')))
        record=dict(sample_id=f"refcoco-val-{info['img_id']}",image_id=info['img_id'],dataset_index=idx,
                    file_name=Path(info['img_path']).name,expressions=expressions,
                    gt_instances_sha256=digest(info['instances']))
        record['sample_hash']=digest(record); records.append(record)
    regression_ids=[x['img_id'] for x in read_json(HERE/'regression-manifest.json')]
    by_image={x['image_id']:x for x in records}
    if [x['image_id'] for x in records[:20]]!=regression_ids: raise ValueError('Regression loader order changed')
    regression=[by_image[i]['sample_id'] for i in regression_ids]
    remaining=sorted([r for r in records if r['sample_id'] not in regression],key=lambda r:r['image_id'])
    random.Random(C['SEED']).shuffle(remaining)
    dev=remaining[:C['DEVELOPMENT_IMAGES']]
    holdout=remaining[C['DEVELOPMENT_IMAGES']:C['DEVELOPMENT_IMAGES']+C['HOLDOUT_IMAGES']]
    sets=dict(regression=regression,probe=[r['sample_id'] for r in dev[:C['PROBE_IMAGES']]],
              stability=[r['sample_id'] for r in dev[:C['STABILITY_IMAGES']]],
              development=[r['sample_id'] for r in dev],holdout=[r['sample_id'] for r in holdout],
              full_validation=[r['sample_id'] for r in records])
    assert not set(sets['development'])&set(sets['holdout'])
    hist={}
    for name,ids in sets.items():
        chosen=[r for r in records if r['sample_id'] in set(ids)]
        hist[name]=dict(images=len(chosen),selected_masks=sum(len(r['expressions']) for r in chosen),
                       one_target=sum(len(r['expressions'])==1 for r in chosen),
                       three_or_more_targets=sum(len(r['expressions'])>=3 for r in chosen))
    manifest=dict(status='PASS',seed=C['SEED'],split='RefCOCO UNC val',text_mode='select_first',concat=True,
                  selection_rule='Seeded simple random shuffle of sorted image IDs, excluding 20 debug images; probe nested in stability nested in development; holdout disjoint',
                  unit='one image inference; one selected first sentence per retained annotation/reference',
                  counts=dict(images=len(records),raw_val_references=sum(r['split']=='val' for r in refs),
                              raw_val_sentences=sum(len(r['sentences']) for r in refs if r['split']=='val'),
                              retained_references=len(selected),selected_target_masks=sum(len(r['expressions']) for r in records)),
                  annotation_zip_sha256=sha256(annotation),instances_sha256=sha256(data/'refcoco/instances.json'),
                  refs_sha256=sha256(data/'refcoco/refs(unc).p'),samples=records,subsets=sets,distribution=hist)
    manifest['manifest_hash']=digest(manifest)
    atomic_json(RESULTS/'dataset_manifest.json',manifest)
    (RESULTS/'refcoco-loader.py').write_text(inspect.getsource(RefCocoDataset))
    (RESULTS/'loader-module.py').write_text(Path(inspect.getfile(RefCocoDataset)).read_text())
    required=set(sets['regression']+sets['development']+sets['holdout'])
    if C['RUN_FULL_VALIDATION']: required=set(sets['full_validation'])
    def image_task(record):
        target=data/'train2014'/record['file_name']
        download('https://s3.amazonaws.com/images.cocodataset.org/train2014/'+record['file_name'],target)
        with Image.open(target) as im: im.verify()
        with Image.open(target) as im: shape=list(im.size)
        return dict(sample_id=record['sample_id'],file_name=record['file_name'],sha256=sha256(target),bytes=target.stat().st_size,size=shape)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        images=list(pool.map(image_task,[r for r in records if r['sample_id'] in required]))
    atomic_json(RESULTS/'image-cache.json',dict(status='PASS',images=images,total_bytes=sum(x['bytes'] for x in images)))
    model_dir=CACHE/'models'; model_dir.mkdir(exist_ok=True)
    os.environ['HF_HUB_ETAG_TIMEOUT']='45'; os.environ['HF_HUB_DOWNLOAD_TIMEOUT']='45'
    snapshot=snapshot_download('deepseek-ai/deepseek-vl-1.3b-chat',revision=C['BACKBONE_REVISION'],
                               local_dir=str(model_dir/'backbone'),allow_patterns=['*.json','*.safetensors'],max_workers=3)
    backbone=Path(snapshot)/'model.safetensors'
    if sha256(backbone)!=C['BACKBONE_SHA256']: raise ValueError('Backbone LFS hash mismatch')
    flmm=hf_hub_download('wusize/flmm','frozen_deepseek_vl_1_3b_chat_unet_sam_l_refcoco_png.pth',revision=C['FLMM_REVISION'],local_dir=str(model_dir/'flmm'))
    if sha256(flmm)!=C['FLMM_SHA256']: raise ValueError('F-LMM hash mismatch')
    download('https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',model_dir/'sam_vit_l_0b3195.pth',C['SAM_SHA256'])
    atomic_json(RESULTS/'cache-manifest.json',dict(status='PASS',models=[dict(path=str(p.relative_to(CACHE)),sha256=sha256(p),bytes=p.stat().st_size) for p in [backbone,Path(flmm),model_dir/'sam_vit_l_0b3195.pth']],data_preparation_seconds=time.monotonic()-started))

def main():
    RESULTS.mkdir(exist_ok=True); CACHE.mkdir(exist_ok=True)
    env=os.environ.copy(); env.update(UV_CACHE_DIR='/kaggle/temp/uv-cache',UV_PYTHON_INSTALL_DIR='/kaggle/temp/python',PIP_NO_CACHE_DIR='1',DS_BUILD_OPS='0',HF_HUB_DISABLE_XET='1',HF_HOME='/kaggle/temp/hf')
    steps=[]; start=time.monotonic()
    def run(name,cmd,timeout=900):
        print('START',name,flush=True); t=time.monotonic()
        with (RESULTS/(name+'.log')).open('w') as f:
            p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
        steps.append(dict(stage=name,status='PASS' if p.returncode==0 else 'FAIL',seconds=time.monotonic()-t))
        atomic_json(RESULTS/'prepare-stages.json',steps)
        print((RESULTS/(name+'.log')).read_text(errors='replace')[-1500:],flush=True)
        if p.returncode: raise RuntimeError(f'{name}: exit {p.returncode}')
    run('uv',[sys.executable,'-m','pip','install','uv'])
    uv=[sys.executable,'-m','uv']; py='/kaggle/temp/flmm-env/bin/python'
    if not Path(py).exists(): run('environment',uv+['venv','--python','3.10.22','/kaggle/temp/flmm-env'])
    # Reuse the successful freeze as constraints, including all transitive versions.
    pip=uv+['pip','install','--python',py,'-c',str(HERE/'constraints.txt')]
    # PyTorch's wheel index does not mirror every transitive PyPI version in the
    # successful freeze. Install its wheels first, then constrain PyPI dependencies.
    run('torch',uv+['pip','install','--python',py,'torch==2.1.2+cu118','torchvision==0.16.2+cu118','--index-url','https://download.pytorch.org/whl/cu118'])
    run('dependencies',pip+['numpy==1.26.4','transformers==4.39.1','accelerate==0.28.0','mmengine==0.10.5','mmdet==3.3.0','mmsegmentation==1.2.2','xtuner==0.1.23','peft==0.10.0','datasets==2.19.2','bitsandbytes==0.43.1','timm==0.9.16','attrdict','einops','sentencepiece','ftfy','opencv-python-headless==4.10.0.84','setuptools<81','pip','wheel','packaging','ninja','psutil','py-cpuinfo','hjson','pydantic<3'])
    run('mmcv',pip+['mmcv==2.1.0','--only-binary','mmcv','--find-links','https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html'])
    run('deepspeed',pip+['--no-build-isolation','deepspeed==0.14.2'])
    repo=CACHE/'F-LMM'
    if not repo.exists(): run('source',['git','clone','https://github.com/wusize/F-LMM.git',str(repo)])
    run('revision',['git','-C',str(repo),'checkout',C['SOURCE_SHA']])
    env['PYTHONPATH']=str(repo)+':'+str(HERE)
    env['LD_LIBRARY_PATH']='/kaggle/temp/flmm-env/lib/python3.10/site-packages/torch/lib:'+env.get('LD_LIBRARY_PATH','')
    run('data-model-cache',[py,str(HERE/'prepare.py'),'worker'],timeout=2400)
    run('freeze',uv+['pip','freeze','--python',py])
    for p in [Path('/kaggle/temp/flmm-env'),Path('/kaggle/temp/python')]:
        print('CACHE_ENV_BYTES',p,sum(x.stat().st_size for x in p.rglob('*') if x.is_file() and not x.is_symlink()),flush=True)
    run('package-environment',['tar','-I','gzip -1','-cf',str(CACHE/'environment.tar.gz'),'-C','/kaggle/temp','flmm-env','python'],timeout=1200)
    atomic_json(CACHE/'environment.json',dict(status='PASS',sha256=sha256(CACHE/'environment.tar.gz'),bytes=(CACHE/'environment.tar.gz').stat().st_size,python='3.10.22',restore_to='/kaggle/temp',freeze_sha256=sha256(RESULTS/'freeze.log')))
    atomic_json(RESULTS/'preparation_summary.json',dict(status='PASS',seconds=time.monotonic()-start,cache_bytes=sum(p.stat().st_size for p in CACHE.rglob('*') if p.is_file() and not p.is_symlink()),steps=steps))
    print('PREPARATION_PASS',flush=True)

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='worker': worker()
    else:
        try: main()
        except Exception as e:
            atomic_json(RESULTS/'preparation_summary.json',dict(status='FAIL',error=repr(e)))
            raise
