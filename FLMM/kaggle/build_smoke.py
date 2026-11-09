"""Package validated setup plus smoke preparation into one Kaggle script."""
import json
from pathlib import Path

base = Path(__file__).resolve().parent
setup = (base / 'setup/setup.py').read_text()
evaluator = (base / 'evaluator/eval_refcoco_subset.py').read_text()
prepare = '''import hashlib, json, os, urllib.request, zipfile
from pathlib import Path
from mmdet.datasets import RefCocoDataset
from huggingface_hub import hf_hub_download
os.chdir('/kaggle/temp/F-LMM')
root=Path('data/coco'); root.mkdir(parents=True, exist_ok=True)
annotation_url='https://web.archive.org/web/20220413011718id_/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip'
with urllib.request.urlopen(annotation_url, timeout=120) as response, open('/kaggle/temp/refcoco.zip','wb') as f:
    import shutil
    shutil.copyfileobj(response,f)
with zipfile.ZipFile('/kaggle/temp/refcoco.zip') as z:
    for name in ('refcoco/instances.json', 'refcoco/refs(unc).p'):
        (root/name).parent.mkdir(parents=True, exist_ok=True)
        (root/name).write_bytes(z.read(name))
dataset=RefCocoDataset(data_root=str(root), ann_file='refcoco/instances.json', split_file='refcoco/refs(unc).p', data_prefix=dict(img_path='train2014/'), split='val', text_mode='select_first', pipeline=[])
sample=dataset.get_data_info(0)
image=Path(sample['img_path']); image.parent.mkdir(parents=True, exist_ok=True)
for i in range(20):
    info=dataset.get_data_info(i)
    image_path=Path(info['img_path'])
    urllib.request.urlretrieve('https://s3.amazonaws.com/images.cocodataset.org/train2014/'+image_path.name, image_path)
    saved=Path('/kaggle/working/smoke-results/inputs'); saved.mkdir(parents=True,exist_ok=True)
    (saved/f'{i}.jpg').write_bytes(image_path.read_bytes())
    (saved/f'{i}.json').write_text(json.dumps(info,indent=2))
Path('/kaggle/working/smoke-results').mkdir(exist_ok=True)
Path('/kaggle/working/smoke-results/input.json').write_text(json.dumps(sample, indent=2))
Path('/kaggle/working/smoke-results/annotation-source.json').write_text(json.dumps(dict(url=annotation_url,sha256=hashlib.sha256(Path('/kaggle/temp/refcoco.zip').read_bytes()).hexdigest(),val_images=len(dataset)),indent=2))
Path('/kaggle/working/smoke-results/input.jpg').write_bytes(image.read_bytes())
ckpt=Path('checkpoints'); ckpt.mkdir(exist_ok=True)
urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth', ckpt/'sam_vit_l_0b3195.pth')
trained=ckpt/'frozen_deepseek_vl_1_3b_chat_unet_sam_l_refcoco_png.pth'
downloaded=hf_hub_download(repo_id='wusize/flmm', filename=trained.name, revision='d70e5b3289ca4255cb6a017a792870aad26c01ce')
trained.symlink_to(downloaded)
manifest=[]
for path in ckpt.glob('*.pth'):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024), b''): h.update(chunk)
    manifest.append(dict(name=path.name, bytes=path.stat().st_size, sha256=h.hexdigest()))
Path('/kaggle/working/smoke-results/checkpoints.json').write_text(json.dumps(manifest,indent=2))
print('DATA_AND_CHECKPOINTS_READY', flush=True)
'''
extra = f'''    Path('/kaggle/temp/prepare_smoke.py').write_text({prepare!r})
    Path('/kaggle/temp/F-LMM/eval_refcoco_subset.py').write_text({evaluator!r})
    run('data-checkpoints', [py, '/kaggle/temp/prepare_smoke.py'], timeout=1800)
    os.chdir('/kaggle/temp/F-LMM')
    env['CUDA_VISIBLE_DEVICES'] = '0'
    run('smoke-inference', [py, 'eval_refcoco_subset.py', 'configs/deepseek_vl/frozen_deepseek_vl_1_3b_chat_unet_sam_l_refcoco_png.py', '--checkpoint', 'checkpoints/frozen_deepseek_vl_1_3b_chat_unet_sam_l_refcoco_png.pth', '--concat', '--max-samples', '20'], timeout=1800)
'''
setup = setup.replace("    report['stage'] = 'COMPLETE'", extra + "    report['stage'] = 'COMPLETE'")
setup = setup.replace("env['DS_BUILD_OPS'] = '0'", "env['DS_BUILD_OPS'] = '0'\nenv['HF_HUB_DISABLE_XET'] = '1'")
compile(setup, 'smoke.py', 'exec')
target = base / 'smoke'
target.mkdir(exist_ok=True)
(target/'smoke.py').write_text(setup, encoding='utf-8')
metadata=json.loads((base/'setup/kernel-metadata.json').read_text())
metadata.update(id='hoofngocjhuy/flmm-refcoco-smoke',title='FLMM RefCOCO Smoke',code_file='smoke.py')
(target/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
print(target)
