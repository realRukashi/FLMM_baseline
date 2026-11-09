"""CPU-only extension of the verified cache to all 1,500 val images."""
import concurrent.futures
import json
import os
from pathlib import Path
import time
import urllib.request
from PIL import Image
from common import atomic_json,read_json,sha256

def main():
    roots=list(Path('/kaggle/input').rglob('cache/environment.json'))
    if len(roots)!=1: raise ValueError('Expected one verified base cache')
    base=roots[0].parent; manifest=read_json(base.parent/'results/dataset_manifest.json')
    existing=read_json(base.parent/'results/image-cache.json')
    by_id={r['sample_id']:r for r in existing['images']}
    target=Path('/kaggle/working/full-data'); (target/'images').mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    def fetch(sample):
        if sample['sample_id'] in by_id: return by_id[sample['sample_id']]
        path=target/'images'/sample['file_name']; tmp=path.with_suffix('.part')
        for attempt in range(3):
            try:
                t=time.monotonic()
                with urllib.request.urlopen('https://s3.amazonaws.com/images.cocodataset.org/train2014/'+sample['file_name'],timeout=45) as r,tmp.open('wb') as f:
                    while True:
                        b=r.read(1024*1024)
                        if not b: break
                        f.write(b)
                        if time.monotonic()-t>180: raise TimeoutError('Image download deadline')
                with Image.open(tmp) as im: im.verify()
                with Image.open(tmp) as im: size=list(im.size)
                os.replace(tmp,path)
                return dict(sample_id=sample['sample_id'],file_name=sample['file_name'],sha256=sha256(path),bytes=path.stat().st_size,size=size)
            except Exception:
                if attempt==2: raise
                time.sleep(2**attempt)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        images=list(pool.map(fetch,manifest['samples']))
    assert len(images)==1500 and len({r['sample_id'] for r in images})==1500
    atomic_json(target/'image-cache.json',dict(status='PASS',images=images,total_bytes=sum(r['bytes'] for r in images)))
    atomic_json(target/'preparation.json',dict(status='PASS',images=1500,reused_images=len(by_id),downloaded_images=1500-len(by_id),
                preparation_s=time.monotonic()-start,downloaded_bytes=sum(p.stat().st_size for p in (target/'images').glob('*.jpg')),
                source_manifest_hash=manifest['manifest_hash'],network_retries=3,timeout_s=45))
    print(read_json(target/'preparation.json'),flush=True)

if __name__=='__main__': main()
