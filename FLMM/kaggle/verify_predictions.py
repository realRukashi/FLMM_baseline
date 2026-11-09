"""Independently verify saved masks against official metric output."""
import argparse
import json
from pathlib import Path
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('result_dir',type=Path)
args=parser.parse_args()
root=args.result_dir
manifest=json.loads((root/'sample-manifest.json').read_text())
metrics=json.loads((root/'metrics.json').read_text())
intersections=[]; unions=[]; examples=[]
for sample in manifest:
    idx=sample['dataset_index']
    data=np.load(root/f'sample-{idx}.npz')
    p=data['prediction'].astype(bool); g=data['ground_truth'].astype(bool)
    assert p.shape == g.shape and p.ndim == 3
    i=(p&g).sum(axis=(1,2)); u=(p|g).sum(axis=(1,2))
    assert np.all(u>0)
    intersections.extend(i.tolist()); unions.extend(u.tolist())
    examples.append(dict(**sample,mask_count=len(p),nonempty_predictions=int(p.reshape(len(p),-1).any(1).sum()),
                         cIoU=float(100*i.sum()/u.sum()),mIoU=float(100*np.mean(i/u))))
i=np.array(intersections,dtype=np.float64); u=np.array(unions,dtype=np.float64)
computed=dict(cIoU=float(100*i.sum()/u.sum()),mIoU=float(100*np.mean(i/u)))
for key,value in computed.items():
    assert abs(value-metrics['metrics'][key])<1e-3,(key,value,metrics)
assert len(manifest)==metrics['sample_count']
report=dict(image_count=len(manifest),mask_count=len(i),recomputed_metrics=computed,
            all_predictions_nonempty=all(x['mask_count']==x['nonempty_predictions'] for x in examples),
            examples=examples,scope='Pipeline smoke check; first N loader images of RefCOCO UNC val; not full benchmark')
(root/'verified-summary.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='examples'},indent=2))
