"""Static scientific overview from verified measured output only."""
import csv
import json
import os
from pathlib import Path
import numpy as np
os.environ['MPLCONFIGDIR']=str(Path(__file__).resolve().parent/'.mplconfig')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent/'results'
verification=json.loads((root/'local_verification.json').read_text())
assert verification['status']=='PASS' and verification['images']==1500
with (root/'per_target_iou.csv').open(encoding='utf-8') as f:
    ious=np.array([float(r['iou']) for r in csv.DictReader(f)])
latency=json.loads((root/'latency.json').read_text())
times=np.array([r['timings']['inference_s'] for r in latency['records']])
assert len(ious)==3811 and len(times)==1500
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
axes[0].hist(ious,bins=np.linspace(0,1,26),color='#287D8E',edgecolor='white')
axes[0].set(xlabel='Per-target IoU',ylabel='Targets',xlim=(0,1),title='3,811 selected target masks')
axes[1].hist(times,bins=35,color='#5166A3',edgecolor='white')
for p,color,label in [(50,'#1A5244','Median'),(95,'#A04B21','p95')]:
    value=np.percentile(times,p); axes[1].axvline(value,color=color,linestyle='--',label=f'{label}: {value:.2f} s')
axes[1].set(xlabel='Synchronized forward time (s/image)',ylabel='Images',title='1,500 images; warm-up excluded')
axes[1].legend(frameon=False)
fig.suptitle('F-LMM DeepSeekVL-1.3B — full RefCOCO UNC val',fontsize=13)
fig.savefig(root/'validation_overview.png',dpi=200)
plt.close(fig)
print(root/'validation_overview.png')
