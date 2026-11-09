"""Non-invasive diagnostics; does not download weights or launch GPU training."""
import os,sys,platform,importlib.metadata
print('python:', sys.version)
print('platform:',platform.platform())
print('cwd:',os.getcwd())
for p in ('torch','transformers','accelerate','xtuner','mmsegmentation','mmdet','mmengine','mmcv'):
 try: print(f'{p}:',importlib.metadata.version(p))
 except importlib.metadata.PackageNotFoundError: print(f'{p}: NOT INSTALLED')
try:
 import torch
 print('torch CUDA:',torch.version.cuda,'available:',torch.cuda.is_available())
 for i in range(torch.cuda.device_count()):
  props=torch.cuda.get_device_properties(i)
  print('GPU',i,props.name,round(props.total_memory/2**30,2),'GiB')
except Exception as e: print('torch diagnostic error:',repr(e))
