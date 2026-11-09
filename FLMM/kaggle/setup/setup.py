"""Bootstrap an isolated F-LMM environment and test imports, without model downloads."""
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

out = Path('/kaggle/working/setup-report')
out.mkdir(exist_ok=True)
env = os.environ.copy()
env['UV_CACHE_DIR'] = '/kaggle/temp/uv-cache'
env['UV_PYTHON_INSTALL_DIR'] = '/kaggle/temp/python'
env['PIP_NO_CACHE_DIR'] = '1'
env['HF_HOME'] = '/kaggle/temp/huggingface'
env['DS_BUILD_OPS'] = '0'
report = {'source_sha': '47d3c2a6f628dd48ba76a1cfdf29b6b2aa8f9eb3', 'stage': 'start'}

def run(label, command, timeout=600):
    report['stage'] = label
    (out / 'status.json').write_text(json.dumps(report, indent=2))
    print('START', label, flush=True)
    with (out / (label + '.log')).open('w') as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    print((out / (label + '.log')).read_text()[-5000:], flush=True)
    if result.returncode:
        raise RuntimeError(f'{label} failed, return code {result.returncode}')
    print('PASS', label, flush=True)

try:
    run('install-uv', [sys.executable, '-m', 'pip', 'install', 'uv'])
    uv = [sys.executable, '-m', 'uv']
    run('python-env', uv + ['venv', '--python', '3.10', '/kaggle/temp/flmm-env'])
    py = '/kaggle/temp/flmm-env/bin/python'
    run('torch', uv + ['pip', 'install', '--python', py, 'torch==2.1.2', 'torchvision==0.16.2',
                      '--index-url', 'https://download.pytorch.org/whl/cu118'], timeout=900)
    run('dependencies', uv + ['pip', 'install', '--python', py,
        'numpy==1.26.4', 'transformers==4.39.1', 'accelerate==0.28.0',
        'mmengine==0.10.5', 'mmdet==3.3.0', 'mmsegmentation==1.2.2',
        'xtuner==0.1.23', 'peft==0.10.0', 'datasets==2.19.2', 'bitsandbytes==0.43.1',
        'timm==0.9.16', 'attrdict', 'einops', 'sentencepiece', 'gdown', 'ftfy',
        'opencv-python-headless==4.10.0.84', 'setuptools<81'], timeout=900)
    run('mmcv', uv + ['pip', 'install', '--python', py, 'mmcv==2.1.0', '--only-binary', 'mmcv',
        '--find-links', 'https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html'], timeout=600)
    run('clone', ['git', 'clone', 'https://github.com/wusize/F-LMM.git', '/kaggle/temp/F-LMM'])
    run('checkout', ['git', '-C', '/kaggle/temp/F-LMM', 'checkout', report['source_sha']])
    env['PYTHONPATH'] = '/kaggle/temp/F-LMM'
    cuda_libs = sorted(Path('/kaggle/temp/flmm-env/lib/python3.10/site-packages/nvidia').glob('*/lib'))
    env['LD_LIBRARY_PATH'] = ':'.join(str(p) for p in cuda_libs) + ':' + env.get('LD_LIBRARY_PATH', '')
    run('deepspeed', uv + ['pip', 'install', '--python', py, 'pip', 'wheel', 'packaging', 'ninja', 'psutil', 'py-cpuinfo', 'hjson', 'pydantic<3'])
    run('deepspeed-package', uv + ['pip', 'install', '--python', py, '--no-build-isolation', 'deepspeed==0.14.2'], timeout=600)
    code = '''import torch
from mmcv.ops import roi_align
from mmdet.datasets import RefCocoDataset
from mmdet.evaluation import RefSegMetric
from mmseg.models import UNet
from xtuner.registry import BUILDER
from flmm.models.frozen_deepseek_vl import FrozenDeepseekVLSAM
from flmm.datasets.transforms import RefCOCO2PNG
from deepseek_vl.models import MultiModalityCausalLM
print('IMPORTS_OK', torch.__version__, torch.cuda.get_device_name(0))
x=torch.ones((2,2), device='cuda', dtype=torch.bfloat16)
print('BF16_MATMUL', (x@x).float().cpu().tolist())
'''
    run('imports', [py, '-c', code])
    run('freeze', uv + ['pip', 'freeze', '--python', py])
    report['stage'] = 'COMPLETE'
except Exception:
    report['error'] = traceback.format_exc()
    print(report['error'], flush=True)
finally:
    (out / 'status.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
