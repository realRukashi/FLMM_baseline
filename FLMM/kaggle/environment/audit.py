"""Read-only environment audit. No model downloads or package installation."""
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

out = Path('/kaggle/working/audit')
out.mkdir(parents=True, exist_ok=True)
report = {'python': sys.version, 'platform': platform.platform()}
report['disk'] = dict(zip(('total', 'used', 'free'), shutil.disk_usage('/kaggle/working')))
report['packages'] = {}
for name in ['torch', 'torchvision', 'transformers', 'accelerate', 'xtuner',
             'mmengine', 'mmcv', 'mmcv-lite', 'mmdet', 'mmsegmentation',
             'numpy', 'timm', 'huggingface-hub']:
    try:
        report['packages'][name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        report['packages'][name] = None
try:
    import torch
    report['cuda'] = torch.version.cuda
    report['gpus'] = [dict(index=i, name=torch.cuda.get_device_name(i),
                           vram_bytes=torch.cuda.get_device_properties(i).total_memory)
                      for i in range(torch.cuda.device_count())]
    report['bf16_supported'] = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
except Exception as exc:
    report['torch_error'] = str(exc)
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=30)
    (out / 'nvidia-smi.txt').write_text(result.stdout + result.stderr)
except Exception as exc:
    report['nvidia_smi_error'] = str(exc)
for label, url in {
    'github': 'https://raw.githubusercontent.com/wusize/F-LMM/main/README.md',
    'huggingface': 'https://huggingface.co/api/models/deepseek-ai/deepseek-vl-1.3b-chat',
}.items():
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            body = response.read()
            report[label] = {'status': response.status, 'bytes': len(body)}
            (out / (label + '.txt')).write_bytes(body)
    except Exception as exc:
        report[label] = {'error': str(exc)}
(out / 'environment.json').write_text(json.dumps(report, indent=2))
freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True, timeout=60)
(out / 'pip-freeze.txt').write_text(freeze.stdout)
print(json.dumps(report, indent=2), flush=True)
print('AUDIT_COMPLETE', flush=True)
