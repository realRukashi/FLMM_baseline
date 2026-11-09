"""Create a minimal auditable patch of the pinned upstream RES evaluator."""
import difflib
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = root / 'upstream/F-LMM/scripts/multiprocess_eval_refcoco.py'
original = source.read_text(encoding='utf-8')
changed = original
replacements = [
    ('    model = BUILDER.build(cfg.model)', '    torch.backends.cuda.enable_flash_sdp(False)\n    torch.backends.cuda.enable_mem_efficient_sdp(False)\n    torch.backends.cuda.enable_math_sdp(True)\n    model = BUILDER.build(cfg.model)'),
    ('import argparse\n', 'import argparse\nimport json\nfrom pathlib import Path\n'),
    ("    args = parser.parse_args()", "    parser.add_argument('--max-samples', type=int, default=1)\n    parser.add_argument('--output-dir', default='/kaggle/working/smoke-results')\n    args = parser.parse_args()\n    if args.max_samples < 1:\n        parser.error('--max-samples must be positive')\n    output_dir = Path(args.output_dir)\n    output_dir.mkdir(parents=True, exist_ok=True)"),
    ('        accelerator.print(f"Unexpected parameters: {unexpected}")',
     '        accelerator.print(f"Unexpected parameters: {unexpected}")\n        accelerator.print(f"Missing parameters: {missing}")\n        (output_dir / "checkpoint-keys.json").write_text(json.dumps(dict(missing=missing, unexpected=unexpected), indent=2))'),
    ('    for name, subset in refcoco_subsets.items():',
     '    refcoco_subsets = {"refcoco_val": refcoco_subsets["refcoco_val"]}\n    for name, subset in refcoco_subsets.items():'),
    ('        results = []',
     '        data_ids = data_ids[:args.max_samples]\n        manifest = [dict(dataset_index=i, img_id=dataset.get_data_info(i).get("img_id"), img_path=dataset.get_data_info(i).get("img_path")) for i in data_ids]\n        (output_dir / "sample-manifest.json").write_text(json.dumps(manifest, indent=2))\n        results = []'),
    ('\n                    pred_masks = pred_masks > 0.5',
     '\n                    pred_masks = pred_masks > 0.5\n                    import numpy as np\n                    np.savez_compressed(output_dir / f"sample-{idx}.npz", prediction=pred_masks.numpy(), ground_truth=gt_masks)'),
    ('            accelerator.print(f"Evaluation results on {name}: {metrics}")',
     '            accelerator.print(f"Evaluation results on {name}: {metrics}")\n            (output_dir / "metrics.json").write_text(json.dumps(dict(split=name, sample_count=len(data_ids), metrics={k: float(v) for k, v in metrics.items()}), indent=2))'),
]
for old, new in replacements:
    if changed.count(old) != 1:
        raise ValueError(f'Unexpected upstream structure for {old!r}')
    changed = changed.replace(old, new)
compile(changed, 'eval_refcoco_subset.py', 'exec')
target = Path(__file__).resolve().parent / 'evaluator'
target.mkdir(exist_ok=True)
(target / 'eval_refcoco_subset.py').write_text(changed, encoding='utf-8')
(target / 'upstream.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True), changed.splitlines(True), fromfile='upstream/scripts/multiprocess_eval_refcoco.py', tofile='eval_refcoco_subset.py')), encoding='utf-8')
print(target)
