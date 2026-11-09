# F-LMM Kaggle: initial reproduction result

Date: 2026-11-05 (ICT). Account: `hoofngocjhuy`. Execution used Kaggle API; no desktop input needed.

## Verified result

Private kernel https://www.kaggle.com/code/hoofngocjhuy/flmm-refcoco-smoke, version 6, completed all 20 images (47 referring target masks) from the beginning of the RefCOCO UNC val loader.

| Metric | Official evaluator | Independent recomputation |
|---|---:|---:|
| cIoU (%) | 79.18694 | 79.18694144 |
| mIoU (%) | 81.68246 | 81.68246098 |

All 47 predictions were non-empty. Source NPZs, sample manifest and metrics are under `results/smoke-v6/smoke-results/`. Verification: `verify_predictions.py`. Ten first-image comparisons (first target in each image, selected by position rather than quality): `results/smoke-v6/smoke-results/prediction-comparison.jpg`.

This is a **pipeline smoke check**, not a full-validation reproduction of the paper's score. Loader reports 1,500 val images; only 20 were evaluated. Text mode is `select_first`, expressions concatenated per image, as in upstream evaluator with `--concat`. One image can contain multiple target masks. These images are debug data and must not be reused as a locked holdout.

## Reproduction settings

- Source: `wusize/F-LMM`, SHA `47d3c2a6f628dd48ba76a1cfdf29b6b2aa8f9eb3`.
- Config: `configs/deepseek_vl/frozen_deepseek_vl_1_3b_chat_unet_sam_l_refcoco_png.py`.
- F-LMM weights: author-owned `wusize/flmm`, revision `d70e5b3289ca4255cb6a017a792870aad26c01ce`, exact matching checkpoint filename.
- Frozen backbone: `deepseek-ai/deepseek-vl-1.3b-chat`; revision not explicitly pinned in this run. This must be pinned and recorded for the larger evaluation.
- SAM: official ViT-L checkpoint. Hashes are in `checkpoints.json`.
- GPU: Tesla T4, GPU 0 only. Two T4s were available; they were not combined.
- Isolated Python 3.10, torch 2.1.2+cu118, transformers 4.39.1, mmcv 2.1.0, mmdet 3.3.0, mmsegmentation 1.2.2, xtuner 0.1.23. Full successful freeze: `results/smoke-v6/setup-report/freeze.log`.
- **Compatibility difference:** forced PyTorch SDPA math backend because BF16 memory-efficient attention failed on T4. Original model dtype, mask threshold and preprocessing retained. Do not compare runtime to an optimized upstream hardware setup.
- Checkpoint loading: no unexpected keys. Missing keys belong to separately loaded frozen DeepSeekVL weights and SAM image encoder; trained head keys present. Raw key list saved.
- Annotation: archive snapshot linked from the official REFER repository's broken-server guidance. ZIP SHA256 `7f924bb7ed8dc4568058e4ff626281918d56e5206f4c868c5a80f088f38c8bf0`. Original UNC host failed DNS. COCO image bucket accessed through valid HTTPS S3 endpoint.

## Observations and limits

- First-target examples include strong overlap on image 278549 (IoU ~0.923) and a clear wrong-region prediction on image 532505 (IoU 0.000). These are individual examples, not a quantified failure category.
- Loader loop took about 39 seconds for 20 images including first inference, preprocessing and artifact saving. This is not a warmed-up median/p95 latency measurement. Peak VRAM was not measured.
- bitsandbytes emitted a missing CUDA library warning. Quantization was not used; full selected inference path completed successfully.
- Kaggle CLI downloaded the artifacts but failed printing the final notebook log under Windows codepage (`charmap`). Stage logs, status, metrics, images and all 20 NPZs were downloaded and checked independently.

## Next experiment

1. Pin backbone revision and freeze an explicit pilot manifest (e.g. 100–300 images), excluding these 20 debug images.
2. Select a disjoint locked holdout manifest before tuning. Record both IDs and selection rule, not only loader positions.
3. Add peak VRAM and synchronized warmed-up latency measurement; save per-case metrics and overlays.
4. Run pilot, quantify a failure/cost category, then preregister one intervention. Full val is a separate reproduction milestone; confirm the matching paper table/model/settings before comparing scores.

No hypothesis about selective SAM refinement has been tested. No training was performed. All submitted GPU jobs have finished.
