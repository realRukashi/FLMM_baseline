# F-LMM Phase 1 evaluation protocol — fixed before new results

Date: 2026-11-06 (ICT). No method improvements, training, threshold tuning, precision changes or resizing experiments in Phase 1.

## Gates and interpretation

1. **G0:** prove checkpoint tensor initialization/loading and conditional evaluator equivalence on the original 20 debug images. Gate is FAIL/BLOCKED until measured, never inferred from import success or key prefixes alone.
2. **Probe:** 50 seeded images, outside the original 20; CUDA-synchronized timings after separate warm-up, peak VRAM/RAM, output bytes, cold load time and data preparation cost.
3. **G1:** measured p95 plus preprocessing/postprocessing, 1.5x safety factor, 20-minute shutdown reserve, disk floor and observed quota. Unknown critical resources cap scope conservatively.
4. **Stability/resume:** nested 100-image set, first job commits at 75, next session mounts its completed output and finishes the remaining 25. Match full fingerprint and file checksums before skipping.
5. **Larger baseline:** only after G0/G1/G2 and a fresh budget gate. Full validation additionally requires all images cached. Development/holdout/full disabled in the initial notebook. Disabled stages are UNVERIFIED, not PASS.

## Fixed computation

- Official source commit: `47d3c2a6f628dd48ba76a1cfdf29b6b2aa8f9eb3`.
- Original DeepSeekVL-1.3B F-LMM config, SAM ViT-L, original resize/postprocess and threshold 0.5, batch one image.
- Backbone pinned to `8f13a8e00dbdc381d614a9d29d61b07e8fe91b3f` and verified against its safetensors LFS SHA256.
- F-LMM trained checkpoint revision `d70e5b3289ca4255cb6a017a792870aad26c01ce` and original successful SHA256; SAM hash unchanged.
- Local cache paths replace URLs; inference runs with Internet disabled and Hugging Face offline.
- Tesla T4 math SDPA backend; BF16 backbone and original FP32 modules retained. No claim of native T4 BF16 tensor-core optimization. Module dtypes and actual language attention class are recorded.
- Random seed `20261106`; no model or threshold selection on pilot/holdout results.

## Evaluation unit and sample selection

The installed `mmdet==3.3.0` loader and annotation files are authoritative. The manifest records actual image/reference/sentence/annotation counts; it does not assume they are equal. `select_first` uses the first sentence of each retained reference, and `--concat` combines them in one image inference. Multiple references pointing to the same annotation may be collapsed by the loader. All excluded sentences remain outside this protocol's coverage.

Map mask position back to `image_id`, `dataset_index`, `ref_id`, `sent_id`, lowercased expression, annotation ID and prediction ID by reproducing the loader's mapping and verifying both text and polygon/RLE identity. Missing source fields are null, never invented.

Preserve original 20 images as regression only. Exclude them from a seeded simple random shuffle of sorted remaining image IDs. First 250 become development; next 400 become holdout. The first 50/100 development IDs form nested feasibility/stability sets. Describe object-count distributions; random selection is not a claim of balanced sampling. Holdout is not used for tuning or exploratory example selection. TestA/testB remain unused.

## Resume and aggregation

Stable sample ID: `refcoco-val-{image_id}`. Each output records source/config/engine/checkpoint/annotation/manifest fingerprint, sample hash, mask file hash, intersection/union arrays and metadata. Atomic NPZ followed by atomic JSON; validate before DONE. Invalid/incomplete/mismatched results rerun or block integrity verification. Event trail is append-only. Duplicate sample IDs cannot be aggregated.

cIoU = 100 * sum(intersection) / sum(union). mIoU averages per-selected-mask IoU, checked against RefSegMetric. Do not average chunk cIoUs. Per-expression latency is explicitly amortized image latency divided by selected masks, not independent single-expression execution.

Every 25 new images creates an atomic export-ready archive. `/kaggle/working` alone is not a cross-session durability guarantee. Completed Kaggle Notebook Output is the durable boundary; bounded separate jobs and the resume test establish that boundary. `/kaggle/temp` is scratch only.

## Failure policy

Network preparation: three attempts, bounded read timeout/backoff, overall worker deadline, image decode checks, checkpoint hashes. No network in GPU evaluation.

Stop on first inference error/OOM; preserve logs and completed samples. Never silently reduce resolution, dtype, batch or threshold. Stop before time reserve or disk floor. Unused stages retain UNVERIFIED status. A failed stage must surface failure in notebook execution; earlier valid artifacts remain usable.

## Budgets and source caveats

Operational defaults: walltime 120 min, evaluation 75 min, reserve 20 min, disk floor 3 GiB. These are user-selected operating limits, not model minima or guaranteed Kaggle quotas.

Quota was measured through official Kaggle API on 2026-11-06: 28.49/30 GPU hours remaining at snapshot. Recheck at submission and treat stale snapshots as UNKNOWN. Kaggle's dynamic docs did not expose a verifiable current platform maximum through the available reader; scheduling relies on the explicitly requested 7,200-second API timeout and a stricter 100-minute processing cutoff, not an assumed 12-hour limit.

Sources: [F-LMM](https://github.com/wusize/F-LMM), [Kaggle notebook docs](https://www.kaggle.com/docs/notebooks), [Kaggle CLI](https://github.com/Kaggle/kaggle-cli), [author checkpoint repository](https://huggingface.co/wusize/flmm), [backbone](https://huggingface.co/deepseek-ai/deepseek-vl-1.3b-chat).
