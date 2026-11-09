# Full RefCOCO UNC validation — protocol and resume

Authorized on 2026-11-06 after Gate 0 audit. Target: all 1,500 UNC val images / 3,811 selected target masks. No method improvement, training, threshold tuning or precision/resolution changes.

## Computation

The inference engine and common aggregation/resume primitives retain the exact hashes recorded by Gate 0. Each build and GPU controller checks them against the G0 evidence. The source commit, backbone revision, trained F-LMM and SAM checkpoint hashes, preprocessing, `select_first`, `--concat`, threshold 0.5 and math attention backend are unchanged.

Two independent Tesla T4 replicas evaluate disjoint image IDs. This is data parallel inference, not combined GPU memory. Every selected target keeps its original reference/sentence/annotation ID and mask index. Both replicas first verify bitwise agreement with the saved 20-image G0 regression outputs.

The full output covers one first sentence per reference, **not all 10,834 sentences**. This is the official released configuration/evaluation semantics with an audited resumable execution driver; it is not an unmodified eight-split upstream script invocation.

## Caching and budget

The existing CPU cache supplies environment/model/checkpoints and 670 images. A separate CPU job downloads only the remaining 830 images, validates decoding/checksums, and publishes a complete 1,500-image cache. GPU inference has Internet disabled.

Each GPU worker runs the fixed seeded feasibility probe (50 images total across replicas). These predictions are part of full validation and reused, not counted twice. The controller gates further work using measured p95, a 1.5x time margin, disk floor, 20-minute export reserve and a recent quota snapshot.

Sequential checkpoint prefetch/hash was added after the first chunk's slow cold load. It changes I/O scheduling only; model tensors/computation remain unchanged. Raw prefetch and model-loading times are retained.

## Durable boundaries and resume

Jobs target prefixes 500, 1,000 and 1,500 in the official loader order. The initial seeded probe can include images beyond those prefixes; reports always state the actual unique completed image/mask count. Every image is assigned to exactly one replica by dataset index modulo worker count.

Workers atomically save compressed prediction/GT arrays followed by a hashed DONE record. A completed prior Kaggle Output is mounted by the next job. A record is skipped only after validating its sample hash, computation fingerprint, record hash and mask-file checksum. Conflicting duplicate outputs cause failure. Global metrics use sums of intersections/unions, never mean chunk cIoUs.

Private per-worker snapshots are written every 25 new images; completed Kaggle chunk outputs form the guaranteed cross-session boundary. If an infrastructure kill prevents the current job from committing any artifact, the last committed chunk remains resumable and only the uncommitted work may need repetition. Ordinary notebook disconnects do not stop the remote job.

To retry an incomplete chunk, attach its committed partial output as `FULL_PRIOR`, retain the same complete data/cache and computation source, and set the desired `FULL_END_INDEX`. The driver validates all prior DONE records before resuming. Refresh `quota_snapshot.json` immediately before submission; stale snapshots intentionally block expansion.

## Output interpretation

- `baseline_metrics.json`: official RefSegMetric and independently recomputed global metrics; PASS only at exactly 1,500 images / 3,811 masks.
- `per_target_iou.csv`: reference/sentence/annotation mapping, IoU and image inference time. Per-target time is an amortized share of image time, not isolated expression execution.
- `latency.json`: CUDA-synchronized per-image forward latency after separate warm-up, preprocessing/postprocessing, memory and per-image records. Downloads, cache restoration and model construction are separate.
- `failure_cases.json` and `failure_cases/`: deterministically selected 20 lowest target IoUs, with input/GT/prediction overlays. These are descriptive low-IoU examples, not an established causal taxonomy.
- `full-records.tar.gz`: every atomic record and compressed binary prediction/GT mask, for independent local verification.
- `full_summary.json`, worker journals and per-chunk metadata: exact coverage, failures, reuse and provenance.

Full validation now includes the previously reserved internal holdout. It must not be treated as untouched for later improvement selection. No improvement or tuning is performed in this run; official testA/testB remain outside this task.
