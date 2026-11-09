# Findings

## Research question
Can we reproduce the pretrained F-LMM DeepSeekVL-1.3B grounding pipeline on RefCOCO, and what measurable limitation is worth testing?

## Current understanding
Full RefCOCO UNC val execution completed on Kaggle using the G0-audited source/config, pinned checkpoints and math attention: 1,500 images / 3,811 selected first-sentence masks. cIoU 74.96344757%, mIoU 76.97119141%; every saved binary mask/record and per-target CSV row was independently verified locally. This is the released evaluator semantics, not all 10,834 sentences and not a claim of matching a paper table. See workspace `FLMM/fullval/REPORT.md`.

## Key results
- Full validation (2026-11-06): cIoU 74.96344757%, mIoU 76.97119141%, 1500 images / 3811 masks. Global intersection 103215624, union 137687931. Median forward 2.274s/image, p95 2.740s/image on two independent T4 replicas; cold load excluded. Full archive and CSV independently verified. All 3811 predictions non-empty; 507 target IoUs below0.5, including112 zero-IoU cases. Twenty deterministic low-IoU visualizations saved.
- One-image smoke check (not a benchmark estimate): image ID 376848, two target masks, cIoU 88.257446%, mIoU 88.271767%. Both predicted masks non-empty. Independently recomputed from saved NPZ and matched official RefSegMetric output. Evidence: workspace `FLMM/kaggle/results/smoke-v5/smoke-results/`.
- This proves the pipeline can execute on Kaggle T4 with the documented environment/backend change. It does not reproduce the paper's full benchmark score.
- Smoke v6 completed 20 val images / 47 target masks: cIoU 79.186941%, mIoU 81.682461%, independently recomputed from saved masks; all predictions non-empty. These first 20 loader images are debug data, not a representative or locked evaluation set. Evidence and limitations: workspace `FLMM/kaggle/RESULTS.md`.

## Patterns and insights
Observed low-IoU examples include wrong-instance selection. These are descriptive cases, not a quantified causal taxonomy or evidence for selective SAM refinement. Full validation has now exposed the former internal holdout; do not treat it as untouched for future tuning.

## Lessons and constraints
- Official repository warns `transformers` changes beyond 4.40 can affect reproduction.
- Actual full-run peak allocated GPU memory was 8870020608 bytes; BF16 backbone with FP32 head/SAM, math attention. Timing is synchronized forward-only after warm-up.
- Sequential checkpoint prefetch/hash reduced observed later cold loading from ~802s to ~10–14s plus ~27s prefetch. This changes I/O only.

## Open questions
- Which quantified failure category is appropriate for the next observational study?
- Which published table/settings match this released first-sentence evaluation protocol?
- Which untouched evaluation split should assess a future selected improvement?

## Experiment trajectory
Environment audit v1 completed; dependency v1 failed and v2 passed; smoke v1–v3 encountered data/checkpoint download errors; smoke v4 loaded the model but failed at attention kernel; smoke v5 succeeded on one image; v6 succeeded on 20 images / 47 masks. All failures retained in workspace `FLMM/kaggle/results/`.
Gate0 passed on 2026-11-06, then user authorized full validation. Three successful resumable chunks completed1500 images; one stale-quota gate attempt preserved all prior records. Full artifacts: workspace `FLMM/fullval/`.
