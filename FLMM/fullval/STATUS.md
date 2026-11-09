# Full-validation task — COMPLETE

Completed and independently verified: **1500 images / 3811 masks**, cIoU74.96344757%, mIoU76.97119141%; median2.274s/image, p952.740s/image. Report: `REPORT.md`. All record/mask hashes, per-target CSV and latency verified; charts and 20 low-IoU cases generated. No more evaluation jobs required. Earlier active-progress details below are historical.

User authorized full RefCOCO UNC val: 1500 images / 3811 selected masks, metrics, per-target IoU, latency, failure cases and resume. Earlier stop-after-G0 instruction was superseded by this explicit request.

- G0 passed; phase1 engine/common hashes remain unchanged. Dedicated fullval drivers do not modify inference.
- CPU `hoofngocjhuy/flmm-full-data` v1 COMPLETE: reused 670 images, downloaded 830, all 1500 cached and verified.
- `hoofngocjhuy/flmm-full-1` v1 COMPLETE: 534 unique images / 1362 masks (prefix500 plus seeded probe outside prefix). Data in `runs/chunk1`.
- `hoofngocjhuy/flmm-full-2` v1 blocked stale quota snapshot (>6h after an interrupted chat); no new inference, all 534 retained. Data in `runs/chunk2-v1`.
- Refreshed quota; `hoofngocjhuy/flmm-full-2` v2 COMPLETE: 1021 images / 2596 masks, 487 newly evaluated, zero worker failures. Data in `runs/chunk2-v2`.
- `hoofngocjhuy/flmm-full-3` v1 finished1500 images /3811 masks, resuming1021 and evaluating only479 new images. Local verification PASS.
- Full fingerprint: `616b053ca5b50e70dbb24da6b973aa315dd4c1892c7be0217f177e60a0d272b7`.
- Last quota snapshot 2026-11-06 13:32UTC; >27 GPU hours available. No additional jobs needed unless a genuine failure requires resume.

## Completed verification workflow (reference)

1. Watch existing job3 via `python -X utf8 -m kaggle kernels status hoofngocjhuy/flmm-full-3` or logs --follow. Do not create duplicate jobs.
2. When finished, download only final metadata/CSV/latency/failure cases, worker metadata and `full-records.tar.gz`, not thousands of individual duplicate mask files. Suggested file regex: `^(full-records\.tar\.gz|results/(?!records/).*|workers/[01]/(worker-summary|regression-check|probe-ready|precision)\.json)$`, page-size200. Download destination `FLMM/fullval` puts `results/` and archive where local verifier expects them.
3. Run `python FLMM/fullval/verify_full.py` (safe tar extraction, all1500 hashes/masks and3811 CSV IDs, recompute official agreement/latency/failure-case order).
4. Run `python FLMM/fullval/plot_full.py` and `python FLMM/fullval/write_report.py`. Inspect plot and a few failure JPGs with view_image.
5. Verify all jobs COMPLETE/no active task GPU jobs, update research log/findings/README and this status. Final concise Vietnamese response with full scores, latency and links REPORT.md / CSV / FAILURE_CASES.md. Do not claim matching paper number or covering all10834 sentences.

## Notes

Sequential prefetch/hash fixed slow safetensors cold FUSE access: first model load802s; later prefetch27s then model load10–14s. Model/precision/threshold unchanged.

Results all use select_first + concat, BF16 DeepSeekVL + FP32 SAM/grounding, math attention on two independent T4 replicas. Full val consumes old internal holdout; no improvements/tuning/testA/testB.

Current final scripts are built/uploaded. `full_finalize.py` will archive exactly3000 files (1500JSON+1500NPZ) once full complete; local verifier expects this. Local plot/report scripts assert full verified completion, never generate invented results.
