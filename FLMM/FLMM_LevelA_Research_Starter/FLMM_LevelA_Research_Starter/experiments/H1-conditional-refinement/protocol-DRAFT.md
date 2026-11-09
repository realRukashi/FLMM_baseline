# H1 — Selective SAM refinement (candidate ONLY)

> DO NOT START until H0 baseline is reproduced and Observation documents a refiner latency/accuracy tradeoff. Existing F-LMM paper already ablates refiner performance; a Level A repeat is valid but is not a novelty claim.

- **Claim to test:** gating expensive refinement based on *a clearly specified coarse-mask statistic* reduces mean latency, with a pre-agreed maximum cIoU loss relative to always-refine.
- **Observation evidence:** TODO (link to actual baseline profiling and error cases).
- **Baseline:** original always-refine configuration (same checkpoint and data).
- **Control ablation:** no-refinement variant if supported without code mistakes.
- **Proposed intervention:** simple gate based on coarse mask confidence/uncertainty or area (choose after validation; do not leak ground-truth into inference).
- **Pre-registered thresholds:** TODO; tune only on pilot, not locked holdout.
- **Primary endpoint:** paired cIoU change and paired warmed-up latency change on locked sample IDs.
- **Falsification criterion:** predefined accuracy loss bound violated or negligible speed gain under matched conditions; the threshold is to be chosen before final eval.
- **Critical confounds:** confidence calibration, varying image size, CUDA asynchronous timings, incorrect refiner bypass.
- **Status:** `ai-suggested`, not a proven limitation or selected direction.
