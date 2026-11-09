# Feasibility — UNVERIFIED pending 50-image probe

Existing smoke: 20 images / 47 masks, about 39 seconds in the loader/inference/write loop. This is not isolated warmed-up latency. Linear extrapolation is planning information only.

The new CPU job prepares reusable environment, exact model weights, annotations and selected images before GPU scheduling. The GPU notebook restores these inputs, audits G0, then measures a seeded 50-image probe. It records cold loading, warm-up, synchronized inference median/p95/max, preprocessing/postprocessing, peak allocated/reserved VRAM, RAM and per-sample output size. A fresh go/no-go uses measured values, quota, free disk, safety factor and shutdown reserve.

No large benchmark should start while G0 or the probe is unresolved. Current code defaults to 100 stability images at most; development and holdout are defined but not evaluated.
