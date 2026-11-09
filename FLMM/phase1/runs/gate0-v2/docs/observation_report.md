# Phase 1 observations

{
  "regression": {
    "status": "PASS",
    "expected_images": 20,
    "image_count": 20,
    "mask_count": 47,
    "intersection": 1378708,
    "union": 1741080,
    "cIoU": 79.18694143864728,
    "mIoU": 81.68246098179696
  },
  "probe": {
    "status": "UNVERIFIED",
    "expected_images": 50,
    "image_count": 0,
    "mask_count": 0,
    "cIoU": null,
    "mIoU": null
  },
  "stability": {
    "status": "UNVERIFIED",
    "expected_images": 100,
    "image_count": 0,
    "mask_count": 0,
    "cIoU": null,
    "mIoU": null
  }
}

Metrics were independently recomputed from saved binary masks. Probe and stability are nested development data, not independent estimates. Holdout is untouched unless explicitly enabled and passed. No method improvement or confidence gate was selected. Memory, timing and output sizes are in profiling.json; do not equate amortized expression time with independent per-expression inference.
