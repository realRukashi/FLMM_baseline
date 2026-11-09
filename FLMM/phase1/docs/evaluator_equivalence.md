# Evaluator equivalence — PASS on 20 regression images

Official upstream control, original modified subset evaluator, and resumable engine produced identical binary masks/GT on the original 20 images. Metrics match within 0.001 percentage point. Controls share one audited model and math backend to avoid needless reload. Only selection/local paths are harnessed; exact diffs are retained. This is not an unmodified eight-split official benchmark run.

{
  "images": 1500,
  "raw_val_references": 3811,
  "raw_val_sentences": 10834,
  "retained_references": 3811,
  "selected_target_masks": 3811
}

Unit: first sentence per retained annotation/reference, grouped and concatenated by image. Other sentences are not evaluated. ref_id/sent_id are derived from annotation and checked against loader text and segmentation. See dataset_manifest.json.
