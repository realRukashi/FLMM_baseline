# F-LMM Kaggle execution

All kernels are private under `hoofngocjhuy`. Credentials are read by Kaggle CLI from the user's existing config, never copied into this directory or uploaded.

- `environment/`: GPU/runtime audit (completed).
- `setup/`: isolated Python 3.10 environment and full inference import check.
- `prepare_evaluator.py`: generate a small diff against pinned official evaluator.
- `evaluator/`: generated subset evaluator and diff. Use `--concat --max-samples 1` for first run.
- `results/`: downloaded run logs and artifacts.

Pinned F-LMM source: `47d3c2a6f628dd48ba76a1cfdf29b6b2aa8f9eb3`.

## Observed runtime

Audit v1: two Tesla T4 GPUs, 15,636,037,632 bytes each; default Python 3.13.16, torch 2.11.0+cu130, transformers 5.20.2. GitHub/Hugging Face connectivity works. L4 request resulted in T4 hardware; report observed hardware instead of requested accelerator.

PyTorch's bf16 capability probe returned true on this runtime, but that alone does not establish native T4 bf16 performance or F-LMM operator compatibility. Isolated runtime must test actual operations and inference.

## Attempts

1. Environment audit: complete; raw evidence in `results/environment-v1`.
2. Dependency check v1: Python 3.10, torch 2.1.2/cu118, mmcv 2.1.0 and package installation succeeded. F-LMM import failed because Xtuner imports DeepSpeed, which was absent. bitsandbytes also reported CUDA library search failure. Raw evidence in `results/setup-v1/setup-report`.
3. Dependency check v2: imports passed and BF16 matmul succeeded. DeepSpeed 0.14.2 installed with native-op build disabled. bitsandbytes library warning persists; no quantization used.
4. Smoke v1 submitted: repeats validated setup, downloads official RefCOCO annotations and one val image, SAM and trained F-LMM checkpoints, then evaluates one image on GPU 0. Result pending. Build with `python FLMM/kaggle/build_smoke.py` from workspace root.
5. Smoke v1 stopped at RefCOCO annotation DNS failure. Official REFER README acknowledges broken server and links issue #14; switched to its Internet Archive snapshot for v2.
6. Smoke v2 downloaded annotation but stopped at COCO image TLS hostname mismatch. Switched to HTTPS S3 path of the same public COCO bucket, validated locally. v3 submitted.
7. Smoke v3 passed image and SAM download, but trained F-LMM Google Drive link failed (gdown retrieval failure on Kaggle; public link returned HTTP 404 locally). Discovered author-owned Hugging Face `wusize/flmm`, exact matching checkpoint filename, revision `d70e5b3289ca4255cb6a017a792870aad26c01ce`. v4 uses this source.
8. Smoke v4 loaded model and reached prediction on RefCOCO val image ID 376848; failed in BF16 SDPA CUTLASS kernel on Tesla T4 (`cutlassF: no kernel found to launch!`). No masks/metrics produced. v5 forces PyTorch math SDPA backend; model dtype and preprocessing stay unchanged. This backend difference is part of the reproduction record.

Verified checkpoint SHA256: F-LMM `02bdf9e878847c68acaa207a916291cb1568ab119cbc658a4dfa94f399dfed0e`; SAM `3adcc4315b642a4d2101128f611684e8734c41232a17c648ed1693702a49a622`. See raw manifests under `results/smoke-v4/smoke-results`.

Smoke v5 successfully produced two target masks on val image 376848. Smoke v6 completed 20 images / 47 masks: cIoU 79.186941%, mIoU 81.682461%, independently recomputed. This is a pipeline smoke check, not full benchmark reproduction. See `RESULTS.md` for evidence and limitations. All submitted GPU jobs finished. A Kaggle run marked COMPLETE can still contain a stage failure: inspect `status.json`, not only kernel status.
