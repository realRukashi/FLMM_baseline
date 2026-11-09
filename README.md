# F-LMM baseline reproduction — Nhóm 10

**Tìm notebook Kaggle và hiểu từng version/file:** xem [Danh mục Kaggle & bản đồ repository](KAGGLE_RUNS_AND_FILES.md). Bắt đầu với [full validation cuối cùng](https://www.kaggle.com/code/hoofngocjhuy/flmm-full-3/1); các chặng trước và phiên bản lỗi đều được giải thích trong danh mục. Notebook chạy được cấu hình private, cần tài khoản có quyền để mở.

Reproduction and evaluation workspace for **F-LMM: Grounding Frozen Large Multimodal Models** (CVPR 2025), using DeepSeekVL-1.3B + SAM ViT-L on Kaggle.

This work evaluates referring **image** segmentation. Video grounding, training and method improvements have not been evaluated.

## Start here

- [Vietnamese nontechnical wrap-up](FLMM/fullval/WRAP_UP_VI.md)
- [Full validation report](FLMM/fullval/REPORT.md)
- [Failure case gallery](FLMM/fullval/FAILURE_CASES.md)
- [Evaluation and resume protocol](FLMM/fullval/PROTOCOL.md)
- [Gate 0 audit](FLMM/phase1/GATE0_AUDIT.md)

## Completed evaluation

RefCOCO UNC val: **1,500 images / 3,811 target masks**, using `select_first` + `concat` evaluator semantics. cIoU **74.9634%**, mIoU **76.9712%**. All saved masks and per-target results independently verified. These results are not a claim of matching a published paper table.

Median synchronized forward latency: **2.274 s/image** after warm-up. Two independent Tesla T4 workers; each image runs on one GPU. Peak allocated/reserved GPU memory: **8.26 / 9.13 GiB**. Timing excludes model loading, downloads and preprocessing/postprocessing.

## Repository contents

- `FLMM/fullval/`: full evaluation code, Kaggle notebooks, reports, per-target CSV, latency, failure visualizations, all prediction/GT records and their archive.
- `FLMM/phase1/`: audited evaluation engine, Gate 0 and intermediate run evidence.
- `FLMM/kaggle/`: initial environment setup and smoke experiments, including failed attempts.
- `FLMM/FLMM_LevelA_Research_Starter/`: research plan, findings, state and append-only research log.
- `FLMM/upstream/F-LMM/`: upstream source snapshot, with its original license and notices retained.

The notebooks package Python scripts; read the protocol before launching. Resume uses previously committed Kaggle outputs. No GPU job is automatically started by cloning this repository.

Local verification (requires NumPy and Pillow):

```bash
python FLMM/fullval/verify_full.py
```

Large model weights, original image datasets, credentials, Python caches and nested Git metadata are not included. Prediction/GT evaluation artifacts and example visualizations are included for auditability.

## Attribution

Original implementation: [wusize/F-LMM](https://github.com/wusize/F-LMM), source commit `47d3c2a6f628dd48ba76a1cfdf29b6b2aa8f9eb3`. The upstream snapshot remains subject to its [S-Lab License](FLMM/upstream/F-LMM/LICENSE); it is not relicensed here. Third-party datasets and model weights retain their respective terms.
