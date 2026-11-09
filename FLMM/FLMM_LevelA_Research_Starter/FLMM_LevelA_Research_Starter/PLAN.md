# F-LMM LEVEL A: RESEARCH EXECUTION PLAN

> Execution update — 2026-11-06: Gate0 passed and full RefCOCO UNC val completed1500 images/3811 selected masks, cIoU74.96344757%, mIoU76.97119141%. All masks/CSV/latency independently verified. See workspace `FLMM/fullval/REPORT.md`; the dated proposed plan below is retained as history. No improvement hypothesis has been tested.

**Date:** 2026-11-05  | **Target first cycle:** 2026-11-15  | **Long-stop:** 2026-12-31  
**Status:** PROPOSED / NOT YET RUN  | **Owner:** group of four (unassigned)

> Source discipline: This is a plan derived from the uploaded AI Research Skills collection (especially autoresearch, brainstorming-research-ideas, ara-research-manager, rigor-reviewer, segment-anything, accelerate and ml-paper-writing), professor's transcript, and official F-LMM / MMDetection documentation. Any technical statement without a measured result is a hypothesis, not an observation. No GPU evaluation has been executed for this starter pack.

## 0. Decision and scope

**Working topic:** Reproducing and Analyzing F-LMM for Referring Image Segmentation (with video grounding as later extension).

**Primary checkpoint/config:** F-LMM with DeepSeekVL-1.3B, `configs/deepseek_vl/frozen_deepseek_vl_1_3b_chat_unet_sam_l_refcoco_png.py`, trained F-LMM `.pth`, pretrained DeepSeekVL and SAM ViT-L weights. The trained F-LMM checkpoint is NOT equivalent to the pretrained DeepSeekVL base model.

**Primary task:** Referring Expression Segmentation (RES), image + referring expression -> predicted binary mask.

**Primary benchmark:** RefCOCO / UNC validation, using official paper evaluation script and cIoU as the primary reproduction metric. A subset result is labelled 'subset' and never compared directly with a full-benchmark number as equivalent.

**Not in first cycle:** training backbone, new video benchmark, user-trust survey, long-context reasoning, fine-tuning a 7B model, claiming novelty from frame sampling, building an app.

**Scientific distinction:** F-LMM's paper contributes frozen-LMM grounding on images. The professor proposed video as an application/extension, not as a confirmed official F-LMM capability.

## 1. Adapted AI Research Skills — what is actually used

| From ZIP | Practical application now | Defer/avoid |
|---|---|---|
| `0-autoresearch-skill` | research-state.yaml, log, findings; inner experiment loop + outer reflection; preregister protocol | No unattended agent cron / autonomous background runs |
| `21-research-ideation/brainstorming-research-ideas` | boundary probing, simplicity test, problem-first after observations | No endless speculative ideation before baseline |
| `21-research-ideation/creative-thinking-for-research` | examine alternatives only after confirmed bottleneck | No grand scope expansion |
| `22-agent-native-research-artifact/research-manager` | provenance: user vs AI suggestion; append-only decision/error trail | Full ARA schema not needed in sprint #1 |
| `22-agent-native-research-artifact/rigor-reviewer` | audit evidence relevance, falsifiability, scope, logic, dead-ends, experimental design | Do not claim peer-reviewed outcome |
| `18-multimodal/segment-anything` | understand SAM masks/prompts; profile mask refiner | SAM standalone is not text grounding |
| `08-distributed-training/accelerate` | reproducible single-GPU evaluation setup; device and seed checks | No 8-GPU distributed training |
| `13-mlops/mlflow` | metadata + parameter/metric/artifact tracking pattern (CSV is enough) | Full MLflow server optional |
| `20-ml-paper-writing/academic-plotting` | per-case error examples; axes/units; evidence-quality figures | Don't produce charts from invented numbers |
| `20-ml-paper-writing/ml-paper-writing` | precise claims with verifiable citations; concise negative-results report | Not a publication sprint |

Other 80+ skills (RLHF, distributed training at scale, serving, RAG, fine-tuning etc.) do not address the current task and are deliberately not activated.

## 2. Hard gates and dates (Vietnam time)

| Date | Phase / gate | Required evidence | Stop/pivot rule |
|---|---|---|---|
| **05–06 Nov** | G0: paper/repo/data/hardware audit | official source URLs, commit SHA, current `nvidia-smi`, package versions, disk/VRAM estimate, checkpoint links and licenses | Do not run training; verify GPU choice in actual Kaggle UI |
| **06–08 Nov** | G1: load model / smoke test | one image + expression -> readable mask, exact command/log, memory usage, run ID | If load fails, capture reproducible traceback; timebox environment debugging, decide alternative by 08–09 Nov |
| **08–10 Nov** | G2: baseline eval | 20–50 debug samples then disjoint fixed pilot subset; official eval metric; raw prediction IDs + cIoU where possible | Missing GT / invalid split -> stop claims about accuracy and fix dataset |
| **10–12 Nov** | G3: observation | ≥1 quantified weakness/cost + 10 representative cases, potential confounds, baseline latency/VRAM | Do not announce an improvement idea as confirmed failure |
| **12–14 Nov** | G4: preregister then test one change | `protocol.md` committed before result; control vs variant same data/config; no tuning on final holdout | A result that is worse/inconclusive still counts as tested |
| **15 Nov** | G5: review mini-report | before/after table, method, examples, caveats, reproducible commands, next decision | If early gates failed, report blocker + actual work; use longer Level A window to 31 Dec |

**Fallback ladder:** F-LMM + DeepSeekVL-1.3B on an available GPU → stable local Linux GPU 24GB+ if accessible → another *paper-backed* accessible grounding baseline with its own evaluation. A pivot must be logged, and results after a pivot must not be labeled as reproduction of F-LMM.

## 3. Execution: Kaggle versus local

1. **Keep code and metadata in Git locally.** Put large checkpoints/datasets outside Git (and outside this ZIP). Use README source URLs, checksums if available, and private cloud storage as needed.
2. **Kaggle is the first feasibility option only if no stronger local GPU is available.** Inspect accelerator currently offered to the account, available quota, disk/network and verify PyTorch/CUDA before installing anything. Dual GPU 16GB does **not** imply single-model 32GB memory.
3. **Prefer persistent Linux machine with ~24 GB+ VRAM** if already available; this is a conservative planning preference, not a verified minimum for F-LMM.
4. **Compatibility risk:** official F-LMM README calls out `transformers==4.39.1` and dependencies Xtuner, MMSegmentation, MMDetection, Accelerate. Do not `pip install -U` blindly. Save a pip freeze on successful load.
5. Run one process first, batch size 1 where supported; use exact official config and data paths. Do not assume model parallelism or 4-bit loading is supported without inspecting implementation.

### First commands (diagnostic only)

```bash
# Environment snapshot
nvidia-smi
python - <<'PYCODE'
import torch, platform
print(platform.platform())
print('torch', torch.__version__, 'CUDA', torch.version.cuda)
print('GPU count:', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_properties(i).total_memory / 2**30)
PYCODE

# Official source, pin revision after review
git clone https://github.com/wusize/F-LMM.git
cd F-LMM
git rev-parse HEAD
```

Once all dependencies, checkpoint files, config references and RefCOCO annotations are checked, the **README's official RES evaluation command** is:

```bash
export PYTHONPATH=.
accelerate launch scripts/multiprocess_eval_refcoco.py \
  configs/deepseek_vl/frozen_deepseek_vl_1_3b_chat_unet_sam_l_refcoco_png.py \
  --checkpoint checkpoints/frozen_deepseek_vl_1_3b_chat_unet_sam_l_refcoco_png.pth --concat
```

This command has **not** been executed for this plan, and may evaluate more data than desired. Before using a subset, inspect script and config to learn how to subset properly and preserve sample IDs.

## 4. Data protocol

**Use official MMDetection RefCOCO preparation** (see sources). Required files include `data/coco/refcoco/instances.json`, `data/coco/refcoco/refs(unc).p` and matching `train2014` images. Inspect the actual loader/config before downloading entire COCO/PNG resources; it may require more paths at initialization.

- **Debug:** ~20–50 samples, openly used for pipeline fixes. No scientific inference.
- **Pilot:** fixed sample IDs from official validation for observation and tuning; choose sample counts based on compute (e.g., 100–300). Record sampling rule, not just a number.
- **Locked evaluation:** disjoint, pre-selected held-out subset of val (e.g., 100–300); evaluate once after selecting variation. Label as *held-out subset of RefCOCO val*, not the original official test set.
- **Official full validation:** run when hardware/compute permits; only this setting is comparable to the reported full-val cIoU if all preprocessing, checkpoint and code match.
- Record any dataset-specific licenses and avoid posting raw data copies without permission.

**Primary metric:** the official script's cIoU; check if cIoU is computed as global intersection over global union. **Secondary:** per-example IoU and failure cases (only if prediction masks and GT can be aligned), median and p95 warmed-up latency (ms/image), GPU peak allocated memory (GiB), GPU hardware and dtype. Report cold model load separately.

Never conflate cIoU with averaged per-sample IoU; never compare numbers from unequal splits/datasets or unpaired hardware settings.

## 5. Observation plan (only after valid baseline)

1. **Quality:** inspect masks on small objects, multi-instance/distraction scenes, location relation, occlusion, long expressions. Counts by category; verify each label available or manually classify as exploratory.
2. **Cost:** profile LMM backbone, mask decoder, SAM refiner separately if code allows; report actual timing not guesses.
3. **Sanity:** check image scaling/normalization, target mapping, non-empty masks and data leakage, and if incorrect predictions are simply due to a missing image or annotation mismatch.
4. **Look for an actionable bottleneck:** one quantified observation that can produce a 1-variable intervention under the same protocol.

**Candidate hypotheses, explicitly NOT findings:**

- H1: A small subset of samples derives most of its benefit from SAM refinement; a confidence-based gate could reduce mean latency at limited accuracy loss. Risk: confidence may be uncalibrated; existing ablations already study refinement.
- H2: Resize/crop strategy changes small-object cIoU versus latency. Risk: resizing may violate original preprocessing; needs apples-to-apples control.
- H3: Query complexity predicts failures more than image content alone. Risk: RefCOCO semantics and wording may confound categories; requires controlled groups.

Choose just one H after observation. A simple ablation without new invention is acceptable for Level A; never advertise novelty prematurely.

## 6. Experimental protocol: integrity rules

- Write prediction/falsification/primary metric/comparison dataset **before** seeing new result; commit protocol separately from results (autoresearch SKILL).
- Fixed source SHA, checkpoint hash or identifier, preprocessing config, device, dtype, batch size, seeds and sample IDs. If nondeterministic, repeat and report spread when feasible.
- Test only one meaningful variable at a time. Use paired comparisons on same sample IDs.
- For performance, warmup and CUDA synchronize, report timing in ms with device details; separate cold start from per-image runtime.
- Quantitative claims must be supported by raw logs/metrics, not a few cherry-picked example masks.
- Track both successes and failures. A negative result is acceptable and can complete thầy's Level A cycle.
- Apply six rigor-review questions: evidence relevance, falsifiability, scope, argument coherence, exploration integrity, methodological rigor.

## 7. Team of four: one baseline, parallel responsibilities

- **Owner A — Environment/Model:** checkout, dependencies, checkpoint integrity, GPU measurements, official evaluation command.
- **Owner B — Data/Evaluation:** RefCOCO files/splits, metric semantics, IDs, quantitative results, leakage checks.
- **Owner C — Observation:** visualizations, per-case failure taxonomy, profiling, drafts one protocol with A/B.
- **Owner D — Literature/Review:** compare F-LMM with SAM 2 / Grounded SAM 2 / VideoGLaMM / DecAF; verify claims, report quality and documentation.

All four share a baseline run and review one another's evidence. Don't ask four people to independently rebuild four large baselines during the first cycle.

## 8. Go / No-go decision

**GO** when: checkpoint loads; exact data loader resolves files; a mask is produced; subset metric computed; results and metadata persist for reproducibility.

**NO-GO for F-LMM within this sprint** when: compatible environment/checkpoint access or GPU memory remains blocked after a timeboxed attempt. Publish the blocker and choose a realistic, paper-backed backup. The level-A cycle can then continue on the backup, but do not conflate results.

**DONE (Level A #1)** when: chosen paper/base is reproduced at an honestly scoped level; an observation is documented; one hypothesis is tested against control; result is evaluated and explained (positive, negative, or inconclusive).

## 9. Official references

- F-LMM GitHub: https://github.com/wusize/F-LMM
- F-LMM CVPR 2025: https://openaccess.thecvf.com/content/CVPR2025/html/Wu_F-LMM_Grounding_Frozen_Large_Multimodal_Models_CVPR_2025_paper.html
- MMDetection RefCOCO preparation: https://github.com/open-mmlab/mmdetection/blob/main/docs/en/user_guides/dataset_prepare.md
- SAM official repo: https://github.com/facebookresearch/segment-anything
- DecAF ICLR 2026 (related): https://proceedings.iclr.cc/paper_files/paper/2026/hash/2fc6b8a3fc23108f184daa4759024c25-Abstract-Conference.html

## 10. Scope / evidence warning

- No code/models/datasets from F-LMM are embedded in the starter pack.
- Reproduction is **not yet verified**; there are no generated benchmark scores.
- SKILL documents are methodology templates, not proof a particular command, dependency version or optimization works on Kaggle.
- Do not install all 98 skills or execute unreviewed scripts. Incompatible autonomy instructions about unattended recurring work are intentionally not applied.
