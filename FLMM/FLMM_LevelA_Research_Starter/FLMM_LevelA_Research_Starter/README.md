# F-LMM Level A Research Starter — 2026-11-05

Read **PLAN.md** first. This is a research-planning and provenance starter adapted from the uploaded `AI-research-SKILLs.zip`, not an installation or a runnable pretrained F-LMM model.

## Files
- `PLAN.md`: dated plan, methodology, timeline, go/no-go gates.
- `research-state.yaml`, `research-log.md`, `findings.md`: two-loop experiment records.
- `literature/survey.md`: key paper matrix.
- `experiments/H0-baseline/protocol.md`: start with this protocol.
- `experiments/H1-conditional-refinement/protocol-DRAFT.md`: future hypothesis only, do not run before actual observation.
- `templates/observations.csv`, `templates/runs.csv`: data-recording templates.
- `templates/review-checklist.md`: six-dimensional evidence audit.
- `scripts/check_environment.py`: quick environment inspection; no model download and no GPU compute.

## Start today
1. Agree on baseline / confirm actual hardware; choose one team owner for reproduction.
2. Clone official F-LMM, record SHA; **do not** run untrusted code blindly.
3. Read official README and inspect config/dependencies; identify data/checkpoint locations.
4. Run `python scripts/check_environment.py` in proposed environment.
5. Fill `experiments/H0-baseline/protocol.md` before any evaluation.
6. Append each failed or successful attempt to `research-log.md`.

### Execution update — 2026-11-05
Kaggle API smoke run completed on Tesla T4: 20 RefCOCO UNC val images / 47 masks, cIoU 79.186941%, mIoU 81.682461%, independently verified from saved masks. This is a debug pipeline check, not full-val reproduction. See workspace `FLMM/kaggle/RESULTS.md` for environment, checkpoint hashes, compatibility changes, failures, and next steps. No training or H1 test performed.

### Full validation update — 2026-11-06
Gate0 passed, then full UNC val completed: **1500 images / 3811 selected masks**, cIoU **74.96344757%**, mIoU **76.97119141%**. All binary masks/CSV/latency independently verified; resumable execution and 20 low-IoU cases saved. See workspace `FLMM/fullval/REPORT.md`. This uses released `select_first + concat` semantics, not all10834 sentences. No training or improvement tested.
