# Danh mục Kaggle, lịch sử version và công dụng các file

Đối chiếu ngày 09/11/2026: danh sách notebook lấy qua Kaggle API của `hoofngocjhuy`; diễn giải từng version dựa trên log, metadata và artifact trong repository. Không chạy lại inference hoặc đổi quyền chia sẻ notebook khi lập danh mục này.

## Mở gì trước?

- Muốn xem lần đánh giá đầy đủ: [Full 3, v1](https://www.kaggle.com/code/hoofngocjhuy/flmm-full-3/1) và [báo cáo kết quả](FLMM/fullval/REPORT.md).
- Muốn hiểu bằng ngôn ngữ dễ đọc: [wrap-up tiếng Việt](FLMM/fullval/WRAP_UP_VI.md).
- Muốn xem máy sai thế nào: [20 failure cases](FLMM/fullval/FAILURE_CASES.md).
- Muốn hiểu quá trình từ đầu đến cuối: đọc bảng version bên dưới theo thứ tự.

Các notebook thực thi được metadata cấu hình **private**. Có link không đồng nghĩa mọi thành viên đều có quyền mở; artifact đã tải về nằm trong repo. Link `/1`, `/2`, ... chỉ version cụ thể; link không có số mở trang notebook hiện tại. File notebook/script trong repo là bản đang lưu, **không phải bản source riêng của mọi version lịch sử**. Những thư mục `results/` và `runs/` lưu bằng chứng theo version; một số lỗi sớm chỉ có log được ghi nhận, không có bộ output đầy đủ.

## Notebook, version và vai trò

Version là một lần lưu/chạy lại cùng notebook; batch là dữ liệu trong một lượt model xử lý. Các chặng full validation dùng cùng model/protocol, hợp thành một lần đánh giá toàn bộ. Không lấy trung bình điểm các chặng.

| Notebook / version | Làm gì và kết quả | Source / bằng chứng trong repo |
|---|---|---|
| [Environment audit v1](https://www.kaggle.com/code/hoofngocjhuy/flmm-environment-audit/1) | Kiểm tra phần cứng/runtime thực nhận: hai T4 và môi trường mặc định. Chưa chạy model. | [Script audit](FLMM/kaggle/environment/audit.py), [output](FLMM/kaggle/results/environment-v1/) |
| [Dependency check v1](https://www.kaggle.com/code/hoofngocjhuy/flmm-dependency-check/1) | Dựng môi trường Python/PyTorch tương thích; import F-LMM thất bại do thiếu DeepSpeed. | [Script setup hiện tại](FLMM/kaggle/setup/setup.py), [output v1](FLMM/kaggle/results/setup-v1/) |
| [Dependency check v2](https://www.kaggle.com/code/hoofngocjhuy/flmm-dependency-check/2) | Bổ sung DeepSpeed, kiểm tra import và phép toán BF16 thành công. Chưa phải đánh giá model. | [output v2](FLMM/kaggle/results/setup-v2/) |
| [Smoke v1](https://www.kaggle.com/code/hoofngocjhuy/flmm-refcoco-smoke/1) | Thử một ảnh; dừng ở tải annotation do lỗi DNS. | [output v1](FLMM/kaggle/results/smoke-v1/) |
| [Smoke v2](https://www.kaggle.com/code/hoofngocjhuy/flmm-refcoco-smoke/2) | Đổi nguồn annotation theo hướng dẫn REFER; dừng ở tải ảnh do lỗi TLS. | [output v2](FLMM/kaggle/results/smoke-v2/) |
| [Smoke v3](https://www.kaggle.com/code/hoofngocjhuy/flmm-refcoco-smoke/3) | Sửa nguồn ảnh; dừng ở tải checkpoint F-LMM từ Google Drive. | [output v3](FLMM/kaggle/results/smoke-v3/) |
| [Smoke v4](https://www.kaggle.com/code/hoofngocjhuy/flmm-refcoco-smoke/4) | Dùng checkpoint từ Hugging Face của tác giả; nạp được model, nhưng attention kernel BF16 lỗi trên T4. Chưa có mask/metric. | [output v4](FLMM/kaggle/results/smoke-v4/) |
| [Smoke v5](https://www.kaggle.com/code/hoofngocjhuy/flmm-refcoco-smoke/5) | Dùng math SDPA; chạy được một ảnh / hai mask. Chứng minh pipeline hoạt động, không đại diện benchmark. | [output v5](FLMM/kaggle/results/smoke-v5/) |
| [Smoke v6](https://www.kaggle.com/code/hoofngocjhuy/flmm-refcoco-smoke/6) | Mở rộng 20 ảnh / 47 mask, kiểm chứng lại độc lập. cIoU 79,19%; đây là debug subset. | [Script smoke hiện tại](FLMM/kaggle/smoke/smoke.py), [output v6](FLMM/kaggle/results/smoke-v6/), [báo cáo](FLMM/kaggle/RESULTS.md) |
| [Phase1 cache v1](https://www.kaggle.com/code/hoofngocjhuy/flmm-phase1-cache/1) | Chuẩn bị cache bằng CPU; lỗi dependency resolver trước khi hoàn thành. | [output v1](FLMM/phase1/runs/cache-v1/) |
| [Phase1 cache v2](https://www.kaggle.com/code/hoofngocjhuy/flmm-phase1-cache/2) | Chuẩn bị cache môi trường/model/dữ liệu có kiểm tra hash. Không dùng GPU inference. | [Notebook](FLMM/phase1/cpu-prep/prepare.ipynb), [output v2](FLMM/phase1/runs/cache-v2/) |
| [Phase1 gates v1](https://www.kaggle.com/code/hoofngocjhuy/flmm-phase1-gates/1) | Lỗi thiếu scratch directory khi khôi phục môi trường, chưa vào audit. | Ghi nhận traceback trong [attempt_history.json](FLMM/phase1/results/attempt_history.json); không có thư mục output v1 riêng |
| [Phase1 gates v2](https://www.kaggle.com/code/hoofngocjhuy/flmm-phase1-gates/2) | Gate 0 PASS: kiểm tra trọng số, precision và mask tương đương trên 20 ảnh / 47 mask. Dừng theo yêu cầu tại thời điểm đó. | [Notebook](FLMM/phase1/gpu-gates/phase1.ipynb), [output](FLMM/phase1/runs/gate0-v2/), [audit](FLMM/phase1/GATE0_AUDIT.md) |
| [Full data v1](https://www.kaggle.com/code/hoofngocjhuy/flmm-full-data/1) | CPU-only: tái dùng 670 ảnh, tải thêm 830 ảnh, kiểm tra đủ cache 1.500 ảnh. | [Notebook](FLMM/fullval/cpu-data/notebook.ipynb), [output](FLMM/fullval/runs/data-v1/) |
| [Full 1 v1](https://www.kaggle.com/code/hoofngocjhuy/flmm-full-1/1) | Chặng đầu: hoàn thành 534 ảnh / 1.362 mask. Có probe nên số ảnh vượt prefix 500. | [Notebook](FLMM/fullval/gpu-1/notebook.ipynb), [output](FLMM/fullval/runs/chunk1/) |
| [Full 2 v1](https://www.kaggle.com/code/hoofngocjhuy/flmm-full-2/1) | Gate từ chối mở rộng vì snapshot quota đã cũ; giữ 534 ảnh, không inference ảnh mới. | [output v1](FLMM/fullval/runs/chunk2-v1/) |
| [Full 2 v2](https://www.kaggle.com/code/hoofngocjhuy/flmm-full-2/2) | Refresh quota, resume; tính 487 ảnh mới, tổng 1.021 ảnh / 2.596 mask. | [Notebook hiện tại](FLMM/fullval/gpu-2/notebook.ipynb), [output v2](FLMM/fullval/runs/chunk2-v2/) |
| [Full 3 v1](https://www.kaggle.com/code/hoofngocjhuy/flmm-full-3/1) | Resume 1.021 ảnh, tính 479 ảnh mới; kết thúc đủ 1.500 ảnh / 3.811 mask. cIoU 74,96%, mIoU 76,97%. | [Notebook](FLMM/fullval/gpu-3/notebook.ipynb), [kết quả cuối](FLMM/fullval/results/), [báo cáo](FLMM/fullval/REPORT.md) |

**Lưu ý lịch sử:** G0 audit ghi “chưa full validation / chưa resume” phản ánh trạng thái khi dừng tại G0. Full validation sau đó đã được người dùng cho phép và hoàn thành; xem báo cáo fullval để biết trạng thái mới nhất.

Hai notebook ban đầu cũng tìm thấy qua API: [F-LMM Level A Research Starter](https://www.kaggle.com/code/hoofngocjhuy/f-lmm-level-a-research-starter) và [notebook728cf5f7cb](https://www.kaggle.com/code/hoofngocjhuy/notebook728cf5f7cb). Notebook thứ hai được ghi trong kế hoạch là tab ban đầu. Chưa có đủ bằng chứng trong repo để gán kết quả benchmark hoặc diễn giải từng version của hai notebook này.

`FLMM/phase1/gpu-resume/phase1.ipynb` và metadata `flmm-phase1-resume` là **file chuẩn bị local**. Không xuất hiện trong danh sách API đối chiếu lần này, và không có bằng chứng đã chạy. Resume thực tế được thực hiện qua Full 2 và Full 3.

## Cách đọc notebook và metadata

Notebook Run All thường có phần mô tả, giải nén code đã đóng gói, khôi phục môi trường/cache rồi gọi script điều khiển. Muốn hiểu thuật toán và resume, đọc các file `.py` thay vì chỉ đếm cell.

Mỗi `kernel-metadata.json` mô tả notebook: `id` là account/slug trên Kaggle; `code_file` là file chạy; `enable_gpu` và `enable_internet` là cấu hình; `kernel_sources` chỉ các output notebook trước cần gắn vào. Đây không phải file chứa kết quả hay lịch sử đầy đủ mọi version.

## Bản đồ file: giai đoạn chạy thử (`FLMM/kaggle/`)

| File / thư mục | Công dụng |
|---|---|
| `README.md`, `RESULTS.md` | Nhật ký thiết lập và kết quả smoke; có mục kế hoạch cũ, đọc cùng báo cáo fullval mới nhất. |
| `environment/audit.py` | Ghi nhận GPU, Python, CUDA, package và kiểm tra môi trường. |
| `setup/setup.py` | Dựng môi trường legacy cô lập, cài dependency và kiểm tra import. |
| `build_smoke.py` | Đóng gói script smoke để gửi lên Kaggle. |
| `smoke/smoke.py` | Điều phối setup, tải model/dữ liệu và chạy subset evaluator. |
| `prepare_evaluator.py` | Tạo evaluator giới hạn số ảnh từ source upstream. |
| `evaluator/eval_refcoco_subset.py` và diff | Bản evaluator subset và bằng chứng thay đổi so với upstream. |
| `verify_predictions.py` | Tính lại metric từ prediction/GT của smoke. |
| `render_predictions.py` | Tạo hình so sánh prediction với GT. |
| `results/environment-v1`, `setup-v*`, `smoke-v*` | Log và artifact của từng attempt; không phải các dataset khác nhau. |

## Bản đồ file: kiểm chứng Gate 0 (`FLMM/phase1/`)

| File / thư mục | Công dụng |
|---|---|
| `GATE0_AUDIT.md` | Bản kết luận audit và các giới hạn tại thời điểm dừng G0. |
| `config.json` | Cấu hình protocol, checkpoint, gate và phạm vi chạy; bản phase1 vẫn khóa theo yêu cầu dừng G0 cũ. |
| `prepare.py` | Chuẩn bị cache, manifest, model và môi trường bằng CPU. |
| `bootstrap.py` | Khôi phục môi trường và chuẩn bị đường dẫn runtime. |
| `engine.py` | Engine đã audit: nạp/kiểm tra model, inference, ghi record, gate và khả năng tiếp tục. |
| `common.py` | Tiện ích chung: hash, fingerprint, ghi atomic, kiểm tra record và tính tổng hợp. |
| `build.py` | Đóng gói code/config/reference thành notebook CPU, GPU gates và resume. |
| `snapshot_quota.py` | Ghi snapshot tài nguyên/quota để resource gate kiểm tra độ mới và ngân sách. |
| `verify_gate0.py` | Kiểm tra độc lập kết quả G0 tại local. |
| `test_common.py` | Kiểm tra logic record, fingerprint, chống duplicate và aggregation. |
| `cpu-prep/prepare.ipynb` | Notebook chuẩn bị cache CPU. |
| `gpu-gates/phase1.ipynb` | Notebook kiểm chứng G0; không tự động đại diện full validation. |
| `gpu-resume/phase1.ipynb` | Notebook dự phòng local; chưa xác nhận đã submit/chạy. |
| `docs/checkpoint_integrity.md` | Giải thích kiểm chứng nguồn và tensor checkpoint. |
| `docs/evaluator_equivalence.md`, `official-control.diff`, `subset-control.diff` | Phạm vi và thay đổi của phép đối chiếu evaluator. |
| `docs/evaluation_protocol.md` | Định nghĩa đơn vị dữ liệu, metric và quy tắc đánh giá. |
| `docs/feasibility_report.md`, `observation_report.md` | Tài liệu giai đoạn phase1; không thay kết luận kết quả fullval. |
| `results/`, `runs/` | Bằng chứng tensor, precision, môi trường, mask regression và lịch sử cache/G0. `loader-module.py`/`refcoco-loader.py` trong output là bản lưu để audit loader. |

## Bản đồ file: đánh giá đầy đủ (`FLMM/fullval/`)

| File / thư mục | Công dụng |
|---|---|
| `REPORT.md` | Kết quả cuối, protocol, số đo, kiểm chứng và giới hạn kết luận. |
| `WRAP_UP_VI.md` | Diễn giải nghiên cứu cho người không chuyên, có đoạn trình bày họp. |
| `PROTOCOL.md` | Cách chia chặng, giữ nguyên phép tính, kiểm tra tài nguyên và điều kiện resume. |
| `STATUS.md` | Trạng thái hoàn tất và ghi chú vận hành lịch sử. |
| `FAILURE_CASES.md` | Danh sách 20 target IoU thấp nhất, link ảnh và câu mô tả. |
| `build.py` | Tạo notebook CPU data và ba notebook GPU; kiểm tra hash engine/common khớp G0 trước đóng gói. |
| `extend_cache.py` | Tái dùng ảnh có sẵn, tải và kiểm tra ảnh còn thiếu bằng CPU. |
| `full_control.py` | Điều phối khôi phục cache, kiểm tra checkpoint, resume record cũ, hai worker và gộp output. |
| `full_worker.py` | Chạy từng replica GPU, kiểm tra regression, probe và inference ảnh còn thiếu. |
| `full_finalize.py` | Đọc mask đã lưu, tổng hợp metric chính thức, CSV, latency, failure cases và archive. |
| `verify_full.py` | Kiểm tra độc lập archive/hash/coverage, pixel IoU, CSV, latency và danh sách failure. |
| `plot_full.py` | Vẽ phân bố IoU và latency từ kết quả đã lưu. |
| `write_report.py` | Tạo báo cáo/index failure khi kiểm chứng local PASS. |
| `cpu-data/`, `gpu-1/`, `gpu-2/`, `gpu-3/` | Source notebook và metadata upload lên Kaggle; số GPU-1/2/3 chỉ chặng, không phải số GPU. |
| `runs/data-v1`, `chunk1`, `chunk2-v1`, `chunk2-v2` | Artifact các chặng trước, gồm cả attempt bị gate chặn. |
| `workers/0`, `workers/1` | Precision, kiểm tra regression, probe và summary của hai worker ở chặng cuối. |
| `full-records.tar.gz` | Archive đủ 1.500 JSON records + 1.500 NPZ prediction/GT; lưu để kiểm tra và phục hồi. |

## Các file kết quả cuối (`FLMM/fullval/results/`)

| File | Dùng để trả lời câu hỏi gì? |
|---|---|
| `baseline_metrics.json` | Điểm đánh giá cuối là bao nhiêu? |
| `per_target_iou.csv` | Đối tượng/câu mô tả nào làm tốt hoặc sai? Có ID và intersection/union/IoU từng target. |
| `latency.json` | Mỗi ảnh mất bao lâu, median/p95 và bộ nhớ quan sát được ra sao? |
| `validation_overview.png` | Xem nhanh phân bố điểm và thời gian xử lý. |
| `failure_cases.json`, `failure_cases/*.jpg` | Các ví dụ IoU thấp nhất; GT xanh, prediction cam. Chưa phải taxonomy nguyên nhân. |
| `records/*.json` | Trạng thái, mapping target, checksum và thông tin xử lý từng ảnh. |
| `records/*.npz` | Mask prediction và GT để xem/tính lại từ pixel. |
| `local_verification.json` | Bộ kết quả đã được kiểm tra độc lập đạt PASS chưa? |
| `full_summary.json` | Coverage, lịch sử các chunk, worker và nguồn code của toàn run. |
| `run_contract.json` | Protocol/computation fingerprint nào xác định record có thể reuse? |
| `run_manifest.jsonl`, `dataset_manifest.json` | Những ảnh và target nào thuộc phạm vi đánh giá? |
| `records_archive.json` | Hash và thông tin kiểm tra archive mask/record. |
| `resource_preflight.json`, `chunk_gate.json` | Vì sao chặng được phép tiếp tục hoặc bị chặn? |
| `checkpoint_prefetch.json` | Bằng chứng kiểm tra/đọc trước checkpoint, tách khỏi đo forward. |
| `full-data-preparation.json` | Cache ảnh đầy đủ được chuẩn bị thế nào? |
| `session.json` | Thông tin phiên chạy cuối. |

## Các phần còn lại

- `FLMM/FLMM_LevelA_Research_Starter/FLMM_LevelA_Research_Starter/`: `PLAN.md`/`KAGGLE_PLAN.md` là kế hoạch; `findings.md` là phát hiện; `research-state.yaml` là trạng thái; `research-log.md` là nhật ký; `experiments/` chứa protocol baseline và giả thuyết chưa kiểm chứng; `templates/` là mẫu ghi chép. Kế hoạch cũ không có nghĩa mọi task đã chạy.
- `FLMM/FLMM_LevelA_Research_Starter.zip`: bản starter đóng gói ban đầu, không thay tài liệu đã cập nhật trong thư mục.
- `FLMM/upstream/F-LMM/`: snapshot source của tác giả, giữ LICENSE gốc. Không phải một run Kaggle khác. Dùng để đối chiếu model/config/evaluator; danh mục này tập trung vào file nghiên cứu và pipeline do dự án sử dụng, không diễn giải từng module thư viện upstream.
- `README.md`, `.gitignore`, `.gitattributes` ở root: cửa vào tài liệu, loại file nhạy cảm/cache khỏi Git và giữ nguyên byte artifact qua các hệ điều hành.

Không tự bấm Run All để “xem kết quả”: thao tác đó có thể tạo job mới và tiêu GPU. Đọc Output/log hoặc artifact trong repo trước. Trạng thái Kaggle COMPLETE cũng chưa đủ để kết luận thí nghiệm PASS; cần xem status/gate và báo cáo của stage.
