# Gate 0 — PASS, đã dừng chờ audit

Ngày: 06/11/2026 (ICT). Kernel private: [FLMM Phase1 Gates, version 2](https://www.kaggle.com/code/hoofngocjhuy/flmm-phase1-gates/2). Không chạy probe 50 ảnh, stability 100 ảnh hoặc benchmark mở rộng.

## Kết luận có bằng chứng

| Nội dung | Trạng thái | Bằng chứng |
|---|---|---|
| Git commit và dependency thực tế | PASS | `results/installed_versions.json`, `results/actual_freeze.txt`, `results/run_contract.json` |
| Backbone pin revision + kiểm tra safetensors SHA256 | PASS | `results/checkpoint_integrity.json` |
| 529 key DeepSeekVL + 345 key SAM image encoder | PASS | 874 bản ghi: so tensor khởi tạo với checkpoint pretrained trên đĩa, rồi xác nhận không đổi sau load F-LMM |
| Tensor checkpoint F-LMM đã train | PASS | 193 tensor khớp giá trị sau load; không có unexpected keys. Bao gồm parameter/buffer, không phải 193 scalar parameters |
| Official control so với subset evaluator | PASS trên cùng 20 ảnh/47 mask | Binary prediction và GT khớp bitwise; metric lệch dưới 0,001 điểm phần trăm |
| Engine resumable so với mask smoke cũ | PASS trên 20 ảnh | Kiểm chứng lại tại máy local bằng `verify_gate0.py` |
| Feasibility 50 / stability 100 / resume qua session | UNVERIFIED | Chưa chạy theo yêu cầu dừng sau G0 |
| Full benchmark reproduction | UNVERIFIED | Chưa chạy |

## Phạm vi phép so sánh

Official control dùng mã evaluator upstream nhưng có harness giới hạn UNC val và đúng 20 ID regression, thay đường dẫn bằng cache pin, dùng cùng model đã audit để tránh load lại, cùng math attention backend. Không thay prediction, interpolation, threshold hoặc RefSegMetric. Đây **không phải** chạy nguyên script tám split không chỉnh sửa.

- [Diff official control](docs/official-control.diff)
- [Diff subset control](docs/subset-control.diff)
- [Báo cáo evaluator](docs/evaluator_equivalence.md)
- [JSON so sánh từng ảnh](results/evaluator_equivalence.json)

Metric của 20 ảnh/47 mask: cIoU **79,186941%**, mIoU **81,682461%**. Tổng intersection = **1.378.708**, union = **1.741.080**. Đã đọc lại file mask và tính độc lập; không lấy trung bình cIoU từng chunk. Đây vẫn là regression/debug set cũ, không phải một đánh giá thống kê mới.

## Đơn vị evaluation đã xác minh

Annotation UNC val thực tế có **1.500 ảnh, 3.811 reference, 10.834 câu**. Loader `select_first` giữ câu đầu mỗi reference; `--concat` gộp các target trong một lần inference ảnh. Toàn split theo protocol này có **3.811 selected target masks**, không bao phủ mọi câu.

`results/per_expression.csv` có 47 hàng, ánh xạ `image_id, dataset_index, ref_id, sent_id, expression, mask_id, prediction_id, annotation_id`. Mapping được kiểm tra cả text lẫn segmentation so với loader chính thức. Manifest toàn split và các tập tương lai được cố định trước các kết quả mới; holdout không bị inference/tuning.

## Precision và attention

- Backbone: BF16; mask head, text projection và SAM: FP32, được xác minh cả dtype parameter lẫn activation warm-up.
- SDPA math backend; flash/memory-efficient SDPA tắt. LlamaSdpaAttention chuyển sang manual attention khi yêu cầu `output_attentions=True`.
- T4 có compute capability 7.5; không nhận BF16 matmul thành công là bằng chứng native BF16 tensor-core support.
- Chi tiết: `results/precision.json` và log `results/audit-stage.log`.

## Các điểm vẫn cần bạn lưu ý khi audit

1. Revision backbone của **smoke lịch sử** không được ghi lại, nên vẫn UNVERIFIED. Revision mới đã pin và tái tạo mask lịch sử bitwise, nhưng điều này không chứng minh ngược revision cũ.
2. Equivalence là conditional trên cùng 20 ảnh và backend nêu trên; chưa xác nhận tương đương mọi setting/table của paper.
3. Chưa đo probe 50 ảnh hoặc chứng minh resume qua session. Unit tests local đã kiểm tra cơ chế chống duplicate, sai fingerprint, output hỏng và aggregation; chúng không thay thế test GPU/resume thực tế.
4. `/kaggle/working` chưa đủ để gọi là durable trước khi job được Kaggle commit. Cache CPU và G0 job đều đã kết thúc; output đã tải về. Chưa thử session thứ hai để resume evaluation.

## Nhật ký và chi phí chuẩn bị

CPU cache v2 mất khoảng **13,27 phút**, gồm khoảng 6,80 phút bước data/model và 4,48 phút đóng gói environment. Cache tổng khoảng **8,83 GB**, environment nén khoảng **3,26 GB**. Không dùng GPU cho việc này. GPU G0 khôi phục environment khoảng **91,11 giây**; model load cộng audit khoảng **77,15 giây** (không gọi đây là cold model-load thuần).

Các lỗi được giữ lại: CPU v1 lỗi resolver; GPU G0 v1 chưa tạo scratch directory, dừng trước audit. G0 v2 PASS. Xem `results/attempt_history.json` và các thư mục `runs/`.

## Nên mở theo thứ tự

1. [Checkpoint integrity](docs/checkpoint_integrity.md), rồi [proof từng tensor](results/checkpoint_integrity.json).
2. [Evaluator equivalence](docs/evaluator_equivalence.md), hai diff harness và [kiểm chứng local](results/gate0_local_verification.json).
3. [Precision](results/precision.json), [protocol](docs/evaluation_protocol.md), [run summary](results/run_summary.json).
4. [Notebook Run All đang khóa ở G0](gpu-gates/phase1.ipynb).

**Trạng thái bàn giao: dừng sau Gate 0 theo yêu cầu người dùng. Không có job mở rộng được gửi.**
