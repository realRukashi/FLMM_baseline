# F-LMM — Full RefCOCO UNC validation

Ngày hoàn tất: 06/11/2026 (ICT). **PASS: 1.500 ảnh / 3.811 selected target masks.** Đã tải và kiểm tra độc lập toàn bộ record, mask, CSV và latency.

| Metric | Kết quả |
|---|---:|
| cIoU (official RefSegMetric) | **74.9634%** |
| mIoU (official RefSegMetric) | **76.9712%** |
| Tổng intersection | 103,215,624 |
| Tổng union | 137,687,931 |
| Median inference / ảnh | 2.274 s |
| p95 inference / ảnh | 2.740 s |
| Max inference / ảnh | 2.971 s |
| Peak GPU allocated | 8.26 GiB |
| Peak GPU reserved | 9.13 GiB |

## Protocol đã chạy

- Official source commit `47d3c2a6f628dd48ba76a1cfdf29b6b2aa8f9eb3`; config DeepSeekVL-1.3B + SAM ViT-L.
- Backbone revision `8f13a8e00dbdc381d614a9d29d61b07e8fe91b3f`; checkpoint F-LMM revision `d70e5b3289ca4255cb6a017a792870aad26c01ce`. SHA256 và computation fingerprint trong `results/run_contract.json`; source engine/common giữ nguyên hash đã PASS G0.
- `text_mode=select_first`, `--concat`, threshold 0.5 và preprocessing không đổi. Chạy toàn bộ UNC val theo semantics của evaluator được phát hành: **không phải toàn bộ 10.834 câu mô tả**, và không phải chạy cả tám split của upstream.
- Hai replica Tesla T4 độc lập, mỗi ảnh trên một GPU. Backbone BF16; grounding head/SAM FP32; math SDPA và manual language-attention fallback như G0. Không training, quantization hoặc đổi phương pháp.
- Cả hai replica đã đối chiếu bitwise với bộ G0 20 ảnh/47 mask trước mở rộng.

## Kiểm chứng và resume

IoU được tính lại từ **toàn bộ mask nhị phân**; CSV có **3.811 prediction ID duy nhất**, khớp ref_id/sent_id/query/mask_id và số đếm intersection/union. Metric độc lập khớp official RefSegMetric trong 0,001 điểm phần trăm; tất cả record/mask hashes hợp lệ.

| Chunk tính ảnh lần đầu | Số ảnh duy nhất |
|---|---:|
| 1 | 534 |
| 2 | 487 |
| 3 | 479 |
| Tổng | **1.500** |

Các chunk sau nạp Kaggle Output đã commit và bỏ qua đúng DONE records khớp fingerprint/checksum. Không lấy trung bình cIoU từng chunk. Có một lần resource gate chặn vì snapshot quota quá 6 giờ; đã refresh quota và resume mà không tính lại ảnh đã lưu. Điều này được giữ trong `runs/chunk2-v1/`.

Archive `full-records.tar.gz` chứa đủ 1.500 JSON records và 1.500 NPZ prediction/GT. Nếu session mới bị ngắt, có thể resume từ output đã commit cuối cùng; hard kill trước commit có thể phải lặp phần chưa commit. Quy trình và điều kiện retry trong [PROTOCOL.md](PROTOCOL.md).

## Latency và failure cases

Latency là forward thời gian thực đo với CUDA synchronize, sau warm-up riêng cho mỗi replica/session; không gồm download, cache restore, model loading, preprocessing hoặc postprocessing. Dữ liệu trải qua ba session độc lập. Cột latency/target trong CSV là **amortized image time**, không phải đo chạy độc lập mỗi câu.

- Target có IoU < 0,5: **507 / 3.811**.
- Target có IoU = 0: **112 / 3.811**.
- Prediction không rỗng: **3811 / 3.811**.
- Đã lưu 20 case có IoU thấp nhất, chọn trước theo quy tắc sort IoU rồi image_id/mask_id, kèm input/GT/prediction. Đây là ví dụ mô tả, chưa phải taxonomy nguyên nhân.

Xem [failure cases](FAILURE_CASES.md) và [biểu đồ phân bố](results/validation_overview.png). Cold load được lưu riêng: chunk đầu khoảng 802 giây; sau bổ sung sequential prefetch/hash khoảng 27 giây, model load ở lần tiếp theo khoảng 10–14 giây. Đây là thay đổi I/O, không đổi model hoặc phép đo.

## Artifacts

- [Metric và coverage](results/baseline_metrics.json)
- [Per-target IoU CSV](results/per_target_iou.csv)
- [Latency/memory từng ảnh](results/latency.json)
- [Kiểm chứng local](results/local_verification.json)
- [Run summary và nguồn code](results/full_summary.json)
- [Computation fingerprint](results/run_contract.json)
- [Kaggle final output](https://www.kaggle.com/code/hoofngocjhuy/flmm-full-3)

Full validation đã quan sát tập holdout nội bộ cũ; không dùng nó như holdout chưa quan sát cho cải tiến sau này. Không tuning/cải tiến trong lần chạy này; testA/testB chưa được dùng. Kết quả trên đây là baseline đo được, không tự khẳng định bằng điểm công bố trong paper khi chưa đối chiếu đúng hàng/setting.
