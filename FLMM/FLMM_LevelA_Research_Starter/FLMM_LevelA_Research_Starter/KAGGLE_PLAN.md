# Kế hoạch tái hiện F-LMM trên Kaggle

Ngày kiểm tra: 2026-11-05 (ICT).

## Trạng thái đã quan sát

- Chrome profile: Huy (Work), theo lựa chọn của người dùng.
- Notebook đang mở ở chế độ chỉnh sửa: https://www.kaggle.com/code/hoofngocjhuy/notebook728cf5f7cb/edit
- Namespace chủ notebook: `hoofngocjhuy`.
- Notebook chứa cell mẫu mặc định, Draft Session đang off, chưa gắn input.
- Chưa xác nhận accelerator khả dụng, quota, Internet hoặc cấu hình runtime. Chưa cài dependencies hay chạy F-LMM.

## Mục tiêu

Tái hiện inference/evaluation của checkpoint F-LMM DeepSeekVL-1.3B trên RefCOCO UNC val. Thành công bước đầu: tạo mask đúng pipeline, chạy được subset có ID cố định và lưu cIoU/mIoU cùng log. Full-val là mốc riêng; chưa training hoặc thay đổi phương pháp.

## 1. Kiểm tra runtime trước khi tải dữ liệu lớn

- Kiểm tra quyền riêng tư notebook trước khi thêm nội dung; dùng notebook private cho thử nghiệm.
- Xem Session options/Settings: accelerator thực tế, quota, Internet, dung lượng.
- Chạy cell chẩn đoán Python/PyTorch/CUDA, GPU name/VRAM, `torch.cuda.is_bf16_supported()`, dung lượng đĩa và package versions. Lưu kết quả, không in biến môi trường hoặc credentials.
- Chỉ khởi động một GPU/process cho lần đầu. Hai GPU không tự gộp VRAM: evaluator chính thức tạo model trên từng process/device.
- Config chính thức tải LMM với bfloat16. Nếu GPU không hỗ trợ phù hợp, cần kiểm tra implementation và ghi riêng mọi thay đổi dtype; không tự nhận là tương đương config gốc.

## 2. Chốt mã nguồn và môi trường

- Clone https://github.com/wusize/F-LMM và lưu commit SHA; kiểm tra mã trước khi chạy.
- Đối chiếu Python/PyTorch/CUDA với Xtuner, MMDetection, MMSegmentation, MMEngine và MMCV; kiểm tra import trước khi tải model lớn.
- Pin `transformers==4.39.1` theo README. Các phiên bản còn lại phải chọn dựa trên compatibility thực tế, chưa có lockfile đã kiểm chứng.
- Khi import thành công, lưu `pip freeze` và install log. Timebox một buổi cho tương thích môi trường, ghi traceback nếu bị chặn.

## 3. Chuẩn bị đúng model và dữ liệu

- Ba tài nguyên model: `deepseek-ai/deepseek-vl-1.3b-chat`, `sam_vit_l_0b3195.pth`, checkpoint F-LMM đã train đúng config DeepSeekVL-1.3B từ link chính thức.
- Lưu URL, tên, dung lượng và SHA256 của file tải được. Base DeepSeekVL không thay thế checkpoint F-LMM.
- RefCOCO: `data/coco/refcoco/instances.json`, `data/coco/refcoco/refs(unc).p`, ảnh COCO train2014 tương ứng.
- Kiểm tra loader và chọn ID trước; chỉ tải ảnh cần cho smoke/pilot nếu có thể, chưa tải toàn bộ tài nguyên training PNG.
- Có thể gắn input dưới `/kaggle/input` và tạo đường dẫn/symlink tương ứng dưới working. Kiểm tra dung lượng trước khi sao chép hoặc giải nén.

## 4. Giới hạn evaluator rõ ràng

Đã đọc script upstream ngày 2026-11-05:

- Script mặc định chạy 8 split: RefCOCO val/testA/testB, RefCOCO+ val/testA/testB, RefCOCOg val/test.
- `--debug` chọn 100 index đầu của MỖI split; không phải một thử nghiệm chỉ RefCOCO val.
- Chưa có CLI `--split` hoặc `--max-samples` trong bản script đã đọc.
- Tạo bản evaluator riêng có diff nhỏ: chỉ RefCOCO val, hỗ trợ manifest ID cố định và giới hạn mẫu; giữ nguyên preprocessing, `--concat`, prediction threshold và RefSegMetric. Lưu diff và commit nguồn.
- Ghi cả missing và unexpected checkpoint keys; script gốc chỉ in unexpected. Xác định key nào hợp lệ trước khi chấp nhận baseline.
- Không chạy evaluator gốc trên dữ liệu RefCOCO-only vì nó sẽ tiếp tục yêu cầu các bộ còn lại.

## 5. Các mốc chạy

| Mốc | Việc làm | Bằng chứng cần lưu |
|---|---|---|
| G0 | Audit runtime và imports | GPU/quota, environment snapshot, SHA, import log |
| G1 | Load model và chạy 1 mẫu val | checkpoint load log, ID, GT/pred mask overlay, peak VRAM |
| G2 | Smoke 20–50 mẫu | IDs, metrics, lỗi và thời gian; chỉ dùng debug pipeline |
| G3 | Pilot cố định, dự kiến 100–300 mẫu theo compute | manifest, raw metrics/log, 10 case đại diện |
| G4 | Full RefCOCO val nếu khả thi | đủ split, cùng config/checkpoint, so đúng hàng trong paper |
| G5 | Phân tích rồi chọn một giả thuyết | observation có số đo, protocol viết trước khi thử biến thể |

Trước G2: điền và chốt `experiments/H0-baseline/protocol.md`. Tách debug/pilot/holdout; không điều chỉnh phương pháp theo holdout. Báo cáo subset với đúng nhãn, không so như full benchmark.

## 6. Điều kiện dừng và lưu kết quả

- OOM: lưu GPU/dtype/sample/config và traceback; xác định thành phần gây lỗi trước khi sửa. Mọi thay đổi ảnh/dtype/refinement phải được ghi là khác biệt.
- Dependency/checkpoint/data bị chặn: ghi blocker và bước tái hiện. Không thay model rồi tiếp tục gọi là tái hiện F-LMM.
- Lưu protocol, IDs, manifest model, environment, patch, logs, metrics và hình minh họa; kiểm tra output được lưu qua cơ chế Save Version trước khi kết thúc session.
- Quota/GPU/runtime hiện vẫn là mục cần xác minh; chưa có kết quả thực nghiệm.

## Nguồn chính thức đã đọc

- https://github.com/wusize/F-LMM
- https://raw.githubusercontent.com/wusize/F-LMM/main/scripts/multiprocess_eval_refcoco.py
- https://raw.githubusercontent.com/wusize/F-LMM/main/configs/deepseek_vl/frozen_deepseek_vl_1_3b_chat_unet_sam_l_refcoco_png.py
