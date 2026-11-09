# Tổng kết F-LMM dành cho buổi họp nhóm

**Chúng ta đã đưa mô hình có sẵn của tác giả chạy được trên Kaggle, kiểm tra trên toàn bộ tập validation đã chọn và lưu đầy đủ bằng chứng. Bây giờ có thể chuyển sang tìm hiểu mô hình sai ở đâu. Việc chọn nó làm baseline chung vẫn cần nhóm thống nhất bài toán, vì thực nghiệm hiện tại xử lý ảnh, chưa xử lý video.**

Hãy hình dung một bức ảnh có nhiều người. Ta đưa cho máy câu “người đứng bên trái” và muốn máy tô đúng phần ảnh thuộc người đó. Đây là bài toán đang thử nghiệm: tìm và phân đoạn đối tượng theo mô tả bằng ngôn ngữ. “Mask” chính là vùng được tô; nó cho biết đối tượng nằm ở những điểm ảnh nào.

F-LMM là phương pháp được chọn để thử làm mốc ban đầu. “Baseline” nghĩa là mốc so sánh: sau này, nếu sửa một thành phần hay đề xuất cách làm mới, ta phải kiểm tra nó có tốt hơn mốc này trong cùng điều kiện hay không. Trong giai đoạn vừa qua, chúng ta dùng mô hình đã được tác giả huấn luyện, chưa tự huấn luyện hay tạo ra phương pháp mới.

Để có kết quả đáng tin, công việc không chỉ là bấm chạy notebook. Chúng ta đã chuẩn bị môi trường, kiểm tra đúng các bộ trọng số, thử trên một lượng ảnh nhỏ và đối chiếu cách tính điểm. Khi các bước kiểm tra đạt yêu cầu, mới mở rộng sang toàn bộ RefCOCO validation theo cách chọn câu mô tả của bộ đánh giá đã dùng.

Lần đánh giá đầy đủ gồm **1.500 ảnh với 3.811 đối tượng cần tìm**. Mỗi dự đoán được lưu lại để có thể xem máy đã tô vùng nào, đối chiếu với đáp án và tính lại điểm. Kết quả cuối đã được kiểm tra lại từ toàn bộ các vùng dự đoán, đồng thời kiểm tra không thiếu hoặc đếm trùng đối tượng.

Hai điểm tổng hợp đạt khoảng **75 và 77 trên thang 100**. Các điểm này đo mức độ chồng khớp giữa vùng máy chọn và vùng đáp án, theo hai cách tổng hợp khác nhau. Điểm càng cao thì vùng dự đoán càng khớp. Chúng **không có nghĩa là máy trả lời đúng 75% hoặc 77% số câu hỏi**. Muốn kết luận có tái hiện được điểm công bố trong paper hay không, vẫn cần đối chiếu đúng cấu hình và cách đánh giá tương ứng.

Về khả năng triển khai, hệ thống đã chạy được trên GPU T4 của Kaggle. Sau khi mô hình được nạp và làm nóng, thời gian xử lý điển hình khoảng **2,3 giây mỗi ảnh**, chưa gồm các bước chuẩn bị dữ liệu và nạp mô hình. Điều này cho thấy việc đánh giá trên ảnh là khả thi với tài nguyên đã dùng; chưa chứng minh khả năng xử lý video trực tiếp hoặc chi phí huấn luyện lại.

Quá trình đánh giá được chia thành nhiều chặng. Có thể hình dung như làm một bài kiểm tra dài và lưu đáp án sau từng chặng: chặng sau đọc lại phần đã hoàn thành rồi làm tiếp. Các notebook/run version là những lần thực thi hoặc thử lại, không phải mỗi version là một mô hình khác. Kết quả cuối được tính từ toàn bộ dự đoán duy nhất. Khả năng chạy tiếp dựa trên output đã lưu thành công trên Kaggle; phần chưa được lưu khi hệ thống bị dừng đột ngột có thể phải tính lại.

Kết quả cũng cho thấy mô hình còn sai. Khoảng **13% đối tượng có vùng dự đoán chồng khớp với đáp án dưới một nửa**. Một số ví dụ đã xem cho thấy máy chọn nhầm người trong ảnh. Chúng ta đã lưu 20 trường hợp có điểm thấp nhất để thảo luận bằng hình ảnh. Tuy nhiên, chưa phân loại lỗi một cách hệ thống nên chưa thể kết luận nguyên nhân chủ yếu là hiểu sai câu, nhầm vị trí hay vẽ sai ranh giới. Những ví dụ tệ nhất cũng không đại diện cho mọi ảnh.

Điểm cần thống nhất với nhóm là khoảng cách từ ảnh sang video. Một ảnh chỉ yêu cầu tìm đúng đối tượng tại một thời điểm. Video còn có chuyển động, che khuất và yêu cầu giữ đúng đối tượng qua nhiều khung hình; một số bài toán còn yêu cầu tìm đúng đoạn thời gian. Chúng ta chưa kiểm tra những khả năng này. Vì vậy, kết quả hiện tại là bằng chứng cho bước xử lý ảnh, chưa phải kết quả tái hiện Video Grounding.

Đề xuất mang vào buổi họp là giữ F-LMM làm ứng viên đã có bằng chứng chạy được. Nếu nhóm chọn bài toán phân đoạn theo mô tả trên ảnh hoặc từng khung hình, kết quả hiện tại có thể làm mốc xuất phát. Nếu nhóm chọn bài toán có yêu cầu rõ về thời gian hoặc theo dõi đối tượng trong video, cần so sánh mức phù hợp với các ứng viên khác trước khi chốt.

Trước buổi họp, cần hoàn thành ba việc: đối chiếu điểm với đúng thiết lập trong paper; chọn vài hình minh họa gồm cả trường hợp làm tốt và làm sai; viết rõ input và output của bài toán nhỏ đề xuất cho nhóm. Tuần tiếp theo nên tập trung phân loại lỗi trên một mẫu có cả kết quả tốt và xấu, định lượng nhóm lỗi rồi mới chọn một giả thuyết cải tiến. Chưa có cải tiến nào được kiểm chứng trong công việc hiện tại.

**Đoạn trình bày ngắn có thể đọc trong buổi họp:**

> Em đang thử F-LMM cho bài toán đưa vào một ảnh và một câu mô tả, rồi yêu cầu máy tô đúng đối tượng được nhắc đến. Em đã dùng mô hình có sẵn của tác giả, chạy đầy đủ tập validation đã chọn trên Kaggle và kiểm tra lại kết quả từ các vùng dự đoán được lưu. Hệ thống đã chạy được, có thể lưu tiến độ và chạy tiếp. Các ví dụ lỗi cho thấy vẫn có trường hợp chọn nhầm đối tượng, nhưng em chưa định lượng nguyên nhân và chưa thử cải tiến. Bước tiếp theo em đề xuất là phân tích lỗi để chọn một vấn đề nhỏ có thể đo và sửa. Em cũng muốn nhóm chốt rõ bài toán video, vì kết quả hiện tại mới ở ảnh, chưa đánh giá theo thời gian và chưa hoàn tất đối chiếu điểm paper.

Tài liệu đối chiếu: [báo cáo kỹ thuật](REPORT.md), [hình các trường hợp lỗi](FAILURE_CASES.md), [điểm từng đối tượng](results/per_target_iou.csv).
