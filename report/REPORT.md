# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 07/11/2026

**Runtime Colab:** GPU – NVIDIA T4

**Python / PyTorch / Ultralytics:** Ultralytics `8.4.145`

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35`

**Thay đổi so với notebook nguồn:** Không.

> Báo cáo không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=**468** ; `class_name`=**cab** ; `rank`=**1** ; `score`=**0.510915** ; `taxonomy_name`=**ImageNet-1K**.
2. Record này nói về **cả bức ảnh**, không phải một vật thể riêng. Đây là tác vụ image classification nên model xếp hạng các lớp mô tả nội dung tổng thể của ảnh, không tạo vị trí riêng cho từng vật thể.
3. Danh sách tên lớp đến từ taxonomy đi kèm checkpoint/model đã huấn luyện (ở đây là ImageNet-1K), không phải model tự nghĩ ra tên lớp mới trong mỗi lần dự đoán.
4. Cần ghi cả `class_id`, `class_name` và `taxonomy_name` để định danh lớp nhất quán: ID là mã tham chiếu, tên lớp giúp con người đọc hiểu, còn taxonomy cho biết mã và tên đó thuộc bộ nhãn nào. Chỉ ghi tên có thể gây nhầm lẫn nếu các bộ nhãn khác nhau có tên giống hoặc gần giống nhau.
5. Guideline cần quy định tiêu chí xác định chủ đề chính, cách xử lý ảnh có nhiều vật thể/chủ đề và thứ tự ưu tiên khi chỉ được chọn một lớp (ví dụ chọn theo mục tiêu/chủ đề nổi bật của ảnh, không tùy ý chọn một vật thể phụ).
6. Không được lấy `score` làm ground truth vì score chỉ thể hiện mức độ tin cậy của model đối với dự đoán. Score cao không chứng minh dự đoán đúng; ground truth phải do người gán nhãn xác định theo guideline.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
JSON đã xuất lưu prediction ở ngưỡng `0.35`.

1. Chép một record: `class_name`=**person** ; `score`=**0.912624** ; `bbox_xyxy`=**[385.33, 69.24, 498.92, 348.92]** ; `bbox_width`=**113.58** ; `bbox_height`=**279.68**.
2. `bbox_xyxy` có dạng `[x1, y1, x2, y2]`: góc trên-trái ở `(385.33, 69.24)` pixel; góc dưới-phải ở `(498.92, 348.92)` pixel. Box rộng **113.58 pixel** và cao **279.68 pixel**.
3. Chọn hai ngưỡng có thể đối chiếu trực tiếp từ JSON đã lưu:
   - Ngưỡng **0.35**: **11 prediction** trong ảnh `kitchen` (đếm số record có `sample_id = kitchen`).
   - Ngưỡng **0.60**: **6 prediction** trong 11 record trên có `score >= 0.60`.
   - Lưu ý: JSON chỉ lưu kết quả được tạo ở ngưỡng `0.35`, vì vậy không thể xác định chính xác số prediction ở ngưỡng `0.20` chỉ từ ZIP này.
4. Khi hạ ngưỡng, model giữ lại nhiều box hơn, trong đó có thể có dự đoán sai, box trùng hoặc box không rõ ràng. Reviewer cần kiểm tra class, vị trí và kích thước box; phát hiện box sai/trùng, xác định vật thể bị bỏ sót và chuyển các trường hợp khó cho annotator sửa hoặc escalation.
5. Quy tắc box: vẽ box bao sát toàn bộ phần nhìn thấy của đúng một vật thể; không cố ý lấy thêm nền hoặc vật thể bên cạnh. Dùng thống nhất thứ tự tọa độ `[x1, y1, x2, y2]`, trong đó điểm đầu là góc trên-trái và điểm cuối là góc dưới-phải.
6. Trên ảnh `kitchen`, có một người ở sát mép trái ảnh chỉ xuất hiện một phần (prediction `person` có box bắt đầu gần `x = 0`). Guideline nên quy định cách gán nhãn vật thể bị cắt mép: chỉ đánh dấu phần nhìn thấy và không tự suy đoán phần nằm ngoài ảnh. Nếu chưa có quy tắc thống nhất về vật thể bị cắt mép thì escalation lên Lab Coach.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn record `kitchen-001`.

1. Chép một record: `instance_id`=**kitchen-001** ; `class_name`=**person** ; `score`=**0.899318** ; số điểm polygon (`polygon_point_count`)=**348** ; 5 điểm đầu của `polygon_xy`=**[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]**.
2. So với box hình chữ nhật, polygon cho biết đường biên và hình dạng cụ thể của vật thể. Polygon có thể bám theo dáng người hoặc vật thể không vuông vắn, thay vì chỉ khoanh vùng chữ nhật bao quanh.
3. `instance_id` dùng để phân biệt từng đối tượng riêng lẻ trong kết quả segmentation. Hai vật thể cùng loại có thể có cùng `class_name` nhưng có `instance_id` khác nhau. Nó không phải class ID vì không đại diện cho loại vật thể; cũng không phải tracking ID vì không theo dõi cùng một vật thể xuyên suốt nhiều khung hình/video.
4. Quy tắc vẽ mask: đi theo đường biên của phần vật thể nhìn thấy, giữ polygon sát biên, không lấy lẫn pixel thuộc nền hoặc vật thể khác. Dùng thống nhất cách xử lý phần bị che khuất và độ chi tiết đường biên trong toàn bộ tập ảnh.
5. Trên ảnh `kitchen`, mask của các vật trên mặt bàn (ví dụ bowl) nằm sát/chồng lấn về mặt thị giác với vùng dining table. Đây là vị trí dễ nhầm ranh giới giữa vật thể đặt trên bàn và mặt bàn. Nếu guideline đã quy định tách riêng từng instance thì làm theo quy tắc đó; nếu biên không thể xác định chắc chắn thì escalation lên Lab Coach, không tự suy đoán.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn lớp cho toàn bộ ảnh, gồm class ID và tên lớp theo taxonomy | Ảnh traffic có top-1 là `cab` với score `0.510915`; prediction là dự đoán của model, không tự chứng minh đó là nhãn đúng | Đọc guideline, xác định chủ đề chính theo tiêu chí và gán một lớp phù hợp | Kiểm tra lớp có phản ánh nội dung ảnh, đúng taxonomy và nhất quán giữa các ảnh |
| Phát hiện vật thể | Một bounding box và class cho từng vật thể; box biểu diễn bằng `[x1, y1, x2, y2]` pixel | Ảnh kitchen có 11 record ở ngưỡng 0.35; có người ở sát mép trái bị cắt một phần; box có thể thiếu/thừa vùng hoặc nhầm vật thể | Vẽ box quanh phần vật thể nhìn thấy, chọn đúng class và tuân thủ quy tắc tọa độ | Kiểm tra class, vị trí/kích thước box, vật thể bị bỏ sót, box trùng và trường hợp bị che/cắt mép |
| Instance segmentation | Một mask/polygon và class cho từng instance; mỗi instance có ID riêng | Biên bowl và dining table ở gần nhau nên dễ lẫn ranh giới; polygon cần bám đúng đường biên | Vẽ mask sát biên phần nhìn thấy, tách riêng từng instance và theo guideline | Kiểm tra độ chính xác đường biên, phần mask lấn/thiếu, việc tách instance và tính nhất quán |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của tôi cho bài lab này là: chỉ sử dụng dữ liệu được cung cấp cho mục đích học tập, không chia sẻ hoặc tải dữ liệu lên nơi không được phép, đồng thời không đưa thông tin cá nhân vào báo cáo/output.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng xử lý và báo cho Lab Coach/giảng viên phụ trách để được hướng dẫn.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
