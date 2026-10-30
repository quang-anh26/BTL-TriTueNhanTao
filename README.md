# Hệ thống Nhận dạng Biển số xe Việt Nam (VNLP)

Hệ thống nhận dạng biển số xe tự động, gồm 2 giai đoạn: **Detection** (YOLOv8 — tìm vùng biển số) và **Recognition** (CRNN — đọc ký tự), hỗ trợ cả biển 1 hàng (ô tô) và biển 2 hàng (ô tô + xe máy).

## Kết quả hiện tại

| Thành phần | Chỉ số | Kết quả |
|---|---|---|
| Detection (YOLOv8) | Precision / Recall / mAP50 | 100% / 99.93% / 99.50% |
| Recognition — one_row | Val Accuracy | 98.59% |
| Recognition — two_row (gồm cả xe máy) | Val Accuracy | 95.33% |

> Lưu ý: đây là kết quả trên tập validation (ảnh crop chuẩn từ dataset VNLP). Với ảnh thực tế (nhiễu, ánh sáng chói, góc nghiêng), độ chính xác sẽ thấp hơn — đây là giới hạn tự nhiên của mọi hệ thống OCR, không phải lỗi hệ thống.

## Cấu trúc thư mục

```
license-plate-recognition/
├── 1_data_preparation/
│   └── prepare_dataset.py         # Tách dataset gốc → train/val/test cho detection + recognition
├── 2_detection/
│   └── train_detection.py         # Train YOLOv8 phát hiện vùng biển số
├── 3_recognition/
│   ├── crnn_model.py              # Kiến trúc CRNN (CNN + BiLSTM + CTC)
│   ├── train_recognition.py       # Train CRNN — tự động train cả one_row lẫn two_row
│   └── runs/
│       ├── recognition_one_row_v2/best_crnn.pt   # Model one_row (48x192)
│       └── recognition_two_row_v2/best_crnn.pt   # Model two_row (48x192, có split_rows)
├── 4_backend/
│   ├── main.py                    # FastAPI API và WebSocket xử lý video trực tiếp
│   ├── video_engine.py            # ROI cấu hình sẵn, theo dõi, OCR và bộ đếm xe
│   ├── tracker.py                 # Gán ID theo dõi ổn định qua các frame
│   ├── models_engine.py           # Bộ suy luận phương tiện, biển số và OCR
│   ├── vehicle_type.onnx          # Phân loại ngữ cảnh biển số xe máy/không phải xe máy
│   ├── train_vehicle_type.py      # Train classifier từ ảnh bối cảnh biển số đã gán loại xe
│   ├── yolov8n.onnx               # YOLOv8n COCO: phát hiện/phân loại phương tiện
│   ├── plate_detector.onnx        # YOLO: phát hiện riêng vùng biển số
│   ├── crnn_one_row.onnx          # CRNN OCR biển số một hàng
│   ├── crnn_two_row.onnx          # CRNN OCR biển số hai hàng
│   └── requirements.txt
└── 5_frontend/                    # React + Vite + Tailwind
    └── src/
        ├── pages/{UploadPage,VideoPage,HistoryPage}.jsx
        ├── components/PlateCard.jsx
        └── services/api.js
```

## Kiến trúc xử lý (pipeline)

```
Ảnh tĩnh: YOLO phát hiện biển số → chọn CRNN theo tỷ lệ crop → OCR

Video:
  YOLOv8n COCO phát hiện xe (car, motorcycle, bus, truck)
      → tracker kết hợp IoU và khoảng cách tâm để duy trì ID qua thay đổi tỷ lệ/hộp
      → YOLO riêng phát hiện biển số → CRNN OCR → gắn biển số vào xe
      → ghi nhận một lần khi điểm đáy/tâm xe nằm trong ROI cấu hình sẵn
      → trình duyệt phát video gốc, canvas chỉ vẽ khung quanh phương tiện theo timestamp từ WebSocket
```

Model phương tiện và model biển số là các model riêng. `yolov8n.onnx` là YOLOv8n COCO dùng để phát hiện phương tiện; `plate_detector.onnx` chỉ tìm vùng biển số; `crnn_*.onnx` đọc ký tự. `vehicle_type.onnx` là classifier nhị phân bổ sung, train từ vùng ảnh quanh biển số trong `detection/two_rows_label_xe_may` (xe máy) và các nhóm biển khác (không phải xe máy). Khi YOLO COCO bỏ sót xe hoặc nhầm xe máy thành ô tô, classifier này dùng ngữ cảnh biển số để xác nhận và tạo bbox ước lượng nếu cần. Kết quả classifier này là bổ sung cho model COCO, không thay thế đánh giá trên camera triển khai.

Classifier được huấn luyện 4 epoch với validation tách theo biển số: accuracy 99.28%; tại ngưỡng xe máy 0.9, precision 99.63% và recall 98.00%. Đây là kết quả nội bộ trên dữ liệu hiện có, không đảm bảo tương đương trên camera khác. Có thể tái tạo bằng `python 4_backend/train_vehicle_type.py` khi đã cài PyTorch và có dữ liệu trong thư mục `detection/`.

Video gốc được phát bằng HTML5 `<video>` ở tốc độ x1; WebSocket gửi tọa độ xe và timestamp của các frame được phân tích để canvas vẽ khung phương tiện theo các mẫu đã xác nhận, nội suy giữa các mẫu và không ngoại suy ra phía trước. Nếu luồng phân tích lệch quá 0.12–0.35 giây so với video, overlay bị ẩn thay vì giữ khung xe cũ trên cảnh mới. ROI chỉ được dùng nội bộ để đếm, không được phủ thành khung lớn trên video. YOLO COCO dùng ngưỡng tối thiểu 0.35 cho ô tô/xe buýt/xe tải và 0.15 riêng cho xe máy; NMS thực hiện riêng theo từng lớp, sau đó loại các box khác lớp gần như trùng khít để xe nhỏ không bị xe lớn khác lớp loại bỏ mà cùng một xe cũng không bị đếm hai lần do nhầm lớp. Tracker kết hợp IoU và khoảng cách tâm để giữ ID ổn định hơn khi hộp đổi tỷ lệ hoặc ít chồng lấp. Track được hiển thị/đếm sau hai mẫu phát hiện liên tiếp, hoặc sớm hơn với độ tin cậy cao (0.7 cho các lớp khác, 0.45 cho xe máy). Backend lấy mẫu phát hiện xe khoảng 5 lần/giây ban đầu và dò biển số mỗi 3 mẫu; khi còn track hoạt động chưa đọc được biển số, detector biển số chạy ở mọi frame được lấy mẫu để có thêm cơ hội OCR. Nếu suy luận chậm, khoảng lấy mẫu tăng để theo kịp video. Vì vậy video không bị giảm tốc, nhưng xe/biển số xuất hiện giữa các mẫu có thể bị bỏ sót. OCR chuẩn hóa dấu phân cách và sửa một số nhầm lẫn OCR chữ/số tại ký tự thứ ba theo định dạng biển Việt Nam; biển số vẫn cần được xác nhận qua ít nhất hai lần đọc.

Vùng đường giám sát và vùng đếm được cấu hình trong `4_backend/video_engine.py` bằng tọa độ chuẩn hóa `PRESET_ROI_NORM`; ROI mẫu bao phủ cả hai hướng lưu thông, chỉ lọc detection và không được vẽ đè lên video. Xe được theo dõi cả khi còn ngoài ROI; overlay chỉ hiện khi điểm đáy/tâm hộp xe nằm trong ROI và khung biến mất khi xe ra khỏi vùng hoặc không còn được phát hiện. Không cần thao tác trên giao diện. Bộ đếm chỉ ghi nhận track một lần khi điểm đáy/tâm xe đi vào vùng đếm cấu hình sẵn.

## Các lỗi lớn đã phát hiện và sửa trong quá trình phát triển

1. **Đường dẫn dữ liệu sai (Windows path trên Colab)** → tạo bản `_colab.py` riêng với path `/content/...`.
2. **Label không chuẩn hóa hoa/thường** → ~50% dữ liệu bị hỏng nhãn ngầm (ký tự thường bị coi là CTC blank) → thêm `.upper()`.
3. **Kiến trúc CNN pool cả chiều rộng lẫn chiều cao ở mọi lớp** → time steps (T) cho CTC quá ngắn (T=8 với ảnh 128px rộng), không đủ chỗ decode biển 8-9 ký tự → sửa: chỉ pool chiều rộng ở 2 lớp đầu, giữ nguyên chiều rộng ở các lớp sau (T=32).
4. **Biển 2 hàng bị đọc sai cấu trúc** — CTC đọc ảnh trái→phải, nhưng biển 2 hàng có ký tự chồng theo chiều dọc tại cùng 1 cột ảnh → sửa: tách ảnh 2 hàng thành nửa trên/dưới, ghép ngang lại trước khi đưa vào model (`split_and_concat_rows`).
5. **Resolution `two_row` quá thấp (32×128)** → tăng lên `48×192`, accuracy nhảy từ 78% → 95.33%.
6. **Resolution `one_row` cũng được tăng lên 48×192** để tăng độ bền với ảnh thực tế nhiễu (dù Val Accuracy trên tập sạch chỉ nhích nhẹ, mục tiêu là độ bền ngoài đời thực, không phải điểm số trên tập sạch).
7. **Backend thiếu endpoint `/api/predict/video`** mà frontend đã gọi sẵn → bổ sung, xử lý theo frame + trả kết quả dạng danh sách.
8. **Đoán loại biển qua `tighten_bbox()` (threshold OTSU)** quá nhạy với ánh sáng/phản chiếu → bỏ, dùng thẳng tỷ lệ bbox thô từ YOLO.
9. **Ngưỡng phân biệt one_row/two_row (2.2) sai so với thực tế** — bbox thật từ ảnh chụp có góc nghiêng/phối cảnh khác hẳn ảnh crop chuẩn lúc train → hiệu chỉnh lại ngưỡng theo dữ liệu thực đo được (1.35).
10. **Logic chọn model ưu tiên sai** — khi cả 2 model đều "hợp lệ theo regex", có thể chọn nhầm kết quả rác (nhưng tình cờ đúng định dạng) của model sai loại → sửa: ưu tiên model được đoán đúng bởi bbox trước, chỉ thử model còn lại khi model chính thất bại.
11. **OCR video dao động giữa các frame** → chỉ xác nhận chuỗi biển số hợp lệ sau khi cùng một track đọc được chuỗi đó ở ít nhất hai frame.

## Hướng dẫn train lại từ đầu (trên Google Colab)

```python
# 1. Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Giải nén dữ liệu gốc
!unzip -q "/content/drive/MyDrive/detection.zip" -d /content
# (nếu bị lồng thư mục detection/detection, chạy thêm:)
!mv /content/detection/detection/* /content/detection/ 2>/dev/null
!rmdir /content/detection/detection 2>/dev/null

# 3. Tạo thư mục code, ghi các file .py bằng %%writefile
#    (prepare_dataset_colab.py, crnn_model.py, train_recognition.py)

# 4. Chuẩn bị dataset
!python /content/1_data_preparation/prepare_dataset_colab.py

# 5. Train recognition (tự động cả one_row + two_row)
import sys
sys.path.append('/content/3_recognition')
from train_recognition import train
train()

# 6. Lưu checkpoint về Drive
import shutil, os
os.makedirs("/content/drive/MyDrive/model_checkpoints", exist_ok=True)
shutil.copy("/content/runs/recognition_one_row_v2/best_crnn.pt",
            "/content/drive/MyDrive/model_checkpoints/one_row_best_crnn.pt")
shutil.copy("/content/runs/recognition_two_row_v2/best_crnn.pt",
            "/content/drive/MyDrive/model_checkpoints/two_row_best_crnn.pt")
```

Train Detection (YOLOv8) tương tự, dùng `train_detection.py` với file `.yaml` trỏ tới `data/processed/detection/`.

## Chạy hệ thống (local)

**Backend:**
```powershell
cd 4_backend
pip install -r requirements.txt
python main.py
```
Kiểm tra: `http://localhost:8000/api/health`

**Frontend:**
```powershell
cd 5_frontend
npm install
npm run dev
```

Vite proxy `/api` và `/ws` tới backend local. Khi frontend/backend chạy trên hai host riêng, đặt `VITE_API_BASE_URL` (ví dụ `https://api.example.com/api`) và `VITE_WS_BASE_URL` (ví dụ `wss://api.example.com/ws`) trước khi build frontend.

## Giới hạn đã biết

- Độ chính xác trên ảnh thực tế (nhiễu, chói sáng, góc nghiêng) thấp hơn đáng kể so với tập validation sạch — đây là giới hạn chung của mọi hệ thống OCR, kể cả các hệ thống thương mại/nhà nước (vốn cũng luôn cần con người xác nhận lại kết quả AI trước khi dùng chính thức).
- Tracker dựa trên IoU, nên xe bị che khuất hoặc mất phát hiện quá lâu có thể nhận ID mới và bị đếm lại.
- Detector phương tiện là YOLO COCO tổng quát, không được fine-tune theo camera này. Backend lọc confidence, vùng đường và kết hợp xác nhận theo track để giảm khung nhầm; các vật thể có hình dáng giống xe và bị nhận nhầm ổn định vẫn cần dữ liệu camera có nhãn để cải thiện triệt để.
- Video phát ở x1; backend lấy mẫu phát hiện để theo kịp năng lực máy. Khi stride tăng, xe/biển số xuất hiện giữa các mẫu có thể bị bỏ sót; vị trí khung được nội suy giữa hai lần phát hiện đã xác nhận và không ngoại suy quá mẫu mới nhất.
- Chỉ hiệu chỉnh nhầm lẫn chữ/số ở vị trí ký tự chữ theo định dạng biển; chưa có hậu xử lý như 0↔D ở các vị trí khác vì rủi ro sửa nhầm trường hợp đúng.
- ROI là vùng mẫu dùng tọa độ tương đối của khung hình; cần sửa `PRESET_ROI_NORM` trong mã nếu góc quay camera khác.
- OCR video ưu tiên tránh hiển thị kết quả không ổn định; biển số chỉ được xác nhận sau ít nhất hai lần đọc hợp lệ của cùng một track.

## Định hướng cải thiện tiếp theo (nếu cần)

- Data augmentation (nhiễu, mờ, thay đổi sáng) khi train để tăng độ bền với ảnh thực tế.
- Fine-tune thêm bằng ảnh CCTV/camera thực tế của chính hệ thống sẽ triển khai.
- Thêm cờ "độ tin cậy thấp — cần xác nhận thủ công" khi cả 2 model đều không cho ra kết quả hợp lệ.