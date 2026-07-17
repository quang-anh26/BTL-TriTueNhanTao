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
│   ├── main.py                    # FastAPI backend — detect + recognize
│   ├── best.pt                    # Model detection đã train
│   └── requirements.txt
└── 5_frontend/                    # React + Vite + Tailwind
    └── src/
        ├── pages/{UploadPage,VideoPage,HistoryPage}.jsx
        ├── components/PlateCard.jsx
        └── services/api.js
```

## Kiến trúc xử lý (pipeline)

```
Ảnh/Video đầu vào
      │
      ▼
[1] YOLOv8 Detection ──→ Tìm bbox vùng biển số
      │
      ▼
[2] Đoán loại biển (tỷ lệ bbox: w/h > 1.35 → one_row, ngược lại → two_row)
      │
      ▼
[3] Recognize bằng model tương ứng
      │  (nếu kết quả không khớp định dạng biển số hợp lệ → thử model còn lại)
      ▼
Text biển số + độ tin cậy
```

Với video: xử lý theo `frame_interval` (bỏ qua N-1 frame giữa mỗi lần xử lý để tăng tốc), sau đó **gộp kết quả bằng Levenshtein distance + voting** — các lần đọc gần giống nhau qua nhiều frame được coi là cùng 1 biển số thật, chọn ra kết quả xuất hiện nhiều nhất.

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
11. **Frontend gộp biển số theo so khớp tuyệt đối** → cùng 1 biển đọc lệch 1-2 ký tự ở các frame khác nhau bị tính thành nhiều biển khác nhau → sửa bằng thuật toán Levenshtein distance + voting.

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

## Giới hạn đã biết

- Độ chính xác trên ảnh thực tế (nhiễu, chói sáng, góc nghiêng) thấp hơn đáng kể so với tập validation sạch — đây là giới hạn chung của mọi hệ thống OCR, kể cả các hệ thống thương mại/nhà nước (vốn cũng luôn cần con người xác nhận lại kết quả AI trước khi dùng chính thức).
- Việc gộp biển số theo video (Levenshtein) có thể nhầm 2 biển số THẬT SỰ khác nhau nhưng chỉ lệch 1-2 ký tự thành cùng 1 biển — trường hợp hiếm gặp trong thực tế.
- Chưa có bước hậu xử lý tự động sửa lỗi ký tự dễ nhầm lẫn (8↔B, 0↔D...) — quyết định không làm vì rủi ro "sửa nhầm" trường hợp đúng.

## Định hướng cải thiện tiếp theo (nếu cần)

- Data augmentation (nhiễu, mờ, thay đổi sáng) khi train để tăng độ bền với ảnh thực tế.
- Fine-tune thêm bằng ảnh CCTV/camera thực tế của chính hệ thống sẽ triển khai.
- Thêm cờ "độ tin cậy thấp — cần xác nhận thủ công" khi cả 2 model đều không cho ra kết quả hợp lệ.