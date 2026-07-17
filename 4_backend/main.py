import io, base64, time, re, sys, os, tempfile
from pathlib import Path
from typing import List
import cv2, numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO
import torch
from torchvision import transforms

sys.path.append(str(Path(__file__).parent.parent / "3_recognition"))
from crnn_model import CRNN, NUM_CLASSES, decode_prediction

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

print("📦 Đang load models...")
detector = YOLO(r"D:\license-plate-recognition\4_backend\best.pt")

# ---------------------------------------------------------------------
# 2 recognizer riêng biệt cho one_row và two_row.
# two_row CẦN bước tách nửa trên/nửa dưới rồi ghép ngang trước khi
# đưa vào model (split_and_concat_rows) — vì model được train với
# tiền xử lý này (đọc trái->phải mới đúng thứ tự ký tự trong label).
# ---------------------------------------------------------------------
RECOGNIZER_CONFIGS = {
    "one_row": {
        "weights":    r"D:\license-plate-recognition\3_recognition\runs\recognition_one_row_v2\best_crnn.pt",
        "img_height": 48,
        "img_width":  192,
        "split_rows": False,
    },
    "two_row": {
        "weights":    r"D:\license-plate-recognition\3_recognition\runs\recognition_two_row_v2\best_crnn.pt",
        "img_height": 48,
        "img_width":  192,
        "split_rows": True,
    },
}

recognizers = {}
transforms_by_type = {}
split_rows_by_type = {}
for plate_type, cfg in RECOGNIZER_CONFIGS.items():
    model = CRNN(img_height=cfg["img_height"], num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(cfg["weights"], map_location="cpu"))
    model.eval()
    recognizers[plate_type] = model
    transforms_by_type[plate_type] = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((cfg["img_height"], cfg["img_width"])),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    split_rows_by_type[plate_type] = cfg["split_rows"]
print("✅ Sẵn sàng!")

class PlateResult(BaseModel):
    plate_text: str
    confidence: float
    plate_type: str
    bbox: List[int]
    image_crop: str

class PredictResponse(BaseModel):
    results: List[PlateResult]
    total_plates: int
    inference_ms: float
    annotated_image: str


# ---------------------------------------------------------------------
# SỬA: bỏ tighten_bbox() dùng threshold OTSU — kỹ thuật này đo vùng chữ
# dựa vào độ sáng/tối, RẤT NHẠY với ánh sáng phản chiếu (biển ô tô sáng
# bóng, ánh đèn...), dễ đo sai tỷ lệ khung hình -> đoán sai loại biển.
# Dùng thẳng tỷ lệ khung hình THÔ từ chính bbox YOLO đã phát hiện —
# đơn giản hơn nhưng ổn định hơn nhiều với ảnh nhiễu/ánh sáng thực tế.
# ---------------------------------------------------------------------
def determine_plate_type_from_bbox(w: int, h: int) -> str:
    """CHỈ dùng làm phương án dự phòng / gợi ý ban đầu.

    NGƯỠNG ĐÃ HIỆU CHỈNH LẠI dựa trên dữ liệu THỰC TẾ (không phải suy
    đoán theo tỷ lệ ảnh crop lúc train, vì ảnh chụp thật có góc
    nghiêng/phối cảnh khiến bbox khác hẳn ảnh crop chuẩn trong dataset):
        - Biển 1 hàng thật (đo thực tế qua debug): ratio ~1.45 - 1.73
        - Biển 2 hàng thật (đo thực tế qua debug): ratio ~1.15 - 1.20
    -> Chọn ngưỡng 1.35 nằm giữa 2 nhóm này.
    """
    ratio = w / max(h, 1)
    return "one_row" if ratio > 1.35 else "two_row"


# Biển số VN (dù 1 hàng hay 2 hàng) đều có cấu trúc chung: 2 số + 1 chữ
# + 4-6 số (dữ liệu train không chứa dấu '-' hay '.'). Dùng để LỌC RA
# kết quả OCR vô nghĩa, không dùng để phân biệt one_row/two_row (vì cấu
# trúc ký tự giống nhau ở cả 2 loại, chỉ khác cách xếp hình).
VALID_PLATE_REGEX = re.compile(r'^\d{2}[A-Z]\d{4,6}$')

def looks_valid(text: str) -> bool:
    return bool(VALID_PLATE_REGEX.match(text.strip().upper()))


def split_and_concat_rows(img: Image.Image) -> Image.Image:
    """PHẢI khớp chính xác với tiền xử lý lúc train two_row:
    tách nửa trên/dưới rồi ghép ngang thành 1 ảnh dài."""
    w, h = img.size
    top    = img.crop((0, 0, w, h // 2))
    bottom = img.crop((0, h // 2, w, h))
    new_img = Image.new(img.mode, (w * 2, h // 2))
    new_img.paste(top, (0, 0))
    new_img.paste(bottom, (w, 0))
    return new_img


def recognize(crop_np, plate_type: str):
    model = recognizers[plate_type]
    transform = transforms_by_type[plate_type]
    pil = Image.fromarray(cv2.cvtColor(crop_np, cv2.COLOR_BGR2RGB))
    if split_rows_by_type[plate_type]:
        pil = split_and_concat_rows(pil)
    tensor = transform(pil).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
    texts = decode_prediction(logits)
    return texts[0] if texts else ""


def process_image(pil_image):
    start = time.time()
    img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    det = detector(img, imgsz=640, conf=0.25)[0]
    results = []
    for box in det.boxes:
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        conf = float(box.conf[0])
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        w, h = x2 - x1, y2 - y1

        # SỬA: Ưu tiên đoán loại biển qua tỷ lệ bbox làm "ứng viên chính"
        # trước, chỉ thử model còn lại khi ứng viên chính đọc ra kết quả
        # KHÔNG hợp lệ. Tránh trường hợp: ảnh biển 1 hàng bị ép đọc qua
        # model two_row (cắt sai vị trí) nhưng vô tình "bịa" ra chuỗi
        # tình cờ đúng định dạng -> thắng nhầm kết quả gần đúng của
        # model one_row (đúng loại) chỉ vì bị lệch 1 ký tự nhỏ.
        guess_type = determine_plate_type_from_bbox(w, h)
        other_type = "two_row" if guess_type == "one_row" else "one_row"

        text_guess = recognize(crop, guess_type)
        if looks_valid(text_guess):
            plate_type, text = guess_type, text_guess
        else:
            text_other = recognize(crop, other_type)
            if looks_valid(text_other):
                plate_type, text = other_type, text_other
            else:
                # Cả 2 đều không hợp lệ -> vẫn tin theo bbox (nhiều khả
                # năng đúng LOẠI biển hơn, dù nội dung có thể sai ký tự)
                plate_type, text = guess_type, text_guess

        _, buf = cv2.imencode(".jpg", crop)
        results.append(PlateResult(
            plate_text=text or "Không đọc được",
            confidence=round(conf, 4),
            plate_type=plate_type,
            bbox=[x1, y1, x2, y2],
            image_crop=base64.b64encode(buf).decode()
        ))
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    _, buf = cv2.imencode(".jpg", img)
    return PredictResponse(
        results=results, total_plates=len(results),
        inference_ms=round((time.time() - start) * 1000, 1),
        annotated_image=base64.b64encode(buf).decode()
    )


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/predict/image", response_model=PredictResponse)
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    return process_image(Image.open(io.BytesIO(contents)).convert("RGB"))


@app.post("/api/predict/video")
async def predict_video(file: UploadFile = File(...), frame_interval: int = 10):
    """
    Xử lý video: đọc từng frame, cứ mỗi `frame_interval` frame thì chạy
    detect + recognize 1 lần (bỏ qua các frame giữa để tăng tốc độ).
    Trả về danh sách các frame đã xử lý, mỗi frame gồm:
      { frame_index, timestamp_ms, results: [PlateResult...] }
    """
    contents = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    frame_results = []
    try:
        cap = cv2.VideoCapture(tmp_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_index % max(frame_interval, 1) == 0:
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                resp = process_image(pil_image)
                timestamp_ms = (frame_index / fps) * 1000
                frame_results.append({
                    "frame_index": frame_index,
                    "timestamp_ms": round(timestamp_ms, 1),
                    "results": [
                        {
                            "plate_text": r.plate_text,
                            "confidence": r.confidence,
                            "plate_type": r.plate_type,
                            "bbox": r.bbox,
                            "image_crop": r.image_crop,
                        }
                        for r in resp.results
                    ],
                })
            frame_index += 1
        cap.release()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return frame_results


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)