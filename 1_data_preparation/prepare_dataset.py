"""
BƯỚC 1: Chuẩn bị dataset detection + recognition từ dữ liệu VNLP gốc

SỬA so với bản gốc:
  1. Tên file ảnh recognition giờ có prefix loại biển:
       "{plate_type}_{plate_text}_{counter}.jpg"
     Ví dụ: "one_row_51G12345_0.jpg", "two_row_51G12345_1.jpg"
     -> Bước train sau này biết CHẮC CHẮN loại biển, không cần đoán
        qua tỷ lệ width/height (có thể sai với ảnh biên/méo).
  2. plate_text được chuẩn hóa .upper() ngay khi lưu, vì biển số
     VN luôn viết hoa nhưng dữ liệu gốc có lẫn chữ thường.
"""

import os
import re
import shutil
import random
from pathlib import Path
from PIL import Image

DATASET_DIR = Path("D:/license-plate-recognition/detection")
OUTPUT_DIR  = Path("D:/license-plate-recognition/data/processed")

TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
SEED        = 42

def parse_filename(filepath):
    name = Path(filepath).stem
    parts = name.split("_")
    plate_idx = None
    for i, p in enumerate(parts):
        if re.search(r'[A-Za-z]', p) and len(p) >= 5:
            plate_idx = i
            break
    if plate_idx is None:
        return None
    plate_text = parts[plate_idx]
    coords = parts[plate_idx + 1:]
    if len(coords) < 4:
        return None
    try:
        x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
    except:
        return None
    return plate_text, x1, y1, x2, y2

def prepare(list_file, image_dir, plate_type):
    if not Path(list_file).exists():
        print(f"Không tìm thấy {list_file}")
        return
    lines = Path(list_file).read_text(encoding="utf-8").strip().splitlines()
    random.seed(SEED)
    random.shuffle(lines)
    n = len(lines)
    train_end = int(n * TRAIN_RATIO)
    val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))
    splits = {
        "train": lines[:train_end],
        "val":   lines[train_end:val_end],
        "test":  lines[val_end:],
    }
    counters = {"train": 0, "val": 0, "test": 0}
    for split, items in splits.items():
        (OUTPUT_DIR / "detection" / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "detection" / "labels" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "recognition" / split).mkdir(parents=True, exist_ok=True)
        for line in items:
            line = line.strip()
            if not line:
                continue
            filename = Path(line).name
            img_path = Path(image_dir) / filename
            if not img_path.exists():
                continue
            result = parse_filename(filename)
            if result is None:
                continue
            plate_text, x1, y1, x2, y2 = result
            plate_text = plate_text.upper()  # chuẩn hóa hoa/thường ngay từ đầu
            try:
                with Image.open(img_path) as img:
                    img = img.convert("RGB")
                    W, H = img.size
                    xc = (x1 + x2) / 2 / W
                    yc = (y1 + y2) / 2 / H
                    bw = (x2 - x1) / W
                    bh = (y2 - y1) / H
                    shutil.copy(img_path, OUTPUT_DIR / "detection" / "images" / split / filename)
                    with open(OUTPUT_DIR / "detection" / "labels" / split / (Path(filename).stem + ".txt"), "w") as f:
                        f.write(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
                    crop = img.crop((x1, y1, x2, y2))
                    crop = crop.resize((160, 128) if plate_type == "two_row" else (320, 64))
                    # Tên file có prefix plate_type để bước train sau biết chắc chắn loại biển
                    out_name = f"{plate_type}_{plate_text}_{counters[split]}.jpg"
                    crop.save(OUTPUT_DIR / "recognition" / split / out_name)
                    counters[split] += 1
            except Exception as e:
                print(f"Lỗi {filename}: {e}")
    print(f"[{plate_type}] train={counters['train']} val={counters['val']} test={counters['test']}")

if __name__ == "__main__":
    print("Bắt đầu chuẩn bị dataset...\n")
    prepare(DATASET_DIR / "list_one_row.txt", DATASET_DIR / "one_row", "one_row")
    prepare(DATASET_DIR / "list_two_row.txt", DATASET_DIR / "two_rows", "two_row")
    prepare(DATASET_DIR / "list_two_rows_label_xe_may.txt", DATASET_DIR / "two_rows_label_xe_may", "two_row")
    print("\nXong! Dataset sẵn sàng để train.")