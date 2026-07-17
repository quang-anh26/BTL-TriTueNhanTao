"""
BƯỚC 3B: Train CRNN nhận dạng ký tự biển số
Phiên bản cuối — train TỰ ĐỘNG cả 2 model (one_row, two_row) trong 1 lần chạy.

Vì sao tách 2 model riêng:
  - Ảnh gốc one_row có tỷ lệ khung hình dài (~5:1, crop gốc 320x64)
  - Ảnh gốc two_row có tỷ lệ gần vuông (~1.25:1, crop gốc 160x128)
  Nếu ép chung 1 kích thước resize, ảnh two_row bị méo nặng -> accuracy
  bị chặn thấp dù loss huấn luyện rất tốt. Mỗi loại cần model + resize
  riêng để giữ đúng hình dạng ký tự.

Dataset: tự động nhận diện 1 trong 2 cấu trúc thư mục sau (ưu tiên
theo thứ tự), không cần chỉnh sửa gì thêm:
  1. Thư mục đã tách sẵn riêng biệt theo loại:
       /content/recognition_one_row/{train,val,test}/*.jpg
       /content/recognition_two_row/{train,val,test}/*.jpg
  2. Thư mục gộp chung, tên file có prefix loại biển (từ bản
     prepare_dataset.py mới):
       /content/recognition/{train,val,test}/one_row_*.jpg
       /content/recognition/{train,val,test}/two_row_*.jpg
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path

from crnn_model import CRNN, CHAR2IDX, NUM_CLASSES, IDX2CHAR, decode_prediction

# ============================================================
# CẤU HÌNH THEO TỪNG LOẠI BIỂN
# ============================================================
PLATE_CONFIGS = {
    "one_row": {
        "data_dir_candidates": ["/content/recognition_one_row", "/content/recognition"],
        "save_dir":   "runs/recognition_one_row",
        "img_height": 32,
        "img_width":  128,   # gần khớp tỷ lệ crop gốc 320x64 (5:1)
    },
    "two_row": {
        "data_dir_candidates": ["/content/recognition_two_row", "/content/recognition"],
        "save_dir":   "runs/recognition_two_row",
        "img_height": 64,
        "img_width":  80,    # gần khớp tỷ lệ crop gốc 160x128 (1.25:1)
    },
}

BASE_CONFIG = {
    "batch_size":  64,
    "epochs":      50,
    "lr":          1e-3,
    "hidden_size": 256,
    "device":      "cuda" if torch.cuda.is_available() else "cpu",
}


# ============================================================
# DATASET
# ============================================================
class PlateDataset(Dataset):
    """
    Hỗ trợ 2 dạng dữ liệu:
      - Thư mục riêng cho từng loại, tên file KHÔNG có prefix:
            "51G12345_0.jpg"
      - Thư mục gộp chung, tên file CÓ prefix loại biển:
            "one_row_51G12345_0.jpg"
    Tự phát hiện dựa trên việc có file nào bắt đầu bằng "{plate_type}_" không.
    """
    def __init__(self, data_dir_candidates, plate_type, split, img_h, img_w):
        root = None
        for cand in data_dir_candidates:
            p = Path(cand) / split
            if p.exists() and any(p.glob("*.jpg")):
                root = p
                break
        if root is None:
            raise FileNotFoundError(
                f"Không tìm thấy thư mục dữ liệu cho '{plate_type}/{split}'. "
                f"Đã thử: {[str(Path(c) / split) for c in data_dir_candidates]}"
            )
        self.root = root
        self.plate_type = plate_type

        all_files = list(root.glob("*.jpg"))
        prefix = f"{plate_type}_"
        prefixed_files = [f for f in all_files if f.stem.startswith(prefix)]
        # Nếu có file mang prefix -> đây là thư mục gộp chung, lọc đúng loại.
        # Nếu không -> đây đã là thư mục tách riêng, dùng toàn bộ file.
        self.files = prefixed_files if prefixed_files else all_files
        self._prefix_len = len(prefix) if prefixed_files else 0

        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((img_h, img_w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        stem = path.stem
        if self._prefix_len:
            stem = stem[self._prefix_len:]
        # "51G12345_0" -> "51G12345"
        label_str = stem.rsplit("_", 1)[0].upper()
        image = self.transform(Image.open(path))
        label = torch.tensor([CHAR2IDX.get(c, 0) for c in label_str], dtype=torch.long)
        return image, label, len(label)


def collate_fn(batch):
    images, labels, lengths = zip(*batch)
    images  = torch.stack(images)
    labels  = torch.cat(labels)
    lengths = torch.tensor(lengths, dtype=torch.long)
    return images, labels, lengths


# ============================================================
# TRAINING CHO 1 LOẠI BIỂN
# ============================================================
def train_one_type(plate_type: str):
    cfg = {**BASE_CONFIG, **PLATE_CONFIGS[plate_type]}
    os.makedirs(cfg["save_dir"], exist_ok=True)
    device = cfg["device"]

    print(f"\n{'='*60}")
    print(f"🚀 Train Recognition Model [{plate_type}] — Device: {device}")
    print(f"   img size: {cfg['img_height']}x{cfg['img_width']}")

    train_ds = PlateDataset(cfg["data_dir_candidates"], plate_type, "train", cfg["img_height"], cfg["img_width"])
    val_ds   = PlateDataset(cfg["data_dir_candidates"], plate_type, "val",   cfg["img_height"], cfg["img_width"])
    print(f"   Nguồn dữ liệu: {train_ds.root.parent}")
    print(f"   Số ảnh train: {len(train_ds)} | val: {len(val_ds)}")

    if len(train_ds) == 0:
        print(f"⚠️  Không có ảnh nào cho '{plate_type}', bỏ qua.")
        return None

    train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,  collate_fn=collate_fn, num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False, collate_fn=collate_fn, num_workers=2)

    model     = CRNN(img_height=cfg["img_height"], num_classes=NUM_CLASSES, hidden_size=cfg["hidden_size"]).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_acc = 0.0

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total_loss = 0
        for images, labels, lengths in train_dl:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            T, B, _ = logits.shape
            log_probs = logits.log_softmax(2)
            input_lengths = torch.full((B,), T, dtype=torch.long)

            loss = criterion(log_probs, labels, input_lengths, lengths)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_dl)

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels, lengths in val_dl:
                images = images.to(device)
                logits = model(images)
                preds  = decode_prediction(logits)

                offset = 0
                for pred, length in zip(preds, lengths.tolist()):
                    true_indices = labels[offset:offset + length].tolist()
                    true_str = "".join(IDX2CHAR.get(i, "") for i in true_indices)
                    if pred == true_str:
                        correct += 1
                    total += 1
                    offset += length

        acc = correct / total if total > 0 else 0
        scheduler.step(avg_loss)

        print(f"[{plate_type}] Epoch [{epoch:02d}/{cfg['epochs']}] Loss: {avg_loss:.4f} | Val Acc: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f"{cfg['save_dir']}/best_crnn.pt")
            print(f"  💾 Saved best model (acc={best_acc:.4f})")

    print(f"✅ [{plate_type}] Train xong! Best Val Accuracy: {best_acc:.4f}")
    print(f"   Best model: {cfg['save_dir']}/best_crnn.pt")
    return best_acc


def train():
    """Train tuần tự cả 2 model: one_row rồi two_row."""
    results = {}
    for plate_type in ["one_row", "two_row"]:
        results[plate_type] = train_one_type(plate_type)

    print(f"\n{'='*60}")
    print("📊 TỔNG KẾT")
    for plate_type, acc in results.items():
        if acc is not None:
            print(f"   {plate_type}: Best Val Accuracy = {acc:.4f}")
    return results


if __name__ == "__main__":
    train()