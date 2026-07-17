"""
BƯỚC 3B: Train CRNN nhận dạng ký tự biển số — train tự động cả 2 model

SỬA QUAN TRỌNG cho two_row: CRNN đọc ảnh theo cột trái->phải, giả định
mỗi cột chỉ có 1 ký tự. Biển 2 dòng có 2 hàng ký tự CHỒNG THEO CHIỀU
DỌC -> tại 1 cột ảnh, model thấy cả ký tự hàng trên LẪN hàng dưới cùng
lúc, trong khi label lại yêu cầu đọc hàng trên xong mới đến hàng dưới.
Đây là xung đột cấu trúc khiến accuracy bị chặn cứng dù loss vẫn giảm.

Cách sửa: tách ảnh two_row làm 2 nửa (trên/dưới) rồi GHÉP NGANG lại
thành 1 ảnh dài duy nhất -> đọc trái->phải giờ đúng thứ tự
[hàng trên][hàng dưới], khớp với thứ tự ký tự trong label.
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

PLATE_CONFIGS = {
    "one_row": {
        "data_dir_candidates": ["/content/recognition_one_row", "/content/license-plate-recognition/data/processed/recognition"],
        "save_dir":   "runs/recognition_one_row",
        "img_height": 32,
        "img_width":  128,
        "split_rows": False,
    },
    "two_row": {
        "data_dir_candidates": ["/content/recognition_two_row", "/content/license-plate-recognition/data/processed/recognition"],
        "save_dir":   "runs/recognition_two_row_v2",
        # Tăng resolution so với 32x128 (đã chững ở ~78%) để giữ nhiều
        # chi tiết ký tự hơn — mỗi ký tự trong biển 2 dòng vốn nhỏ hơn.
        "img_height": 48,
        "img_width":  192,
        "split_rows": True,
    },
}

BASE_CONFIG = {
    "batch_size":  64,
    "epochs":      50,
    "lr":          1e-3,
    "hidden_size": 256,
    "device":      "cuda" if torch.cuda.is_available() else "cpu",
}


def split_and_concat_rows(img: Image.Image) -> Image.Image:
    """Tách ảnh biển 2 dòng làm nửa trên/nửa dưới rồi ghép NGANG lại
    thành 1 ảnh dài, để thứ tự đọc trái->phải khớp với thứ tự
    hàng-trên-trước-hàng-dưới-sau trong label."""
    w, h = img.size
    top    = img.crop((0, 0, w, h // 2))
    bottom = img.crop((0, h // 2, w, h))
    new_img = Image.new(img.mode, (w * 2, h // 2))
    new_img.paste(top, (0, 0))
    new_img.paste(bottom, (w, 0))
    return new_img


class PlateDataset(Dataset):
    def __init__(self, data_dir_candidates, plate_type, split, img_h, img_w, split_rows=False):
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
        self.split_rows = split_rows

        all_files = list(root.glob("*.jpg"))
        prefix = f"{plate_type}_"
        prefixed_files = [f for f in all_files if f.stem.startswith(prefix)]
        self.files = prefixed_files if prefixed_files else all_files
        self._prefix_len = len(prefix) if prefixed_files else 0

        self.img_h = img_h
        self.img_w = img_w
        self.resize_transform = transforms.Compose([
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
        label_str = stem.rsplit("_", 1)[0].upper()

        img = Image.open(path)
        if self.split_rows:
            img = split_and_concat_rows(img)
        image = self.resize_transform(img)

        label = torch.tensor([CHAR2IDX.get(c, 0) for c in label_str], dtype=torch.long)
        return image, label, len(label)


def collate_fn(batch):
    images, labels, lengths = zip(*batch)
    images  = torch.stack(images)
    labels  = torch.cat(labels)
    lengths = torch.tensor(lengths, dtype=torch.long)
    return images, labels, lengths


def train_one_type(plate_type: str):
    cfg = {**BASE_CONFIG, **PLATE_CONFIGS[plate_type]}
    os.makedirs(cfg["save_dir"], exist_ok=True)
    device = cfg["device"]

    print(f"\n{'='*60}")
    print(f"🚀 Train Recognition Model [{plate_type}] — Device: {device}")
    print(f"   img size: {cfg['img_height']}x{cfg['img_width']} | split_rows: {cfg['split_rows']}")

    train_ds = PlateDataset(cfg["data_dir_candidates"], plate_type, "train", cfg["img_height"], cfg["img_width"], cfg["split_rows"])
    val_ds   = PlateDataset(cfg["data_dir_candidates"], plate_type, "val",   cfg["img_height"], cfg["img_width"], cfg["split_rows"])
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
    return best_acc


def train(types=("one_row", "two_row")):
    results = {}
    for plate_type in types:
        results[plate_type] = train_one_type(plate_type)
    print(f"\n{'='*60}")
    print("📊 TỔNG KẾT")
    for plate_type, acc in results.items():
        if acc is not None:
            print(f"   {plate_type}: Best Val Accuracy = {acc:.4f}")
    return results


if __name__ == "__main__":
    train()