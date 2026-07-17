"""
BƯỚC 3A: Định nghĩa model CRNN (CNN + BiLSTM + CTC)
Kiến trúc: VGG-style CNN → BiLSTM × 2 → CTC Loss

SỬA: Chỉ pool cả H và W ở 2 lớp CNN đầu tiên. Các lớp pool sau đó
chỉ nén chiều CAO (height), giữ nguyên chiều RỘNG (width) — vì width
chính là số time steps (T) đưa vào CTC.
"""

import torch
import torch.nn as nn

CHARS = "0123456789ABCDEFGHIKLMNPRSTUVXYZ-"
CHAR2IDX = {c: i + 1 for i, c in enumerate(CHARS)}
IDX2CHAR = {i + 1: c for i, c in enumerate(CHARS)}
NUM_CLASSES = len(CHARS) + 1


class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.fc  = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out)


class CRNN(nn.Module):
    def __init__(self, img_height=32, num_classes=NUM_CLASSES, hidden_size=256):
        super().__init__()
        assert img_height % 16 == 0, "img_height phải chia hết cho 16"

        def conv_block(in_c, out_c, pool=None):
            layers = [
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            ]
            if pool == "both":
                layers.append(nn.MaxPool2d(2, 2))
            elif pool == "height":
                layers.append(nn.MaxPool2d((2, 1), (2, 1)))
            return nn.Sequential(*layers)

        self.cnn = nn.Sequential(
            conv_block(1,   64,  pool="both"),
            conv_block(64,  128, pool="both"),
            conv_block(128, 256, pool=None),
            conv_block(256, 256, pool="height"),
            conv_block(256, 512, pool=None),
            conv_block(512, 512, pool="height"),
            conv_block(512, 512, pool=None),
            nn.Conv2d(512, 512, kernel_size=(img_height // 16, 1)),
        )

        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, num_classes),
        )

    def forward(self, x):
        feat = self.cnn(x)
        feat = feat.squeeze(2)
        feat = feat.permute(2, 0, 1)
        out = self.rnn(feat)
        return out


def decode_prediction(logits):
    _, preds = logits.max(2)
    preds = preds.transpose(0, 1)
    results = []
    for pred in preds:
        chars = []
        prev = 0
        for p in pred.tolist():
            if p != prev and p != 0:
                chars.append(IDX2CHAR.get(p, ""))
            prev = p
        results.append("".join(chars))
    return results