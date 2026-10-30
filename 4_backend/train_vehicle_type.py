import copy
import random
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "detection"
MODEL_PATH = Path(__file__).resolve().parent / "vehicle_type.onnx"
IMAGE_SIZE = 96
SEED = 7


def plate_group(image_path: Path) -> str:
    parts = image_path.stem.split("_")
    plate = parts[3] if len(parts) > 3 else image_path.stem
    return re.sub(r"[^A-Z0-9]", "", plate.upper())


class PlateContextDataset(Dataset):
    def __init__(self, examples, augment=False):
        self.examples = examples
        self.augment = augment

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        path, label = self.examples[index]
        parts = path.stem.split("_")
        x1, y1, x2, y2 = map(int, parts[-4:])
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"Could not read training image: {path}")

        height, width = image.shape[:2]
        box_width = max(1, x2 - x1)
        box_height = max(1, y2 - y1)
        center_x = (x1 + x2) // 2
        crop = image[
            max(0, int(y1 - 2 * box_height)):min(height, int(y2 + box_height)),
            max(0, int(center_x - 1.6 * box_width)):min(width, int(center_x + 1.6 * box_width)),
        ]
        if crop.size == 0:
            raise RuntimeError(f"Invalid plate context crop: {path}")

        crop = cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        if self.augment:
            if random.random() < 0.5:
                crop = cv2.flip(crop, 1)
            if random.random() < 0.6:
                crop = np.clip(
                    crop.astype(np.float32) * random.uniform(0.75, 1.25)
                    + random.uniform(-20, 20),
                    0,
                    255,
                ).astype(np.uint8)

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1))
        return tensor, torch.tensor(label, dtype=torch.long)


class VehicleContextClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        input_channels = 3
        for output_channels in (16, 32, 64, 96):
            layers.extend([
                nn.Conv2d(
                    input_channels,
                    output_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(output_channels),
                nn.SiLU(inplace=True),
            ])
            input_channels = output_channels

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(96, 2),
        )

    def forward(self, image):
        return self.classifier(self.features(image))


def build_splits():
    motorcycle = list((DATA_DIR / "two_rows_label_xe_may").glob("*.jpg"))
    other = list((DATA_DIR / "two_rows").glob("*.jpg"))
    other.extend((DATA_DIR / "one_row").glob("*.jpg"))
    if not motorcycle or not other:
        raise FileNotFoundError(f"Vehicle context images were not found under {DATA_DIR}")

    random.seed(SEED)
    random.shuffle(other)
    examples = [(path, 1) for path in motorcycle]
    examples.extend((path, 0) for path in other[:len(motorcycle)])

    groups = {}
    for example in examples:
        groups.setdefault(plate_group(example[0]), []).append(example)
    conflicting_groups = [
        group for group in groups.values()
        if len({label for _, label in group}) > 1
    ]
    if conflicting_groups:
        raise ValueError(
            f"Found {len(conflicting_groups)} license plates with conflicting vehicle labels"
        )

    group_names = list(groups)
    random.shuffle(group_names)
    validation_groups = set(group_names[:max(1, int(len(group_names) * 0.2))])
    training = []
    validation = []
    for group_name, group_examples in groups.items():
        target = validation if group_name in validation_groups else training
        target.extend(group_examples)
    return training, validation


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(min(4, torch.get_num_threads()))

    training_examples, validation_examples = build_splits()
    training = DataLoader(
        PlateContextDataset(training_examples, augment=True),
        batch_size=64,
        shuffle=True,
    )
    validation = DataLoader(
        PlateContextDataset(validation_examples),
        batch_size=128,
        shuffle=False,
    )

    model = VehicleContextClassifier()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()
    best_accuracy = 0.0
    best_state = None

    for epoch in range(1, 5):
        model.train()
        total_loss = 0.0
        for images, labels in training:
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct = total = true_positive = false_positive = false_negative = 0
        threshold_counts = {
            threshold: [0, 0, 0]
            for threshold in (0.8, 0.9, 0.95)
        }
        with torch.no_grad():
            for images, labels in validation:
                probabilities = torch.softmax(model(images), dim=1)[:, 1]
                predictions = probabilities >= 0.5
                correct += (predictions == labels.bool()).sum().item()
                total += labels.numel()
                true_positive += ((predictions) & (labels == 1)).sum().item()
                false_positive += ((predictions) & (labels == 0)).sum().item()
                false_negative += ((~predictions) & (labels == 1)).sum().item()
                for threshold, counts in threshold_counts.items():
                    selected = probabilities >= threshold
                    counts[0] += (selected & (labels == 1)).sum().item()
                    counts[1] += (selected & (labels == 0)).sum().item()
                    counts[2] += ((~selected) & (labels == 1)).sum().item()

        accuracy = correct / max(total, 1)
        print({
            "epoch": epoch,
            "loss": round(total_loss / max(len(training), 1), 4),
            "accuracy": round(accuracy, 4),
            "motorcycle_precision": round(
                true_positive / max(true_positive + false_positive, 1), 4
            ),
            "motorcycle_recall": round(
                true_positive / max(true_positive + false_negative, 1), 4
            ),
            "thresholds": {
                threshold: {
                    "precision": round(tp / max(tp + fp, 1), 4),
                    "recall": round(tp / max(tp + fn, 1), 4),
                }
                for threshold, (tp, fp, fn) in threshold_counts.items()
            },
        }, flush=True)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("Vehicle classifier training did not produce a valid model")

    model.load_state_dict(best_state)
    model.eval()
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE),
        str(MODEL_PATH),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"Saved vehicle type classifier to {MODEL_PATH}")


if __name__ == "__main__":
    main()
