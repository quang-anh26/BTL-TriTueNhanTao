import os, re, cv2, base64
import numpy as np
import onnxruntime as ort
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

CHARS = "0123456789ABCDEFGHIKLMNPRSTUVXYZ-"
IDX2CHAR = {i + 1: c for i, c in enumerate(CHARS)}
VALID_PLATE_REGEX = re.compile(r'^\d{2}[A-Z]{1,2}\d{4,6}$')

def normalize_plate_text(text: str) -> str:
    normalized = re.sub(r'[^A-Z0-9]', '', text.upper())
    if len(normalized) > 2:
        normalized = normalized[:2] + {
            '0': 'D',
            '1': 'I',
            '2': 'Z',
            '5': 'S',
            '6': 'G',
            '8': 'B',
        }.get(normalized[2], normalized[2]) + normalized[3:]
    return normalized


def looks_valid(text: str) -> bool:
    return bool(VALID_PLATE_REGEX.fullmatch(normalize_plate_text(text)))

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    h0, w0 = img.shape[:2]
    r = min(new_shape[0] / h0, new_shape[1] / w0)
    nw, nh = int(round(w0 * r)), int(round(h0 * r))
    pad_x = (new_shape[1] - nw) // 2
    pad_y = (new_shape[0] - nh) // 2
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_shape[0], new_shape[1], 3), color, dtype=np.uint8)
    canvas[pad_y : pad_y + nh, pad_x : pad_x + nw] = resized
    blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(blob, axis=0), r, pad_x, pad_y

class ModelsEngine:
    def __init__(self):
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = min(2, os.cpu_count() or 1)
        opts.inter_op_num_threads = 1
        available_providers = ort.get_available_providers()
        providers = (
            ['CUDAExecutionProvider', 'CPUExecutionProvider']
            if 'CUDAExecutionProvider' in available_providers
            else ['CPUExecutionProvider']
        )

        yolo_path = ROOT_DIR / 'yolov8n.onnx'
        if not yolo_path.exists():
            yolo_path = BASE_DIR / 'yolov8n.onnx'
        self.vehicle_session = ort.InferenceSession(str(yolo_path), opts, providers=providers)

        plate_det_path = BASE_DIR / 'plate_detector.onnx'
        self.plate_det_session = ort.InferenceSession(str(plate_det_path), opts, providers=providers)
        vehicle_type_path = BASE_DIR / 'vehicle_type.onnx'
        self.vehicle_type_session = ort.InferenceSession(
            str(vehicle_type_path), opts, providers=providers
        )

        crnn_one_path = BASE_DIR / 'crnn_one_row.onnx'
        crnn_two_path = BASE_DIR / 'crnn_two_row.onnx'
        self.crnn_one_session = ort.InferenceSession(str(crnn_one_path), opts, providers=providers)
        self.crnn_two_session = ort.InferenceSession(str(crnn_two_path), opts, providers=providers)

        # COCO mapping for vehicles
        self.vehicle_classes = {
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck'
        }

    def classify_motorcycle_context(self, img_bgr, plate_bbox):
        x1, y1, x2, y2 = plate_bbox
        height, width = img_bgr.shape[:2]
        box_width = max(1, x2 - x1)
        box_height = max(1, y2 - y1)
        center_x = (x1 + x2) // 2
        crop = img_bgr[
            max(0, int(y1 - 2 * box_height)):min(height, int(y2 + box_height)),
            max(0, int(center_x - 1.6 * box_width)):min(width, int(center_x + 1.6 * box_width)),
        ]
        if crop.size == 0:
            return 0.0

        resized = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_AREA)
        blob = resized[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        logits = self.vehicle_type_session.run(
            None,
            {self.vehicle_type_session.get_inputs()[0].name: blob[np.newaxis, :]},
        )[0][0]
        probabilities = np.exp(logits - np.max(logits))
        probabilities /= probabilities.sum()
        return float(probabilities[1])

    def suppress_vehicle_overlaps(self, candidates, nms_thresh):
        by_class = {}
        for candidate in candidates:
            by_class.setdefault(candidate['class_name'], []).append(candidate)

        class_filtered = []
        for class_candidates in by_class.values():
            boxes = []
            scores = []
            for candidate in class_candidates:
                x1, y1, x2, y2 = candidate['bbox']
                boxes.append([x1, y1, x2 - x1, y2 - y1])
                scores.append(candidate['conf'])

            indices = cv2.dnn.NMSBoxes(
                boxes, scores, 0.0, nms_thresh
            )
            class_filtered.extend(
                class_candidates[int(index)] for index in indices
            )

        class_filtered.sort(key=lambda item: item['conf'], reverse=True)
        kept = []
        for candidate in class_filtered:
            x1, y1, x2, y2 = candidate['bbox']
            area = max(0, x2 - x1) * max(0, y2 - y1)
            duplicate = False
            for existing in kept:
                ex1, ey1, ex2, ey2 = existing['bbox']
                intersection = (
                    max(0, min(x2, ex2) - max(x1, ex1))
                    * max(0, min(y2, ey2) - max(y1, ey1))
                )
                existing_area = max(0, ex2 - ex1) * max(0, ey2 - ey1)
                union = area + existing_area - intersection
                if union and intersection / union >= 0.8:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(candidate)

        return kept

    def detect_vehicles(self, img_bgr, conf_thresh=0.35, nms_thresh=0.45):
        h0, w0 = img_bgr.shape[:2]
        blob, r, pad_x, pad_y = letterbox(img_bgr)
        out = self.vehicle_session.run(None, {'images': blob})[0][0].T  # [8400, 84]

        boxes = []
        scores = []
        classes = []

        for p in out:
            class_scores = p[4:]
            cls_id = int(np.argmax(class_scores))
            if cls_id in self.vehicle_classes:
                conf = float(class_scores[cls_id])
                class_name = self.vehicle_classes[cls_id]
                class_threshold = (
                    0.15 if class_name == 'motorcycle' else conf_thresh
                )
                if conf >= class_threshold:
                    cx, cy, w, h = p[:4]
                    x1 = int(round((cx - w / 2 - pad_x) / r))
                    y1 = int(round((cy - h / 2 - pad_y) / r))
                    x2 = int(round((cx + w / 2 - pad_x) / r))
                    y2 = int(round((cy + h / 2 - pad_y) / r))

                    x1 = max(0, min(w0, x1))
                    y1 = max(0, min(h0, y1))
                    x2 = max(0, min(w0, x2))
                    y2 = max(0, min(h0, y2))
                    if x2 > x1 and y2 > y1:
                        boxes.append([x1, y1, x2 - x1, y2 - y1])
                        scores.append(conf)
                        classes.append(class_name)

        if not boxes:
            return []

        candidates = []
        for box, score, class_name in zip(boxes, scores, classes):
            x, y, width, height = box
            candidates.append({
                'bbox': [x, y, x + width, y + height],
                'conf': round(score, 3),
                'class_name': class_name
            })

        return self.suppress_vehicle_overlaps(candidates, nms_thresh)

    def detect_plates(self, img_bgr, conf_thresh=0.25, nms_thresh=0.45):
        h0, w0 = img_bgr.shape[:2]
        blob, r, pad_x, pad_y = letterbox(img_bgr)
        out = self.plate_det_session.run(None, {'images': blob})[0][0].T  # [8400, 5]

        boxes = []
        scores = []
        for p in out:
            conf = float(p[4])
            if conf >= conf_thresh:
                cx, cy, w, h = p[:4]
                x1 = int(round((cx - w / 2 - pad_x) / r))
                y1 = int(round((cy - h / 2 - pad_y) / r))
                x2 = int(round((cx + w / 2 - pad_x) / r))
                y2 = int(round((cy + h / 2 - pad_y) / r))

                x1 = max(0, min(w0, x1))
                y1 = max(0, min(h0, y1))
                x2 = max(0, min(w0, x2))
                y2 = max(0, min(h0, y2))
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2 - x1, y2 - y1])
                    scores.append(conf)

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thresh, nms_thresh)
        results = []
        for idx in indices:
            x, y, w, h = boxes[idx]
            results.append({
                'bbox': [x, y, x + w, y + h],
                'conf': round(scores[idx], 3)
            })
        return results

    def decode_ctc(self, logits):
        # logits: [48, 1, 35]
        preds = np.argmax(logits, axis=2)[:, 0]
        chars = []
        prev = 0
        for p in preds:
            if p != prev and p != 0:
                chars.append(IDX2CHAR.get(p, ''))
            prev = p
        return ''.join(chars)

    def recognize_crop(self, crop_bgr, plate_type: str):
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        if plate_type == 'two_row':
            ch, cw = gray.shape
            half = ch // 2
            if half > 0:
                top = gray[:half, :]
                bottom = gray[half:half * 2, :]
                gray = np.concatenate([top, bottom], axis=1)

        resized = cv2.resize(gray, (192, 48), interpolation=cv2.INTER_LINEAR)
        norm = ((resized.astype(np.float32) / 255.0) - 0.5) / 0.5
        inp = norm[np.newaxis, np.newaxis, :, :]
        sess = self.crnn_one_session if plate_type == 'one_row' else self.crnn_two_session
        logits = sess.run(None, {'input': inp})[0]
        return self.decode_ctc(logits)

    def recognize_plate(self, crop_bgr):
        if crop_bgr.size == 0 or crop_bgr.shape[0] < 5 or crop_bgr.shape[1] < 5:
            return "", 0.0, "unknown"

        h, w = crop_bgr.shape[:2]
        ratio = w / max(h, 1)
        guess_type = "one_row" if ratio > 1.35 else "two_row"
        other_type = "two_row" if guess_type == "one_row" else "one_row"

        text_guess = self.recognize_crop(crop_bgr, guess_type)
        normalized_guess = normalize_plate_text(text_guess)
        if looks_valid(normalized_guess):
            return normalized_guess, 0.95, guess_type

        text_other = self.recognize_crop(crop_bgr, other_type)
        normalized_other = normalize_plate_text(text_other)
        if looks_valid(normalized_other):
            return normalized_other, 0.90, other_type

        # Neither looks strictly valid by regex; keep guess_type
        return normalized_guess or normalized_other, 0.50, guess_type
