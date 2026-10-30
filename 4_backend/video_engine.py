import cv2, time, base64, os
import numpy as np
from typing import Dict, List, Tuple
from models_engine import ModelsEngine, looks_valid, normalize_plate_text
from tracker import VehicleTracker

VIETNAMESE_NAMES = {
    'car': 'Ô tô',
    'motorcycle': 'Xe máy',
    'bus': 'Xe buýt',
    'truck': 'Xe tải'
}

# Predefined Camera Monitoring Zone (Preset ROI in normalized coordinates 0.0 - 1.0)
# Covers the primary camera monitoring area on the road
PRESET_ROI_NORM = [
    (0.20, 0.30),
    (0.97, 0.30),
    (0.98, 0.95),
    (0.02, 0.95)
]
MIN_VEHICLE_CONFIDENCE = 0.35
MIN_TRACK_CONFIRMATION_HITS = 2

def format_timestamp(ms: float) -> str:
    total_sec = int(ms / 1000)
    minutes = total_sec // 60
    seconds = total_sec % 60
    return f"{minutes:02d}:{seconds:02d}"

class VideoProcessor:
    def __init__(self, models_engine: ModelsEngine):
        self.engine = models_engine
        self.tracker = VehicleTracker(max_lost=12, iou_threshold=0.25)
        self.counts = {
            'total': 0,
            'car': 0,
            'motorcycle': 0,
            'bus': 0,
            'truck': 0
        }
        self.records = []  # List of recorded vehicle entry events
        self.records_by_id = {} # track_id -> record
        self.records_revision = 0
        self.status = "idle"  # idle, processing, paused, stopped, completed
        self.paused = False
        self.stopped = False

    def get_pixel_roi(self, width: int, height: int) -> np.ndarray:
        pts = [(int(x * width), int(y * height)) for x, y in PRESET_ROI_NORM]
        return np.array(pts, dtype=np.int32)

    def is_in_roi(self, point: Tuple[int, int], roi_pts: np.ndarray) -> bool:
        # Measure point (cx, y_bottom) against polygon
        res = cv2.pointPolygonTest(roi_pts, (float(point[0]), float(point[1])), False)
        return res >= 0

    def bbox_intersects_roi(self, bbox: List[int], roi_pts: np.ndarray) -> bool:
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            return False
        vehicle_polygon = np.array(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            dtype=np.float32,
        ).reshape((-1, 1, 2))
        roi_polygon = roi_pts.astype(np.float32).reshape((-1, 1, 2))
        intersection_area, _ = cv2.intersectConvexConvex(
            vehicle_polygon, roi_polygon
        )
        return intersection_area > 0

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp_ms: float,
        detect_plates: bool = True,
    ) -> Tuple[List[dict], List[List[int]]]:
        h, w = frame.shape[:2]
        roi_pts = self.get_pixel_roi(w, h)

        # 1. Detect vehicles
        vehicle_dets = self.engine.detect_vehicles(
            frame, conf_thresh=MIN_VEHICLE_CONFIDENCE
        )

        # 2. Detect plates and use their surrounding image context to confirm motorcycles.
        plate_dets = (
            self.engine.detect_plates(frame, conf_thresh=0.25)
            if detect_plates
            else []
        )
        plate_dets = [
            detection for detection in plate_dets
            if self.bbox_intersects_roi(detection['bbox'], roi_pts)
        ]
        if detect_plates:
            for pdet in plate_dets:
                motorcycle_probability = self.engine.classify_motorcycle_context(
                    frame, pdet['bbox']
                )
                if motorcycle_probability < 0.9:
                    continue

                px1, py1, px2, py2 = pdet['bbox']
                pcx = (px1 + px2) // 2
                pcy = (py1 + py2) // 2
                matching_detection = next(
                    (
                        detection for detection in vehicle_dets
                        if detection['bbox'][0] <= pcx <= detection['bbox'][2]
                        and detection['bbox'][1] <= pcy <= detection['bbox'][3]
                    ),
                    None,
                )
                if matching_detection is not None:
                    matching_detection['class_name'] = 'motorcycle'
                    matching_detection['conf'] = max(
                        matching_detection['conf'], motorcycle_probability
                    )
                    matching_detection['motorcycle_context'] = True
                    continue

                matching_track = min(
                    (
                        track for track in self.tracker.tracks
                        if track.bbox[0] <= pcx <= track.bbox[2]
                        and track.bbox[1] <= pcy <= track.bbox[3]
                    ),
                    key=lambda track: (
                        (track.bbox[2] - track.bbox[0])
                        * (track.bbox[3] - track.bbox[1])
                    ),
                    default=None,
                )
                if matching_track is not None:
                    matching_track.motorcycle_confirmed = True
                    matching_track.class_name = 'motorcycle'
                    continue

                box_width = max(1, px2 - px1)
                box_height = max(1, py2 - py1)
                if box_height >= 0.2 * h or box_width >= 0.3 * w:
                    # The plate detector can return the whole motorcycle as its box.
                    context_bbox = [px1, py1, px2, py2]
                else:
                    context_bbox = [
                        max(0, int(pcx - 1.6 * box_width)),
                        max(0, int(py1 - 2 * box_height)),
                        min(w, int(pcx + 1.6 * box_width)),
                        min(h, int(py2 + box_height)),
                    ]
                vehicle_dets.append({
                    'bbox': context_bbox,
                    'conf': motorcycle_probability,
                    'class_name': 'motorcycle',
                    'motorcycle_context': True,
                })

        # 3. Update multi-object tracker
        tracks = self.tracker.update(vehicle_dets)

        # 4. Associate plates with tracked vehicles
        for pdet in plate_dets:
            px1, py1, px2, py2 = pdet['bbox']
            pcx = (px1 + px2) // 2
            pcy = (py1 + py2) // 2

            # Find vehicle containing this plate center
            best_track = None
            min_dist = float('inf')
            for track in tracks:
                vx1, vy1, vx2, vy2 = track.bbox
                # Expand slightly
                if vx1 - 10 <= pcx <= vx2 + 10 and vy1 - 10 <= pcy <= vy2 + 10:
                    vcx = (vx1 + vx2) // 2
                    vcy = (vy1 + vy2) // 2
                    dist = (pcx - vcx) ** 2 + (pcy - vcy) ** 2
                    if dist < min_dist:
                        min_dist = dist
                        best_track = track

            # Crop and OCR
            crop = frame[max(0, py1):min(h, py2), max(0, px1):min(w, px2)]
            if crop.size > 0:
                text, conf, _ = self.engine.recognize_plate(crop)
                normalized_text = normalize_plate_text(text)
                if best_track is not None and looks_valid(normalized_text):
                    _, buf = cv2.imencode(".jpg", crop)
                    crop_b64 = base64.b64encode(buf).decode()
                    previous_plate = best_track.plate_text
                    best_track.update_plate(normalized_text, conf, crop_b64)
                    if (
                        best_track.track_id in self.records_by_id
                        and best_track.plate_text != previous_plate
                    ):
                        rec = self.records_by_id[best_track.track_id]
                        rec['plate_text'] = best_track.plate_text
                        rec['plate_crop'] = best_track.plate_crop
                        self.records_revision += 1

        # 5. Check ROI and count
        active_vehicles = []
        for track in tracks:
            vx1, vy1, vx2, vy2 = track.bbox
            in_roi = self.is_in_roi(
                ((vx1 + vx2) // 2, vy2),
                roi_pts,
            )
            track.in_roi = in_roi
            vtype = track.class_name

            is_confirmed_vehicle = (
                track.consecutive_hits >= MIN_TRACK_CONFIRMATION_HITS
                or track.conf >= 0.7
                or (
                    track.class_name == 'motorcycle'
                    and track.conf >= 0.45
                )
            )
            if in_roi and not track.counted and is_confirmed_vehicle:
                track.counted = True
                track.entry_time_ms = timestamp_ms

                # Crop vehicle thumbnail if possible
                v_crop = frame[max(0, vy1):min(h, vy2), max(0, vx1):min(w, vx2)]
                v_b64 = ""
                if v_crop.size > 0:
                    small = cv2.resize(v_crop, (min(160, v_crop.shape[1]), min(120, v_crop.shape[0])))
                    _, vbuf = cv2.imencode(".jpg", small)
                    v_b64 = base64.b64encode(vbuf).decode()

                # Increment counts
                if vtype in self.counts:
                    self.counts[vtype] += 1
                else:
                    self.counts['car'] += 1
                self.counts['total'] += 1

                record = {
                    'id': track.track_id,
                    'track_id': track.track_id,
                    'class_name': vtype,
                    'class_vn': VIETNAMESE_NAMES.get(vtype, vtype),
                    'plate_text': track.plate_text if track.plate_text else "chưa đọc được",
                    'timestamp_ms': round(timestamp_ms, 1),
                    'timestamp_str': format_timestamp(timestamp_ms),
                    'plate_crop': track.plate_crop,
                    'vehicle_crop': v_b64
                }
                self.records.append(record)
                self.records_by_id[track.track_id] = record
                self.records_revision += 1
            elif track.track_id in self.records_by_id:
                record = self.records_by_id[track.track_id]
                if record['class_name'] != vtype:
                    old_type = record['class_name']
                    if old_type in self.counts:
                        self.counts[old_type] = max(0, self.counts[old_type] - 1)
                    if vtype in self.counts:
                        self.counts[vtype] += 1
                    else:
                        self.counts['car'] += 1
                    record['class_name'] = vtype
                    record['class_vn'] = VIETNAMESE_NAMES.get(vtype, vtype)
                    self.records_revision += 1

            if in_roi and track.lost_frames == 0 and is_confirmed_vehicle:
                active_vehicles.append({
                    'track_id': track.track_id,
                    'class_name': track.class_name,
                    'class_vn': VIETNAMESE_NAMES.get(track.class_name, track.class_name),
                    'bbox': track.bbox,
                    'plate_text': track.plate_text if track.plate_text else "chưa đọc được",
                    'in_roi': in_roi,
                    'counted': track.counted,
                    'missed_frames': track.lost_frames
                })

        return active_vehicles, roi_pts.tolist()
