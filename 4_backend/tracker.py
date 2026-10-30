import numpy as np

def compute_iou(box1, box2):
    # box: [x1, y1, x2, y2]
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / max(union, 1e-6)

class TrackedVehicle:
    def __init__(self, track_id, bbox, conf, class_name):
        self.track_id = track_id
        self.bbox = list(bbox)  # [x1, y1, x2, y2]
        self.conf = conf
        self.class_name = class_name
        self.motorcycle_confirmed = False
        self.class_votes = {
            class_name: {"count": 1, "confidence": conf}
        }
        self.plate_text = ""
        self.plate_conf = 0.0
        self.plate_crop = ""
        self.plate_votes = {}
        self.lost_frames = 0
        self.hits = 1
        self.consecutive_hits = 1
        self.counted = False
        self.in_roi = False
        self.entry_time_ms = 0

    def update(self, bbox, conf, class_name):
        self.consecutive_hits = (
            self.consecutive_hits + 1 if self.lost_frames == 0 else 1
        )
        # Smooth bbox update
        alpha = 0.7
        self.bbox = [
            int(alpha * n + (1 - alpha) * o)
            for n, o in zip(bbox, self.bbox)
        ]
        self.conf = max(self.conf, conf)
        vote = self.class_votes.setdefault(
            class_name, {"count": 0, "confidence": 0.0}
        )
        vote["count"] += 1
        vote["confidence"] = max(vote["confidence"], conf)
        self.class_name = (
            "motorcycle"
            if self.motorcycle_confirmed
            else max(
                self.class_votes.items(),
                key=lambda item: (item[1]["count"], item[1]["confidence"]),
            )[0]
        )
        self.lost_frames = 0
        self.hits += 1

    def update_plate(self, plate_text, plate_conf, plate_crop=None):
        if not plate_text:
            return

        plate_text = plate_text.strip().upper()
        vote = self.plate_votes.setdefault(
            plate_text, {"count": 0, "confidence": 0.0, "crop": ""}
        )
        vote["count"] += 1
        vote["confidence"] = max(vote["confidence"], plate_conf)
        if plate_crop:
            vote["crop"] = plate_crop

        best_text, best_vote = max(
            self.plate_votes.items(),
            key=lambda item: (item[1]["count"], item[1]["confidence"]),
        )
        if best_vote["count"] < 2:
            return

        self.plate_text = best_text
        self.plate_conf = best_vote["confidence"]
        self.plate_crop = best_vote["crop"]

class VehicleTracker:
    def __init__(self, max_lost=15, iou_threshold=0.25):
        self.max_lost = max_lost
        self.iou_threshold = iou_threshold
        self.next_id = 1
        self.tracks = []  # list of TrackedVehicle

    def update(self, detections):
        # detections: list of dict(bbox=[x1, y1, x2, y2], conf=float, class_name=str)
        matched_tracks = set()
        matched_dets = set()

        if self.tracks and detections:
            # Combine box overlap and center proximity so perspective/scale changes
            # do not immediately split a moving vehicle into a new counted track.
            match_matrix = np.full(
                (len(self.tracks), len(detections)), -1.0, dtype=np.float32
            )
            for t_idx, track in enumerate(self.tracks):
                tx1, ty1, tx2, ty2 = track.bbox
                track_center = ((tx1 + tx2) / 2, (ty1 + ty2) / 2)
                track_diagonal = np.hypot(tx2 - tx1, ty2 - ty1)
                for d_idx, det in enumerate(detections):
                    dx1, dy1, dx2, dy2 = det['bbox']
                    detection_center = ((dx1 + dx2) / 2, (dy1 + dy2) / 2)
                    detection_diagonal = np.hypot(dx2 - dx1, dy2 - dy1)
                    center_distance = np.hypot(
                        track_center[0] - detection_center[0],
                        track_center[1] - detection_center[1],
                    )
                    max_distance = max(
                        48.0,
                        0.8 * max(track_diagonal, detection_diagonal),
                    )
                    iou = compute_iou(track.bbox, det['bbox'])
                    if iou < 0.05 and center_distance > max_distance:
                        continue

                    score = iou + 0.45 * max(
                        0.0, 1.0 - center_distance / max_distance
                    )
                    if track.class_name == det['class_name']:
                        score += 0.05
                    match_matrix[t_idx, d_idx] = score

            # Greedy matching
            while True:
                max_val = np.max(match_matrix)
                if max_val < self.iou_threshold:
                    break
                t_idx, d_idx = np.unravel_index(
                    np.argmax(match_matrix), match_matrix.shape
                )
                if t_idx in matched_tracks or d_idx in matched_dets:
                    match_matrix[t_idx, d_idx] = -1.0
                    continue

                matched_tracks.add(t_idx)
                matched_dets.add(d_idx)
                det = detections[d_idx]
                track = self.tracks[t_idx]
                track.update(det['bbox'], det['conf'], det['class_name'])
                if det.get('motorcycle_context'):
                    track.motorcycle_confirmed = True
                    track.class_name = "motorcycle"
                match_matrix[t_idx, :] = -1.0
                match_matrix[:, d_idx] = -1.0

        # Increment lost frames for unmatched tracks
        for t_idx, track in enumerate(self.tracks):
            if t_idx not in matched_tracks:
                track.lost_frames += 1
                track.consecutive_hits = 0

        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.lost_frames <= self.max_lost]

        # Create new tracks for unmatched detections
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_dets:
                new_track = TrackedVehicle(
                    track_id=self.next_id,
                    bbox=det['bbox'],
                    conf=det['conf'],
                    class_name=det['class_name']
                )
                new_track.motorcycle_confirmed = bool(det.get('motorcycle_context'))
                self.next_id += 1
                self.tracks.append(new_track)

        return self.tracks
