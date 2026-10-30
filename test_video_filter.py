import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "4_backend"))

from video_engine import MIN_TRACK_CONFIRMATION_HITS, MIN_VEHICLE_CONFIDENCE, VideoProcessor
from models_engine import ModelsEngine


class FakeModelsEngine:
    def __init__(self, detections):
        self.detections = iter(detections)
        self.confidence_thresholds = []

    def detect_vehicles(self, frame, conf_thresh):
        self.confidence_thresholds.append(conf_thresh)
        return next(self.detections)

class FakeVehicleSession:
    def __init__(self, output):
        self.output = output

    def run(self, _outputs, _inputs):
        return [self.output]


class VideoDetectionFilterTests(unittest.TestCase):
    def test_requires_two_consecutive_detections_before_showing_or_counting(self):
        vehicle = {"bbox": [100, 220, 200, 320], "conf": 0.6, "class_name": "car"}
        outside = {"bbox": [100, 30, 200, 100], "conf": 0.6, "class_name": "car"}
        models = FakeModelsEngine(
            [[vehicle], [], [vehicle], [vehicle], [outside], [outside], []]
        )
        processor = VideoProcessor(models)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        first, _ = processor.process_frame(frame, 0, detect_plates=False)
        self.assertEqual(first, [])
        self.assertEqual(processor.counts["total"], 0)

        missing, _ = processor.process_frame(frame, 100, detect_plates=False)
        self.assertEqual(missing, [])

        reappeared, _ = processor.process_frame(frame, 200, detect_plates=False)
        self.assertEqual(reappeared, [])
        self.assertEqual(processor.counts["total"], 0)

        confirmed, _ = processor.process_frame(frame, 300, detect_plates=False)
        self.assertEqual(len(confirmed), 1)
        self.assertTrue(confirmed[0]["in_roi"])
        self.assertEqual(processor.counts["total"], 1)

        outside_first, _ = processor.process_frame(frame, 400, detect_plates=False)
        self.assertEqual(outside_first, [])
        outside_confirmed, _ = processor.process_frame(
            frame, 500, detect_plates=False
        )
        self.assertEqual(outside_confirmed, [])
        self.assertEqual(processor.counts["total"], 1)

        lost, _ = processor.process_frame(frame, 600, detect_plates=False)
        self.assertEqual(lost, [])
        self.assertTrue(
            all(
                threshold == MIN_VEHICLE_CONFIDENCE
                for threshold in models.confidence_thresholds
            )
        )
        self.assertEqual(MIN_TRACK_CONFIRMATION_HITS, 2)

    def test_tracker_keeps_small_vehicle_across_low_overlap_samples(self):
        tracker = VideoProcessor(FakeModelsEngine([])).tracker
        first = {"bbox": [278, 418, 324, 440], "conf": 0.44, "class_name": "car"}
        shifted = {"bbox": [270, 401, 315, 422], "conf": 0.57, "class_name": "car"}

        tracks = tracker.update([first])
        first_id = tracks[0].track_id
        tracks = tracker.update([shifted])

        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].track_id, first_id)
        self.assertEqual(tracks[0].consecutive_hits, 2)

    def test_nested_different_vehicle_classes_are_not_suppressed(self):
        engine = ModelsEngine.__new__(ModelsEngine)
        detections = [
            {"bbox": [0, 0, 300, 300], "conf": 0.8, "class_name": "car"},
            {"bbox": [120, 120, 180, 240], "conf": 0.4, "class_name": "motorcycle"},
        ]

        kept = engine.suppress_vehicle_overlaps(detections, 0.45)

        self.assertEqual(len(kept), 2)

    def test_duplicate_boxes_from_different_classes_are_suppressed(self):
        engine = ModelsEngine.__new__(ModelsEngine)
        detections = [
            {"bbox": [20, 20, 220, 220], "conf": 0.8, "class_name": "car"},
            {"bbox": [22, 22, 218, 218], "conf": 0.7, "class_name": "truck"},
        ]

        kept = engine.suppress_vehicle_overlaps(detections, 0.45)

        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["class_name"], "car")

    def test_high_confidence_motorcycle_is_not_delayed_for_second_sample(self):
        motorcycle = {
            "bbox": [200, 220, 250, 300],
            "conf": 0.5,
            "class_name": "motorcycle",
        }
        processor = VideoProcessor(FakeModelsEngine([[motorcycle]]))
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        visible, _ = processor.process_frame(frame, 0, detect_plates=False)

        self.assertEqual(len(visible), 1)
        self.assertEqual(visible[0]["class_name"], "motorcycle")
        self.assertEqual(processor.counts["motorcycle"], 1)

    def test_roi_includes_vehicles_in_both_directional_lanes(self):
        truck = {
            "bbox": [476, 230, 576, 356],
            "conf": 0.848,
            "class_name": "truck",
        }
        opposite_lane_motorcycle = {
            "bbox": [107, 326, 133, 367],
            "conf": 0.515,
            "class_name": "motorcycle",
        }
        processor = VideoProcessor(
            FakeModelsEngine([[truck, opposite_lane_motorcycle]])
        )
        frame = np.zeros((1024, 576, 3), dtype=np.uint8)

        visible, _ = processor.process_frame(
            frame, 0, detect_plates=False
        )

        self.assertEqual(
            {vehicle["class_name"] for vehicle in visible},
            {"truck", "motorcycle"},
        )
        self.assertEqual(processor.counts["truck"], 1)
        self.assertEqual(processor.counts["motorcycle"], 1)

    def test_low_confidence_motorcycle_detection_is_retained(self):
        output = np.zeros((1, 84, 8400), dtype=np.float32)
        output[0, :4, 0] = [320, 320, 60, 100]
        output[0, 7, 0] = 0.2
        engine = ModelsEngine.__new__(ModelsEngine)
        engine.vehicle_session = FakeVehicleSession(output)
        engine.vehicle_classes = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

        detections = engine.detect_vehicles(
            np.zeros((480, 640, 3), dtype=np.uint8),
            conf_thresh=MIN_VEHICLE_CONFIDENCE,
        )

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["class_name"], "motorcycle")
        self.assertAlmostEqual(detections[0]["conf"], 0.2)


if __name__ == "__main__":
    unittest.main()
