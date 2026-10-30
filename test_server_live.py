import sys, os, time, threading, requests, json, asyncio
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="backslashreplace")

# Add 4_backend to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent / "4_backend"))

import uvicorn
import websockets
import cv2
from main import app, engine
from models_engine import looks_valid, normalize_plate_text
from tracker import TrackedVehicle
from video_engine import VideoProcessor

assert normalize_plate_text("90-826.1274") == "90B261274"
assert looks_valid("90-826.1274")

class_vote_test = TrackedVehicle(1, [0, 0, 100, 100], 0.9, "car")
class_vote_test.update([1, 1, 101, 101], 0.8, "truck")
assert class_vote_test.class_name == "car"
class_vote_test.update([2, 2, 102, 102], 0.8, "truck")
assert class_vote_test.class_name == "truck"

sample_capture = cv2.VideoCapture("sample_traffic.mp4")
sample_capture.set(cv2.CAP_PROP_POS_FRAMES, 21)
sample_ok, sample_frame = sample_capture.read()
sample_capture.release()
assert sample_ok
sample_plate = engine.detect_plates(sample_frame, 0.25)[0]
sample_plate_box = sample_plate["bbox"]
plate_crop = sample_frame[
    sample_plate_box[1]:sample_plate_box[3],
    sample_plate_box[0]:sample_plate_box[2],
]
assert engine.recognize_plate(plate_crop)[0] == "90B261274"
assert engine.classify_motorcycle_context(sample_frame, sample_plate_box) >= 0.9

context_capture = cv2.VideoCapture("sample_traffic.mp4")
context_capture.set(cv2.CAP_PROP_POS_FRAMES, 26)
context_ok, context_frame = context_capture.read()
context_capture.release()
assert context_ok
context_processor = VideoProcessor(engine)
context_processor.process_frame(context_frame, 2600, detect_plates=True)
assert context_processor.counts["motorcycle"] > 0
context_track = context_processor.tracker.tracks[0]
context_track_area = (
    (context_track.bbox[2] - context_track.bbox[0])
    * (context_track.bbox[3] - context_track.bbox[1])
)
assert context_track_area < context_frame.shape[0] * context_frame.shape[1] * 0.15

track_capture = cv2.VideoCapture("sample_traffic.mp4")
track_processor = VideoProcessor(engine)
for frame_index in range(18, 22):
    track_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    track_ok, track_frame = track_capture.read()
    assert track_ok
    track_processor.process_frame(track_frame, frame_index * 100, True)
track_capture.release()
assert track_processor.counts["total"] == 1
assert track_processor.counts["motorcycle"] == 1

def run_server():
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="warning")

server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()
time.sleep(2.0)

print("--- 1. Testing GET /api/health ---")
r = requests.get("http://127.0.0.1:8001/api/health")
print("Health response:", r.status_code, r.json())
assert r.status_code == 200

print("--- 2. Testing POST /api/predict/image ---")
test_img_path = "data/processed/detection/images/val/1_1008_0_36A38584_93_232_153_298.jpg"
with open(test_img_path, "rb") as f:
    files = {"file": ("test.jpg", f, "image/jpeg")}
    r_img = requests.post("http://127.0.0.1:8001/api/predict/image", files=files)
print("Image predict status:", r_img.status_code)
img_json = r_img.json()
print("Total plates:", img_json.get("total_plates"))
for res in img_json.get("results", []):
    print("  Plate:", res["plate_text"], "conf:", res["confidence"], "type:", res["plate_type"])
assert r_img.status_code == 200
assert img_json.get("total_plates") >= 1

print("--- 3. Testing POST /api/video/upload ---")
with open("sample_traffic.mp4", "rb") as f:
    files = {"file": ("sample_traffic.mp4", f, "video/mp4")}
    r_vid = requests.post("http://127.0.0.1:8001/api/video/upload", files=files)
print("Upload status:", r_vid.status_code)
vid_json = r_vid.json()
print("Upload response:", vid_json)
assert r_vid.status_code == 200
video_id = vid_json["video_id"]
raw_response = requests.get(f"http://127.0.0.1:8001{vid_json['stream_url']}", stream=True)
print("Raw video response:", raw_response.status_code, raw_response.headers.get("content-type"))
assert raw_response.status_code == 200
assert raw_response.headers.get("content-type") == "video/mp4"
raw_response.close()
range_response = requests.get(
    f"http://127.0.0.1:8001{vid_json['stream_url']}",
    headers={"Range": "bytes=0-1023"},
    stream=True,
)
assert range_response.status_code == 206
range_response.close()

print("--- 4. Testing WebSocket live video streaming ---")
async def test_ws():
    uri = f"ws://127.0.0.1:8001/ws/video/{video_id}"
    async with websockets.connect(uri) as ws:
        print("Connected to WebSocket, sending start...")
        await ws.send(json.dumps({"action": "start"}))
        received_frames = 0
        records = []
        pause_verified = False
        resume_verified = False
        fallback_frame_received = False
        last_frame_idx = -1
        while received_frames < 6:
            msg = await ws.recv()
            data = json.loads(msg)
            if data.get("type") == "frame":
                received_frames += 1
                f_idx = data["frame_idx"]
                assert f_idx > last_frame_idx
                assert data["analysis_stride_frames"] >= 1
                last_frame_idx = f_idx
                t_str = data["timestamp_str"]
                prog = data["progress_pct"]
                fps = data["fps"]
                counts = data["counts"]
                records = data.get("records", records)
                has_overlay = "active_vehicles" in data and len(data.get("roi", [])) == 4
                assert data["source_fps"] > 0
                assert data["frame_width"] > 0 and data["frame_height"] > 0
                if received_frames < 2:
                    assert "frame_image" not in data
                if data.get("frame_image"):
                    fallback_frame_received = True
                print(f"WS Frame #{f_idx} [{t_str}] prog: {prog}% fps: {fps} total_counted: {counts['total']} overlay_ok: {has_overlay}")
                assert has_overlay
                if records:
                    print(f"  Records ({len(records)}): {[r['class_vn'] + ': ' + r['plate_text'] for r in records[:3]]}")
                if received_frames == 2:
                    await ws.send(json.dumps({"action": "pause"}))
                    await ws.send(json.dumps({"action": "frame_fallback"}))
            elif data.get("type") == "status_update" and data.get("status") == "paused":
                pause_verified = True
                await ws.send(json.dumps({"action": "resume"}))
                resume_verified = True
        print("Sending stop...")
        await ws.send(json.dumps({"action": "stop"}))
        assert pause_verified and resume_verified
        assert fallback_frame_received
        print("WebSocket streaming verified successfully!")

asyncio.run(test_ws())

async def test_full_ws():
    uri = f"ws://127.0.0.1:8001/ws/video/{video_id}"
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({"action": "start"}))
        frame_count = 0
        completion = None
        while completion is None:
            data = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
            if data.get("type") == "frame":
                frame_count += 1
                assert "active_vehicles" in data
                assert data["analysis_stride_frames"] >= 1
            elif data.get("type") == "completed":
                completion = data
            elif data.get("type") == "error":
                raise AssertionError(data.get("message", "Video processing failed"))

        assert frame_count > 1
        assert completion["counts"]["total"] > 0
        print(
            "Full WebSocket run:",
            frame_count,
            "frames, counts:",
            completion["counts"],
            "records:",
            [
                (record["class_name"], record["plate_text"])
                for record in completion["records"]
            ],
        )

try:
    asyncio.run(test_full_ws())
finally:
    uploaded_video_path = (
        Path(__file__).resolve().parent
        / "4_backend"
        / "temp_videos"
        / f"{video_id}.mp4"
    )
    if uploaded_video_path.exists():
        uploaded_video_path.unlink()

print("All E2E backend tests passed successfully!")
