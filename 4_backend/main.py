import io, base64, time, re, sys, os, uuid, asyncio, tempfile, logging, math, mimetypes
from pathlib import Path
from typing import List, Dict, Optional
import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from models_engine import ModelsEngine, looks_valid
from video_engine import VideoProcessor, format_timestamp, VIETNAMESE_NAMES

BASE_DIR = Path(__file__).resolve().parent
TEMP_VIDEO_DIR = BASE_DIR / "temp_videos"
TEMP_VIDEO_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="VNLP License Plate & Traffic Recognition")
logger = logging.getLogger(__name__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("Loading ONNX models engine")
engine = ModelsEngine()
logger.info("ONNX models are ready")

# Existing Data Models for compatibility with UploadPage and existing frontend
class PlateResult(BaseModel):
    plate_text: str
    confidence: float
    plate_type: str
    bbox: List[int]
    image_crop: str

class PredictResponse(BaseModel):
    results: List[PlateResult]
    total_plates: int
    inference_ms: float
    annotated_image: str

def process_single_image(img_bgr):
    start = time.time()
    h0, w0 = img_bgr.shape[:2]
    plate_dets = engine.detect_plates(img_bgr, conf_thresh=0.25)
    results = []
    annotated = img_bgr.copy()

    for pdet in plate_dets:
        x1, y1, x2, y2 = pdet['bbox']
        conf = pdet['conf']
        crop = img_bgr[max(0, y1):min(h0, y2), max(0, x1):min(w0, x2)]
        if crop.size == 0:
            continue

        text, r_conf, ptype = engine.recognize_plate(crop)
        display_text = text if text else "Không đọc được"

        _, buf = cv2.imencode(".jpg", crop)
        crop_b64 = base64.b64encode(buf).decode()

        results.append(PlateResult(
            plate_text=display_text,
            confidence=round(conf, 4),
            plate_type=ptype,
            bbox=[x1, y1, x2, y2],
            image_crop=crop_b64
        ))

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, display_text, (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    _, buf = cv2.imencode(".jpg", annotated)
    annotated_b64 = base64.b64encode(buf).decode()
    inference_ms = round((time.time() - start) * 1000, 1)

    return PredictResponse(
        results=results,
        total_plates=len(results),
        inference_ms=inference_ms,
        annotated_image=annotated_b64
    )

@app.get("/api/health")
def health():
    return {"status": "ok", "engine": "onnx"}

def get_video_path(video_id: str):
    if not re.fullmatch(r"[a-f0-9]{32}", video_id):
        return None
    return next(TEMP_VIDEO_DIR.glob(f"{video_id}.*"), None)

# Existing Image Prediction Endpoint
@app.post("/api/predict/image", response_model=PredictResponse)
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Không thể đọc file ảnh")
    return process_single_image(img)

# Existing Batch Video Prediction Endpoint (preserved for backward compatibility)
@app.post("/api/predict/video")
async def predict_video(file: UploadFile = File(...), frame_interval: int = 10):
    contents = await file.read()
    temp_path = TEMP_VIDEO_DIR / f"batch_{uuid.uuid4().hex[:8]}.mp4"
    with open(temp_path, "wb") as f:
        f.write(contents)

    frame_results = []
    try:
        cap = cv2.VideoCapture(str(temp_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_index % max(frame_interval, 1) == 0:
                resp = process_single_image(frame)
                timestamp_ms = (frame_index / fps) * 1000
                frame_results.append({
                    "frame_index": frame_index,
                    "timestamp_ms": round(timestamp_ms, 1),
                    "results": [
                        {
                            "plate_text": r.plate_text,
                            "confidence": r.confidence,
                            "plate_type": r.plate_type,
                            "bbox": r.bbox,
                            "image_crop": r.image_crop,
                        }
                        for r in resp.results
                    ],
                })
            frame_index += 1
        cap.release()
    finally:
        if temp_path.exists():
            temp_path.unlink()

    return frame_results

# Video Upload for Real-time Streaming
@app.post("/api/video/upload")
async def upload_video(file: UploadFile = File(...)):
    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()
    if suffix not in {'.mp4', '.avi', '.mov', '.mkv', '.webm'}:
        raise HTTPException(status_code=400, detail="Định dạng video không được hỗ trợ")

    video_id = uuid.uuid4().hex
    target_path = TEMP_VIDEO_DIR / f"{video_id}{suffix}"

    try:
        with target_path.open("wb") as target:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                target.write(chunk)
    except OSError as exc:
        target_path.unlink(missing_ok=True)
        logger.exception("Could not save uploaded video")
        raise HTTPException(status_code=500, detail="Không thể lưu video tải lên") from exc

    cap = cv2.VideoCapture(str(target_path))
    if not cap.isOpened():
        target_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Video bị lỗi hoặc không thể giải mã")

    readable, _ = cap.read()
    if not readable:
        cap.release()
        target_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Video không có frame hợp lệ")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        fps = 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration_sec = total_frames / fps if total_frames > 0 else 0
    cap.release()

    return {
        "video_id": video_id,
        "filename": filename,
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "duration_sec": round(duration_sec, 2),
        "width": width,
        "height": height,
        "stream_url": f"/api/video/raw/{video_id}"
    }

# Endpoint for HTML5 Replay of Original Video
@app.get("/api/video/raw/{video_id}")
async def get_raw_video(video_id: str):
    file_path = get_video_path(video_id)
    if file_path is None or not file_path.exists():
        raise HTTPException(status_code=404, detail="Không tìm thấy video")
    media_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    return FileResponse(file_path, media_type=media_type)

# Real-Time WebSocket Streaming Endpoint
@app.websocket("/ws/video/{video_id}")
async def websocket_video_stream(websocket: WebSocket, video_id: str):
    await websocket.accept()
    video_path = get_video_path(video_id)
    if video_path is None or not video_path.exists():
        await websocket.send_json({"type": "error", "message": "Video không tồn tại trên hệ thống"})
        await websocket.close()
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        await websocket.send_json({"type": "error", "message": "Không thể mở file video"})
        await websocket.close()
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        fps = 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    processor = VideoProcessor(engine)

    state = {
        "running": False,
        "paused": False,
        "stopped": False,
        "send_frame_images": False
    }

    # Background task to listen for control commands (pause, resume, stop)
    async def listen_commands():
        try:
            while not state["stopped"]:
                data = await websocket.receive_json()
                if not isinstance(data, dict):
                    state["error"] = "Lệnh điều khiển video không hợp lệ"
                    state["stopped"] = True
                    return
                action = data.get("action")
                if action == "start":
                    state["running"] = True
                    state["paused"] = False
                    state["send_frame_images"] = bool(data.get("frame_fallback", False))
                elif action == "pause":
                    state["paused"] = True
                elif action == "resume":
                    state["paused"] = False
                elif action == "frame_fallback":
                    state["send_frame_images"] = True
                elif action == "stop":
                    state["stopped"] = True
                    state["running"] = False
                    break
                else:
                    state["error"] = "Lệnh điều khiển video không được hỗ trợ"
                    state["stopped"] = True
        except WebSocketDisconnect:
            state["stopped"] = True
        except (ValueError, RuntimeError) as exc:
            logger.warning("Invalid WebSocket video command: %s", exc)
            state["error"] = "Không thể đọc lệnh điều khiển video"
            state["stopped"] = True

    cmd_task = asyncio.create_task(listen_commands())

    try:
        # Wait until client sends "start" command
        while not state["running"] and not state["stopped"]:
            await asyncio.sleep(0.05)

        frame_idx = 0
        processed_frames = 0
        base_frame_stride = max(1, int(round(fps / 5.0)))
        frame_stride = base_frame_stride
        average_proc_time = 0.0
        stream_started_at = None
        first_timestamp_ms = 0.0
        last_time = time.time()
        fps_smooth = 0.0
        last_records_revision = -1

        while not state["stopped"] and not state.get("error"):
            if state["paused"]:
                await websocket.send_json({
                    "type": "status_update",
                    "status": "paused",
                    "counts": processor.counts,
                    "records": processor.records
                })
                while state["paused"] and not state["stopped"]:
                    await asyncio.sleep(0.1)
                if state["stopped"]:
                    break

            ret, frame = cap.read()
            if not ret:
                # Video reached the end
                break

            timestamp_ms = (frame_idx / max(fps, 1.0)) * 1000

            # Keep checking unread tracks so a sparse first OCR attempt can recover.
            has_unread_tracks = any(
                not track.plate_text and track.lost_frames < 3
                for track in processor.tracker.tracks
            )
            run_plate_detection = processed_frames % 3 == 0 or has_unread_tracks
            t0 = time.time()
            active_vehicles, roi_points = await asyncio.to_thread(
                processor.process_frame,
                frame,
                timestamp_ms,
                run_plate_detection,
            )
            if state["stopped"]:
                break
            proc_time = time.time() - t0
            average_proc_time = (
                proc_time if average_proc_time == 0
                else 0.8 * average_proc_time + 0.2 * proc_time
            )
            frame_stride = max(
                base_frame_stride,
                int(math.ceil(average_proc_time * fps)),
            )

            # Calc FPS
            now = time.time()
            dt = now - last_time
            last_time = now
            if dt > 0:
                measured_fps = 1.0 / dt
                fps_smooth = measured_fps if frame_idx == 0 else (
                    0.8 * fps_smooth + 0.2 * measured_fps
                )

            progress_pct = round((frame_idx / max(total_frames, 1)) * 100, 1)

            payload = {
                "type": "frame",
                "frame_idx": frame_idx,
                "total_frames": total_frames,
                "timestamp_ms": round(timestamp_ms, 1),
                "timestamp_str": format_timestamp(timestamp_ms),
                "progress_pct": progress_pct,
                "fps": round(fps_smooth, 1),
                "source_fps": round(fps, 2),
                "analysis_stride_frames": frame_stride,
                "plate_detection_sampled": run_plate_detection,
                "frame_width": frame.shape[1],
                "frame_height": frame.shape[0],
                "roi": roi_points,
                "counts": processor.counts,
                "active_vehicles": active_vehicles,
                "status": "processing"
            }
            if processor.records_revision != last_records_revision:
                payload["records"] = processor.records
                last_records_revision = processor.records_revision
            if state["send_frame_images"]:
                h, w = frame.shape[:2]
                target_w = min(w, 800)
                display_frame = frame
                if target_w < w:
                    target_h = int(h * (target_w / w))
                    display_frame = cv2.resize(
                        frame, (target_w, target_h), interpolation=cv2.INTER_AREA
                    )
                encoded, buf = cv2.imencode(
                    ".jpg", display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 78]
                )
                if not encoded:
                    raise RuntimeError("Không thể mã hóa frame video")
                payload["frame_image"] = (
                    "data:image/jpeg;base64," + base64.b64encode(buf).decode()
                )

            await websocket.send_json(payload)
            if stream_started_at is None:
                stream_started_at = time.monotonic()
                first_timestamp_ms = timestamp_ms
            else:
                target_elapsed = (timestamp_ms - first_timestamp_ms) / 1000.0
                remaining = target_elapsed - (time.monotonic() - stream_started_at)
                if remaining > 0:
                    await asyncio.sleep(remaining)

            processed_frames += 1
            for _ in range(frame_stride - 1):
                if not cap.grab():
                    break
            frame_idx += frame_stride

        if state.get("error"):
            await websocket.send_json({"type": "error", "message": state["error"]})
        elif not state["stopped"]:
            # Send completion payload
            await websocket.send_json({
                "type": "completed",
                "frame_idx": frame_idx,
                "total_frames": total_frames,
                "progress_pct": 100.0,
                "counts": processor.counts,
                "records": processor.records,
                "status": "completed"
            })

    except WebSocketDisconnect:
        logger.info("Video WebSocket disconnected for %s", video_id)
    except Exception as e:
        logger.exception("Video stream failed for %s", video_id)
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except (WebSocketDisconnect, RuntimeError):
            logger.debug("Could not send video stream error to disconnected client")
    finally:
        state["stopped"] = True
        cmd_task.cancel()
        try:
            await cmd_task
        except asyncio.CancelledError:
            pass
        cap.release()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
