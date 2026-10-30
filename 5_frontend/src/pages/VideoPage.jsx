import { useState, useRef, useEffect } from "react";
import { uploadVideo, getVideoRawUrl, getVideoWsUrl } from "../services/api";

const VEHICLE_META = {
  car: { name: "Ô tô", icon: "🚗", color: "text-blue-400", bg: "bg-blue-500/10", border: "border-blue-500/30" },
  motorcycle: { name: "Xe máy", icon: "🏍️", color: "text-emerald-400", bg: "bg-emerald-500/10", border: "border-emerald-500/30" },
  bus: { name: "Xe buýt", icon: "🚌", color: "text-amber-400", bg: "bg-amber-500/10", border: "border-amber-500/30" },
  truck: { name: "Xe tải", icon: "🚚", color: "text-purple-400", bg: "bg-purple-500/10", border: "border-purple-500/30" },
};

const VEHICLE_COLORS = {
  car: "#60a5fa",
  motorcycle: "#34d399",
  bus: "#fbbf24",
  truck: "#c084fc",
};

export default function VideoPage({ addHistory }) {
  const [file, setFile] = useState(null);
  const [videoInfo, setVideoInfo] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [status, setStatus] = useState("idle"); // idle, uploading, ready, processing, paused, completed, stopped, error
  const [error, setError] = useState(null);

  // Live streaming states
  const [frameImage, setFrameImage] = useState(null);
  const [videoFallback, setVideoFallback] = useState(false);
  const [currentTimeStr, setCurrentTimeStr] = useState("00:00");
  const [progressPct, setProgressPct] = useState(0);
  const [frameIdx, setFrameIdx] = useState(0);
  const [totalFrames, setTotalFrames] = useState(0);
  const [fps, setFps] = useState(0);
  const [samplingStride, setSamplingStride] = useState(1);
  const [activeVehiclesCount, setActiveVehiclesCount] = useState(0);

  // Counters & Records
  const [counts, setCounts] = useState({
    total: 0,
    car: 0,
    motorcycle: 0,
    bus: 0,
    truck: 0,
  });
  const [records, setRecords] = useState([]);

  const fileInputRef = useRef(null);
  const videoRef = useRef(null);
  const fallbackImageRef = useRef(null);
  const overlayCanvasRef = useRef(null);
  const overlayFrameRef = useRef(null);
  const videoFallbackRef = useRef(false);
  const playbackStartedRef = useRef(false);
  const pendingPlaybackRef = useRef(null);
  const wsRef = useRef(null);

  const drawOverlay = () => {
    const canvas = overlayCanvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    const bounds = canvas.getBoundingClientRect();
    const pixelRatio = window.devicePixelRatio || 1;
    const canvasWidth = Math.round(bounds.width * pixelRatio);
    const canvasHeight = Math.round(bounds.height * pixelRatio);
    if (canvas.width !== canvasWidth || canvas.height !== canvasHeight) {
      canvas.width = canvasWidth;
      canvas.height = canvasHeight;
    }

    const context = canvas.getContext("2d");
    if (!context) return;
    context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    context.clearRect(0, 0, bounds.width, bounds.height);

    const frame = overlayFrameRef.current;
    const fallbackImage = fallbackImageRef.current;
    const mediaReady = videoFallbackRef.current
      ? fallbackImage?.complete
      : video.readyState >= 2;
    const mediaTime = videoFallbackRef.current
      ? frame?.timestampSec
      : video.currentTime;
    if (!frame || !mediaReady || frame.width <= 0 || frame.height <= 0) return;

    if (!videoFallbackRef.current) {
      const sampleInterval = frame.analysisStrideFrames
        / Math.max(frame.sourceFps, 1);
      const maxOverlayAge = Math.min(0.35, Math.max(0.12, sampleInterval * 1.5));
      if (
        mediaTime > frame.timestampSec + maxOverlayAge
        || mediaTime < frame.previousTimestampSec - maxOverlayAge
      ) {
        return;
      }
    }

    const scale = Math.min(bounds.width / frame.width, bounds.height / frame.height);
    const renderedWidth = frame.width * scale;
    const renderedHeight = frame.height * scale;
    const offsetX = (bounds.width - renderedWidth) / 2;
    const offsetY = (bounds.height - renderedHeight) / 2;

    context.save();
    context.translate(offsetX, offsetY);
    context.scale(scale, scale);

    const fontSize = Math.max(14, frame.width * 0.022);
    context.font = `600 ${fontSize}px system-ui, sans-serif`;
    frame.vehicles.forEach((vehicle) => {
      const previousVehicle = frame.previousVehicles.find(
        (previous) => previous.track_id === vehicle.track_id
      );
      const detectionInterval = frame.timestampSec - frame.previousTimestampSec;
      const shouldInterpolate = previousVehicle
        && detectionInterval > 0
        && mediaTime <= frame.timestampSec;
      const interpolation = shouldInterpolate
        ? Math.max(0, Math.min(
          1,
          (mediaTime - frame.previousTimestampSec) / detectionInterval
        ))
        : 1;
      const bbox = shouldInterpolate
        ? vehicle.bbox.map((coordinate, index) => {
          return Math.round(
            previousVehicle.bbox[index]
            + (coordinate - previousVehicle.bbox[index]) * interpolation
          );
        })
        : vehicle.bbox;
      if (vehicle.missed_frames > 5) return;
      const [x1, y1, x2, y2] = bbox;
      const color = VEHICLE_COLORS[vehicle.class_name] || "#4ade80";
      const label = `#${vehicle.track_id} ${vehicle.class_vn}: ${vehicle.plate_text || "chưa đọc được"}`;
      const labelWidth = context.measureText(label).width + 12;
      const labelHeight = fontSize + 10;
      const labelY = Math.max(0, y1 - labelHeight);

      context.strokeStyle = color;
      context.lineWidth = Math.max(2, frame.width * 0.003);
      context.globalAlpha = Math.max(0.35, 1 - vehicle.missed_frames * 0.12);
      context.strokeRect(x1, y1, x2 - x1, y2 - y1);
      context.fillStyle = color;
      const labelX = Math.max(0, Math.min(x1, frame.width - labelWidth));
      context.fillRect(labelX, labelY, labelWidth, labelHeight);
      context.fillStyle = "#111827";
      context.fillText(label, labelX + 6, labelY + fontSize + 1);
      context.globalAlpha = 1;
    });

    context.restore();
  };

  useEffect(() => {
    let animationId = null;
    const renderFrame = () => {
      drawOverlay();
      animationId = null;
      if (videoRef.current && !videoRef.current.paused && !videoRef.current.ended) {
        animationId = window.requestAnimationFrame(renderFrame);
      }
    };
    const startRendering = () => {
      if (animationId === null) {
        animationId = window.requestAnimationFrame(renderFrame);
      }
    };
    const stopRendering = () => {
      if (animationId !== null) {
        window.cancelAnimationFrame(animationId);
        animationId = null;
      }
      drawOverlay();
    };
    const video = videoRef.current;
    drawOverlay();
    video?.addEventListener("play", startRendering);
    video?.addEventListener("pause", stopRendering);
    video?.addEventListener("timeupdate", drawOverlay);
    video?.addEventListener("seeked", drawOverlay);
    video?.addEventListener("loadedmetadata", drawOverlay);
    window.addEventListener("resize", drawOverlay);
    return () => {
      if (animationId !== null) window.cancelAnimationFrame(animationId);
      video?.removeEventListener("play", startRendering);
      video?.removeEventListener("pause", stopRendering);
      video?.removeEventListener("timeupdate", drawOverlay);
      video?.removeEventListener("seeked", drawOverlay);
      video?.removeEventListener("loadedmetadata", drawOverlay);
      window.removeEventListener("resize", drawOverlay);
    };
  }, [videoInfo]);

  // Cleanup WebSocket on unmount
  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  const handleSelectFile = async (selectedFile) => {
    if (!selectedFile) return;
    if (!selectedFile.type.startsWith("video/") && !selectedFile.name.match(/\.(mp4|avi|mov|mkv|webm)$/i)) {
      setError("Vui lòng chọn một file video hợp lệ (.mp4, .avi, .mov, .mkv)");
      return;
    }

    // Stop current stream if running
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setFile(selectedFile);
    setVideoInfo(null);
    setError(null);
    setFrameImage(null);
    setVideoFallback(false);
    videoFallbackRef.current = false;
    videoRef.current?.pause();
    overlayFrameRef.current = null;
    playbackStartedRef.current = false;
    pendingPlaybackRef.current = null;
    setProgressPct(0);
    setFrameIdx(0);
    setTotalFrames(0);
    setCurrentTimeStr("00:00");
    setFps(0);
    setSamplingStride(1);
    setActiveVehiclesCount(0);
    setCounts({ total: 0, car: 0, motorcycle: 0, bus: 0, truck: 0 });
    setRecords([]);

    // Upload to server to get video_id and metadata
    setUploading(true);
    setStatus("uploading");
    try {
      const data = await uploadVideo(selectedFile);
      setVideoInfo(data);
      setTotalFrames(data.total_frames || 0);
      setStatus("ready");
    } catch (err) {
      setError(err.message || "Không thể tải video lên server. Hãy chắc chắn backend đang chạy.");
      setStatus("error");
    } finally {
      setUploading(false);
    }
  };

  const startPendingPlayback = () => {
    const video = videoRef.current;
    const pending = pendingPlaybackRef.current;
    if (!video || !pending || video.readyState < 1) return;

    video.currentTime = pending.timestampSec;
    video.playbackRate = 1;
    pendingPlaybackRef.current = null;
    playbackStartedRef.current = true;
    video.play().catch(() => {
      setError("Trình duyệt không thể phát video. Hãy dùng video có codec được trình duyệt hỗ trợ.");
    });
  };

  const handleStart = () => {
    if (!videoInfo) return;
    setError(videoFallbackRef.current
      ? "Codec gốc không được trình duyệt hỗ trợ; đang dùng luồng frame trực tiếp."
      : null);
    setFrameImage(null);
    overlayFrameRef.current = null;
    playbackStartedRef.current = false;
    pendingPlaybackRef.current = null;
    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.currentTime = 0;
    }
    setCurrentTimeStr("00:00");
    setProgressPct(0);
    setFrameIdx(0);
    setTotalFrames(videoInfo.total_frames || 0);
    setFps(0);
    setSamplingStride(1);
    setActiveVehiclesCount(0);
    setCounts({ total: 0, car: 0, motorcycle: 0, bus: 0, truck: 0 });
    setRecords([]);
    setStatus("processing");

    // Connect WebSocket
    const wsUrl = getVideoWsUrl(videoInfo.video_id);
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      // Send start action
      ws.send(JSON.stringify({
        action: "start",
        frame_fallback: videoFallbackRef.current,
      }));
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        if (data.type === "frame") {
          const sourceFps = data.source_fps || videoInfo.fps || 25;
          if (videoFallbackRef.current) {
            setFrameImage(data.frame_image || null);
          }
          overlayFrameRef.current = {
            timestampSec: data.timestamp_ms / 1000,
            previousTimestampSec: overlayFrameRef.current?.timestampSec
              ?? data.timestamp_ms / 1000,
            previousVehicles: overlayFrameRef.current?.vehicles || [],
            sourceFps,
            analysisStrideFrames: data.analysis_stride_frames || 1,
            width: data.frame_width,
            height: data.frame_height,
            roi: data.roi || [],
            vehicles: data.active_vehicles || [],
          };
          setCurrentTimeStr(data.timestamp_str || "00:00");
          setProgressPct(data.progress_pct || 0);
          setFrameIdx(data.frame_idx || 0);
          setTotalFrames(data.total_frames || videoInfo.total_frames);
          setFps(data.fps || 0);
          setSamplingStride(data.analysis_stride_frames || 1);
          if (data.counts) setCounts(data.counts);
          if (data.records) setRecords(data.records);
          if (data.active_vehicles) setActiveVehiclesCount(data.active_vehicles.length);
          if (!videoFallbackRef.current && !playbackStartedRef.current) {
            pendingPlaybackRef.current = {
              timestampSec: data.timestamp_ms / 1000,
            };
            startPendingPlayback();
          }
          setStatus("processing");
        } else if (data.type === "status_update") {
          if (data.counts) setCounts(data.counts);
          if (data.records) setRecords(data.records);
          if (data.status) setStatus(data.status);
        } else if (data.type === "completed") {
          setStatus("completed");
          setProgressPct(100);
          if (data.counts) setCounts(data.counts);
          if (data.records) {
            setRecords(data.records);
            // Save to history
            if (addHistory) {
              const uniquePlates = [...new Set(data.records.map((r) => r.plate_text).filter((p) => p && p !== "chưa đọc được"))];
              addHistory({
                type: "video",
                filename: videoInfo.filename,
                plates: uniquePlates,
                totalCount: data.counts?.total || data.records.length,
              });
            }
          }
        } else if (data.type === "error") {
          setError(data.message || "Lỗi xử lý video từ backend");
          setStatus("error");
        }
      } catch (e) {
        console.error("Lỗi parse WS packet:", e);
      }
    };

    ws.onerror = () => {
      setError("Mất kết nối WebSocket tới server. Vui lòng kiểm tra lại backend.");
      setStatus("error");
    };

    ws.onclose = () => {
      setStatus((current) => current === "processing" ? "stopped" : current);
    };
  };

  const handlePause = () => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: "pause" }));
      videoRef.current?.pause();
      setStatus("paused");
    }
  };

  const handleResume = () => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: "resume" }));
      if (!videoFallbackRef.current) videoRef.current?.play().catch(() => {
        setError("Không thể tiếp tục phát video trong trình duyệt.");
      });
      setStatus("processing");
    }
  };

  const handleStop = () => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: "stop" }));
      wsRef.current.close();
      wsRef.current = null;
    }
    videoRef.current?.pause();
    setStatus("stopped");
  };

  const handleVideoError = () => {
    if (videoFallbackRef.current) {
      setError("Không thể phát video hoặc luồng frame thay thế.");
      return;
    }

    videoFallbackRef.current = true;
    setVideoFallback(true);
    setError("Codec video gốc không được trình duyệt hỗ trợ; sẽ dùng luồng frame trực tiếp khi phân tích.");
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: "frame_fallback" }));
    }
  };

  const handleReplay = () => {
    const video = videoRef.current;
    if (!video) return;
    if (videoFallbackRef.current) {
      setError("Video này chỉ có thể phát lại nếu trình duyệt hỗ trợ codec gốc.");
      return;
    }
    overlayFrameRef.current = null;
    video.playbackRate = 1;
    video.currentTime = 0;
    video.play().catch(() => {
      setError("Trình duyệt không thể phát lại video này.");
    });
  };

  return (
    <div className="space-y-6">
      {/* Top Header & File Input */}
      <div className="bg-gray-900/90 rounded-2xl border border-gray-800 p-5 shadow-xl">
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={uploading || status === "processing"}
              className="bg-blue-600 hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed
                text-white font-semibold px-5 py-2.5 rounded-xl text-sm flex items-center gap-2 transition-all shadow-md"
            >
              📂 {file ? "Đổi video khác" : "Chọn video giao thông"}
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept="video/*"
              className="hidden"
              onChange={(e) => handleSelectFile(e.target.files[0])}
            />

            {file && (
              <div className="flex items-center gap-2 bg-gray-800/80 px-3.5 py-1.5 rounded-lg border border-gray-700 text-sm">
                <span className="text-gray-200 font-medium truncate max-w-xs">{file.name}</span>
                {videoInfo && (
                  <span className="text-xs text-blue-400 bg-blue-950/60 px-2 py-0.5 rounded border border-blue-800/50">
                    {videoInfo.duration_sec}s · {videoInfo.fps} FPS · {videoInfo.width}x{videoInfo.height}
                  </span>
                )}
              </div>
            )}
          </div>

          {/* Region info pill */}
          <div className="flex items-center gap-2 text-xs bg-cyan-950/40 text-cyan-300 border border-cyan-800/50 px-3.5 py-1.5 rounded-lg">
            <span className="inline-block w-2 h-2 rounded-full bg-cyan-400 animate-pulse"></span>
            <span>Chỉ phương tiện được xác nhận trong vùng đường giám sát mới hiện khung; vào <strong>vùng đếm</strong> mới được ghi nhận</span>
          </div>
        </div>

        {uploading && (
          <div className="mt-4 flex items-center gap-3 text-sm text-blue-400">
            <div className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
            Đang tải và chuẩn bị video...
          </div>
        )}

        {error && (
          <div className="mt-4 bg-red-950/50 border border-red-800 text-red-300 rounded-xl p-3 text-sm flex items-center gap-2">
            <span>⚠️</span>
            <span>{error}</span>
          </div>
        )}
      </div>

      {/* Main 2-Column Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* ================= LEFT COLUMN: Video Stream & Controls (7 Cols) ================= */}
        <div className="lg:col-span-7 space-y-4">
          <div className="bg-gray-900 rounded-2xl border border-gray-800 overflow-hidden shadow-2xl flex flex-col">
            {/* Live Header Status Bar */}
            <div className="bg-gray-950/90 px-4 py-2.5 border-b border-gray-800 flex items-center justify-between text-xs">
              <div className="flex items-center gap-2">
                {status === "processing" && (
                  <span className="flex items-center gap-1.5 text-red-400 font-bold bg-red-950/60 border border-red-800/60 px-2 py-0.5 rounded">
                    <span className="w-2 h-2 rounded-full bg-red-500 animate-ping"></span>
                    LIVE PHÂN TÍCH
                  </span>
                )}
                {status === "paused" && (
                  <span className="text-amber-400 font-medium bg-amber-950/60 border border-amber-800/60 px-2 py-0.5 rounded">
                    ⏸ ĐÃ TẠM DỪNG
                  </span>
                )}
                {status === "completed" && (
                  <span className="text-emerald-400 font-medium bg-emerald-950/60 border border-emerald-800/60 px-2 py-0.5 rounded">
                    ✓ HOÀN THÀNH
                  </span>
                )}
                {status === "ready" && (
                  <span className="text-gray-400 bg-gray-800 px-2 py-0.5 rounded">
                    SẴN SÀNG
                  </span>
                )}
                {status === "idle" && (
                  <span className="text-gray-500 bg-gray-800/50 px-2 py-0.5 rounded">
                    CHƯA CHỌN VIDEO
                  </span>
                )}

                {status === "processing" && (
                  <span className="text-gray-400 pl-2 border-l border-gray-800">
                    Phân tích: <strong>{fps} FPS</strong> · Mỗi <strong>{samplingStride} frame</strong>
                    {" "}· Xe trên hình: <strong>{activeVehiclesCount}</strong>
                  </span>
                )}
              </div>

              {/* Toggle Replay mode after completed */}
              {status === "completed" && videoInfo && !videoFallback && (
                <button
                  onClick={handleReplay}
                  className="bg-gray-800 hover:bg-gray-700 text-blue-400 px-2.5 py-1 rounded text-xs transition-all font-medium border border-gray-700"
                >
                  🎬 Phát lại video gốc
                </button>
              )}
            </div>

            {/* Video Viewport */}
            <div className="relative aspect-[4/3] md:aspect-video bg-black flex items-center justify-center overflow-hidden">
              {videoInfo ? (
                <>
                  <video
                    ref={videoRef}
                    src={getVideoRawUrl(videoInfo.video_id)}
                    controls={!videoFallback && ["ready", "completed", "stopped", "error"].includes(status)}
                    muted
                    playsInline
                    onLoadedMetadata={() => {
                      setError(null);
                      startPendingPlayback();
                    }}
                    onTimeUpdate={drawOverlay}
                    onError={handleVideoError}
                    className={`w-full h-full object-contain ${videoFallback ? "hidden" : ""}`}
                  />
                  {videoFallback && frameImage && (
                    <img
                      ref={fallbackImageRef}
                      src={frameImage}
                      alt="Luồng phân tích video trực tiếp"
                      onLoad={drawOverlay}
                      className="w-full h-full object-contain"
                    />
                  )}
                  <canvas
                    ref={overlayCanvasRef}
                    aria-hidden="true"
                    className="absolute inset-0 w-full h-full pointer-events-none"
                  />
                </>
              ) : (
                <div className="text-center p-8 space-y-3">
                  <div className="text-5xl opacity-40">🚦</div>
                  <p className="text-gray-400 text-sm font-medium">
                    {file
                      ? "Bấm nút 'Bắt đầu phân tích' bên dưới để phát và quét trực tiếp"
                      : "Chọn một video giao thông để bắt đầu"}
                  </p>
                  <p className="text-xs text-gray-600 max-w-sm mx-auto">
                    Hệ thống sẽ tự động phát hiện phương tiện, đọc biển số và ghi nhận lượt xe khi đi vào vùng camera.
                  </p>
                </div>
              )}
            </div>

            {/* Control Bar & Progress */}
            <div className="bg-gray-950 p-4 border-t border-gray-800 space-y-3">
              {/* Progress Slider Bar */}
              <div className="space-y-1">
                <div className="flex justify-between text-xs text-gray-400">
                  <span>Thời gian: <strong className="text-gray-200">{currentTimeStr}</strong></span>
                  <span>Frame: <strong className="text-gray-200">{frameIdx}</strong> / {totalFrames} ({progressPct}%)</span>
                </div>
                <div className="w-full bg-gray-800 h-2 rounded-full overflow-hidden">
                  <div
                    className="bg-gradient-to-r from-blue-600 to-cyan-400 h-full transition-all duration-150"
                    style={{ width: `${progressPct}%` }}
                  />
                </div>
              </div>

              {/* Action Buttons */}
              <div className="flex items-center justify-between pt-1">
                <div className="flex items-center gap-2">
                  {(status === "ready" || status === "stopped" || status === "completed") && (
                    <button
                      onClick={handleStart}
                      disabled={!videoInfo || uploading}
                      className="bg-blue-600 hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed
                        text-white font-semibold px-6 py-2.5 rounded-xl text-sm transition-all shadow-lg shadow-blue-900/40 flex items-center gap-2"
                    >
                      <span>▶</span>
                      <span>{status === "completed" ? "Phân tích lại" : "Bắt đầu phân tích"}</span>
                    </button>
                  )}

                  {status === "processing" && (
                    <button
                      onClick={handlePause}
                      className="bg-amber-600 hover:bg-amber-500 text-white font-semibold px-5 py-2.5 rounded-xl text-sm transition-all flex items-center gap-2"
                    >
                      <span>⏸</span>
                      <span>Tạm dừng</span>
                    </button>
                  )}

                  {status === "paused" && (
                    <button
                      onClick={handleResume}
                      className="bg-emerald-600 hover:bg-emerald-500 text-white font-semibold px-5 py-2.5 rounded-xl text-sm transition-all flex items-center gap-2"
                    >
                      <span>▶</span>
                      <span>Tiếp tục</span>
                    </button>
                  )}

                  {(status === "processing" || status === "paused") && (
                    <button
                      onClick={handleStop}
                      className="bg-gray-800 hover:bg-red-900/60 hover:text-red-300 text-gray-300 px-4 py-2.5 rounded-xl text-sm transition-all border border-gray-700"
                    >
                      ⏹ Dừng
                    </button>
                  )}
                </div>

                <div className="text-xs text-gray-500">
                  {status === "processing" && "Đang gửi luồng dữ liệu..."}
                  {status === "completed" && "Đã xử lý xong toàn bộ video"}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* ================= RIGHT COLUMN: Counters & Recognized Vehicle List (5 Cols) ================= */}
        <div className="lg:col-span-5 space-y-5">
          {/* 1. Counter Card */}
          <div className="bg-gray-900 rounded-2xl border border-gray-800 p-5 shadow-xl space-y-4">
            <div className="flex items-center justify-between border-b border-gray-800 pb-3">
              <div>
                <h3 className="font-semibold text-gray-200 text-base">Bộ đếm xe vào vùng camera</h3>
                <p className="text-xs text-gray-500">Không đếm trùng lặp cùng 1 xe</p>
              </div>
              <div className="text-right">
                <span className="text-3xl font-extrabold text-blue-400 font-mono tracking-tight">
                  {counts.total}
                </span>
                <span className="block text-[10px] text-gray-500 uppercase tracking-wider">Tổng lượt xe</span>
              </div>
            </div>

            {/* Categorized Counters */}
            <div className="grid grid-cols-2 gap-3">
              {/* Car */}
              <div className="bg-gray-950/60 rounded-xl p-3 border border-blue-900/40 flex items-center justify-between">
                <div className="flex items-center gap-2.5">
                  <span className="text-xl">🚗</span>
                  <div>
                    <span className="text-xs text-gray-400 font-medium block">Ô tô</span>
                    <span className="text-xs text-blue-400/80">Car</span>
                  </div>
                </div>
                <span className="text-2xl font-bold font-mono text-white">{counts.car}</span>
              </div>

              {/* Motorcycle */}
              <div className="bg-gray-950/60 rounded-xl p-3 border border-emerald-900/40 flex items-center justify-between">
                <div className="flex items-center gap-2.5">
                  <span className="text-xl">🏍️</span>
                  <div>
                    <span className="text-xs text-gray-400 font-medium block">Xe máy</span>
                    <span className="text-xs text-emerald-400/80">Motorbike</span>
                  </div>
                </div>
                <span className="text-2xl font-bold font-mono text-white">{counts.motorcycle}</span>
              </div>

              {/* Bus */}
              <div className="bg-gray-950/60 rounded-xl p-3 border border-amber-900/40 flex items-center justify-between">
                <div className="flex items-center gap-2.5">
                  <span className="text-xl">🚌</span>
                  <div>
                    <span className="text-xs text-gray-400 font-medium block">Xe buýt</span>
                    <span className="text-xs text-amber-400/80">Bus</span>
                  </div>
                </div>
                <span className="text-2xl font-bold font-mono text-white">{counts.bus}</span>
              </div>

              {/* Truck */}
              <div className="bg-gray-950/60 rounded-xl p-3 border border-purple-900/40 flex items-center justify-between">
                <div className="flex items-center gap-2.5">
                  <span className="text-xl">🚚</span>
                  <div>
                    <span className="text-xs text-gray-400 font-medium block">Xe tải</span>
                    <span className="text-xs text-purple-400/80">Truck</span>
                  </div>
                </div>
                <span className="text-2xl font-bold font-mono text-white">{counts.truck}</span>
              </div>
            </div>
          </div>

          {/* 2. Real-time Vehicle Records List */}
          <div className="bg-gray-900 rounded-2xl border border-gray-800 p-5 shadow-xl flex flex-col h-[480px]">
            <div className="flex items-center justify-between border-b border-gray-800 pb-3 mb-3">
              <div className="flex items-center gap-2">
                <h3 className="font-semibold text-gray-200 text-sm">Danh sách xe đã nhận diện</h3>
                <span className="bg-gray-800 text-gray-300 text-xs px-2 py-0.5 rounded-full font-mono">
                  {records.length}
                </span>
              </div>
              <span className="text-xs text-gray-500">Cập nhật lúc phát</span>
            </div>

            {/* Scrollable list */}
            <div className="flex-1 overflow-y-auto space-y-2.5 pr-1.5 scrollbar-thin scrollbar-thumb-gray-800">
              {records.length === 0 ? (
                <div className="h-full flex flex-col items-center justify-center text-center p-6 text-gray-500 space-y-2">
                  <div className="text-3xl opacity-30">📋</div>
                  <p className="text-sm">Chưa có xe nào đi vào vùng camera</p>
                  <p className="text-xs text-gray-600">Khi xe đi qua vùng được cấu hình sẵn, kết quả sẽ xuất hiện ngay tại đây</p>
                </div>
              ) : (
                records.map((r, idx) => {
                  const meta = VEHICLE_META[r.class_name] || VEHICLE_META.car;
                  const hasPlate = r.plate_text && r.plate_text !== "chưa đọc được" && r.plate_text !== "Không đọc được";

                  return (
                    <div
                      key={r.id || idx}
                      className="bg-gray-950/80 rounded-xl p-3 border border-gray-800/80 hover:border-gray-700 transition-all flex items-center justify-between gap-3 shadow-sm"
                    >
                      {/* Left: Icon & Type */}
                      <div className="flex items-center gap-3">
                        <div className={`w-10 h-10 rounded-xl ${meta.bg} ${meta.border} border flex items-center justify-center text-xl shrink-0`}>
                          {meta.icon}
                        </div>
                        <div>
                          <div className="flex items-center gap-2">
                            <span className="font-bold text-gray-200 text-sm">{r.class_vn || meta.name}</span>
                            <span className="text-[11px] font-mono text-gray-500 bg-gray-800/70 px-1.5 py-0.2 rounded">
                              #{r.track_id}
                            </span>
                          </div>
                          <span className="text-xs text-gray-500">
                            Vào lúc {r.timestamp_str || "00:00"}
                          </span>
                        </div>
                      </div>

                      {/* Right: License Plate Badge & Thumbnail */}
                      <div className="flex items-center gap-2">
                        {hasPlate ? (
                          <div className="bg-white text-gray-950 font-bold px-2.5 py-1 rounded border-2 border-gray-900 font-mono text-xs tracking-wider shadow-sm">
                            {r.plate_text}
                          </div>
                        ) : (
                          <div className="bg-gray-800 text-gray-400 text-xs px-2 py-1 rounded font-medium border border-gray-700 italic">
                            chưa đọc được
                          </div>
                        )}

                        {r.plate_crop && (
                          <img
                            src={`data:image/jpeg;base64,${r.plate_crop}`}
                            alt="Crop biển số"
                            className="w-12 h-6 object-cover rounded border border-gray-700"
                          />
                        )}
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
