const BASE_URL = (import.meta.env.VITE_API_BASE_URL || "/api").replace(/\/$/, "");
const WS_BASE_URL = import.meta.env.VITE_WS_BASE_URL ||
  `${window.location.protocol === "https:" ? "wss:" : "ws:"}//${window.location.host}/ws`;

export async function predictImage(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE_URL}/predict/image`, { method: "POST", body: form });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function predictVideo(file, frameInterval = 10) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE_URL}/predict/video?frame_interval=${frameInterval}`, {
    method: "POST", body: form,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function uploadVideo(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE_URL}/video/upload`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    let errText = "Lỗi khi tải video lên server";
    try {
      const errJson = await res.json();
      errText = errJson.detail || errText;
    } catch (_) {
      errText = await res.text() || errText;
    }
    throw new Error(errText);
  }
  return res.json();
}

export function getVideoRawUrl(videoId) {
  return `${BASE_URL}/video/raw/${videoId}`;
}

export function getVideoWsUrl(videoId) {
  return `${WS_BASE_URL}/video/${videoId}`;
}

export async function checkHealth() {
  const res = await fetch(`${BASE_URL}/health`);
  return res.json();
}
