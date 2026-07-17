const BASE_URL = "http://localhost:8000/api";

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

export async function checkHealth() {
  const res = await fetch(`${BASE_URL}/health`);
  return res.json();
}
