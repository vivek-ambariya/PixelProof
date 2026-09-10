// The only place that talks to the backend.
// Paths are relative, so the same build works behind `uvicorn src.api:app`
// (served from /) and behind the Vite dev proxy.

// Must match MAX_UPLOAD_BYTES in src/api.py, so the client rejects a file with
// the same number the server would quote back.
export const MAX_UPLOAD_MB = 12
export const MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1_000_000

/** Human-readable reason a file cannot be scored, or null if it is fine. */
export function rejectReason(file) {
  if (!file) return 'No file selected.'
  if (!file.type.startsWith('image/')) {
    return `${file.name} is not an image.`
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    return `${file.name} is ${(file.size / 1e6).toFixed(1)} MB; the limit is ${MAX_UPLOAD_MB} MB.`
  }
  return null
}

async function errorFrom(res) {
  let detail = `HTTP ${res.status}`
  try {
    const body = await res.json()
    if (body && body.detail) detail = body.detail
  } catch {
    /* a non-JSON error body is not worth reporting verbatim */
  }
  const err = new Error(detail)
  err.status = res.status
  return err
}

/** POST /predict -> { probability, verdict, heatmap_base64, explanation, ... } */
export async function predictImage(file, { signal } = {}) {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch('/predict', { method: 'POST', body: form, signal })
  if (!res.ok) throw await errorFrom(res)
  return res.json()
}

/** GET /api/health -> model + build status. Never throws; returns null instead. */
export async function fetchHealth() {
  try {
    const res = await fetch('/api/health')
    if (!res.ok) return null
    return await res.json()
  } catch {
    return null
  }
}
