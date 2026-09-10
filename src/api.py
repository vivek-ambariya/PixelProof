"""FastAPI service: POST /predict, plus the built frontend on /.

One command runs everything:

    uvicorn src.api:app        ->  http://localhost:8000

``src/frontend/dist`` is committed, so a judge needs no Node install. If the
build is missing the API still starts and ``/`` explains how to build it, rather
than failing at import time.

Scoring goes through ``src/inference.py`` -- the same module ``model/predict.py``
uses. Nothing about inference is reimplemented here; this file is transport only.

Uploaded images are held in memory for the duration of the request and never
written to disk.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from inference import Predictor  # noqa: E402

DIST = ROOT / "src" / "frontend" / "dist"

# 12 MB decimal, matching the "MAX 12 MB" the UI prints and the same constant in
# src/frontend/src/lib/api.js. Deliberately not 12 MiB: 12*1024*1024/1e6 is
# 12.58, which the error message would round to "13 MB" and contradict the UI.
MAX_UPLOAD_MB = 12
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1_000_000

app = FastAPI(
    title="PixelProof",
    description="Estimates the likelihood that an image was AI-generated, and "
                "shows where in the frame the estimate comes from.",
    version="0.3.0",
)

# The Vite dev server runs on another origin; harmless for a local-only tool.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loaded lazily so the process starts (and /api/health can explain itself) even
# with no checkpoint present.
_predictor: Predictor | None = None
_load_error: str | None = None


def get_predictor() -> Predictor:
    global _predictor, _load_error
    if _predictor is None:
        try:
            _predictor = Predictor()
            _load_error = None
        except Exception as e:
            _load_error = f"{type(e).__name__}: {e}"
            raise HTTPException(
                status_code=503,
                detail=f"model unavailable: {_load_error}",
            )
    return _predictor


@app.get("/api/health")
def health() -> JSONResponse:
    """Model and build status. Never raises, so the UI can always render."""
    try:
        info = get_predictor().info()
        model_ok = True
    except HTTPException:
        info, model_ok = {"error": _load_error}, False
    return JSONResponse({
        "status": "ok" if model_ok else "degraded",
        "model_loaded": model_ok,
        "model": info,
        "frontend_built": DIST.is_dir(),
    })


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    """Score one uploaded image.

    Returns ``probability`` (calibrated P(AI-generated)), ``verdict``,
    ``heatmap_base64`` (PNG) and ``explanation``, plus extra context fields.
    """
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty upload")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"file is {len(raw) / 1e6:.1f} MB; the limit is "
                   f"{MAX_UPLOAD_MB} MB",
        )

    try:
        with Image.open(io.BytesIO(raw)) as im:
            im.load()
            image = im.convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        raise HTTPException(status_code=400,
                            detail=f"could not decode image: {type(e).__name__}")

    pred = get_predictor()
    try:
        result = pred.predict(image)
    except Exception as e:
        # The image is already discarded at this point; say so plainly.
        raise HTTPException(status_code=500,
                            detail=f"inference failed: {type(e).__name__}: {e}")

    result["filename"] = file.filename or "upload"
    return JSONResponse(result)


@app.get("/api/model")
def model_info() -> JSONResponse:
    return JSONResponse(get_predictor().info())


_MISSING_BUILD_HTML = """<!doctype html>
<title>PixelProof - frontend not built</title>
<style>
 body{background:#0b0908;color:#ece7e1;font:15px/1.6 system-ui,sans-serif;
      margin:0;padding:48px;max-width:60ch}
 code{background:#1a1614;padding:2px 6px;border-radius:2px;color:#4fd8c4}
 h1{font-size:28px;letter-spacing:-.02em;margin:0 0 18px}
 a{color:#4fd8c4}
 .k{color:#8a807a;font:11px/1.8 ui-monospace,monospace;letter-spacing:.12em;
    text-transform:uppercase}
</style>
<div class="k">PixelProof / api running</div>
<h1>The frontend has not been built</h1>
<p>The API is up and <code>POST /predict</code> works. To build the UI:</p>
<pre><code>cd src/frontend
npm install
npm run build</code></pre>
<p>Then reload this page. Meanwhile you can check
   <a href="/api/health">/api/health</a> or
   <a href="/docs">/docs</a>.</p>
"""


# Mounted LAST and at "/", because a mount at the root captures every path that
# earlier routes did not already claim.
if DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(DIST), html=True), name="frontend")
else:
    @app.get("/")
    def missing_build() -> HTMLResponse:
        return HTMLResponse(_MISSING_BUILD_HTML, status_code=200)
