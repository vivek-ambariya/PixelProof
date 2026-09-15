# PixelProof — single-process image: FastAPI serves both /predict and the
# pre-built frontend (src/frontend/dist, already committed — no Node needed
# here). Built for Hugging Face Spaces' Docker SDK, which expects the app to
# listen on port 7860; works unchanged on any other Docker host too, just
# publish a different port at `docker run -p`.

FROM python:3.13-slim

# opencv-python-headless still links against a few shared libs that a slim
# base doesn't ship (libGL is NOT needed — that's the point of "headless" —
# but glib/SM/Xext/Xrender are pulled in transitively on some wheels).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Installed before the rest of the source so this layer only rebuilds when
# requirements.txt changes, not on every code edit.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Only what src/inference.py actually needs at runtime: the app code, the
# committed head (model/weights/), and the small non-checkpoint scripts under
# model/. model/checkpoints/*.pth is git- and docker-ignored on purpose — the
# frozen CLIP backbone comes from timm instead (see README §2).
COPY src/ src/
COPY model/ model/

# Pre-fetch the frozen CLIP ViT-B/16 backbone at build time (timm -> HF Hub)
# rather than on the first live request, so a judge's first upload isn't the
# thing waiting on a ~350 MB download. Comment out if you'd rather keep the
# image smaller and eat that latency once on first use instead.
RUN python -c "\
import sys; sys.path.insert(0, 'src'); \
from PIL import Image; \
from inference import Predictor; \
p = Predictor(); \
p.predict(Image.new('RGB', (224, 224))); \
print('backbone cached:', p.info())"

ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT}"]
