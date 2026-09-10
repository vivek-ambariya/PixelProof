"""The single shared inference path.

**Everything** that scores an image goes through this module: ``model/predict.py``
(what the organisers run), ``src/api.py`` (what users hit), and
``model/evaluate.py`` (metrics). Duplicating any of this elsewhere is how a CLI
and an API end up disagreeing about the same picture, so the loading,
preprocessing, calibration and wording all live here once.

Probability
    ``sigmoid(logit / temperature)`` -- never a bare sigmoid. The temperature is
    fitted by ``model/calibrate.py`` on the validation split. Without it the
    number is a raw score that looks like a probability and is not one.

Verdict wording
    Always "Likely AI-generated" or "Likely real", with the probability. Never
    "fake", "certain" or "definitely". Nothing is claimed about who made an
    image or about any person in it. ``confidence_band`` is reported separately
    so a UI can show an inconclusive state without changing that wording.

Checkpoint resolution, in order:
    1. an explicit path
    2. ``model/checkpoints/<backbone>_best.pth`` -- what training writes
    3. ``model/weights/head.safetensors`` -- the tiny committed head, so a fresh
       clone can predict with no training run. The frozen backbone is pulled
       from timm on first use and cached; only the head is version-controlled.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
_HERE = Path(__file__).resolve().parent

# Both directories go on the path explicitly. This module is imported two
# different ways -- as `inference` by the CLIs (which add src/ themselves) and as
# `src.inference` by `uvicorn src.api:app`, which puts only the project root on
# sys.path. Without src/ here, the `import explain` below fails under uvicorn.
for _p in (str(ROOT / "model"), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model import DEFAULT_BACKBONE, PixelProofNet, pick_device  # noqa: E402

import explain  # noqa: E402

CHECKPOINT_DIR = ROOT / "model" / "checkpoints"
WEIGHTS_DIR = ROOT / "model" / "weights"

VERDICT_AI = "Likely AI-generated"
VERDICT_REAL = "Likely real"

# Probabilities inside this band are reported as inconclusive. The design's
# footer states that scores in the middle should not be acted on, and this is
# where that lives. It never changes the two verdict strings.
INCONCLUSIVE_LO = 0.35
INCONCLUSIVE_HI = 0.65

MAX_SIDE = 1600  # cap the long edge before Grad-CAM, to bound memory and time


def _find_checkpoint() -> tuple[Path | None, str]:
    """Locate a usable checkpoint. Returns (path, kind)."""
    # An explicit override wins, so the API can be pointed at another
    # checkpoint without editing code: PIXELPROOF_CHECKPOINT=... uvicorn src.api:app
    env = os.environ.get("PIXELPROOF_CHECKPOINT")
    if env:
        p = Path(env)
        if not p.is_file():
            raise FileNotFoundError(f"PIXELPROOF_CHECKPOINT is set but missing: {p}")
        return p, "head" if p.suffix == ".safetensors" else "checkpoint"
    if CHECKPOINT_DIR.is_dir():
        for name in (f"{DEFAULT_BACKBONE}_best.pth", "resnet50_best.pth"):
            p = CHECKPOINT_DIR / name
            if p.is_file():
                return p, "checkpoint"
        found = sorted(CHECKPOINT_DIR.glob("*_best.pth"))
        if found:
            return found[0], "checkpoint"
    head = WEIGHTS_DIR / "head.safetensors"
    if head.is_file():
        return head, "head"
    return None, "none"


class Predictor:
    """Loads a model once and scores images. Not thread-safe; the API holds one."""

    def __init__(self, checkpoint: str | Path | None = None,
                 calibration: str | Path | None = None,
                 device: str = "auto",
                 half: bool | None = None) -> None:
        self.device = pick_device(device)
        # Autocast fp16 roughly doubles throughput on MPS (14 -> 27 img/s for
        # ViT-B/16 measured on an M2). It is deliberately ONE setting shared by
        # every scoring path below, so the CLI and the API cannot end up on
        # different numerics and disagree about the same image. CPU stays fp32,
        # where autocast buys nothing.
        self.half = (self.device.type in ("mps", "cuda")) if half is None else half

        if checkpoint is None:
            path, kind = _find_checkpoint()
            if path is None:
                raise FileNotFoundError(
                    "no model weights found. Train one:\n"
                    "  python model/train.py --backbone clip_vit_b16\n"
                    f"or place a committed head at {WEIGHTS_DIR / 'head.safetensors'}"
                )
        else:
            path = Path(checkpoint)
            if not path.is_file():
                raise FileNotFoundError(f"checkpoint not found: {path}")
            kind = "head" if path.suffix == ".safetensors" else "checkpoint"

        self.checkpoint_path = path
        self.checkpoint_kind = kind

        if kind == "checkpoint":
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            self.backbone_name = ckpt.get("backbone_name", DEFAULT_BACKBONE)
            self.net = PixelProofNet(self.backbone_name, pretrained=True)
            self.net.load_state_dict(ckpt["model"])
            self.train_epoch = ckpt.get("epoch")
        else:
            from safetensors.torch import load_file
            meta_path = path.with_name("head_config.json")
            meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
            self.backbone_name = meta.get("backbone", DEFAULT_BACKBONE)
            self.net = PixelProofNet(self.backbone_name, pretrained=True)
            self.net.head.load_state_dict(load_file(str(path)))
            self.train_epoch = meta.get("epoch")

        self.net.to(self.device).eval()

        # Calibration. Absent means temperature 1.0 and threshold 0.5, and
        # `calibrated` reports False so callers can surface that honestly.
        if calibration is None:
            calibration = os.environ.get("PIXELPROOF_CALIBRATION") or None
        if calibration is None:
            cand = [path.parent / f"{self.backbone_name}_calibration.json",
                    WEIGHTS_DIR / "calibration.json"]
            calibration = next((c for c in cand if c.is_file()), None)
        self.temperature, self.threshold, self.calibrated = 1.0, 0.5, False
        self.calibration_path = None
        if calibration is not None and Path(calibration).is_file():
            cal = json.loads(Path(calibration).read_text())
            self.temperature = float(cal.get("temperature", 1.0))
            self.threshold = float(cal.get("threshold", 0.5))
            self.calibrated = True
            self.calibration_path = Path(calibration)

    # --- info ------------------------------------------------------------
    def info(self) -> dict:
        return {
            "backbone": self.backbone_name,
            "weights": str(self.checkpoint_path.name),
            "weights_kind": self.checkpoint_kind,
            "device": str(self.device),
            "img_size": self.net.img_size,
            "calibrated": self.calibrated,
            "temperature": round(self.temperature, 4),
            "threshold": round(self.threshold, 4),
            "train_epoch": self.train_epoch,
        }

    # --- preprocessing ---------------------------------------------------
    def _tensor(self, image: Image.Image) -> torch.Tensor:
        """Resize to the backbone's input and normalise. Returns (1, 3, H, W)."""
        arr = np.asarray(image.convert("RGB"))
        s = self.net.img_size
        arr = cv2.resize(arr, (s, s), interpolation=cv2.INTER_CUBIC)
        x = arr.astype(np.float32) / 255.0
        x = (x - np.array(self.net.mean, np.float32)) / np.array(self.net.std, np.float32)
        return torch.from_numpy(x.transpose(2, 0, 1))[None]

    def _calibrate(self, logits: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-logits / self.temperature))

    @torch.no_grad()
    def _logits(self, x: torch.Tensor) -> np.ndarray:
        """The single forward used by every scoring path. Returns float32 logits."""
        x = x.to(self.device)
        if self.half:
            with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                out = self.net(x)
        else:
            out = self.net(x)
        return out.float().cpu().numpy()

    # --- scoring ---------------------------------------------------------
    @torch.no_grad()
    def probabilities(self, images: list[Image.Image],
                      batch_size: int = 64) -> np.ndarray:
        """Calibrated P(AI-generated) for a list of images. No heatmaps.

        This is the batched path used by predict.py and evaluate.py.
        """
        out: list[np.ndarray] = []
        for i in range(0, len(images), batch_size):
            chunk = images[i:i + batch_size]
            out.append(self._logits(torch.cat([self._tensor(im) for im in chunk])))
        if not out:
            return np.zeros(0, dtype=np.float32)
        return self._calibrate(np.concatenate(out))

    @torch.no_grad()
    def probability(self, image: Image.Image) -> float:
        return float(self.probabilities([image])[0])

    def verdict_for(self, p: float) -> tuple[str, str]:
        """(verdict, confidence_band) for a calibrated probability."""
        verdict = VERDICT_AI if p >= self.threshold else VERDICT_REAL
        band = ("inconclusive" if INCONCLUSIVE_LO <= p <= INCONCLUSIVE_HI
                else "clear")
        return verdict, band

    def predict(self, image: Image.Image, want_heatmap: bool = True,
                want_explanation: bool = True) -> dict:
        """Full single-image result: probability, verdict, heatmap, explanation.

        This is what POST /predict returns. The image is never written to disk.
        """
        t0 = time.time()
        image = image.convert("RGB")
        orig_w, orig_h = image.size

        # Bound the work for very large uploads. The model only ever sees
        # img_size anyway; this caps the Grad-CAM overlay cost.
        work = image
        if max(orig_w, orig_h) > MAX_SIDE:
            scale = MAX_SIDE / max(orig_w, orig_h)
            work = image.resize((max(1, int(orig_w * scale)),
                                 max(1, int(orig_h * scale))), Image.LANCZOS)

        x = self._tensor(work)
        # Same _logits path as the batched CLI, so both report the same number.
        logit = float(self._logits(x)[0])
        p = float(self._calibrate(np.array([logit]))[0])
        verdict, band = self.verdict_for(p)

        result = {
            "probability": round(p, 6),
            "verdict": verdict,
            "heatmap_base64": "",
            "explanation": "",
            # everything below is extra context, not part of the required contract
            "confidence_band": band,
            "threshold": round(self.threshold, 4),
            "calibrated": self.calibrated,
            "logit": round(logit, 6),
            "image": {"width": orig_w, "height": orig_h},
            "model": {"backbone": self.backbone_name,
                      "weights_kind": self.checkpoint_kind},
        }

        cam = None
        if want_heatmap or want_explanation:
            try:
                cam = explain.gradcam_map(self.net, x, self.device,
                                         out_size=work.size)
            except Exception as e:  # a failed CAM must not fail the prediction
                result["heatmap_error"] = f"{type(e).__name__}: {e}"

        if cam is not None and want_heatmap:
            result["heatmap_base64"] = explain.to_base64_png(
                explain.overlay_heatmap(work, cam))

        if want_explanation:
            if cam is not None:
                text, measurements = explain.build_explanation(
                    p, work, cam, threshold=self.threshold)
                result["explanation"] = text
                result["measurements"] = measurements
            else:
                result["explanation"] = (
                    f"{verdict}, at an estimated {p * 100:.1f}% probability of "
                    f"being AI-generated. The saliency map could not be computed "
                    f"for this image, so no region-level evidence is available. "
                    f"Read this as a likelihood, not a finding."
                )

        if not self.calibrated:
            result["warning"] = (
                "no calibration file found, so this probability is an "
                "uncalibrated raw score. Run model/calibrate.py."
            )

        result["elapsed_ms"] = int((time.time() - t0) * 1000)
        return result

    def predict_path(self, path: str | Path, **kw) -> dict:
        with Image.open(path) as im:
            out = self.predict(im, **kw)
        out["filename"] = Path(path).name
        return out


_CACHED: Predictor | None = None


def get_predictor(**kw) -> Predictor:
    """Process-wide singleton, so the API loads the backbone once."""
    global _CACHED
    if _CACHED is None:
        _CACHED = Predictor(**kw)
    return _CACHED


def export_head(checkpoint: str | Path, out_dir: Path = WEIGHTS_DIR) -> dict:
    """Write the trainable head to safetensors so it can be committed.

    A frozen backbone means the head is the only thing training produced:
    197,121 parameters, about 770 KB at fp32, against ~344 MB for the full
    checkpoint whose backbone timm can fetch on demand. That is what lets a
    fresh clone predict without a training run while .gitignore still excludes
    every real checkpoint.
    """
    from safetensors.torch import save_file

    ckpt = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    backbone = ckpt.get("backbone_name", DEFAULT_BACKBONE)
    head = {k: v.contiguous() for k, v in ckpt["head"].items()}
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(head, str(out_dir / "head.safetensors"))
    meta = {
        "backbone": backbone,
        "epoch": ckpt.get("epoch"),
        "monitor_split": ckpt.get("monitor_split"),
        "monitor_auc": ckpt.get("monitor_auc"),
        "note": "trainable head only; the frozen backbone comes from timm",
    }
    (out_dir / "head_config.json").write_text(json.dumps(meta, indent=2) + "\n")
    size = (out_dir / "head.safetensors").stat().st_size
    print(f"head -> {out_dir / 'head.safetensors'} ({size / 1024:.1f} KB)")
    print(f"meta -> {out_dir / 'head_config.json'}")

    cal = Path(checkpoint).parent / f"{backbone}_calibration.json"
    if cal.is_file():
        (out_dir / "calibration.json").write_text(cal.read_text())
        print(f"calibration -> {out_dir / 'calibration.json'}")
    else:
        print(f"note: no calibration at {cal}; run model/calibrate.py first")
    return meta


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="inspect or export the shared model")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--export-head", action="store_true",
                    help="write model/weights/head.safetensors for committing")
    ap.add_argument("--image", default=None, help="score one image and print the result")
    args = ap.parse_args()

    if args.export_head:
        ck = args.checkpoint or _find_checkpoint()[0]
        if ck is None:
            raise SystemExit("no checkpoint to export")
        export_head(ck)
        raise SystemExit(0)

    pred = Predictor(checkpoint=args.checkpoint)
    print(json.dumps(pred.info(), indent=2))
    if args.image:
        r = pred.predict_path(args.image)
        r.pop("heatmap_base64", None)
        r.pop("measurements", None)
        print(json.dumps(r, indent=2))
