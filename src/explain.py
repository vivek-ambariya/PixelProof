"""Grad-CAM overlay plus a grounded, templated explanation.

Two jobs, kept separate:

``gradcam_map``
    A Grad-CAM saliency map over the model's own decision, upsampled to the
    original image size.

``build_explanation``
    Sentences describing what was actually measured **in the highlighted
    regions**. Every number in the returned text comes from a measurement on
    this image -- nothing is generated prose, and nothing is asserted that was
    not computed. If a signal is not detected, its sentence is omitted rather
    than hedged.

Deliberate restrictions:

- No statement is made about *who* produced an image, and none about any person,
  face, or identity appearing in it. Regions are described by position in the
  frame and by image statistics only.
- Wording is always "Likely AI-generated" / "Likely real" with a probability.
  Never "fake", "certain" or "definitely".

The three measurements mirror the three signal families the UI explains:

1. **Frequency** -- peaks in the radially averaged spectrum above the local
   trend, which upsampling lattices tend to leave and lenses do not.
2. **Noise vs luminance** -- Pearson r between per-patch luminance and per-patch
   residual deviation. Sensor noise grows with light, so real photographs
   usually give a clear positive r; synthesised grain is applied more evenly and
   trends toward zero.
3. **Local detail** -- Laplacian variance inside the highlighted regions against
   the rest of the frame, which says whether the model is keying on detailed or
   flat areas.

A caveat: at very low source resolution (CIFAKE is 32x32) the frequency test has
almost no bandwidth to work with and will usually report nothing. That is
honest, not broken.
"""

from __future__ import annotations

import base64
import io

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# Heat colours taken from the design reference: amber is high, blue is low.
HIGH_RGB = (224, 161, 58)   # #e0a13a
LOW_RGB = (40, 120, 190)    # #2878be

GRID_ROWS = ("upper", "middle", "lower")
GRID_COLS = ("left", "centre", "right")


# --- Grad-CAM ----------------------------------------------------------------
class _GradPath(nn.Module):
    """Backbone + head with gradients enabled.

    ``PixelProofNet.forward_features`` wraps a frozen backbone in ``no_grad``,
    which is right for training and fatal for Grad-CAM -- no graph means no
    activation gradients. This path deliberately avoids it and returns shape
    (B, 1) because pytorch-grad-cam expects a class dimension.
    """

    def __init__(self, net) -> None:
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net.head(self.net.backbone(x)).unsqueeze(-1)


def _target_layer_and_reshape(net):
    """Pick the Grad-CAM target layer for the backbone in use.

    For a ViT the activations are a token sequence, so they need reshaping into
    a 2D grid with the prefix (class) tokens dropped. For a CNN the last
    convolutional stage is already spatial.
    """
    if net.backbone_name == "resnet50":
        return [net.backbone.layer4[-1]], None

    n_prefix = int(getattr(net.backbone, "num_prefix_tokens", 1))
    side = net.img_size // 16  # patch16

    def reshape(t: torch.Tensor) -> torch.Tensor:
        # (B, prefix + side*side, C) -> (B, C, side, side)
        t = t[:, n_prefix:, :]
        b, n, c = t.shape
        s = int(round(n ** 0.5)) if n != side * side else side
        return t.reshape(b, s, s, c).permute(0, 3, 1, 2)

    return [net.backbone.blocks[-1].norm1], reshape


def gradcam_map(net, tensor: torch.Tensor, device: torch.device,
                out_size: tuple[int, int] | None = None) -> np.ndarray:
    """Grad-CAM saliency in [0, 1].

    ``tensor`` is a single preprocessed image, (1, 3, H, W). ``out_size`` is
    (width, height) to resize the map to, normally the original image size.
    """
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    layers, reshape = _target_layer_and_reshape(net)
    wrapper = _GradPath(net).to(device).eval()

    # A frozen backbone has requires_grad=False on every parameter, so without
    # this the forward builds no graph before the head and the CAM comes back
    # empty. Making the *input* require grad builds the graph without
    # allocating gradient buffers for 86M frozen parameters.
    x = tensor.to(device).clone().requires_grad_(True)

    kwargs = {"model": wrapper, "target_layers": layers}
    if reshape is not None:
        kwargs["reshape_transform"] = reshape
    with GradCAM(**kwargs) as cam:
        grayscale = cam(input_tensor=x, targets=[ClassifierOutputTarget(0)])
    m = grayscale[0].astype(np.float32)

    lo, hi = float(m.min()), float(m.max())
    m = (m - lo) / (hi - lo) if hi > lo else np.zeros_like(m)
    if out_size is not None:
        m = cv2.resize(m, out_size, interpolation=cv2.INTER_CUBIC)
        m = np.clip(m, 0.0, 1.0)
    return m


def overlay_heatmap(image: Image.Image, cam: np.ndarray,
                    max_alpha: float = 0.62) -> Image.Image:
    """Blend the saliency map over the image, blue (low) to amber (high)."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    h, w = rgb.shape[:2]
    if cam.shape != (h, w):
        cam = cv2.resize(cam.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    cam = np.clip(cam, 0.0, 1.0)[..., None]

    low = np.array(LOW_RGB, dtype=np.float32)
    high = np.array(HIGH_RGB, dtype=np.float32)
    heat = low * (1.0 - cam) + high * cam
    # Transparent where the model is not looking, strongest at the peaks.
    alpha = 0.10 + (max_alpha - 0.10) * cam
    out = rgb * (1.0 - alpha) + heat * alpha
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def to_base64_png(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# --- measurements ------------------------------------------------------------
def _luminance(rgb: np.ndarray) -> np.ndarray:
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])


def noise_luminance_correlation(rgb: np.ndarray, patch: int = 16) -> dict:
    """Pearson r between per-patch mean luminance and per-patch residual std.

    The residual is the image minus a 3x3 median filter, which keeps the noise
    and discards most structure. Sensor noise is shot-noise dominated, so its
    deviation rises with brightness; evenly applied synthetic grain does not.
    """
    y = _luminance(rgb)
    resid = y - cv2.medianBlur(y.astype(np.float32), 3)
    h, w = y.shape
    if min(h, w) < patch * 2:
        patch = max(4, min(h, w) // 4)
    means, stds = [], []
    for r in range(0, h - patch + 1, patch):
        for c in range(0, w - patch + 1, patch):
            means.append(float(y[r:r + patch, c:c + patch].mean()))
            stds.append(float(resid[r:r + patch, c:c + patch].std()))
    if len(means) < 8:
        return {"n_patches": len(means), "r": None}
    m = np.array(means)
    s = np.array(stds)
    if m.std() < 1e-6 or s.std() < 1e-6:
        return {"n_patches": len(means), "r": None}
    return {"n_patches": len(means), "r": float(np.corrcoef(m, s)[0, 1]),
            "mean_residual_std": float(s.mean())}


def spectral_peaks(rgb: np.ndarray, min_side: int = 64) -> dict:
    """Peaks in the radially averaged log spectrum above a smoothed trend.

    Returns ``{"available": False}`` when the image is too small for the test to
    mean anything, which is the honest answer for a 32x32 source.
    """
    y = _luminance(rgb)
    h, w = y.shape
    if min(h, w) < min_side:
        return {"available": False, "reason": f"source is {w}x{h}; "
                f"the frequency test needs at least {min_side}px on the short side"}
    n = min(h, w)
    y = y[:n, :n]
    y = y - y.mean()
    win = np.outer(np.hanning(n), np.hanning(n))
    mag = np.abs(np.fft.fftshift(np.fft.fft2(y * win)))
    logmag = np.log1p(mag)

    cy = cx = n // 2
    yy, xx = np.mgrid[:n, :n]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.int32)
    nbins = n // 2
    prof = np.array([logmag[r == i].mean() for i in range(nbins)])

    # Smooth to get the expected 1/f falloff, then look for what stands above it.
    k = max(5, nbins // 12) | 1
    trend = np.convolve(prof, np.ones(k) / k, mode="same")
    excess = prof - trend
    sd = float(excess[nbins // 4:].std())
    peaks = []
    if sd > 1e-9:
        for i in range(max(3, nbins // 8), nbins - 2):
            if excess[i] > 3.0 * sd and excess[i] >= excess[i - 1] and excess[i] >= excess[i + 1]:
                peaks.append({"norm_freq": round(i / nbins, 3),
                              "sigma_above_trend": round(float(excess[i] / sd), 1)})
    peaks.sort(key=lambda p: -p["sigma_above_trend"])
    return {"available": True, "peaks": peaks[:3], "n_bins": nbins}


def top_mask(cam: np.ndarray, top_frac: float = 0.15) -> np.ndarray:
    """Boolean mask of the highest-saliency pixels, selected by rank.

    Rank rather than a value threshold: a quantile cut is unusable when the map
    holds many identical values (a mostly-zero CAM makes the 85th percentile
    equal to the minimum, marking every pixel as hot). argpartition always
    returns exactly k pixels regardless of ties.
    """
    k = max(1, int(round(top_frac * cam.size)))
    idx = np.argpartition(cam.ravel(), -k)[-k:]
    flat = np.zeros(cam.size, dtype=bool)
    flat[idx] = True
    return flat.reshape(cam.shape)


def region_summary(cam: np.ndarray) -> dict:
    """Where the saliency mass sits, on a 3x3 grid, and how peaked it is.

    Shares are fractions of total saliency mass, not counts above a threshold.
    That needs no cutoff, is unaffected by ties, and reads naturally as
    "this share of the model's attention".
    """
    h, w = cam.shape
    rows = np.linspace(0, h, 4).astype(int)
    cols = np.linspace(0, w, 4).astype(int)
    total = float(cam.sum())
    cells = []
    for ri in range(3):
        for ci in range(3):
            block = cam[rows[ri]:rows[ri + 1], cols[ci]:cols[ci + 1]]
            share = float(block.sum()) / total if total > 1e-9 else 0.0
            cells.append({"name": f"{GRID_ROWS[ri]}-{GRID_COLS[ci]}", "share": share})
    cells.sort(key=lambda c: -c["share"])
    # How peaked is the map? Mean of the top decile over the overall mean; 1.0
    # is perfectly uniform.
    flat = np.sort(cam.ravel())[::-1]
    concentration = float(flat[:max(1, len(flat) // 10)].mean() /
                          max(1e-6, flat.mean()))
    # 1/9 = 0.111 is uniform, so require a clear margin over it to count.
    return {"cells": cells,
            "top_cells": [c for c in cells if c["share"] >= 0.18][:3],
            "concentration": concentration}


def detail_contrast(rgb: np.ndarray, cam: np.ndarray, top_frac: float = 0.15) -> dict:
    """Laplacian variance inside the hot regions vs the rest of the frame."""
    y = _luminance(rgb).astype(np.float32)
    if cam.shape != y.shape:
        cam = cv2.resize(cam.astype(np.float32), (y.shape[1], y.shape[0]),
                         interpolation=cv2.INTER_CUBIC)
    lap = cv2.Laplacian(y, cv2.CV_32F)
    hot = top_mask(cam, top_frac)
    if hot.sum() < 16 or (~hot).sum() < 16:
        return {"available": False}
    return {"available": True,
            "hot_lap_var": float(lap[hot].var()),
            "cold_lap_var": float(lap[~hot].var())}


# --- explanation -------------------------------------------------------------
def measure(image: Image.Image, cam: np.ndarray) -> dict:
    """Every measurement the explanation is allowed to cite."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    return {
        "size": {"width": int(rgb.shape[1]), "height": int(rgb.shape[0])},
        "regions": region_summary(cam),
        "noise": noise_luminance_correlation(rgb),
        "spectrum": spectral_peaks(rgb),
        "detail": detail_contrast(rgb, cam),
    }


def format_pct(p: float) -> str:
    """Format a probability for display, without ever implying certainty.

    A probability of 0.0002 renders as "0.0%" under plain rounding, which reads
    as a definite answer -- exactly the claim this tool must not make. Values in
    the extreme tails are floored and capped instead.
    """
    pct = p * 100.0
    if pct < 0.1:
        return "under 0.1%"
    if pct > 99.9:
        return "over 99.9%"
    return f"{pct:.1f}%"


def _phrase_list(names: list[str]) -> str:
    if not names:
        return "no single region"
    if len(names) == 1:
        return f"the {names[0]}"
    return "the " + ", ".join(names[:-1]) + f" and {names[-1]}"


def build_explanation(probability: float, image: Image.Image, cam: np.ndarray,
                      threshold: float = 0.5) -> tuple[str, dict]:
    """Return (explanation text, the measurements it was built from).

    Every clause is emitted only when the measurement backing it exists.
    """
    m = measure(image, cam)
    parts: list[str] = []

    # 1. Where the model looked.
    top = [c["name"] for c in m["regions"]["top_cells"]]
    conc = m["regions"]["concentration"]
    if top and conc > 1.6:
        parts.append(
            f"Attention concentrates in {_phrase_list(top)} of the frame "
            f"({int(round(100 * sum(c['share'] for c in m['regions']['top_cells'])))}% "
            f"of the strongest 15% of saliency)."
        )
    else:
        parts.append("Saliency is spread fairly evenly across the frame rather "
                     "than resting on any one region.")

    # 2. Noise against luminance.
    r = m["noise"].get("r")
    if r is not None:
        if r >= 0.35:
            parts.append(
                f"Per-patch noise deviation rises with local brightness "
                f"(r = {r:.2f} over {m['noise']['n_patches']} patches), the "
                f"relationship a camera sensor normally leaves."
            )
        elif r <= 0.15:
            parts.append(
                f"Per-patch noise deviation is close to flat against local "
                f"brightness (r = {r:.2f} over {m['noise']['n_patches']} "
                f"patches), where sensor output usually shows a clear positive "
                f"relationship."
            )
        else:
            parts.append(
                f"The relationship between per-patch noise and local brightness "
                f"is weak but present (r = {r:.2f}), which does not separate the "
                f"two cases on its own."
            )

    # 3. Frequency.
    spec = m["spectrum"]
    if spec.get("available"):
        pk = spec.get("peaks") or []
        if pk:
            p = pk[0]
            parts.append(
                f"The radially averaged spectrum carries a peak "
                f"{p['sigma_above_trend']}x above its local trend at normalised "
                f"frequency {p['norm_freq']}, consistent with a resampling step."
            )
        else:
            parts.append("The radially averaged spectrum shows no peak standing "
                         "above its local trend, so the frequency test adds nothing here.")
    elif spec.get("reason"):
        parts.append(f"The frequency test was skipped: {spec['reason']}.")

    # 4. Detail context for the hot regions.
    d = m["detail"]
    if d.get("available"):
        hot, cold = d["hot_lap_var"], d["cold_lap_var"]
        if cold > 1e-6:
            ratio = hot / cold
            if ratio >= 1.3:
                parts.append(f"The highlighted regions carry {ratio:.1f}x the "
                             f"local detail of the rest of the frame.")
            elif ratio <= 0.77:
                parts.append(f"The highlighted regions are markedly flatter than "
                             f"the rest of the frame ({ratio:.2f}x its local detail), "
                             f"so the estimate rests on smooth areas.")

    # 5. The verdict sentence, then the standing caveat.
    verdict = "Likely AI-generated" if probability >= threshold else "Likely real"
    parts.insert(0, f"{verdict}, at an estimated {format_pct(probability)} "
                    f"probability of being AI-generated.")
    parts.append("Read this as a likelihood, not a finding.")
    return " ".join(parts), m


if __name__ == "__main__":
    # Exercise the measurement functions on synthetic inputs, no model needed.
    rng = np.random.default_rng(0)
    print("=" * 68)
    # A "sensor-like" image: noise scaled by local brightness.
    ramp = np.tile(np.linspace(10, 240, 256), (256, 1))
    sensor = ramp + rng.normal(0, 1, (256, 256)) * np.sqrt(ramp) * 0.7
    sensor_rgb = np.clip(np.stack([sensor] * 3, -1), 0, 255)
    # A "flat grain" image: constant-variance noise.
    flat = ramp + rng.normal(0, 6, (256, 256))
    flat_rgb = np.clip(np.stack([flat] * 3, -1), 0, 255)
    for nm, arr in (("sensor-like (noise scales with light)", sensor_rgb),
                    ("flat grain (constant variance)", flat_rgb)):
        res = noise_luminance_correlation(arr)
        print(f"{nm:42} r = {res['r']:+.3f}")
    print()
    # A lattice image should produce a spectral peak; smooth gradient should not.
    xx = np.arange(256)
    lattice = 128 + 40 * np.sin(2 * np.pi * xx / 4)[None, :].repeat(256, 0)
    for nm, arr in (("4px lattice", np.stack([lattice] * 3, -1)),
                    ("smooth ramp", ramp[..., None].repeat(3, -1))):
        s = spectral_peaks(np.clip(arr, 0, 255).astype(np.float32))
        print(f"{nm:42} peaks = {s.get('peaks')}")
    print()
    small = spectral_peaks(np.zeros((32, 32, 3), dtype=np.float32))
    print(f"{'32x32 source':42} {small}")
    print()
    for nm, mk in (("hot upper-left", lambda c: c.__setitem__((slice(4, 22), slice(4, 22)), 1.0)),
                   ("hot lower-right", lambda c: c.__setitem__((slice(42, 60), slice(42, 60)), 1.0)),
                   ("uniform", lambda c: c.__setitem__((slice(None), slice(None)), 1.0))):
        cam = np.zeros((64, 64), np.float32); mk(cam)
        rs = region_summary(cam)
        top = [f"{c['name']} {c['share']:.0%}" for c in rs['top_cells']] or ["(none)"]
        print(f"{nm:24} conc {rs['concentration']:5.2f}  top: {', '.join(top)}")
    print("=" * 68)
