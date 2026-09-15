"""Degradation-vs-accuracy analysis -- the measurement behind bonus module C.

Module C asks for accuracy *maintained* when images are "JPEG-compressed,
resized, screenshotted, or lightly edited", and explicitly asks to **show a
degradation-vs-accuracy analysis**. Training with augmentation is not that
analysis; this file is. It scores the same calibrated models over a grid of
(split x degradation) and reports what each condition costs.

Where the degradation is applied
    At **224px display scale**, not at the stored file's 32x32. Every evaluated
    image is upscaled once to 224 -- "the image as displayed or shared" -- and
    the degradation is applied there, before normal preprocessing. JPEG-q30 on a
    32x32 thumbnail, or a downscale to 8x8, are not the real-world conditions
    module C is about; the same operations on 224px of content are. The honest
    caveat, stated in the generated report too: that 224px detail was upscaled
    from 32px, so these are degradations of an upscaled image, not of a natively
    high-resolution one.

Threshold
    Each model keeps the operating point calibrated on **clean** ``val``. It is
    deliberately not re-fitted per condition: at deploy time you do not know
    which degradation arrived, so re-calibrating per condition would report an
    accuracy nobody could actually obtain. Reporting AUC alongside accuracy and
    FPR at one fixed, stated threshold is also what the challenge's §4.2 asks
    for.

Scoring runs through ``src/inference.py`` -- the same path the CLI, the API and
``evaluate.py`` use -- and the metrics come from ``evaluate.metrics_for``, so a
robustness number and a headline number can never be computed two different
ways.

Usage:
    python model/robustness.py --self-check        # verify the degradations
    python model/robustness.py                     # the full grid
    python model/robustness.py --models clip --limit 500 --conditions clean jpeg_q30
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
for _p in (str(ROOT / "src"), str(ROOT / "model")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evaluate import metrics_for  # noqa: E402
from inference import Predictor  # noqa: E402

# The scale the degradations act at. 224 is both the backbones' input size and a
# reasonable stand-in for "the size this image is actually looked at".
DISPLAY_SIZE = 224

CHECKPOINT_DIR = ROOT / "model" / "checkpoints"

DEFAULT_SPLITS = ["test", "val_unseen_generator"]

# Known models, in report order. Each entry is (key, label, checkpoint filename).
KNOWN_MODELS = [
    ("clip", "CLIP ViT-B/16 + linear head (primary)", "clip_vit_b16_best.pth"),
    ("resnet", "ResNet50, fine-tuned end-to-end (baseline)", "resnet50_best.pth"),
]


# --- degradations --------------------------------------------------------
#
# Every function takes a 224x224 RGB image and returns a 224x224 RGB image, so
# each condition is self-contained and does not depend on how the inference
# preprocessor happens to resample. All are deterministic -- no random
# parameters -- so a rerun reproduces the table exactly.

def to_display(image: Image.Image) -> Image.Image:
    """The shared starting point: every source image at one display scale."""
    return image.convert("RGB").resize((DISPLAY_SIZE, DISPLAY_SIZE),
                                       Image.Resampling.BICUBIC)


def _clean(im: Image.Image) -> Image.Image:
    return im.copy()


def _jpeg(quality: int) -> Callable[[Image.Image], Image.Image]:
    def fn(im: Image.Image) -> Image.Image:
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        with Image.open(buf) as out:
            out.load()
            return out.convert("RGB")
    return fn


def _resize_roundtrip(frac: float) -> Callable[[Image.Image], Image.Image]:
    """Downscale by ``frac`` then back up -- a resize round-trip loses detail."""
    def fn(im: Image.Image) -> Image.Image:
        small = max(8, int(round(DISPLAY_SIZE * frac)))
        return (im.resize((small, small), Image.Resampling.BICUBIC)
                  .resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.Resampling.BILINEAR))
    return fn


def _screenshot(im: Image.Image) -> Image.Image:
    """Screenshot-then-share: an off-integer rescale followed by a JPEG save."""
    small = int(round(DISPLAY_SIZE * 0.85))
    shot = im.resize((small, small), Image.Resampling.BICUBIC)
    shot = _jpeg(75)(shot)
    return shot.resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.Resampling.BILINEAR)


def _light_edit(im: Image.Image) -> Image.Image:
    """A mild edit of the kind a person makes before posting: brighten, add a
    little contrast and saturation, sharpen slightly. No crop or geometry
    change, so the label is untouched."""
    out = ImageEnhance.Brightness(im).enhance(1.08)
    out = ImageEnhance.Contrast(out).enhance(1.12)
    out = ImageEnhance.Color(out).enhance(1.06)
    return out.filter(ImageFilter.UnsharpMask(radius=1.2, percent=60, threshold=2))


# (key, table label, description, fn). Order is the order the tables print in.
CONDITIONS: list[tuple[str, str, str, Callable[[Image.Image], Image.Image]]] = [
    ("clean", "clean", "upscaled to 224 only -- the reference row", _clean),
    ("jpeg_q90", "JPEG q90", "JPEG re-encode, quality 90", _jpeg(90)),
    ("jpeg_q70", "JPEG q70", "JPEG re-encode, quality 70", _jpeg(70)),
    ("jpeg_q50", "JPEG q50", "JPEG re-encode, quality 50", _jpeg(50)),
    ("jpeg_q30", "JPEG q30", "JPEG re-encode, quality 30", _jpeg(30)),
    ("resize_50", "resize 1/2", "downscale to 112 then back to 224",
     _resize_roundtrip(0.5)),
    ("resize_25", "resize 1/4", "downscale to 56 then back to 224",
     _resize_roundtrip(0.25)),
    ("screenshot", "screenshot", "0.85 rescale then JPEG q75 -- screenshot then share",
     _screenshot),
    ("light_edit", "light edit", "brightness/contrast/saturation nudge + slight sharpen",
     _light_edit),
]

CONDITION_INDEX = {key: (label, desc, fn) for key, label, desc, fn in CONDITIONS}


# --- self-check ----------------------------------------------------------

def self_check() -> int:
    """Verify every degradation before trusting an hour of GPU time to it.

    Asserts each condition returns a 224x224 RGB image, that every non-clean
    condition actually changes pixels, and that all of them are deterministic.
    Returns a process exit code.
    """
    rng = np.random.default_rng(0)
    # A structured source, not noise: JPEG and resize barely touch pure noise,
    # which would make "did it change anything" a meaningless test.
    yy, xx = np.mgrid[0:32, 0:32]
    arr = ((np.sin(xx / 2.0) + np.cos(yy / 3.0) + 2.0) * 60).astype(np.uint8)
    arr = np.stack([arr, np.roll(arr, 5, axis=0), np.roll(arr, 9, axis=1)], axis=-1)
    arr = np.clip(arr.astype(int) + rng.integers(-8, 9, arr.shape), 0, 255).astype(np.uint8)
    base = to_display(Image.fromarray(arr, mode="RGB"))

    failures: list[str] = []
    print(f"self-check: {len(CONDITIONS)} conditions at {DISPLAY_SIZE}px")
    for key, label, _desc, fn in CONDITIONS:
        try:
            out = fn(base)
            twice = fn(base)
        except Exception as e:  # a broken condition must not reach the grid
            failures.append(f"{key}: raised {type(e).__name__}: {e}")
            print(f"  {key:12} FAIL  raised {type(e).__name__}")
            continue

        problems = []
        if out.size != (DISPLAY_SIZE, DISPLAY_SIZE):
            problems.append(f"size {out.size} != {(DISPLAY_SIZE, DISPLAY_SIZE)}")
        if out.mode != "RGB":
            problems.append(f"mode {out.mode} != RGB")

        a = np.asarray(out, dtype=np.int16)
        b = np.asarray(base, dtype=np.int16)
        delta = float(np.abs(a - b).mean()) if a.shape == b.shape else float("nan")
        if key == "clean":
            if delta != 0.0:
                problems.append(f"clean changed pixels (mean |delta| {delta:.3f})")
        elif not delta > 0.0:
            problems.append("did not change any pixel")

        if np.asarray(out).tobytes() != np.asarray(twice).tobytes():
            problems.append("not deterministic across two calls")

        if problems:
            failures.extend(f"{key}: {p}" for p in problems)
            print(f"  {key:12} FAIL  {'; '.join(problems)}")
        else:
            print(f"  {key:12} ok    mean |delta| vs clean {delta:6.3f}")

    if failures:
        print(f"\nself-check FAILED ({len(failures)} problem(s))")
        return 1
    print("\nself-check passed")
    return 0


# --- the grid ------------------------------------------------------------

def _load_rows(df: pd.DataFrame) -> tuple[list[Image.Image], list[float]]:
    """Load a chunk of rows to display scale. Unreadable rows are dropped."""
    images, labels = [], []
    for _, row in df.iterrows():
        p = Path(row["path"])
        p = p if p.is_absolute() else ROOT / p
        try:
            with Image.open(p) as im:
                im.load()
                images.append(to_display(im))
            labels.append(float(row["label"]))
        except Exception:
            continue
    return images, labels


def run_grid(pred: Predictor, df: pd.DataFrame, condition_keys: list[str],
             batch_size: int, chunk: int, split_name: str) -> dict:
    """Score one split under every condition, reading each image from disk once."""
    probs: dict[str, list[np.ndarray]] = {k: [] for k in condition_keys}
    labels: list[float] = []
    t0 = time.time()

    for start in range(0, len(df), chunk):
        images, chunk_labels = _load_rows(df.iloc[start:start + chunk])
        if not images:
            continue
        labels.extend(chunk_labels)
        for key in condition_keys:
            fn = CONDITION_INDEX[key][2]
            variants = [fn(im) for im in images]
            probs[key].append(pred.probabilities(variants, batch_size=batch_size))
            for v in variants:
                v.close()
        for im in images:
            im.close()

        done = min(start + chunk, len(df))
        rate = (done * len(condition_keys)) / max(time.time() - t0, 1e-6)
        eta = ((len(df) - done) * len(condition_keys)) / max(rate, 1e-6)
        print(f"    {split_name}: {done}/{len(df)} images  "
              f"{rate:.1f} scored/s  eta {eta / 60:.1f} min", flush=True)

    y = np.array(labels, dtype=np.float32)
    out: dict[str, dict] = {}
    for key in condition_keys:
        p = np.concatenate(probs[key]) if probs[key] else np.zeros(0)
        out[key] = metrics_for(p, y, pred.threshold)
    return out


def _resolve_models(which: list[str], explicit: str | None) -> list[tuple[str, str, Path]]:
    """(key, label, checkpoint path) for each model that can actually be loaded."""
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            raise FileNotFoundError(f"checkpoint not found: {p}")
        return [("custom", f"custom checkpoint ({p.name})", p)]

    resolved = []
    for key, label, filename in KNOWN_MODELS:
        if key not in which:
            continue
        path = CHECKPOINT_DIR / filename
        if path.is_file():
            resolved.append((key, label, path))
            continue
        # No .pth on a fresh clone. The committed head still gives the primary
        # model; there is no committed baseline, so resnet is skipped loudly.
        head = ROOT / "model" / "weights" / "head.safetensors"
        if key == "clip" and head.is_file():
            print(f"note: {filename} not present; using the committed head "
                  f"({head.name}) for the primary model")
            resolved.append((key, label + " [from committed head]", head))
        else:
            print(f"note: {filename} not present -- skipping {key}. "
                  f"Train it with: python model/train.py --backbone "
                  f"{'resnet50' if key == 'resnet' else 'clip_vit_b16'}")
    return resolved


# --- reporting -----------------------------------------------------------

def _delta(m: dict, clean: dict, field: str) -> float | None:
    if field not in m or field not in clean:
        return None
    return m[field] - clean[field]


def _split_table(split: dict, condition_keys: list[str]) -> list[str]:
    clean = split.get("clean", {})
    lines = ["| condition | ROC-AUC | ΔAUC vs clean | accuracy | FPR | recall (AI) |",
             "|---|---|---|---|---|---|"]
    for key in condition_keys:
        m = split.get(key, {})
        label = CONDITION_INDEX[key][0]
        if "roc_auc" not in m:
            lines.append(f"| {label} | n/a | n/a | n/a | n/a | n/a |")
            continue
        d = _delta(m, clean, "roc_auc")
        d_str = "—" if key == "clean" or d is None else f"{d:+.4f}"
        lines.append(f"| {label} | {m['roc_auc']:.4f} | {d_str} | "
                     f"{m['accuracy']:.4f} | {m['fpr']:.4f} | {m['tpr_recall']:.4f} |")
    return lines


def _drift_line(label: str, sname: str, split: dict) -> str | None:
    """Describe *which way* the fixed threshold drifts under heavy compression.

    The headline AUC drop understates what a user experiences, because AUC is
    threshold-free. If compression shifts every probability the same direction,
    ranking survives while the calibrated operating point silently stops
    meaning what it meant on clean data -- which is the failure a deployed
    detector actually shows. Both numbers come from the grid; the direction is
    read off their signs, not assumed.
    """
    clean = split.get("clean", {})
    heaviest = next((k for k in ("jpeg_q30", "jpeg_q50", "jpeg_q70", "jpeg_q90")
                     if k in split and "roc_auc" in split[k]), None)
    if heaviest is None or "roc_auc" not in clean:
        return None
    m = split[heaviest]
    d_auc = m["roc_auc"] - clean["roc_auc"]
    d_recall = m["tpr_recall"] - clean["tpr_recall"]
    d_fpr = m["fpr"] - clean["fpr"]

    if d_recall < 0 and d_fpr <= 0:
        direction = ("**\"likely real\"** — compressed AI images stop being "
                     "flagged, and the miss rate is where the cost lands")
    elif d_recall > 0 and d_fpr >= 0:
        direction = ("**\"likely AI-generated\"** — compressed *real photos* "
                     "start being flagged, which §4.2 calls the costly error")
    else:
        direction = "no single direction (recall and FPR moved the same way)"

    return (f"- **{label} / `{sname}` under {CONDITION_INDEX[heaviest][0]}** — "
            f"AUC moves {d_auc:+.4f}, but recall moves {d_recall:+.4f} and FPR "
            f"{d_fpr:+.4f}. The ranking largely survives; what shifts is the "
            f"fixed operating point, and it drifts toward {direction}.")


def _findings(payload: dict) -> list[str]:
    """Statements derived only from the numbers in the grid, not asserted."""
    lines: list[str] = []
    per_model_drop: dict[str, float] = {}

    for mkey, model in payload["models"].items():
        for sname, split in model["splits"].items():
            clean = split.get("clean", {})
            if "roc_auc" not in clean:
                continue
            drops = [(k, split[k]["roc_auc"] - clean["roc_auc"])
                     for k in split if k != "clean" and "roc_auc" in split[k]]
            if not drops:
                continue
            worst_key, worst_d = min(drops, key=lambda kv: kv[1])
            mean_d = float(np.mean([d for _, d in drops]))
            per_model_drop.setdefault(mkey, 0.0)
            per_model_drop[mkey] += mean_d / max(len(model["splits"]), 1)

            fprs = [(k, split[k]["fpr"]) for k in split if "fpr" in split[k]]
            worst_fpr_key, worst_fpr = max(fprs, key=lambda kv: kv[1])

            lines.append(
                f"- **{model['label']} / `{sname}`** — clean AUC "
                f"{clean['roc_auc']:.4f}. Mean change across the "
                f"{len(drops)} degraded conditions: {mean_d:+.4f}. Worst "
                f"condition: **{CONDITION_INDEX[worst_key][0]}** at "
                f"{split[worst_key]['roc_auc']:.4f} ({worst_d:+.4f}). Highest "
                f"false-positive rate on real photos: "
                f"{CONDITION_INDEX[worst_fpr_key][0]} at {worst_fpr:.4f} "
                f"(clean {clean['fpr']:.4f}).")

    if len(per_model_drop) > 1:
        best = max(per_model_drop.items(), key=lambda kv: kv[1])
        worst = min(per_model_drop.items(), key=lambda kv: kv[1])
        best_label = payload["models"][best[0]]["label"]
        worst_label = payload["models"][worst[0]]["label"]
        lines.append(
            f"- **Across backbones**, averaged over every split and condition, "
            f"{best_label} loses least ({best[1]:+.4f} mean AUC) and "
            f"{worst_label} loses most ({worst[1]:+.4f}). The gap between them "
            f"is {abs(best[1] - worst[1]):.4f} AUC.")

    drift = [ln for model in payload["models"].values()
             for sname, split in model["splits"].items()
             if (ln := _drift_line(model["label"], sname, split))]
    if drift:
        lines.append("")
        lines.append("**Where the damage actually is — the threshold, not the "
                     "discriminator:**")
        lines.append("")
        lines.extend(drift)
    return lines


def write_markdown(payload: dict, path: Path, condition_keys: list[str]) -> None:
    lines = [
        "# PixelProof — degradation-vs-accuracy analysis",
        "",
        "*Bonus module C. Generated by `python model/robustness.py`; every number "
        "below is produced by that script, not transcribed by hand.*",
        "",
        "## What was measured",
        "",
        f"Each evaluated image is upscaled once to **{DISPLAY_SIZE}×{DISPLAY_SIZE}** — "
        "the image as it would be displayed or shared — and the degradation is "
        "applied at that scale before normal preprocessing. Degrading the stored "
        "32×32 file instead would mean JPEG-q30 on a thumbnail and a downscale to "
        "8×8, which are not the conditions module C describes.",
        "",
        "Each model keeps the operating point calibrated on **clean** `val`. The "
        "threshold is deliberately *not* re-fitted per condition: at deploy time "
        "you do not know which degradation arrived, so a per-condition threshold "
        "would report an accuracy no user could obtain.",
        "",
        "| condition | what it does |",
        "|---|---|",
    ]
    for key in condition_keys:
        label, desc, _ = CONDITION_INDEX[key]
        lines.append(f"| {label} | {desc} |")

    lines += ["", "## Results", ""]
    for model in payload["models"].values():
        lines.append(f"### {model['label']}")
        info = model["model"]
        lines.append("")
        lines.append(f"*{info['backbone']} — threshold {info['threshold']}, "
                     f"T={info['temperature']}*")
        lines.append("")
        for sname, split in model["splits"].items():
            n = split.get("clean", {}).get("n", 0)
            lines.append(f"**`{sname}`** (n={n})")
            lines.append("")
            lines += _split_table(split, condition_keys)
            lines.append("")

    findings = _findings(payload)
    if findings:
        lines += ["## What the numbers say", ""] + findings + [""]

    lines += [
        "## Honest caveats",
        "",
        f"- **The {DISPLAY_SIZE}px detail is upscaled from 32×32 sources.** These "
        "are degradations of an upscaled image, not of a natively "
        "high-resolution photo. A natively-512px test set would be a stronger "
        "measurement; CIFAKE does not provide one, and the native-resolution "
        "ProGAN images have no native-resolution real counterpart to pair with, "
        "so no AUC can be computed from them alone.",
        "- **The `clean` row is this grid's own baseline, not the headline "
        "metric.** It differs from `report/metrics_final.json` because of the "
        "sample size used here and the extra upscale-then-preprocess round trip.",
        "- **Degradations are applied one at a time.** Real images often arrive "
        "compressed *and* resized *and* re-saved; compounded degradation is not "
        "measured here.",
        "- Conditions are deterministic and fixed, so this table is reproducible, "
        "but it is a fixed ladder rather than a random sample of real-world "
        "handling.",
        "",
        f"*Grid: {payload['n_conditions']} conditions × "
        f"{payload['n_splits']} split(s) × {len(payload['models'])} model(s). "
        f"Generated {payload['generated']}.*",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


# --- entry point ---------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data" / "csv"))
    ap.add_argument("--splits", nargs="*", default=DEFAULT_SPLITS)
    ap.add_argument("--models", nargs="*", default=["clip", "resnet"],
                    choices=["clip", "resnet"])
    ap.add_argument("--checkpoint", default=None,
                    help="score one explicit checkpoint instead of --models")
    ap.add_argument("--conditions", nargs="*", default=[k for k, *_ in CONDITIONS],
                    help="subset of conditions to run; 'clean' is always included")
    ap.add_argument("--limit", type=int, default=4000,
                    help="cap rows per split (the CSVs are shuffled, so this is "
                         "a random balanced sample). 0 means no cap.")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=250,
                    help="images held in memory per disk pass")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-json", default=str(ROOT / "report" / "robustness.json"))
    ap.add_argument("--out-md", default=str(ROOT / "report" / "robustness.md"))
    ap.add_argument("--self-check", action="store_true",
                    help="verify the degradations and exit")
    ap.add_argument("--rebuild-report", action="store_true",
                    help="regenerate the markdown from an existing --out-json "
                         "without re-scoring anything")
    args = ap.parse_args()

    if args.self_check:
        sys.exit(self_check())

    if args.rebuild_report:
        # Scoring the grid costs hours; rewording the write-up should not.
        src = Path(args.out_json)
        if not src.is_file():
            print(f"no grid to rebuild from: {src}")
            sys.exit(1)
        payload = json.loads(src.read_text())
        keys = [c["key"] for c in payload["conditions"]]
        write_markdown(payload, Path(args.out_md), keys)
        print(f"rebuilt {args.out_md} from {src}")
        sys.exit(0)

    unknown = [c for c in args.conditions if c not in CONDITION_INDEX]
    if unknown:
        ap.error(f"unknown condition(s): {', '.join(unknown)}. "
                 f"Available: {', '.join(CONDITION_INDEX)}")
    # 'clean' is the reference every delta is measured against, so it is not
    # optional -- without it the table has nothing to compare to.
    condition_keys = [k for k, *_ in CONDITIONS
                      if k == "clean" or k in args.conditions]

    data_dir = Path(args.data)
    models = _resolve_models(args.models, args.checkpoint)
    if not models:
        print("no usable checkpoints -- nothing to evaluate")
        sys.exit(1)

    payload: dict = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "display_size": DISPLAY_SIZE,
        "limit_per_split": args.limit or None,
        "conditions": [{"key": k, "label": CONDITION_INDEX[k][0],
                        "description": CONDITION_INDEX[k][1]}
                       for k in condition_keys],
        "n_conditions": len(condition_keys),
        "models": {},
    }

    split_frames: dict[str, pd.DataFrame] = {}
    for name in args.splits:
        csv_path = data_dir / f"{name}.csv"
        if not csv_path.is_file():
            print(f"{name}: no {csv_path.name}, skipping")
            continue
        df = pd.read_csv(csv_path)
        if len(df) == 0:
            print(f"{name}: empty split, skipping")
            continue
        if args.limit and args.limit < len(df):
            df = df.iloc[:args.limit].reset_index(drop=True)
        split_frames[name] = df
    payload["n_splits"] = len(split_frames)

    if not split_frames:
        print("no usable splits -- nothing to evaluate")
        sys.exit(1)

    for mkey, mlabel, mpath in models:
        print("=" * 78)
        print(f"{mlabel}  <-  {mpath.name}")
        pred = Predictor(checkpoint=mpath, device=args.device)
        info = pred.info()
        print(f"  device {info['device']}   threshold {info['threshold']}   "
              f"T {info['temperature']}")
        print("=" * 78)

        model_entry = {"label": mlabel, "model": info, "splits": {}}
        for sname, df in split_frames.items():
            print(f"\n  {sname}: {len(df)} images x {len(condition_keys)} conditions "
                  f"(AI frac {df['label'].mean():.3f})")
            split_result = run_grid(pred, df, condition_keys, args.batch_size,
                                    args.chunk, sname)
            model_entry["splits"][sname] = split_result

            clean = split_result.get("clean", {})
            print(f"\n    {'condition':12} {'AUC':>8} {'dAUC':>9} {'acc':>8} "
                  f"{'FPR':>8} {'recall':>8}")
            for key in condition_keys:
                m = split_result[key]
                if "roc_auc" not in m:
                    print(f"    {CONDITION_INDEX[key][0]:12} {'n/a':>8}")
                    continue
                d = _delta(m, clean, "roc_auc")
                d_str = "—" if key == "clean" else f"{d:+.4f}"
                print(f"    {CONDITION_INDEX[key][0]:12} {m['roc_auc']:8.4f} "
                      f"{d_str:>9} {m['accuracy']:8.4f} {m['fpr']:8.4f} "
                      f"{m['tpr_recall']:8.4f}")

        payload["models"][mkey] = model_entry
        del pred

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n")

    out_md = Path(args.out_md)
    write_markdown(payload, out_md, condition_keys)

    print("\n" + "=" * 78)
    for line in _findings(payload):
        print("  " + line.lstrip("- ").replace("**", ""))
    print("=" * 78)
    print(f"\ngrid  -> {out_json}")
    print(f"report -> {out_md}")


if __name__ == "__main__":
    main()
