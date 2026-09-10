"""Batch prediction CLI.

Scores a single image or a folder of images and writes ``filename,probability``,
where probability is the calibrated P(AI-generated).

All scoring goes through ``src/inference.py``, the same module ``src/api.py``
uses, so this CLI and the web UI can never disagree about the same image.

Usage:
    python model/predict.py --input path/to/image.jpg --out preds.csv
    python model/predict.py --input path/to/folder   --out preds.csv
    python model/predict.py --input folder --out preds.csv --verbose
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from inference import Predictor  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def collect(target: Path, recursive: bool = True) -> list[Path]:
    if target.is_file():
        return [target]
    if not target.is_dir():
        raise SystemExit(f"--input not found: {target}")
    it = target.rglob("*") if recursive else target.glob("*")
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="image file or folder")
    ap.add_argument("--out", default="preds.csv")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-recursive", action="store_true",
                    help="do not descend into subfolders")
    ap.add_argument("--verbose", action="store_true",
                    help="add verdict, confidence_band and dimensions to the CSV")
    ap.add_argument("--sort-by-probability", action="store_true",
                    help="highest probability first, instead of by filename")
    args = ap.parse_args()

    paths = collect(Path(args.input), recursive=not args.no_recursive)
    if not paths:
        raise SystemExit(f"no images found under {args.input}")

    pred = Predictor(checkpoint=args.checkpoint, calibration=args.calibration,
                     device=args.device)
    info = pred.info()
    print(f"model {info['backbone']} ({info['weights']})  device {info['device']}")
    if not info["calibrated"]:
        print("WARNING  no calibration found: probabilities are uncalibrated raw "
              "scores.\n         Run model/calibrate.py for real probabilities.")
    else:
        print(f"calibrated  T={info['temperature']}  threshold={info['threshold']}")
    print(f"scoring {len(paths)} image(s)")

    rows: list[dict] = []
    skipped: list[tuple[Path, str]] = []
    t0 = time.time()

    # Score in batches, reopening per batch so a huge folder never holds every
    # decoded image in memory at once.
    for i in range(0, len(paths), args.batch_size):
        chunk = paths[i:i + args.batch_size]
        images, keep = [], []
        for p in chunk:
            try:
                im = Image.open(p)
                im.load()
                images.append(im.convert("RGB"))
                keep.append(p)
            except Exception as e:
                skipped.append((p, f"{type(e).__name__}: {e}"))
        if not images:
            continue
        probs = pred.probabilities(images, batch_size=args.batch_size)
        for p, im, prob in zip(keep, images, probs):
            verdict, band = pred.verdict_for(float(prob))
            row = {"filename": p.name, "probability": round(float(prob), 6)}
            if args.verbose:
                row.update({"verdict": verdict, "confidence_band": band,
                            "width": im.width, "height": im.height,
                            "path": str(p)})
            rows.append(row)
            im.close()
        done = min(i + args.batch_size, len(paths))
        if len(paths) > args.batch_size:
            rate = done / max(time.time() - t0, 1e-6)
            print(f"  {done}/{len(paths)}  {rate:.1f} img/s", flush=True)

    if args.sort_by_probability:
        rows.sort(key=lambda r: -r["probability"])
    else:
        rows.sort(key=lambda r: r["filename"])

    # filename,probability first, so the required contract is the leading columns.
    fields = ["filename", "probability"]
    if args.verbose:
        fields += ["verdict", "confidence_band", "width", "height", "path"]
    out = Path(args.out)
    if out.parent != Path(""):
        out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"\nwrote {len(rows)} row(s) -> {out}")
    if skipped:
        print(f"skipped {len(skipped)} unreadable file(s):")
        for p, why in skipped[:5]:
            print(f"  {p.name}: {why}")
    if rows:
        ai = sum(1 for r in rows if r["probability"] >= pred.threshold)
        print(f"at threshold {pred.threshold:.3f}: {ai} likely AI-generated, "
              f"{len(rows) - ai} likely real")


if __name__ == "__main__":
    main()
