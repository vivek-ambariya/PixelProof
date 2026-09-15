"""Carve a native-resolution held-out test set out of the fetched GenImage data.

Why a separate split
    ``test``, ``val_unseen_content`` and ``val_unseen_generator`` are all
    32x32-native by construction (CIFAKE, and ProGAN resolution-matched down to
    32x32 on purpose). None of them can detect the failure mode that matters
    most in deployment: a detector trained only on 32x32 sources learns
    "sharp and detailed" as a stand-in for "AI-generated", because no real
    training image was ever sharp. On a genuine native-resolution photograph
    that shortcut inverts and real photos are confidently called fake.

    ``test_native`` is the split that can see it: native-resolution photographs
    against native-resolution generated images, never resized down.

How the holdout stays honest
    The images are physically **moved out** of ``data/raw/`` before
    ``prepare.py`` runs, so they cannot be indexed into train or val by any
    later rebuild. Everything left under ``data/raw/`` is fair game for
    training; everything under ``data/holdout_native/`` is not.

``evaluate.py`` reads ``<data-dir>/<split>.csv`` by name, so the CSV written
here needs no change anywhere else:

    python model/evaluate.py --splits test_native

Usage:
    python data/make_native_testset.py                 # 400 per class
    python data/make_native_testset.py --per-class 250
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# Columns prepare.py writes, matched exactly so evaluate.py sees a normal split.
COLUMNS = ["path", "label", "generator", "source", "content_class", "content_group"]

# (source dir under data/raw, label, generator, source) for each half.
SOURCES = [
    (ROOT / "data" / "raw" / "real" / "genimage_imagenet", 0, "none", "genimage_imagenet"),
    (ROOT / "data" / "raw" / "ai" / "midjourney_genimage", 1, "midjourney", "genimage_midjourney"),
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-class", type=int, default=400,
                    help="images moved to the holdout per class")
    ap.add_argument("--holdout-dir", default=str(ROOT / "data" / "holdout_native"))
    ap.add_argument("--out-csv", default=str(ROOT / "data" / "csv" / "test_native.csv"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    holdout_root = Path(args.holdout_dir)
    rows: list[dict] = []

    for src_dir, label, generator, source in SOURCES:
        dest_dir = holdout_root / source
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Anything already moved counts toward the target, so re-running is safe
        # and does not keep eating into the training pool.
        already = sorted(p for p in dest_dir.iterdir()
                         if p.suffix.lower() in IMAGE_EXTS)
        need = args.per_class - len(already)

        moved = list(already)
        if need > 0:
            if not src_dir.is_dir():
                raise SystemExit(
                    f"{src_dir} not found. Fetch the data first:\n"
                    f"  python data/fetch_genimage.py --generator Midjourney"
                )
            pool = sorted(p for p in src_dir.iterdir()
                          if p.suffix.lower() in IMAGE_EXTS)
            if len(pool) < need:
                raise SystemExit(
                    f"{src_dir} has {len(pool)} images but {need} more are "
                    f"needed for the holdout. Fetch more, or lower --per-class."
                )
            rng.shuffle(pool)
            for p in pool[:need]:
                target = dest_dir / p.name
                shutil.move(str(p), str(target))
                moved.append(target)

        remaining = len([p for p in src_dir.iterdir()
                         if p.suffix.lower() in IMAGE_EXTS]) if src_dir.is_dir() else 0
        print(f"{source:22} holdout {len(moved):5}   left for training {remaining:5}")

        for p in moved:
            rows.append({
                "path": str(p.relative_to(ROOT)),
                "label": label,
                "generator": generator,
                "source": source,
                "content_class": "unknown",
                "content_group": -1,
            })

    rng.shuffle(rows)
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=COLUMNS).to_csv(out, index=False)

    ai = sum(r["label"] for r in rows)
    print(f"\nwrote {len(rows)} rows ({ai} AI / {len(rows) - ai} real) -> {out}")
    print("\nThese files now live outside data/raw/, so prepare.py cannot index")
    print("them into train or val. Rebuild the other splits next:")
    print("  python data/prepare.py --holdout-generators progan")


if __name__ == "__main__":
    main()
