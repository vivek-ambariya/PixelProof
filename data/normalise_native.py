"""Strip resolution, aspect ratio and format cues from the fetched GenImage halves.

The problem this exists to prevent
    GenImage's two halves are trivially separable by geometry alone. Measured on
    the fetched sample:

        real (ImageNet)     median 500x375, range 80x60 .. 3072x2304, mostly 4:3
        fake (Midjourney)   1024x1024 for EVERY image, always 1:1

    A detector can score "is this a 1024x1024 square?" and be almost perfectly
    right, having learned nothing about generation. That is the same failure
    that made the CIFAKE-trained models call real photographs fake -- a cue
    correlated with the label inside the dataset and meaningless outside it --
    just pointing the other way. Training on the raw halves would swap one
    shortcut for another and the resulting accuracy would not survive contact
    with a real image.

What this does
    Both halves go through one identical pipeline: centre-crop to square,
    downscale to a single edge length, re-encode at one quality. Afterwards the
    two classes are indistinguishable by size, shape or container, so the only
    thing left to separate them is content and generator artefacts.

Why 256, and why images below it are dropped rather than upscaled
    Everything must be *downscaled*, never upscaled. Upscaling a small image
    reintroduces exactly the blur signature that taught the original models
    "blurry means real", so a small real photo stretched to 256 would rebuild
    the confound this is meant to remove. Images whose short side is under the
    target are therefore discarded (~2.6% of the ImageNet half). 256 also sits
    just above the 224 the backbones consume, so the training distribution
    lands where inference actually operates.

Idempotent: an image already at the target size and format is left alone, so
re-running after fetching more images only processes the new ones.

Usage:
    python data/normalise_native.py                # 256px, both halves
    python data/normalise_native.py --size 320 --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent

TARGETS = [
    ROOT / "data" / "raw" / "real" / "genimage_imagenet",
    ROOT / "data" / "raw" / "ai" / "midjourney_genimage",
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def centre_square(im: Image.Image) -> Image.Image:
    w, h = im.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return im.crop((left, top, left + side, top + side))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, default=256,
                    help="edge length both halves are normalised to")
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--dirs", nargs="*", default=None,
                    help="override the directories to normalise")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change without writing anything")
    args = ap.parse_args()

    dirs = [Path(d) for d in args.dirs] if args.dirs else TARGETS

    grand_kept = grand_dropped = grand_skipped = 0
    for d in dirs:
        if not d.is_dir():
            print(f"{d}: not found, skipping")
            continue
        files = sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        kept = dropped = skipped = 0

        for p in files:
            try:
                with Image.open(p) as im:
                    im.load()
                    w, h = im.size
                    # Already normalised -- leave it and its bytes alone.
                    if (w, h) == (args.size, args.size) and p.suffix.lower() == ".jpg":
                        skipped += 1
                        continue
                    if min(w, h) < args.size:
                        # Upscaling would rebuild the blur confound; drop it.
                        dropped += 1
                        if not args.dry_run:
                            p.unlink()
                        continue
                    out = centre_square(im.convert("RGB")).resize(
                        (args.size, args.size), Image.Resampling.LANCZOS)
                if not args.dry_run:
                    out.save(p.with_suffix(".jpg"), format="JPEG",
                             quality=args.quality)
                    if p.suffix.lower() != ".jpg":
                        p.unlink()
                kept += 1
            except Exception as e:
                print(f"  {p.name}: {type(e).__name__}, dropping")
                dropped += 1
                if not args.dry_run:
                    p.unlink(missing_ok=True)

        total = kept + dropped + skipped
        print(f"{d.name:22} {total:5} in   {kept:5} normalised   "
              f"{skipped:5} already ok   {dropped:5} dropped (short side < {args.size})")
        grand_kept += kept
        grand_dropped += dropped
        grand_skipped += skipped

    print(f"\n{'DRY RUN -- nothing written' if args.dry_run else 'done'}: "
          f"{grand_kept} normalised, {grand_skipped} untouched, "
          f"{grand_dropped} dropped")

    if not args.dry_run:
        print("\nBoth halves are now identical in size, shape and container, so")
        print("geometry and format can no longer stand in for the label.")
        print("Next:")
        print("  python data/make_native_testset.py")
        print("  python data/prepare.py --holdout-generators progan")


if __name__ == "__main__":
    main()
