"""Export the trainable head from a frozen-backbone checkpoint, for a fresh clone.

``.gitignore`` excludes every ``.pth``, so a judge cloning the repo has no
checkpoint. For a *frozen* backbone that is not a problem: the backbone comes
from timm on first run, so only the trained head has to be version-controlled --
~770 KB instead of 344 MB. ``src/inference.py`` prefers a real checkpoint when
one exists and falls back to this head otherwise, and the two must produce
identical probabilities.

That invariant breaks silently whenever the model is retrained and this file is
not regenerated, which is exactly what a stale committed head looks like: a
fresh clone quietly scores images with an older model than the one the report
describes. Run this after every retrain of a frozen-backbone model.

Usage:
    python model/export_head.py
    python model/export_head.py --checkpoint model/checkpoints/clip_vit_b16_best.pth
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model"))

from model import BACKBONES  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint",
                    default=str(ROOT / "model" / "checkpoints" / "clip_vit_b16_best.pth"))
    ap.add_argument("--calibration", default=None,
                    help="defaults to <backbone>_calibration.json beside the checkpoint")
    ap.add_argument("--out-dir", default=str(ROOT / "model" / "weights"))
    args = ap.parse_args()

    ck = Path(args.checkpoint)
    if not ck.is_file():
        raise SystemExit(f"checkpoint not found: {ck}")

    blob = torch.load(ck, map_location="cpu", weights_only=False)
    backbone = blob.get("backbone_name", "clip_vit_b16")
    if not BACKBONES.get(backbone, {}).get("frozen", False):
        raise SystemExit(
            f"{backbone} is fine-tuned end to end, so its head alone cannot "
            f"reproduce it. Only a frozen backbone can be shipped this way."
        )

    head = {k.split("head.", 1)[1]: v
            for k, v in blob["model"].items() if k.startswith("head.")}
    if not head:
        raise SystemExit("no head.* tensors found in the checkpoint")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from safetensors.torch import save_file
    save_file(head, str(out_dir / "head.safetensors"))

    cal_src = Path(args.calibration) if args.calibration else \
        ck.with_name(f"{backbone}_calibration.json")
    if cal_src.is_file():
        shutil.copyfile(cal_src, out_dir / "calibration.json")
    else:
        print(f"WARNING  no calibration at {cal_src}; the committed head will "
              f"fall back to T=1.0 / threshold=0.5")

    (out_dir / "head_config.json").write_text(json.dumps({
        "backbone": backbone,
        "epoch": blob.get("epoch"),
        "monitor_split": blob.get("monitor_split"),
        "monitor_auc": blob.get("monitor_auc"),
        "note": "trainable head only; the frozen backbone comes from timm",
    }, indent=2) + "\n")

    n = sum(v.numel() for v in head.values())
    size = (out_dir / "head.safetensors").stat().st_size
    print(f"exported {len(head)} tensors, {n:,} params, {size/1e3:.0f} KB "
          f"-> {out_dir/'head.safetensors'}")
    print(f"from {ck.name} (epoch {blob.get('epoch')})")


if __name__ == "__main__":
    main()
