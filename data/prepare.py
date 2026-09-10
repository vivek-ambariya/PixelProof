"""Build the train/val/test CSVs for PixelProof.

Reads a raw image root and writes one CSV per split to ``data/csv/``. Two raw
layouts are understood:

1. **CIFAKE** — ``<raw>/cifake/{train,test}/{REAL,FAKE}/*.jpg``. The real half is
   CIFAR-10, the AI half is Stable Diffusion 1.4. CIFAKE was assembled by merging
   ten per-class folders into one, which left a Windows-style ``name (k).jpg``
   suffix; ``k`` therefore identifies the CIFAR-10 class. That was verified by
   per-group colour statistics on both halves independently (groups 1 and 9 are
   the only ones with mean blue above mean red -- airplane and ship -- and group 7
   is the most green-dominant -- frog). We record the raw group index as well as
   the name so a content holdout keys on group identity rather than on the name.

2. **Generic convention** — ``<raw>/real/<source>/**`` and ``<raw>/ai/<generator>/**``.
   Adding a new generator is a folder drop; no code change.

Splits written:

- ``train``                 -- what the model fits.
- ``val``                   -- in-distribution validation, also used by calibrate.py.
- ``val_unseen_generator``  -- generators named in ``--holdout-generators``, removed
  from train entirely. This is the split the spec asks for. CIFAKE ships a single
  generator (sd14), so with CIFAKE alone this split is EMPTY by construction and
  evaluate.py reports it as N/A rather than inventing a number. Drop a second
  generator into ``<raw>/ai/<name>/`` and it populates with no code change.
- ``val_unseen_content``    -- whole semantic classes held out (default horse, ship),
  removed from train entirely. Not a substitute for an unseen-generator test; it
  measures generalisation to unseen *content*, and is named for what it is.
- ``test``                  -- CIFAKE's own test directory, never touched by
  training or calibration.

Usage:
    python data/prepare.py                          # defaults
    python data/prepare.py --holdout-content horse,ship --val-frac 0.1
    python data/prepare.py --limit 400              # tiny CSVs for smoke tests
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

# Project root is the parent of the directory holding this file.
ROOT = Path(__file__).resolve().parent.parent

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# CIFAR-10 class order. Index i corresponds to CIFAKE filename group i+1.
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# Columns every CSV carries. `label` is 1 for AI-generated, 0 for real.
COLUMNS = ["path", "label", "generator", "source", "content_class", "content_group"]

_GROUP_RE = re.compile(r"\((\d+)\)\s*$")


def _content_group(stem: str) -> int:
    """Recover the merged-folder index from a CIFAKE filename stem.

    ``0042 (7)`` -> 7, and a bare ``0042`` -> 1 (the first folder's files kept
    their original names when the folders were merged).
    """
    m = _GROUP_RE.search(stem)
    return int(m.group(1)) if m else 1


def _rel(p: Path) -> str:
    """Path relative to the project root, so CSVs stay portable."""
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p.resolve())


def index_cifake(cifake_root: Path) -> list[dict]:
    """Index a CIFAKE tree. Returns rows tagged with their origin split."""
    rows: list[dict] = []
    for origin in ("train", "test"):
        for folder, label in (("REAL", 0), ("FAKE", 1)):
            d = cifake_root / origin / folder
            if not d.is_dir():
                continue
            for f in sorted(d.iterdir()):
                if f.suffix.lower() not in IMAGE_EXTS:
                    continue
                g = _content_group(f.stem)
                rows.append({
                    "path": _rel(f),
                    "label": label,
                    # CIFAKE's AI half is entirely Stable Diffusion 1.4.
                    "generator": "sd14" if label else "none",
                    "source": "cifake" if label else "cifar10",
                    "content_class": CIFAR10_CLASSES[g - 1] if 1 <= g <= 10 else "unknown",
                    "content_group": g,
                    "_origin": origin,
                })
    return rows


def index_generic(raw_root: Path) -> list[dict]:
    """Index the generic convention: raw/real/<source>/** and raw/ai/<generator>/**."""
    rows: list[dict] = []
    for top, label in (("real", 0), ("ai", 1)):
        base = raw_root / top
        if not base.is_dir():
            continue
        for bucket in sorted(p for p in base.iterdir() if p.is_dir()):
            for f in sorted(bucket.rglob("*")):
                if not f.is_file() or f.suffix.lower() not in IMAGE_EXTS:
                    continue
                rows.append({
                    "path": _rel(f),
                    "label": label,
                    "generator": bucket.name if label else "none",
                    "source": bucket.name,
                    "content_class": "unknown",
                    "content_group": 0,
                    "_origin": "train",
                })
    return rows


def _stratified_split(rows: list[dict], frac: float, rng: random.Random):
    """Split rows into (majority, minority) with `frac` going to the minority.

    Strata are (label, content_group) so both class balance and content mix are
    preserved on each side.
    """
    strata: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        strata[(r["label"], r["content_group"])].append(r)
    big, small = [], []
    for key in sorted(strata):
        group = strata[key][:]
        rng.shuffle(group)
        n = int(round(len(group) * frac))
        small.extend(group[:n])
        big.extend(group[n:])
    return big, small


def build_splits(rows: list[dict], args, rng: random.Random) -> dict[str, list[dict]]:
    holdout_gens = {g.strip() for g in args.holdout_generators.split(",") if g.strip()}
    holdout_content = {c.strip().lower() for c in args.holdout_content.split(",") if c.strip()}
    # Accept either class names or raw group indices for the content holdout.
    holdout_groups = set()
    for c in holdout_content:
        if c.isdigit():
            holdout_groups.add(int(c))
        elif c in CIFAR10_CLASSES:
            holdout_groups.add(CIFAR10_CLASSES.index(c) + 1)
        else:
            raise SystemExit(f"--holdout-content: unknown class {c!r}. "
                             f"Choose from {CIFAR10_CLASSES} or a group index 1-10.")

    splits: dict[str, list[dict]] = {}

    # test is CIFAKE's own test directory, kept entirely out of training.
    splits["test"] = [r for r in rows if r["_origin"] == "test"]
    pool = [r for r in rows if r["_origin"] == "train"]

    # 1. Unseen-generator split: the held-out generators' AI images, plus a
    #    matched number of real images so the split is scoreable on its own.
    unseen_gen = [r for r in pool if r["label"] == 1 and r["generator"] in holdout_gens]
    if unseen_gen:
        pool = [r for r in pool if not (r["label"] == 1 and r["generator"] in holdout_gens)]
        reals = [r for r in pool if r["label"] == 0]
        rng.shuffle(reals)
        matched = reals[:len(unseen_gen)]
        matched_ids = {id(r) for r in matched}
        pool = [r for r in pool if id(r) not in matched_ids]
        unseen_gen = unseen_gen + matched
    splits["val_unseen_generator"] = unseen_gen

    # 2. Unseen-content split: whole semantic classes, both labels, out of train.
    unseen_content = [r for r in pool if r["content_group"] in holdout_groups]
    if unseen_content:
        pool = [r for r in pool if r["content_group"] not in holdout_groups]
    splits["val_unseen_content"] = unseen_content

    # 3. Whatever remains becomes train/val.
    train, val = _stratified_split(pool, args.val_frac, rng)
    splits["train"] = train
    splits["val"] = val
    return splits


def _summarise(name: str, rows: list[dict]) -> dict:
    n = len(rows)
    ai = sum(r["label"] for r in rows)
    return {
        "split": name,
        "n": n,
        "ai": ai,
        "real": n - ai,
        "ai_frac": round(ai / n, 4) if n else None,
        "generators": sorted({r["generator"] for r in rows if r["label"] == 1}),
        "content_classes": sorted({r["content_class"] for r in rows}),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-root", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "csv"))
    ap.add_argument("--val-frac", type=float, default=0.1,
                    help="fraction of the remaining pool held out as in-distribution val")
    ap.add_argument("--holdout-generators", default="",
                    help="comma-separated generator names to hold out entirely. "
                         "CIFAKE has only 'sd14'; holding it out would leave no AI "
                         "training data, so the default is empty.")
    ap.add_argument("--holdout-content", default="horse,ship",
                    help="comma-separated CIFAR-10 class names (or group indices) "
                         "to hold out entirely as an unseen-content split")
    ap.add_argument("--limit", type=int, default=0,
                    help="if >0, subsample each split to at most this many rows "
                         "(for smoke tests; class balance preserved)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    raw_root = Path(args.raw_root)
    if not raw_root.is_dir():
        raise SystemExit(f"raw root not found: {raw_root}")

    rows: list[dict] = []
    cifake_root = raw_root / "cifake"
    if cifake_root.is_dir():
        rows += index_cifake(cifake_root)
    rows += index_generic(raw_root)
    if not rows:
        raise SystemExit(
            f"no images found under {raw_root}. Expected either "
            f"{raw_root}/cifake/{{train,test}}/{{REAL,FAKE}}/ or "
            f"{raw_root}/{{real,ai}}/<name>/."
        )

    print(f"indexed {len(rows)} images from {raw_root}")
    print(f"  by label     : {dict(Counter('ai' if r['label'] else 'real' for r in rows))}")
    print(f"  by generator : {dict(Counter(r['generator'] for r in rows if r['label']))}")

    splits = build_splits(rows, args, rng)

    if args.limit:
        for name, rs in splits.items():
            if len(rs) > args.limit:
                big, small = _stratified_split(rs, args.limit / len(rs), rng)
                splits[name] = small

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "seed": args.seed,
        "val_frac": args.val_frac,
        "holdout_generators": args.holdout_generators,
        "holdout_content": args.holdout_content,
        "limit": args.limit,
        "label_meaning": {"1": "AI-generated", "0": "real"},
        "splits": [],
    }

    print()
    hdr = f"{'split':24} {'n':>7} {'AI':>7} {'real':>7} {'AI frac':>8}  generators"
    print(hdr); print("-" * len(hdr))
    for name in ("train", "val", "val_unseen_generator", "val_unseen_content", "test"):
        rs = splits.get(name, [])
        # Shuffle before writing. Indexing and splitting both produce rows grouped
        # by label, so an unshuffled CSV makes any head-slice single-class -- which
        # would quietly hand `--limit` and dataset.make_loader(limit=...) a subset
        # with only one label in it. Seeded, so the CSVs stay reproducible.
        rng.shuffle(rs)
        df = pd.DataFrame(rs, columns=COLUMNS) if rs else pd.DataFrame(columns=COLUMNS)
        df.to_csv(out_dir / f"{name}.csv", index=False)
        s = _summarise(name, rs)
        manifest["splits"].append(s)
        frac = f"{s['ai_frac']:.3f}" if s["ai_frac"] is not None else "  --  "
        gens = ",".join(s["generators"]) or "(none)"
        print(f"{name:24} {s['n']:>7} {s['ai']:>7} {s['real']:>7} {frac:>8}  {gens}")

    if not splits.get("val_unseen_generator"):
        print("\nNOTE  val_unseen_generator is empty: the only generator present is "
              "'sd14',\n      so there is no second generator to hold out. evaluate.py "
              "will report it\n      as N/A. Drop another generator into "
              "data/raw/ai/<name>/ to populate it.")

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nwrote 5 CSVs + manifest.json to {out_dir}")

    # Sanity: no path may appear in more than one split.
    seen: dict[str, str] = {}
    for name, rs in splits.items():
        for r in rs:
            if r["path"] in seen:
                raise SystemExit(f"LEAK: {r['path']} in both {seen[r['path']]} and {name}")
            seen[r["path"]] = name
    print(f"leak check passed: {len(seen)} unique paths across all splits")


if __name__ == "__main__":
    main()
