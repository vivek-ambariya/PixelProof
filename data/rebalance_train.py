"""Set the native-resolution share of train.csv explicitly, instead of by accident.

The problem
    CIFAKE contributes ~71,000 training rows; the native-resolution GenImage
    sample contributes ~2,800. Left alone, and with the usual --limit-train
    24000, native-resolution images end up around 4% of the training set. The
    loss is then ~96% dominated by 32x32 thumbnails -- the exact distribution
    whose shortcut ("sharp means generated") is what we are trying to unlearn.
    A handful of counterexamples buried under that majority is unlikely to move
    the decision boundary where it matters.

What this does
    Rewrites train.csv as an explicit mixture: a capped random sample of the
    CIFAKE rows plus the native-resolution rows repeated ``--oversample`` times,
    so the native share is a number chosen on purpose and recorded, rather than
    whatever falls out of the relative corpus sizes.

    Oversampling rather than discarding CIFAKE keeps the in-distribution
    performance the existing splits measure, so `test` and the two unseen splits
    stay comparable to the previously reported runs.

Class balance is preserved within each source, and the result is reshuffled, so
no head-slice of the CSV is single-source or single-class.

Usage:
    python data/rebalance_train.py                       # 20k CIFAKE + 2x native
    python data/rebalance_train.py --cifake 24000 --oversample 3
    python data/rebalance_train.py --report              # show the mix, write nothing
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# Sources that count as native resolution. Everything else is treated as the
# 32x32 CIFAKE corpus.
NATIVE_SOURCES = {"genimage_imagenet", "genimage_midjourney", "midjourney_genimage"}


def describe(df: pd.DataFrame, label: str) -> None:
    native = df["source"].isin(NATIVE_SOURCES)
    n = len(df)
    print(f"{label}")
    print(f"  rows            {n:,}")
    print(f"  native-res      {native.sum():,} ({100 * native.mean():.1f}%)")
    print(f"  AI fraction     {df['label'].mean():.3f}")
    if native.any():
        sub = df[native]
        print(f"    native AI frac  {sub['label'].mean():.3f}")
    if (~native).any():
        sub = df[~native]
        print(f"    cifake AI frac  {sub['label'].mean():.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=str(ROOT / "data" / "csv" / "train.csv"))
    ap.add_argument("--cifake", type=int, default=20000,
                    help="how many CIFAKE rows to keep (class-balanced)")
    ap.add_argument("--oversample", type=int, default=2,
                    help="how many times to repeat each native-resolution row")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report", action="store_true",
                    help="print the current mix and exit without writing")
    args = ap.parse_args()

    path = Path(args.csv)
    df = pd.read_csv(path)
    describe(df, f"before -- {path}")

    if args.report:
        return

    native = df[df["source"].isin(NATIVE_SOURCES)]
    cifake = df[~df["source"].isin(NATIVE_SOURCES)]

    if native.empty:
        raise SystemExit(
            "no native-resolution rows found in train.csv. Fetch and index them "
            "first:\n"
            "  python data/fetch_genimage.py --generator Midjourney\n"
            "  python data/normalise_native.py\n"
            "  python data/prepare.py --holdout-generators progan"
        )

    # Sample CIFAKE per class so the cap cannot skew the label balance.
    keep = []
    for lbl, grp in cifake.groupby("label"):
        take = min(len(grp), args.cifake // 2)
        keep.append(grp.sample(n=take, random_state=args.seed))
    cifake_s = pd.concat(keep)

    repeated = pd.concat([native] * args.oversample, ignore_index=True)
    out = pd.concat([cifake_s, repeated], ignore_index=True)
    out = out.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    print()
    describe(out, f"after  -- {path}")
    out.to_csv(path, index=False)
    print(f"\nwrote {len(out):,} rows -> {path}")
    print("\nNote: the native rows are repeated, so the feature cache will hold")
    print("duplicate vectors for them. That is the intended cost of the reweighting.")


if __name__ == "__main__":
    main()
