"""Evaluate the detector and print the metrics the report needs.

Reported per split:

- **ROC-AUC** -- threshold-free ranking quality. Reported for the overall test
  split and separately for the unseen split, because the gap between them is the
  number that actually matters: a detector that only works on the generator it
  trained against is the failure this project exists to avoid.
- **Macro-F1** -- unweighted mean of the per-class F1 scores.
- **Confusion matrix** -- at the calibrated operating threshold.
- **Accuracy and FPR at the stated threshold** -- FPR is the expensive error
  here, since wrongly flagging a real photograph is worse than missing a
  generated one.

Scoring runs through ``src/inference.py``, the same path the CLI and the API
use, so these numbers are what a user would actually get -- calibration
included -- rather than a separately-implemented evaluation forward pass.

Usage:
    python model/evaluate.py
    python model/evaluate.py --splits test val_unseen_content --limit 2000
    python model/evaluate.py --out report/metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (confusion_matrix, f1_score, precision_recall_fscore_support,
                             roc_auc_score)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from inference import Predictor  # noqa: E402

DEFAULT_SPLITS = ["test", "val_unseen_generator", "val_unseen_content", "val"]


def score_split(pred: Predictor, df: pd.DataFrame, batch_size: int,
                label: str) -> tuple[np.ndarray, np.ndarray]:
    """Calibrated probabilities and labels for a split dataframe."""
    probs, labels = [], []
    t0 = time.time()
    for i in range(0, len(df), batch_size):
        chunk = df.iloc[i:i + batch_size]
        images, keep = [], []
        for _, row in chunk.iterrows():
            p = Path(row["path"])
            p = p if p.is_absolute() else ROOT / p
            try:
                im = Image.open(p)
                im.load()
                images.append(im.convert("RGB"))
                keep.append(float(row["label"]))
            except Exception:
                continue
        if not images:
            continue
        probs.append(pred.probabilities(images, batch_size=batch_size))
        labels.extend(keep)
        for im in images:
            im.close()
        done = min(i + batch_size, len(df))
        if done % (batch_size * 20) == 0:
            rate = done / max(time.time() - t0, 1e-6)
            eta = (len(df) - done) / max(rate, 1e-6)
            print(f"    {label}: {done}/{len(df)}  {rate:.1f} img/s  "
                  f"eta {eta / 60:.1f} min", flush=True)
    if not probs:
        return np.zeros(0), np.zeros(0)
    return np.concatenate(probs), np.array(labels, dtype=np.float32)


def metrics_for(probs: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    """Every metric the report quotes, at one stated threshold."""
    n = len(labels)
    if n == 0:
        return {"n": 0, "note": "split is empty"}
    out: dict = {"n": int(n), "ai_frac": float(labels.mean()),
                 "threshold": float(threshold)}
    if len(np.unique(labels)) < 2:
        out["note"] = "only one class present; AUC and F1 are undefined"
        return out

    out["roc_auc"] = float(roc_auc_score(labels, probs))
    pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, pred, labels=[0, 1], zero_division=0)
    out.update({
        "accuracy": float((tp + tn) / n),
        "macro_f1": float(f1_score(labels, pred, average="macro", zero_division=0)),
        "fpr": float(fp / max(tn + fp, 1)),
        "tpr_recall": float(tp / max(tp + fn, 1)),
        "precision_ai": float(prec[1]),
        "recall_ai": float(rec[1]),
        "f1_real": float(f1[0]),
        "f1_ai": float(f1[1]),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "mean_prob_real": float(probs[labels == 0].mean()),
        "mean_prob_ai": float(probs[labels == 1].mean()),
    })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data" / "csv"))
    ap.add_argument("--splits", nargs="*", default=DEFAULT_SPLITS)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap rows per split (the CSVs are shuffled, so this is "
                         "a random balanced sample)")
    ap.add_argument("--out", default=str(ROOT / "report" / "metrics.json"))
    args = ap.parse_args()

    data_dir = Path(args.data)
    pred = Predictor(checkpoint=args.checkpoint, calibration=args.calibration,
                     device=args.device)
    info = pred.info()
    print("=" * 78)
    print(f"model      : {info['backbone']}  ({info['weights']})")
    print(f"device     : {info['device']}")
    print(f"calibration: {'T=%.4f  threshold=%.4f' % (info['temperature'], info['threshold'])}"
          if info["calibrated"] else
          "calibration: NONE -- probabilities are uncalibrated raw scores")
    print("=" * 78)

    results: dict[str, dict] = {}
    for name in args.splits:
        csv_path = data_dir / f"{name}.csv"
        if not csv_path.is_file():
            print(f"\n{name}: no {csv_path.name}, skipping")
            continue
        df = pd.read_csv(csv_path)
        if len(df) == 0:
            print(f"\n{name}: EMPTY split -- reported as N/A")
            results[name] = {"n": 0, "note": (
                "split is empty. CIFAKE ships a single generator (sd14), so there "
                "is no second generator to hold out. Not a failure -- there is "
                "genuinely nothing to measure until another generator is added.")}
            continue
        if args.limit and args.limit < len(df):
            df = df.iloc[:args.limit].reset_index(drop=True)
        print(f"\n{name}: {len(df)} images (AI frac {df['label'].mean():.3f})")
        probs, labels = score_split(pred, df, args.batch_size, name)
        m = metrics_for(probs, labels, pred.threshold)
        results[name] = m
        if m.get("n", 0) == 0 or "roc_auc" not in m:
            print(f"  {m.get('note', 'no metrics')}")
            continue
        cm = m["confusion_matrix"]
        print(f"  ROC-AUC          {m['roc_auc']:.4f}")
        print(f"  accuracy         {m['accuracy']:.4f}   (at threshold "
              f"{m['threshold']:.4f})")
        print(f"  macro-F1         {m['macro_f1']:.4f}   "
              f"(real {m['f1_real']:.4f} / AI {m['f1_ai']:.4f})")
        print(f"  FPR              {m['fpr']:.4f}   <- real photos wrongly flagged")
        print(f"  recall (AI)      {m['tpr_recall']:.4f}")
        print(f"  confusion        tn {cm['tn']}  fp {cm['fp']}  "
              f"fn {cm['fn']}  tp {cm['tp']}")
        print(f"  mean P(AI)       real {m['mean_prob_real']:.3f} | "
              f"AI {m['mean_prob_ai']:.3f}")

    # The headline comparison: in-distribution against generalisation.
    print("\n" + "=" * 78)
    print("GENERALISATION GAP")
    print("=" * 78)
    base = results.get("test", {}).get("roc_auc")
    if base is None:
        print("  no test AUC to compare against")
    else:
        print(f"  test (in-distribution)      ROC-AUC {base:.4f}")
        for name in ("val_unseen_generator", "val_unseen_content"):
            a = results.get(name, {}).get("roc_auc")
            if a is None:
                print(f"  {name:27} N/A")
            else:
                print(f"  {name:27} ROC-AUC {a:.4f}   "
                      f"gap {base - a:+.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": info, "splits": results,
               "limit_per_split": args.limit or None}
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nmetrics -> {out}")


if __name__ == "__main__":
    main()
