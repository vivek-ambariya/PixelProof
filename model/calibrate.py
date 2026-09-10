"""Temperature scaling and operating-threshold selection on the validation set.

A trained network's raw sigmoid is not a probability -- BCE-trained models are
usually over-confident, so 0.9 does not mean "nine times out of ten". PixelProof
shows a number to a person and calls it a likelihood, so that number has to be
calibrated.

Two things are fitted here, both on the **validation** split only (never on test):

1. **Temperature** ``T``, minimising NLL of ``sigmoid(logit / T)``. A single
   scalar, so it cannot change the ranking of predictions and therefore cannot
   change ROC-AUC -- it only rescales confidence. ``src/inference.py`` must apply
   it, or the probability a user sees is a bare sigmoid.

2. **An operating threshold** at **FPR <= 5%**. Wrongly flagging a real
   photograph as AI-generated is the expensive error for this tool, so the
   threshold is chosen by capping that error rather than by maximising accuracy.
   The report needs accuracy and FPR quoted at a stated threshold; this is it.

Writes ``<backbone>_calibration.json`` next to the checkpoint, holding the
temperature, the threshold, and the metrics at that threshold.

Usage:
    python model/calibrate.py --checkpoint model/checkpoints/clip_vit_b16_best.pth
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import PixelProofNet, pick_device
from train import ImageRows, build_eval_transform

ROOT = Path(__file__).resolve().parent.parent

TARGET_FPR = 0.05


def rel_to_root(p: Path) -> str:
    """Path relative to the project root when it is inside it, else absolute.

    `Path.relative_to` raises for anything outside the root, and being absolute
    is not the same as being under ROOT -- a checkpoint kept in /tmp is both.
    """
    p = Path(p).resolve()
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


@torch.no_grad()
def collect_logits(net: PixelProofNet, df: pd.DataFrame, device: torch.device,
                   batch_size: int, num_workers: int) -> tuple[np.ndarray, np.ndarray]:
    """Raw (uncalibrated) logits and labels for a split, clean images only."""
    tf = build_eval_transform(net.img_size, net.mean, net.std)
    loader = DataLoader(ImageRows(df, tf), batch_size=batch_size, shuffle=False,
                        num_workers=num_workers)
    net.eval()
    logits, labels = [], []
    for xb, yb in loader:
        logits.append(net(xb.to(device)).float().cpu().numpy())
        labels.append(yb.numpy())
    return np.concatenate(logits), np.concatenate(labels)


def fit_temperature(logits: np.ndarray, labels: np.ndarray,
                    max_iter: int = 200) -> tuple[float, float, float]:
    """Fit T by minimising NLL with LBFGS. Returns (T, nll_before, nll_after)."""
    z = torch.tensor(logits, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    crit = nn.BCEWithLogitsLoss()
    nll_before = crit(z, y).item()

    # Optimise log_T so T stays strictly positive.
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = crit(z / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    T = float(log_t.exp().item())
    nll_after = crit(z / T, y).item()
    return T, nll_before, nll_after


def pick_threshold(probs: np.ndarray, labels: np.ndarray,
                   target_fpr: float = TARGET_FPR) -> dict:
    """Lowest threshold whose FPR is still <= target_fpr.

    A lower threshold flags more images, so among the thresholds meeting the FPR
    cap we want the most sensitive one -- that maximises recall subject to the
    constraint. roc_curve returns thresholds in decreasing order alongside
    non-decreasing FPR, so the last index satisfying the cap is the answer.
    """
    fpr, tpr, thr = roc_curve(labels, probs)
    ok = np.where(fpr <= target_fpr)[0]
    if len(ok) == 0:
        return {"threshold": 0.5, "note": "no threshold met the FPR cap; using 0.5"}
    i = ok[-1]
    t = float(thr[i])
    # roc_curve prepends an infinite threshold; guard against it.
    if not np.isfinite(t):
        t = 1.0
    return {"threshold": t, "fpr_at_threshold": float(fpr[i]),
            "tpr_at_threshold": float(tpr[i])}


def metrics_at(probs: np.ndarray, labels: np.ndarray, t: float) -> dict:
    pred = (probs >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(t),
        "accuracy": float((tp + tn) / max(len(labels), 1)),
        "fpr": float(fp / max(tn + fp, 1)),
        "tpr_recall": float(tp / max(tp + fn, 1)),
        "precision": float(tp / max(tp + fp, 1)) if (tp + fp) else 0.0,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def ece(probs: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error -- the headline number temperature scaling moves."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (probs > lo) & (probs <= hi)
        if m.sum() == 0:
            continue
        total += (m.sum() / len(probs)) * abs(labels[m].mean() - probs[m].mean())
    return float(total)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", default=str(ROOT / "data" / "csv"))
    ap.add_argument("--split", default="val",
                    help="split to fit on; must NOT be test")
    ap.add_argument("--target-fpr", type=float, default=TARGET_FPR)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.split == "test":
        raise SystemExit("refusing to calibrate on test; that leaks the test set")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise SystemExit(f"checkpoint not found: {ckpt_path}\nTrain first.")
    csv_path = Path(args.data) / f"{args.split}.csv"
    if not csv_path.is_file():
        raise SystemExit(f"{csv_path} not found; run data/prepare.py")

    device = pick_device(args.device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    backbone = ckpt.get("backbone_name", "clip_vit_b16")
    net = PixelProofNet(backbone, pretrained=True).to(device)
    net.load_state_dict(ckpt["model"])
    print(f"loaded {ckpt_path.name}  backbone={backbone}  "
          f"epoch={ckpt.get('epoch')}  "
          f"{ckpt.get('monitor_split')} AUC={ckpt.get('monitor_auc')}")

    df = pd.read_csv(csv_path)
    print(f"fitting on {args.split}: {len(df)} images "
          f"(AI frac {df['label'].mean():.3f})")
    logits, labels = collect_logits(net, df, device, args.batch_size, args.num_workers)

    T, nll_before, nll_after = fit_temperature(logits, labels)
    p_raw = 1.0 / (1.0 + np.exp(-logits))
    p_cal = 1.0 / (1.0 + np.exp(-logits / T))

    auc_raw = float(roc_auc_score(labels, p_raw))
    auc_cal = float(roc_auc_score(labels, p_cal))

    print("\n--- temperature scaling ---")
    print(f"  T                 : {T:.4f}   "
          f"({'over-confident before' if T > 1 else 'under-confident before'})")
    print(f"  NLL   before/after: {nll_before:.4f} / {nll_after:.4f}")
    print(f"  ECE   before/after: {ece(p_raw, labels):.4f} / {ece(p_cal, labels):.4f}")
    print(f"  AUC   before/after: {auc_raw:.4f} / {auc_cal:.4f}  "
          f"(a scalar cannot change ranking; these must match)")

    pick = pick_threshold(p_cal, labels, args.target_fpr)
    t = pick["threshold"]
    at_t = metrics_at(p_cal, labels, t)
    at_half = metrics_at(p_cal, labels, 0.5)

    print(f"\n--- operating threshold at FPR <= {args.target_fpr:.0%} ---")
    print(f"  threshold         : {t:.4f}")
    for nm, m in (("at chosen threshold", at_t), ("at 0.5 for reference", at_half)):
        cm = m["confusion_matrix"]
        print(f"  {nm:22} acc {m['accuracy']:.4f}  FPR {m['fpr']:.4f}  "
              f"recall {m['tpr_recall']:.4f}  "
              f"[tn {cm['tn']} fp {cm['fp']} fn {cm['fn']} tp {cm['tp']}]")

    out = Path(args.out) if args.out else ckpt_path.parent / f"{backbone}_calibration.json"
    payload = {
        "checkpoint": rel_to_root(ckpt_path),
        "backbone": backbone,
        "fitted_on_split": args.split,
        "n_images": int(len(df)),
        "temperature": T,
        "target_fpr": args.target_fpr,
        "threshold": t,
        "nll_before": nll_before,
        "nll_after": nll_after,
        "ece_before": ece(p_raw, labels),
        "ece_after": ece(p_cal, labels),
        "auc": auc_cal,
        "metrics_at_threshold": at_t,
        "metrics_at_0.5": at_half,
        "note": ("probability = sigmoid(logit / temperature); verdict is "
                 "'Likely AI-generated' when probability >= threshold, "
                 "else 'Likely real'"),
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\ncalibration -> {out}")


if __name__ == "__main__":
    main()
