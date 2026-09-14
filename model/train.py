"""Train the PixelProof detector.

Two training paths, picked by whether the backbone is frozen:

**Cached path** (``--backbone clip_vit_b16``, the default). The backbone never
updates, so its features are computed once and reused. Augmentation must happen
*before* a frozen backbone, which conflicts with caching -- so instead of
augmenting on the fly we precompute features for K fixed, deterministic variants
of every training image and sample one variant per image per epoch. K=4:

    0  clean                 resize only
    1  jpeg_q40              JPEG re-encode at quality 40
    2  downscale_upscale     halve the resolution and scale back up
    3  blur                  mild Gaussian blur

These are the degradation families the model is scored on, so the same variants
that provide augmentation also buy robustness to them. The cache is keyed by a
hash of the backbone, input size, normalisation and the full variant spec, so
changing any of them produces a different key and a stale cache can never be
silently reused.

One documented cost of caching: horizontal flip cannot be applied, because
features of a flipped image are not recoverable from the features of the
original. It would double the cache to add it. The end-to-end path does use it.

**End-to-end path** (``--backbone resnet50``). Weights move every step, so
features cannot be cached and augmentation is applied on the fly with the random
ranges in ``build_train_transform``.

Early stopping watches AUC on the *unseen* split, not overall val AUC -- a
detector that only works on the generator it trained against is the failure mode
this project exists to avoid. Note that CIFAKE ships a single generator, so
``val_unseen_generator`` is empty; the script falls back to
``val_unseen_content`` and says so loudly rather than early-stopping on nothing.

Usage:
    python model/train.py --backbone clip_vit_b16 --epochs 20 --batch-size 64
    python model/train.py --overfit 200          # wiring check, must reach ~0 loss
    python model/train.py --backbone resnet50 --epochs 10
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from probe_training import ProbeTrainer, create_indexed_dataloader
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import DEFAULT_BACKBONE, PixelProofNet, count_params, describe, pick_device

ROOT = Path(__file__).resolve().parent.parent

# --- augmentation ------------------------------------------------------------
# The K deterministic variants used to build the feature cache. p=1.0 everywhere
# and every range collapsed to a single value, so each variant is reproducible.
VARIANT_SPEC: list[dict] = [
    {"name": "clean", "ops": []},
    {"name": "jpeg_q40", "ops": [{"t": "ImageCompression", "quality_range": [40, 40]}]},
    {"name": "downscale_upscale", "ops": [{"t": "Downscale", "scale_range": [0.5, 0.5]}]},
    {"name": "blur", "ops": [{"t": "GaussianBlur", "blur_limit": [3, 3],
                              "sigma_limit": [1.0, 1.0]}]},
]
K_VARIANTS = len(VARIANT_SPEC)


def _op(spec: dict) -> A.BasicTransform:
    t = spec["t"]
    if t == "ImageCompression":
        return A.ImageCompression(compression_type="jpeg",
                                  quality_range=tuple(spec["quality_range"]), p=1.0)
    if t == "Downscale":
        return A.Downscale(scale_range=tuple(spec["scale_range"]), p=1.0)
    if t == "GaussianBlur":
        return A.GaussianBlur(blur_limit=tuple(spec["blur_limit"]),
                              sigma_limit=tuple(spec["sigma_limit"]), p=1.0)
    raise ValueError(f"unknown op {t!r}")


def build_variant_transform(idx: int, img_size: int, mean, std) -> A.Compose:
    """Deterministic transform for cache variant `idx`."""
    ops = [_op(o) for o in VARIANT_SPEC[idx]["ops"]]
    return A.Compose(ops + [
        A.Resize(img_size, img_size, interpolation=cv2.INTER_CUBIC),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
        ToTensorV2(),
    ])


def build_train_transform(img_size: int, mean, std) -> A.Compose:
    """Random augmentation for the end-to-end path.

    Horizontal flip only -- a vertical flip would invert the directional
    lighting and shading cues the third signal family depends on.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ImageCompression(compression_type="jpeg", quality_range=(30, 95), p=0.5),
        A.OneOf([
            A.Downscale(scale_range=(0.35, 0.75), p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0.3, 1.5), p=1.0),
        ], p=0.4),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        A.Resize(img_size, img_size, interpolation=cv2.INTER_CUBIC),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
        ToTensorV2(),
    ])


def build_eval_transform(img_size: int, mean, std) -> A.Compose:
    return A.Compose([
        A.Resize(img_size, img_size, interpolation=cv2.INTER_CUBIC),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
        ToTensorV2(),
    ])


# --- datasets ----------------------------------------------------------------
class ImageRows(Dataset):
    """Yields (tensor, label) for one dataframe under a fixed transform."""

    def __init__(self, df: pd.DataFrame, transform: A.Compose) -> None:
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        p = Path(row["path"])
        p = p if p.is_absolute() else ROOT / p
        with Image.open(p) as im:
            arr = np.asarray(im.convert("RGB"))
        return self.transform(image=arr)["image"], np.float32(row["label"])


class CachedFeatures(Dataset):
    """Cached features with per-epoch random variant sampling.

    ``feats`` is (N, K, D). Training samples one of the K variants per item;
    evaluation always uses variant 0 (clean).
    """

    def __init__(self, feats: np.ndarray, labels: np.ndarray, train: bool,
                 seed: int = 0) -> None:
        self.feats = feats
        self.labels = labels
        self.train = train
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        k = int(self.rng.integers(self.feats.shape[1])) if self.train else 0
        return (torch.from_numpy(self.feats[i, k].astype(np.float32)),
                np.float32(self.labels[i]))


# --- feature cache -----------------------------------------------------------
def cache_key(backbone: str, img_size: int, mean, std, n_variants: int) -> str:
    payload = json.dumps({
        "backbone": backbone,
        "img_size": img_size,
        "mean": [round(float(v), 6) for v in mean],
        "std": [round(float(v), 6) for v in std],
        "variants": VARIANT_SPEC[:n_variants],
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


@torch.no_grad()
def compute_features(net: PixelProofNet, df: pd.DataFrame, n_variants: int,
                     device: torch.device, batch_size: int, num_workers: int,
                     label: str, half: bool = True) -> np.ndarray:
    """Return (N, n_variants, D) float16 features.

    ``half`` runs the backbone under autocast fp16. Measured on an M2 this lifts
    CLIP ViT-B/16 from 22 to 30 img/s, and since the cache is stored as float16
    anyway it costs no precision that survives storage. Autocast rather than
    ``.half()`` on the weights, so the model is left untouched for later use.
    """
    out = np.zeros((len(df), n_variants, net.feat_dim), dtype=np.float16)
    net.eval()
    use_ac = half and device.type in ("mps", "cuda")
    for k in range(n_variants):
        tf = build_variant_transform(k, net.img_size, net.mean, net.std)
        loader = DataLoader(ImageRows(df, tf), batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, drop_last=False)
        done, t0 = 0, time.time()
        for x, _ in loader:
            if use_ac:
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    f = net.forward_features(x.to(device))
            else:
                f = net.forward_features(x.to(device))
            out[done:done + len(f), k] = f.detach().float().cpu().numpy().astype(np.float16)
            done += len(f)
            if done % (batch_size * 40) == 0:
                rate = done / max(time.time() - t0, 1e-6)
                print(f"    {label} variant {k} ({VARIANT_SPEC[k]['name']}): "
                      f"{done}/{len(df)}  {rate:.0f} img/s", flush=True)
        print(f"    {label} variant {k} ({VARIANT_SPEC[k]['name']}): done in "
              f"{time.time() - t0:.1f}s", flush=True)
    return out


def get_features(net: PixelProofNet, df: pd.DataFrame, split: str, n_variants: int,
                 cache_dir: Path, device: torch.device, batch_size: int,
                 num_workers: int, use_cache: bool = True,
                 half: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Load features from cache, or compute and store them."""
    key = cache_key(net.backbone_name, net.img_size, net.mean, net.std, n_variants)
    cache_dir.mkdir(parents=True, exist_ok=True)
    f = cache_dir / f"{split}_{net.backbone_name}_k{n_variants}_{key}_n{len(df)}.npz"
    labels = df["label"].to_numpy(dtype=np.float32)
    if use_cache and f.is_file():
        z = np.load(f, allow_pickle=False)
        if z["feats"].shape[0] == len(df):
            print(f"  cache hit  {f.name}  {tuple(z['feats'].shape)}")
            return z["feats"], z["labels"]
        print(f"  cache size mismatch, recomputing {f.name}")
    print(f"  computing features for {split}: {len(df)} images x {n_variants} variant(s)")
    feats = compute_features(net, df, n_variants, device, batch_size, num_workers,
                             split, half=half)
    if use_cache:
        np.savez(f, feats=feats, labels=labels)
        print(f"  cached -> {f.name} ({f.stat().st_size / 1e6:.0f} MB)")
    return feats, labels


# --- metrics / schedule ------------------------------------------------------
def safe_auc(y: np.ndarray, s: np.ndarray) -> float:
    """ROC-AUC, or nan when only one class is present."""
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def cosine_warmup(optimizer, total_steps: int, warmup_frac: float = 0.1):
    warmup = max(1, int(total_steps * warmup_frac))

    def fn(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


def leakage_check(splits: dict[str, pd.DataFrame]) -> None:
    """Report exact-path overlap between splits, which would be real leakage.

    Basename collisions are reported separately and are expected: CIFAKE names
    files per source folder, so train/REAL/0000.jpg and test/REAL/0000.jpg are
    different images that happen to share a name. Warning on basenames alone
    would fire on essentially every file and teach you to ignore the check.
    """
    names = [n for n in ("train", "val", "val_unseen_generator", "val_unseen_content",
                         "test") if n in splits and len(splits[n])]
    problems = 0
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pa, pb = set(splits[a]["path"]), set(splits[b]["path"])
            shared = pa & pb
            if shared:
                problems += 1
                print(f"  LEAKAGE  {a} and {b} share {len(shared)} identical path(s), "
                      f"e.g. {sorted(shared)[:3]}")
    if not problems:
        print("  no identical paths shared between any two splits")
    tr = {Path(p).name for p in splits["train"]["path"]}
    va = {Path(p).name for p in splits.get("val", pd.DataFrame({"path": []}))["path"]}
    dup = tr & {Path(p).name for p in va} if va else set()
    if dup:
        print(f"  note: {len(dup)} basename(s) appear in both train and val at "
              f"different paths (expected for CIFAKE's per-folder naming, not leakage)")


# --- training ----------------------------------------------------------------
def evaluate(module: nn.Module, loader: DataLoader,
             device: torch.device) -> tuple[float, float]:
    """Return (mean BCE loss, AUC) over a loader."""
    module.eval()
    crit = nn.BCEWithLogitsLoss()
    ys, ss, losses = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = module(xb)
            losses.append(crit(logits, yb).item() * len(yb))
            ys.append(yb.cpu().numpy())
            ss.append(torch.sigmoid(logits).cpu().numpy())
    y = np.concatenate(ys)
    s = np.concatenate(ss)
    return float(np.sum(losses) / len(y)), safe_auc(y, s)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # data/prepare.py writes its CSVs to data/csv, so that is the default here.
    ap.add_argument("--data", default=str(ROOT / "data" / "csv"),
                    help="directory holding the split CSVs from data/prepare.py")
    ap.add_argument("--backbone", default=DEFAULT_BACKBONE,
                    choices=["clip_vit_b16", "resnet50", "3-stream"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--use-probe", action="store_true",
                help="run PROBE hard-negative refinement after standard training")
    ap.add_argument("--probe-iterations", type=int, default=3,
                    help="number of PROBE refinement iterations")
    ap.add_argument("--probe-percentile", type=int, default=10,
                    help="hard-negative percentile")
    ap.add_argument("--probe-epochs", type=int, default=5,
                    help="epochs per PROBE iteration")
    ap.add_argument("--lr", type=float, default=None,
                    help="default 1e-3 for the cached head, 1e-4 end to end")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-frac", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=5,
                    help="epochs without unseen-split AUC improvement before stopping")
    ap.add_argument("--out", default=str(ROOT / "model" / "checkpoints"))
    ap.add_argument("--cache-dir", default=str(ROOT / "data" / "cache"))
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit-train", type=int, default=0,
                    help="cap the number of training rows (for quick runs)")
    ap.add_argument("--no-half", action="store_true",
                    help="disable autocast fp16 during feature extraction "
                         "(slower; fp32 measured 22 img/s vs 30 on an M2)")
    ap.add_argument("--overfit", type=int, default=0,
                    help="wiring check: fit this many images with no augmentation "
                         "and no dropout; loss must approach zero")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = pick_device(args.device)
    data_dir = Path(args.data)
    if not data_dir.is_dir():
        raise SystemExit(f"--data directory not found: {data_dir}\n"
                         f"Run `python data/prepare.py` first.")

    splits: dict[str, pd.DataFrame] = {}
    for name in ("train", "val", "val_unseen_generator", "val_unseen_content", "test"):
        p = data_dir / f"{name}.csv"
        if p.is_file():
            splits[name] = pd.read_csv(p)
    if "train" not in splits:
        raise SystemExit(f"no train.csv in {data_dir}")

    print("=" * 70)
    print(f"device: {device}   seed: {args.seed}")
    print("=" * 70)

    net = PixelProofNet(args.backbone, pretrained=True).to(device)
    print(describe(net))
    total, trainable = count_params(net)
    if net.frozen and trainable >= total:
        raise SystemExit("backbone freeze did not take effect")
    print("=" * 70)

    # ---- overfit wiring check -------------------------------------------
    if args.overfit:
        n = args.overfit
        df = splits["train"].iloc[:n].reset_index(drop=True)
        print(f"OVERFIT CHECK on {len(df)} images "
              f"(label mean {df['label'].mean():.3f})")
        if df["label"].nunique() < 2:
            raise SystemExit("overfit subset is single-class; re-run data/prepare.py")
        # Dropout off and no augmentation: this measures wiring, not generalisation.
        for m in net.head.modules():
            if isinstance(m, nn.Dropout):
                m.p = 0.0

        if net.frozen:
            feats, labels = get_features(net, df, "overfit", 1, Path(args.cache_dir),
                                         device, args.batch_size, args.num_workers,
                                         use_cache=False, half=not args.no_half)
            ds = CachedFeatures(feats, labels, train=False, seed=args.seed)
            module = net.head
        else:
            ds = ImageRows(df, build_eval_transform(net.img_size, net.mean, net.std))
            module = net
        loader = DataLoader(ds, batch_size=min(args.batch_size, len(ds)), shuffle=True,
                            num_workers=0 if net.frozen else args.num_workers)
        lr = args.lr if args.lr is not None else (1e-3 if net.frozen else 1e-4)
        opt = torch.optim.AdamW([p for p in module.parameters() if p.requires_grad],
                                lr=lr, weight_decay=0.0)
        crit = nn.BCEWithLogitsLoss()
        epochs = max(args.epochs, 60) if net.frozen else args.epochs
        what = "head only" if net.frozen else "whole network end to end"
        n_fit = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"  fitting {what} ({n_fit:,} params) with lr={lr}, "
              f"no dropout, no augmentation, {epochs} epochs")
        module.train()
        hist = []
        for ep in range(1, epochs + 1):
            tot, cnt, correct = 0.0, 0, 0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                logits = module(xb)
                loss = crit(logits, yb)
                loss.backward()
                opt.step()
                tot += loss.item() * len(yb)
                cnt += len(yb)
                correct += ((torch.sigmoid(logits) > 0.5).float() == yb).sum().item()
            hist.append(tot / cnt)
            if ep <= 3 or ep % 10 == 0 or ep == epochs:
                print(f"    epoch {ep:3}  loss {tot / cnt:.6f}  "
                      f"train acc {correct / cnt:.4f}")
        final = hist[-1]
        print(f"\n  first-epoch loss {hist[0]:.6f} -> final loss {final:.6f}")
        ok = final < 0.05
        print(f"  RESULT: {'PASS' if ok else 'FAIL'} "
              f"- final loss {final:.6f} {'<' if ok else '>='} 0.05")
        if not ok:
            print("  The model cannot memorise a tiny subset, so something is "
                  "wired wrong.\n  Check the head, the optimiser and the label "
                  "column before training.")
        raise SystemExit(0 if ok else 1)

    # ---- full training ---------------------------------------------------
    print("leakage check")
    leakage_check(splits)
    print("=" * 70)

    if args.limit_train:
        splits["train"] = splits["train"].iloc[:args.limit_train].reset_index(drop=True)

    # The split early stopping watches. Prefer a genuine unseen-generator split.
    if len(splits.get("val_unseen_generator", [])) > 0:
        monitor = "val_unseen_generator"
    elif len(splits.get("val_unseen_content", [])) > 0:
        monitor = "val_unseen_content"
        print("WARNING  val_unseen_generator is EMPTY, so early stopping falls back")
        print("         to val_unseen_content. CIFAKE ships a single generator")
        print("         (sd14), so there is no second generator to hold out. This")
        print("         measures generalisation to unseen CONTENT, which is a")
        print("         weaker guarantee than unseen-generator generalisation.")
        print("         Add a generator under data/raw/ai/<name>/ to fix this.")
    else:
        monitor = "val"
        print("WARNING  no unseen split available; early stopping on overall val AUC.")
    print(f"early stopping monitors: {monitor} AUC")
    print("=" * 70)

    eval_splits = [n for n in ("val", "val_unseen_generator", "val_unseen_content")
                   if len(splits.get(n, [])) > 0]

    lr = args.lr if args.lr is not None else (1e-3 if net.frozen else 1e-4)
    cache_dir = Path(args.cache_dir)

    if net.frozen:
        print(f"cached path: precomputing features (K={K_VARIANTS} for train, "
              f"1 for eval)")
        tr_f, tr_y = get_features(net, splits["train"], "train", K_VARIANTS, cache_dir,
                                  device, args.batch_size, args.num_workers,
                                  half=not args.no_half)
        train_ds = CachedFeatures(tr_f, tr_y, train=True, seed=args.seed)
        eval_loaders = {}
        for n in eval_splits:
            f, y = get_features(net, splits[n], n, 1, cache_dir, device,
                                args.batch_size, args.num_workers,
                                half=not args.no_half)
            eval_loaders[n] = DataLoader(CachedFeatures(f, y, train=False),
                                         batch_size=512, shuffle=False, num_workers=0)
        module = net.head
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=0)
    else:
        print("end-to-end path: on-the-fly augmentation, no feature cache")
        train_loader = DataLoader(
            ImageRows(splits["train"], build_train_transform(net.img_size, net.mean, net.std)),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
            drop_last=True)
        ev = build_eval_transform(net.img_size, net.mean, net.std)
        eval_loaders = {n: DataLoader(ImageRows(splits[n], ev), batch_size=args.batch_size,
                                      shuffle=False, num_workers=args.num_workers)
                        for n in eval_splits}
        module = net

    # Class imbalance. prepare.py produces balanced splits, so this is usually 1.
    # Read it from the dataframe on both paths -- the cached path's labels are
    # derived from the same column, and branching here only invites drift.
    train_labels = splits["train"]["label"].to_numpy(dtype=np.float32)
    n_pos = float(train_labels.sum())
    n_neg = float(len(train_labels) - n_pos)
    pos_weight = None
    if n_pos > 0 and abs(n_neg / n_pos - 1.0) > 0.05:
        pos_weight = torch.tensor([n_neg / n_pos], device=device)
        print(f"class imbalance {n_neg:.0f}:{n_pos:.0f} -> pos_weight "
              f"{n_neg / n_pos:.3f}")
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    params = [p for p in module.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=args.weight_decay)
    sched = cosine_warmup(opt, args.epochs * max(1, len(train_loader)), args.warmup_frac)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.backbone
    log_path = out_dir / f"{tag}_trainlog.csv"
    ckpt_path = out_dir / f"{tag}_best.pth"
    cfg_path = out_dir / f"{tag}_config.json"

    cfg = {
        **vars(args),
        "monitor_split": monitor,
        "k_variants": K_VARIANTS if net.frozen else 0,
        "variant_spec": VARIANT_SPEC if net.frozen else [],
        "lr_used": lr,
        "feat_dim": net.feat_dim,
        "img_size": net.img_size,
        "mean": list(net.mean),
        "std": list(net.std),
        "total_params": total,
        "trainable_params": trainable,
        "pos_weight": None if pos_weight is None else float(pos_weight.item()),
        "train_rows": len(splits["train"]),
        "device": str(device),
    }
    cfg_path.write_text(json.dumps(cfg, indent=2, default=str) + "\n")
    print(f"config -> {cfg_path.name}")

    fields = ["epoch", "train_loss", "lr", "seconds"] + \
             [f"{n}_{m}" for n in eval_splits for m in ("loss", "auc")]
    with log_path.open("w", newline="") as fh:
        csv.DictWriter(fh, fieldnames=fields).writeheader()

    best, best_ep, since = -1.0, 0, 0
    print("=" * 70)
    for ep in range(1, args.epochs + 1):
        module.train()
        t0, tot, cnt = time.time(), 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(module(xb), yb)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(yb)
            cnt += len(yb)
        train_loss = tot / max(cnt, 1)

        row = {"epoch": ep, "train_loss": round(train_loss, 6),
               "lr": round(opt.param_groups[0]["lr"], 8),
               "seconds": round(time.time() - t0, 2)}
        for n in eval_splits:
            l, a = evaluate(module, eval_loaders[n], device)
            row[f"{n}_loss"], row[f"{n}_auc"] = round(l, 6), round(a, 6)
        with log_path.open("a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=fields).writerow(row)

        msg = (f"epoch {ep:3}/{args.epochs}  loss {train_loss:.4f}  "
               + "  ".join(f"{n.replace('val_unseen_', 'unseen_')} AUC "
                           f"{row[f'{n}_auc']:.4f}" for n in eval_splits)
               + f"  {row['seconds']:.1f}s")
        score = row[f"{monitor}_auc"]
        if not math.isnan(score) and score > best:
            best, best_ep, since = score, ep, 0
            torch.save({"model": net.state_dict(), "head": net.head.state_dict(),
                        "backbone_name": net.backbone_name, "epoch": ep,
                        "monitor_split": monitor, "monitor_auc": score,
                        "config": cfg}, ckpt_path)
            msg += "  <- best, saved"
        else:
            since += 1
        print(msg, flush=True)
        if since >= args.patience:
            print(f"early stop: no {monitor} AUC improvement in {args.patience} epochs")
            break
    if args.use_probe:
        print("\n" + "=" * 70)
        print("PROBE adversarial hard-negative refinement")
        print("=" * 70)

        indexed_train_loader = create_indexed_dataloader(train_loader)
        indexed_val_loader = create_indexed_dataloader(eval_loaders["val"])

        probe_trainer = ProbeTrainer(
            model=net,
            device=device,
            learning_rate=1e-4,
            weight_decay=1e-5,
        )

        probe_history = probe_trainer.train_with_probe(
            train_dataloader=indexed_train_loader,
            val_dataloader=indexed_val_loader,
            num_iterations=args.probe_iterations,
            percentile_hard=args.probe_percentile,
            epochs_per_iteration=args.probe_epochs,
        )

        print("PROBE refinement complete.")       
    print("=" * 70)
    print(f"best {monitor} AUC {best:.4f} at epoch {best_ep}")
    print(f"checkpoint -> {ckpt_path}")
    print(f"log        -> {log_path}")
    # Not relative_to: --out may point outside the project, and that raises.
    shown = (ckpt_path.resolve().relative_to(ROOT)
             if ckpt_path.resolve().is_relative_to(ROOT) else ckpt_path.resolve())
    print(f"\nNext: python model/calibrate.py --checkpoint {shown}")


if __name__ == "__main__":
    main()
