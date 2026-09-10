"""Torch Dataset over the CSVs written by ``data/prepare.py``.

Each CSV row carries ``path,label,generator,source,content_class,content_group``,
where ``label`` is 1 for AI-generated and 0 for real. Paths are stored relative to
the project root so the CSVs stay portable.

Augmentation policy
-------------------
The cues a synthetic-image detector relies on are subtle: upsampling lattices in
the frequency domain, and noise variance that fails to scale with luminance.
Aggressive augmentation destroys exactly those cues, so the train pipeline stays
deliberately mild:

- ``HorizontalFlip`` -- geometry-only, leaves frequency and noise statistics intact.
- ``ImageCompression`` -- mild JPEG re-encoding. Included on purpose: real uploads
  arrive re-compressed, and this is the one augmentation that buys robustness to it.

Two common transforms are deliberately NOT used:

- ``GaussNoise`` would inject synthetic noise with luminance-independent variance --
  the exact statistic that distinguishes generated images from sensor output. Adding
  it would teach the model the opposite of what it needs to learn.
- Blur/sharpen resample the image and smear the frequency peaks the first signal
  family depends on.

Note on CIFAKE: its images are 32x32 and get upscaled to the backbone's input size.
That upscale imprints its own resampling pattern, and it does so on both classes
equally, so it does not leak label information -- but it does attenuate the original
generator lattice. This is a real limitation of training on CIFAKE and is recorded
in report/model_report.md.
"""

from __future__ import annotations

from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent

# Sensible defaults; model.py overrides these from the backbone's pretrained_cfg
# so normalisation always matches how the backbone was trained.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(
    img_size: int = 224,
    mean: tuple[float, ...] = IMAGENET_MEAN,
    std: tuple[float, ...] = IMAGENET_STD,
    train: bool = False,
) -> A.Compose:
    """Build the albumentations pipeline. See the module docstring for the policy.

    Augmentations run at native resolution and the resize comes last, so JPEG
    artefacts land at the scale a real re-upload would produce them.
    """
    steps: list = []
    if train:
        steps += [
            A.HorizontalFlip(p=0.5),
            A.ImageCompression(compression_type="jpeg", quality_range=(75, 100), p=0.3),
        ]
    steps += [
        # Bicubic to match the interpolation timm's ViT configs expect.
        A.Resize(img_size, img_size, interpolation=cv2.INTER_CUBIC),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
        ToTensorV2(),
    ]
    return A.Compose(steps)


class PixelProofDataset(Dataset):
    """Reads one split CSV and yields dicts.

    Each item is ``{"image": FloatTensor[3,H,W], "label": FloatTensor[],
    "path": str, "generator": str, "content_class": str}``. Strings collate into
    plain lists under the default collate_fn, which is what evaluate.py and
    predict.py want for per-file reporting.
    """

    def __init__(
        self,
        csv_path: str | Path,
        transform: A.Compose | None = None,
        root: Path = ROOT,
        img_size: int = 224,
    ) -> None:
        self.csv_path = Path(csv_path)
        if not self.csv_path.is_file():
            raise FileNotFoundError(
                f"{self.csv_path} not found. Run `python data/prepare.py` first."
            )
        self.df = pd.read_csv(self.csv_path)
        self.root = Path(root)
        self.transform = transform or build_transforms(img_size=img_size, train=False)

        missing = {"path", "label"} - set(self.df.columns)
        if missing:
            raise ValueError(f"{self.csv_path} is missing column(s): {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.df)

    def _resolve(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else self.root / p

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        path = self._resolve(row["path"])
        # PIL decodes the widest range of inputs; convert("RGB") normalises
        # greyscale, palette and RGBA images to three channels.
        with Image.open(path) as im:
            arr = np.asarray(im.convert("RGB"))
        image = self.transform(image=arr)["image"]
        return {
            "image": image,
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "path": str(row["path"]),
            "generator": str(row.get("generator", "unknown")),
            "content_class": str(row.get("content_class", "unknown")),
        }


def make_loader(
    csv_path: str | Path,
    img_size: int = 224,
    batch_size: int = 64,
    train: bool = False,
    mean: tuple[float, ...] = IMAGENET_MEAN,
    std: tuple[float, ...] = IMAGENET_STD,
    num_workers: int = 4,
    shuffle: bool | None = None,
    limit: int = 0,
) -> DataLoader:
    """Convenience constructor used by train.py / evaluate.py / calibrate.py."""
    ds = PixelProofDataset(
        csv_path,
        transform=build_transforms(img_size=img_size, mean=mean, std=std, train=train),
        img_size=img_size,
    )
    if limit and limit < len(ds):
        # A head-slice is a valid random sample because prepare.py shuffles each
        # split before writing it, so the slice keeps both labels.
        ds.df = ds.df.iloc[:limit].reset_index(drop=True)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train if shuffle is None else shuffle,
        num_workers=num_workers,
        pin_memory=False,  # no effect on MPS, and avoids a warning
        drop_last=False,
        persistent_workers=bool(num_workers),
    )


if __name__ == "__main__":
    # Smoke check: load each split, pull one batch, report shapes and balance.
    csv_dir = ROOT / "data" / "csv"
    print(f"reading CSVs from {csv_dir}\n")
    hdr = f"{'split':24} {'rows':>7} {'batch image shape':>22} {'label mean':>11}"
    print(hdr); print("-" * len(hdr))
    for name in ("train", "val", "val_unseen_generator", "val_unseen_content", "test"):
        p = csv_dir / f"{name}.csv"
        if not p.is_file():
            print(f"{name:24} {'MISSING':>7}")
            continue
        ds = PixelProofDataset(p)
        if len(ds) == 0:
            print(f"{name:24} {0:>7} {'(empty split)':>22} {'--':>11}")
            continue
        loader = make_loader(p, batch_size=8, num_workers=0, train=(name == "train"))
        b = next(iter(loader))
        print(f"{name:24} {len(ds):>7} {str(tuple(b['image'].shape)):>22} "
              f"{b['label'].mean().item():>11.3f}")
    batch = next(iter(make_loader(csv_dir / "val.csv", batch_size=2, num_workers=0)))
    print("\nper-item keys:", sorted(batch.keys()))
    # A tiny head-slice must contain both labels, or step 2's overfit check is
    # meaningless. Assert it rather than leave it to be noticed later.
    tiny = make_loader(csv_dir / "train.csv", batch_size=32, num_workers=0, limit=32)
    labels = next(iter(tiny))["label"]
    assert 0 < labels.mean().item() < 1, (
        f"a 32-row slice of train.csv is single-class (label mean "
        f"{labels.mean().item():.3f}); re-run data/prepare.py"
    )
    print(f"tiny-subset label balance: {labels.mean().item():.3f} (both classes present)")
