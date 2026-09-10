"""Model definitions for PixelProof.

Two backbones, chosen with ``--backbone``:

``clip_vit_b16`` (primary, default)
    A **frozen** CLIP ViT-B/16 image tower via timm, used purely as a feature
    extractor, with a small trainable head on top. Frozen CLIP features plus a
    linear probe generalise to unseen generators markedly better than an
    end-to-end fine-tuned CNN (Ojha et al. 2023, arXiv:2302.10174). Because the
    backbone never updates, its features can be precomputed once and the head
    trained in seconds -- see train.py's feature cache.

``resnet50`` (baseline)
    torchvision ResNet50 with ImageNet weights, fine-tuned end to end. Included
    because the report needs a baseline, and because the expected result is the
    interesting one: it tends to score *higher* on generators it trained against
    and *lower* on unseen ones. No feature caching is possible here -- the
    weights move, so features must be recomputed every batch.

Both backbones feed the identical ``DetectorHead``, so a comparison between them
is a comparison of representations rather than of head capacity.

Output convention: ``forward`` returns raw **logits** of shape ``(B,)``. A higher
logit means more likely AI-generated. Probabilities come from applying a sigmoid,
and the user-facing probability additionally applies the temperature fitted by
calibrate.py -- never a bare sigmoid.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# --- backbone registry -------------------------------------------------------
# `frozen` decides both whether gradients flow into the backbone and whether
# train.py may use the feature cache.
BACKBONES: dict[str, dict] = {
    "clip_vit_b16": {
        "timm_name": "vit_base_patch16_clip_224.openai",
        "source": "timm",
        "frozen": True,
        "feat_dim": 768,
    },
    "resnet50": {
        # torchvision's stronger recipe; documented so the baseline is reproducible.
        "weights": "IMAGENET1K_V2",
        "source": "torchvision",
        "frozen": False,
        "feat_dim": 2048,
    },
}

DEFAULT_BACKBONE = "clip_vit_b16"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DetectorHead(nn.Module):
    """The trainable part: Dropout -> Linear -> GELU -> Linear -> 1 logit."""

    def __init__(self, feat_dim: int, hidden: int = 256, p_drop: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(p_drop),
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # (B, feat_dim) -> (B,)
        return self.net(feats).squeeze(-1)


class PixelProofNet(nn.Module):
    """Backbone + head. ``forward`` returns logits of shape (B,)."""

    def __init__(
        self,
        backbone: str = DEFAULT_BACKBONE,
        pretrained: bool = True,
        hidden: int = 256,
        p_drop: float = 0.3,
    ) -> None:
        super().__init__()
        if backbone not in BACKBONES:
            raise ValueError(
                f"unknown backbone {backbone!r}; choose from {sorted(BACKBONES)}"
            )
        self.backbone_name = backbone
        spec = BACKBONES[backbone]
        self.frozen = spec["frozen"]

        if spec["source"] == "timm":
            import timm

            self.backbone = timm.create_model(
                spec["timm_name"], pretrained=pretrained, num_classes=0
            )
            cfg = getattr(self.backbone, "pretrained_cfg", {}) or {}
            self.img_size = (cfg.get("input_size") or (3, 224, 224))[-1]
            self.mean = tuple(cfg.get("mean") or IMAGENET_MEAN)
            self.std = tuple(cfg.get("std") or IMAGENET_STD)
            feat_dim = self.backbone.num_features
        else:
            from torchvision import models

            self.backbone = models.resnet50(
                weights=spec["weights"] if pretrained else None
            )
            feat_dim = self.backbone.fc.in_features
            # Identity in place of the classifier gives pooled 2048-d features.
            self.backbone.fc = nn.Identity()
            self.img_size = 224
            self.mean, self.std = IMAGENET_MEAN, IMAGENET_STD

        if feat_dim != spec["feat_dim"]:
            # Not fatal, but the registry is used to size the cache, so say so.
            print(f"  note: {backbone} feat_dim is {feat_dim}, "
                  f"registry said {spec['feat_dim']}")
        self.feat_dim = feat_dim
        self.head = DetectorHead(feat_dim, hidden=hidden, p_drop=p_drop)

        if self.frozen:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        """Disable gradients for every backbone parameter and keep it in eval mode.

        eval() matters independently of requires_grad: it stops any BatchNorm
        running statistics from drifting, which would otherwise change a
        "frozen" backbone's outputs between epochs.
        """
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    def train(self, mode: bool = True):  # noqa: A003 - matches nn.Module API
        """Keep a frozen backbone in eval mode even when the model is training."""
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return self

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Pooled backbone features, (B, feat_dim).

        Gradients are suppressed only for a frozen backbone. Note this is
        ``no_grad`` on the *feature* path used during training; explain.py needs
        activation gradients for Grad-CAM and therefore calls the backbone
        directly rather than going through here.
        """
        if self.frozen:
            with torch.no_grad():
                return self.backbone(x)
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


def count_params(model: nn.Module) -> tuple[int, int]:
    """(total, trainable) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def describe(model: PixelProofNet) -> str:
    total, trainable = count_params(model)
    pct = 100.0 * trainable / total if total else 0.0
    return (
        f"backbone       : {model.backbone_name} "
        f"({'FROZEN' if model.frozen else 'fine-tuned end to end'})\n"
        f"input size     : {model.img_size}x{model.img_size}\n"
        f"normalisation  : mean={tuple(round(v, 4) for v in model.mean)} "
        f"std={tuple(round(v, 4) for v in model.std)}\n"
        f"feature dim    : {model.feat_dim}\n"
        f"total params   : {total:,}\n"
        f"TRAINABLE      : {trainable:,}  ({pct:.3f}% of total)"
    )


def pick_device(prefer: str = "auto") -> torch.device:
    """MPS on Apple silicon, else CUDA, else CPU."""
    if prefer != "auto":
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


if __name__ == "__main__":
    # Build both backbones, print parameter counts, and push one batch through.
    # The trainable count is the check that freezing actually took effect.
    torch.manual_seed(0)
    for name in ("clip_vit_b16", "resnet50"):
        print("=" * 66)
        m = PixelProofNet(name, pretrained=True)
        print(describe(m))
        m.eval()
        x = torch.randn(2, 3, m.img_size, m.img_size)
        with torch.no_grad():
            out = m(x)
        print(f"forward        : {tuple(x.shape)} -> logits {tuple(out.shape)}  "
              f"values {[round(v, 4) for v in out.tolist()]}")
    print("=" * 66)
