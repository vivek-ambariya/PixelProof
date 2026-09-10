# PixelProof — Model Report

*SIH 2026 Internal Hackathon, Problem Statement 2 (SignalScope). One page, per
the submission contract §7.3.*

| Field | |
|---|---|
| **Task** | Binary real-vs-AI-generated image classification, calibrated confidence score. Bonus A attempted (faithful explanation: Grad-CAM heat-map + grounded, measurement-derived text — see `report/explanation_samples/`). |
| **Data & split** | **Train/val/in-distribution test**: [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) (Bird & Lotfi 2023) — real half is CIFAR-10, AI half is Stable Diffusion 1.4, all 32×32. Used 24,000 train images for the primary (CLIP) model, 12,000 for the baseline (ResNet50), out of 100,000 available, plus 8,000 val and 20,000 test (both untouched by training). **Unseen-content split** (proxy, not a substitute for cross-generator testing): whole CIFAR-10 classes (horse, ship — 19,801 images) held out of train entirely. **Unseen-generator split** (the primary generalisation metric): 1,000 [ProGAN](https://huggingface.co/datasets/frp94/progan_val) images, resolution-matched to 32×32 to remove a resolution confound, held out of train entirely, matched with 1,000 CIFAR-10 reals (2,000 total). All five CSVs are built by `data/prepare.py`; a leak check confirms zero path overlap across all 121,000 indexed images. |
| **Model / approach** | **Primary:** frozen `vit_base_patch16_clip_224.openai` (timm) — 85,996,545 total params, **197,121 trainable (0.229%)**, verified — feeding `Dropout(0.3) → Linear(768,256) → GELU → Linear(256,1)`. Frozen backbone lets features be precomputed once across 4 fixed augmentation variants (clean / JPEG-q40 / downscale-upscale / blur) so the head trains in <1s/epoch; chosen per Ojha et al. 2023 (arXiv:2302.10174), which reports frozen-CLIP-features-plus-linear-probe generalising across generators better than end-to-end fine-tuning. **Baseline:** `resnet50` (torchvision, ImageNet weights), same head, fine-tuned end to end (24,032,833/24,032,833 trainable). AdamW + cosine LR with warmup, `BCEWithLogitsLoss`, early stopping on **`val_unseen_content` AUC** (not overall val AUC — see Limitations). Calibration: temperature scaling fit on `val`, threshold chosen for FPR ≤ 5% on `val`. |
| **Metric & result** | See table below. Primary ranking metric is **unseen-generator AUC**. |
| **Baseline** | ResNet50 (fine-tuned end-to-end) scores **higher** in-distribution but **also higher on the true unseen-generator split** than the frozen-CLIP primary model — see Limitations for why this reverses an earlier (weaker) proxy measurement. |
| **Limitations** | See below. |

## Metric & result

Threshold chosen at **FPR ≤ 5%** on `val` (a wrongly-flagged real photo is the
costly error). AUC is unaffected by calibration (a scalar cannot reorder
predictions), so pre/post-calibration AUC is identical by construction.

### Frozen CLIP ViT-B/16 + linear head (primary) — T=0.7813, threshold=0.7833

| split | n | ROC-AUC | macro-F1 | accuracy | FPR | recall (AI) | tn / fp / fn / tp |
|---|---|---|---|---|---|---|---|
| test (in-distribution) | 20,000 | 0.9851 | 0.9362 | 0.9363 | 0.0491 | 0.9216 | 9509 / 491 / 784 / 9216 |
| val_unseen_content (proxy) | 20,000 | 0.9723 | 0.8982 | 0.8986 | 0.0390 | 0.8362 | 9610 / 390 / 1638 / 8362 |
| **val_unseen_generator (ProGAN)** | 2,000 | **0.9410** | 0.8336 | 0.8355 | 0.0570 | 0.7280 | 943 / 57 / 272 / 728 |

### ResNet50, fine-tuned end-to-end (baseline) — T=0.9006, threshold=0.5287

| split | n | ROC-AUC | macro-F1 | accuracy | FPR | recall (AI) | tn / fp / fn / tp |
|---|---|---|---|---|---|---|---|
| test (in-distribution) | 20,000 | 0.9884 | 0.9459 | 0.9459 | 0.0503 | 0.9422 | 9497 / 503 / 578 / 9422 |
| val_unseen_content (proxy) | 20,000 | 0.9685 | 0.8889 | 0.8894 | 0.0431 | 0.8220 | 9569 / 431 / 1780 / 8220 |
| **val_unseen_generator (ProGAN)** | 2,000 | **0.9519** | 0.8625 | 0.8635 | 0.0500 | 0.7770 | 950 / 50 / 223 / 777 |

Both models substantially beat the 18–31% accuracy range naive detectors are
reported to score on generators they weren't trained against. `val_unseen_generator`
is the split that should be weighted most heavily per the challenge's own
scoring rules (§4.2, §9, §10): on that split, **ResNet50 (0.9519) currently
edges out the frozen-CLIP primary model (0.9410)**.

## Baseline comparison, in full

The design rationale (frozen backbone generalises better) is a published
finding, not something we assumed true without checking — and checking it
against a genuine unseen generator (rather than a proxy) tells a more nuanced
story:

- **Content-holdout proxy** (whole CIFAR-10 classes withheld, same generator):
  frozen CLIP generalises *better* than ResNet50 (0.9723 vs 0.9685 AUC).
- **True unseen-generator holdout** (ProGAN, a GAN never seen in training):
  ResNet50 generalises *slightly better* than frozen CLIP (0.9519 vs 0.9410 AUC).

The proxy predicted the wrong winner. This is reported plainly rather than
adjusted after the fact, because the entire point of the unseen-generator
metric is to catch exactly this kind of gap between "generalises well by one
measure" and "generalises well against a genuinely new generator."

## Limitations

- **A content-holdout is a weaker proxy for cross-generator generalisation
  than an actual second generator**, evidenced directly above — before ProGAN
  data was available, the wrong model would have looked like the better
  generaliser.
- **Recall drops hardest on the truly unseen generator.** CLIP: 0.92 (test) →
  0.73 (ProGAN). A borderline ProGAN example scored 0.766 — correctly leaning
  toward AI-generated but just under the real-data-calibrated 0.783 threshold
  (`report/explanation_samples/sample_progan_ai.json`) — a representative
  failure mode, not a cherry-picked one.
- **CIFAKE's source images are 32×32**, upscaled ~7× to each backbone's 224px
  input. This measurably limits the Grad-CAM heat-map's spatial resolution (a
  14×14 attention grid over what was 32px of real content) and likely
  attenuates the frequency-domain lattice cue the explainer describes,
  independent of model accuracy.
- **The ProGAN holdout is 1,000 images from one additional generator** — real
  evidence of generalisation, not a comprehensive cross-generator benchmark.
  Its exact provenance/licence is not fully confirmed (see README §3).
- Expect further degradation on generators released after this build, on
  compression/resizing/screenshotting beyond the augmentation used in
  training, and on image content very unlike CIFAR-10's ten classes.
- Per the challenge's scope rules, PixelProof makes **no claim about the
  identity of any person** in an image and is not built or evaluated for
  face-swap deepfakes.
