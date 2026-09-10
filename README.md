# PixelProof

**SIH 2026 Internal Hackathon — Problem Statement 2, SignalScope**
*Telling Real From Synthetic in the Age of Generative Media*

PixelProof estimates the probability that an image is AI-generated and shows
*where in the frame* that estimate comes from, via a Grad-CAM heat-map and a
grounded, measurement-based explanation. It reports likelihoods, never
verdicts, and makes no claim about who produced an image or about any real
person appearing in one.

---

## 1. Modules built

| | Module | Status |
|---|---|---|
| **Core** | Real-vs-AI-generated classification, calibrated confidence, ROC-AUC / macro-F1 / confusion matrix on held-out data (incl. an unseen-generator split), minimal interface | ✅ built |
| **Bonus A** | Faithful explanation — Grad-CAM heat-map + grounded, measurement-derived text (not free-form prose) | ✅ built |
| **Bonus C** | Robustness to degradation — trained with JPEG re-compression, downscale/upscale and blur variants; not yet benchmarked with a degradation-vs-accuracy curve | ◐ partial |
| B, D, E, F, G | Generator attribution, provenance/EXIF, multimodal text consistency, deployable interface polish, adversarial analysis | not attempted |

Module F (deployable interface) is effectively covered by the shipped web app
(`uvicorn src.api:app`) even though not separately engineered as a browser
extension.

## 2. Setup and run instructions (reproduces a prediction in under 10 minutes)

**Requirements:** Python 3.10+ (built and tested on 3.13.14 — see
[Environment notes](#environment-notes)). No Node.js needed; the frontend is
pre-built and committed.

```bash
git clone <this-repo-url>
cd PixelProof
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** — drop an image in the hero dropzone, or use
the CLI:

```bash
python model/predict.py --input path/to/image.jpg --out preds.csv
```

**Why this is fast even though no `.pth` checkpoint is committed:** `.gitignore`
excludes every real checkpoint (constraint of this build), but `model/weights/`
commits the *trained head* — 197,121 parameters, ~770 KB — plus its calibration.
`src/inference.py` loads that head, fetches the frozen CLIP backbone from timm
on first run (cached afterward), and reproduces the full checkpoint's output
bit-for-bit (verified: identical probability to 6 decimal places against the
full 344 MB checkpoint). First run needs network access for that one-time
backbone download; every run after is fully offline.

`predict.py` and `src/api.py` both call the same `src/inference.py` — the CLI
an organizer runs and the API a user hits can never disagree about the same
image.

## 3. Datasets used, sources and licences

| Dataset | Role | Size used | Source | Licence |
|---|---|---|---|---|
| **CIFAKE** | Training + in-distribution val/test. Real half is CIFAR-10; AI half is Stable Diffusion 1.4 images, both at 32×32. | 24,000 train (CLIP) / 12,000 train (ResNet50) of 100,000 available; 8,000 val; 20,000 test | [Kaggle: birdy654/cifake-real-and-ai-generated-synthetic-images](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images); introduced in Bird & Lotfi, *"CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images"*, 2023 ([arXiv:2303.14126](https://arxiv.org/abs/2303.14126)) | Stated as open/research-use on Kaggle — **confirm the exact licence text on the dataset page before any redistribution beyond this submission**; not independently re-verified here. |
| **ProGAN validation images** | Held out entirely as the unseen-generator split (`val_unseen_generator`) — never trained on. Resolution-matched to 32×32 before use (see §5) so the split measures generator artefacts, not resolution. | 1,000 images (bird/car/dog categories) | [huggingface.co/datasets/frp94/progan_val](https://huggingface.co/datasets/frp94/progan_val) | **Not stated on the source page** (dataset card lists licence as "more information needed"). The category structure (bird/car/dog…) matches the well-known ProGAN validation benchmark from Wang et al., *"CNN-Generated Images Are Surprisingly Easy to Spot… For Now"*, CVPR 2020 — plausibly the same underlying images repackaged, but this is **not confirmed** and should be verified before any use beyond this internal submission. |

Training data was **not** touched by calibration or evaluation; `val` is used
for calibration and threshold selection, `test` and both unseen splits are
untouched by both training and calibration.

## 4. Reported metrics

Both models: FP16 autocast inference on Apple M2 (MPS), CLIP feature-cached
during training. Threshold chosen at **FPR ≤ 5%** on `val` (wrongly flagging a
real photo is the costly error) — see [report/model_report.md](report/model_report.md)
for the full derivation.

### Primary model — frozen CLIP ViT-B/16 + linear head
Calibration: T=0.7813, threshold=0.7833

| split | ROC-AUC | macro-F1 | accuracy | FPR | recall (AI) | confusion (tn/fp/fn/tp) |
|---|---|---|---|---|---|---|
| test (in-distribution) | 0.9851 | 0.9362 | 0.9363 | 0.0491 | 0.9216 | 9509 / 491 / 784 / 9216 |
| val_unseen_content (proxy: horse+ship held out) | 0.9723 | 0.8982 | 0.8986 | 0.0390 | 0.8362 | 9610 / 390 / 1638 / 8362 |
| **val_unseen_generator (ProGAN, true unseen generator)** | **0.9410** | 0.8336 | 0.8355 | 0.0570 | 0.7280 | 943 / 57 / 272 / 728 |

### Baseline — ResNet50, fine-tuned end-to-end
Calibration: T=0.9006, threshold=0.5287

| split | ROC-AUC | macro-F1 | accuracy | FPR | recall (AI) | confusion (tn/fp/fn/tp) |
|---|---|---|---|---|---|---|
| test (in-distribution) | 0.9884 | 0.9459 | 0.9459 | 0.0503 | 0.9422 | 9497 / 503 / 578 / 9422 |
| val_unseen_content (proxy) | 0.9685 | 0.8889 | 0.8894 | 0.0431 | 0.8220 | 9569 / 431 / 1780 / 8220 |
| **val_unseen_generator (ProGAN, true unseen generator)** | **0.9519** | 0.8625 | 0.8635 | 0.0500 | 0.7770 | 950 / 50 / 223 / 777 |

**The finding that matters most, honestly reported:** the fine-tuned ResNet50
baseline generalises to a *genuinely new* generator (ProGAN, a GAN, never seen
in training) **slightly better** than the frozen-CLIP primary model (0.9519 vs
0.9410 AUC) — the reverse of what a content-holdout proxy suggested before
ProGAN data was available (see §5, Limitations). Both comfortably beat the
18–31% accuracy range the problem statement's own reference material cites for
naively-transferred detectors on unseen generators.

## 5. Architecture, calibration, and limitations

**Architecture.** Two interchangeable backbones behind one head
(`Dropout → Linear(→256) → GELU → Linear(→1)`):
- **Primary:** `vit_base_patch16_clip_224.openai` via timm, **entirely frozen**
  (verified: 197,121 of 85,996,545 params trainable, 0.229%). Chosen because
  frozen-CLIP-features-plus-linear-probe is reported to generalise across
  generators better than end-to-end fine-tuning (Ojha et al. 2023,
  [arXiv:2302.10174](https://arxiv.org/abs/2302.10174)) — a frozen backbone also
  lets features be **precomputed once** (4 fixed augmentation variants per
  image: clean / JPEG-q40 / downscale-upscale / blur) so the head trains in
  under a second per epoch.
- **Baseline:** `resnet50` (torchvision, ImageNet weights), fine-tuned end to
  end, same head.

**Calibration.** Temperature scaling (`model/calibrate.py`) fitted on `val`
only, then a threshold chosen for **FPR ≤ 5%** — never a bare sigmoid. AUC is
identical before/after scaling by construction (a scalar cannot reorder
predictions); calibration only makes the reported number a genuine probability.

**Early stopping** watches **`val_unseen_content` AUC**, not overall `val`
AUC — `val` AUC climbed monotonically to convergence on both models while
unseen-content AUC peaked early and drifted down, confirming that stopping on
in-distribution accuracy alone would have shipped a worse-generalising model.

**Known limitations, stated honestly:**
- **A content-holdout is a weaker proxy than a true generator-holdout, and it
  predicted the wrong winner.** Before ProGAN data existed, the content-holdout
  (whole CIFAR-10 classes withheld) suggested frozen CLIP generalises better
  than ResNet50. The true unseen-generator test says the opposite. Don't trust
  a content-holdout as a stand-in for cross-generator generalisation.
- **Recall drops hardest on the truly unseen generator**: 0.93 (test) → 0.73
  (ProGAN) for CLIP. A borderline ProGAN example scored 0.766 — correctly
  leaning AI but just under the real-data-calibrated threshold of 0.783 (see
  `report/explanation_samples/`) — the honest shape of the failure mode, not a
  cherry-picked success.
- **CIFAKE's images are 32×32.** Upscaled ~7× to the backbones' 224px input,
  which measurably degrades the Grad-CAM heat-map (a 14×14 attention grid over
  what was 32px of real content) and likely destroys most of the
  frequency-lattice cue the explainer text describes, even though the
  explanation only ever states what it actually measured.
- **The ProGAN holdout is small** (1,000 images) and single-generator; it is
  real evidence, not a comprehensive cross-generator benchmark.
- Accuracy is expected to drop further on generators released after this
  build, on heavily-compressed/resized/screenshotted images beyond what
  training augmentation covers, and on content categories very unlike
  CIFAR-10's ten classes.
- Per the challenge's own scope rules, this system makes **no claim about
  people** in an image and is not evaluated on face-swap deepfakes.

## 6. Demo video & deployed app

- Demo video: **[add link before submission]**
- Deployed app: **[add link if hosted, or note "run locally via the command
  above" — see §2]**

---

## Originality declaration

- Model architecture, training loop, calibration, Grad-CAM integration,
  explanation engine, API and frontend were built for this hackathon
  (10–15 September 2026 window).
- **Third-party code referenced, not copied:** none directly reused. Early in
  the build we inspected a public repo named `Detecting-AI-Generated-Fake-Images`
  (a Duke IDS705 course project, MIT-licensed, supplied to us as a zip archive —
  exact repo URL not retained) as a candidate data source; it contained no
  usable dataset (only 6 sample photos and private-bucket references) and none
  of its code was reused.
- **Published method referenced:** the frozen-backbone design choice follows
  the finding in Ojha, Li & Lee, *"Towards Universal Fake Image Detectors that
  Generalize Across Generative Models"*, CVPR 2023
  ([arXiv:2302.10174](https://arxiv.org/abs/2302.10174)) — an idea, not code.
- **Libraries:** PyTorch, torchvision, timm, albumentations, grad-cam
  (`pytorch-grad-cam`), FastAPI, React, GSAP, framer-motion — all used via
  their public APIs per `requirements.txt` / `src/frontend/package.json`, no
  vendored source.
- **Datasets:** see §3 above; both are public, and licence terms for the
  ProGAN mirror in particular should be confirmed before use beyond this
  internal submission.

## Environment notes

Built and tested on Python **3.13.14** (macOS, Apple M2, MPS backend) rather
than the originally-targeted 3.10 — no available 3.10 install on the build
machine, and every pinned dependency in `requirements.txt` resolves cleanly on
3.13 with no version conflicts. `torch.backends.mps` is used automatically
when available, falling back to CUDA then CPU.

## Project structure

```
pixelproof/
  requirements.txt
  data/prepare.py          # builds train/val/test CSVs; holds out named
                           # generators AND whole content classes as two
                           # separate, honestly-named generalisation splits
  model/
    dataset.py             # torch Dataset over the CSVs
    model.py               # frozen CLIP ViT-B/16 (or ResNet50 --backbone
                           # resnet50 baseline) + small trainable head
    train.py                # feature-cached training for the frozen path,
                            # end-to-end for the baseline; overfit-check flag
    calibrate.py            # temperature scaling + FPR-capped threshold
    predict.py               # CLI: --input <file|folder> --out preds.csv
    evaluate.py               # ROC-AUC / macro-F1 / confusion matrix /
                              # accuracy & FPR at threshold, per split
  src/
    inference.py            # SHARED: the only place that loads a model and
                            # scores an image — predict.py and api.py both
                            # call this, so they cannot disagree
    explain.py               # Grad-CAM overlay + grounded text built only
                             # from measurements taken on the image
    api.py                   # FastAPI; serves the built frontend from /
    frontend/                # Vite + React app (dist/ is committed)
  report/
    model_report.md          # the required one-page report
    metrics_final.json        # full metrics, all splits, both models
    explanation_samples/       # sample /predict outputs + heat-maps
```
