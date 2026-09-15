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
| **Bonus C** | Robustness to degradation — a full degradation-vs-accuracy grid (9 conditions × 2 splits × both models) covering JPEG compression, resizing, screenshotting and light edits, with the failure mode analysed: [report/robustness.md](report/robustness.md) | ✅ built |
| **Bonus F** | Deployable interface — the shipped web app (`uvicorn src.api:app`), containerised (`Dockerfile`) and deployed live: React frontend on Vercel, FastAPI backend on Render. **Caveat:** free-tier hosts cannot hold the CLIP backbone in memory, so upload-and-analyse runs locally but is degraded on the hosted URL — see [§6](#6-demo-video--deployed-app) | ✅ built (local) · ⚠️ hosted demo degraded |
| B, D, E, G | Generator attribution, provenance/EXIF, multimodal text consistency, adversarial analysis | not attempted |

Module F was not separately engineered as a browser extension; the deployed web
app is the interface.

## 2. Setup and run instructions (reproduces a prediction in under 10 minutes)

**Requirements:** Python 3.10+ (built and tested on 3.13.14 — see
[Environment notes](#environment-notes)). No Node.js needed; the frontend is
pre-built and committed.

```bash
git clone https://github.com/vivek-ambariya/PixelProof.git
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
| **GenImage** — Midjourney generations + ImageNet photographs, both at **native resolution** | Fixes the all-32×32 training distribution (see *Why a second dataset was added* below). Supplies native-resolution examples to **both** classes in `train`/`val`, plus `test_native` — a held-out native split physically moved out of `data/raw/` so it cannot leak into training. | **1,590 train** (798 Midjourney / 792 ImageNet real) · **170 val** (82 / 88) · **800 held out** (400 / 400) → `data/csv/test_native.csv` | [huggingface.co/datasets/shimei123/Genimage](https://huggingface.co/datasets/shimei123/Genimage) — a mirror of the GenImage benchmark (Zhu et al., *"GenImage: A Million-Scale Benchmark for Detecting AI-Generated Images"*, NeurIPS 2023 Datasets & Benchmarks track). Pulled by HTTP-range streaming rather than full download — `data/fetch_genimage.py`. | **Not stated on the mirror's page.** The challenge brief names GenImage as permitted extra training data (§4.1) **provided it is cited** — this row is that citation. Confirm the upstream licence before any use beyond this internal submission. |

**Why a second dataset was added.** Every CIFAKE image is 32×32-native, which
let *"sharp, detailed image"* stand in for *"AI-generated"* inside the training
distribution — no real training example was ever sharp. On a genuine
native-resolution photograph that shortcut **inverts**, and the detector calls
real photos fake with high confidence. Augmentation cannot repair this:
blurring or re-compressing a 32×32 image cannot add real high-frequency detail
to the *real* class. GenImage supplies both halves at native resolution, so the
shortcut is no longer available. **The effect of this fix is not yet measured** —
`test_native` exists and is correctly held out, but none of the models reported
in §4 have been evaluated on it, so no claim is made here about how much it
helped.

Training data was **not** touched by calibration or evaluation; `val` is used
for calibration and threshold selection, `test`, `test_native` and the unseen
splits are untouched by both training and calibration.

## 4. Reported metrics

Both models: FP16 autocast inference on Apple M2 (MPS), CLIP feature-cached
during training. Threshold chosen at **FPR ≤ 5%** on `val` (wrongly flagging a
real photo is the costly error) — see [report/model_report.md](report/model_report.md)
for the full derivation.

### 3-stream (spatial + frequency + noise) — highest raw score, but see the caveat
Calibration: T=1.5212, threshold=0.7749. Trained 8 epochs; a fine-tuned
ResNet50 spatial stream concatenated with 10 hand-built FFT/DCT frequency
features and 8 Laplacian/residual noise features (2066-d fused), one head.

| split | ROC-AUC | macro-F1 | accuracy | FPR | recall (AI) | confusion (tn/fp/fn/tp) |
|---|---|---|---|---|---|---|
| test (in-distribution) | 0.9849 | 0.9374 | 0.9374 | 0.0480 | 0.9228 | 9520 / 480 / 772 / 9228 |
| val_unseen_content (proxy: horse+ship held out) | 0.9564 | 0.8630 | 0.8640 | 0.0431 | 0.7729 | 9379 / 422 / 2271 / 7729 |
| **val_unseen_generator (ProGAN, true unseen generator)** | **0.9662** | 0.8869 | 0.8875 | 0.0400 | 0.8150 | 960 / 40 / 185 / 815 |

> **⚠️ Read this number with the caveat below — it is not directly comparable
> to the other two models.** This run selected its best epoch on
> `val_unseen_generator` (`monitor_split: val_unseen_generator`), *the same
> split reported here*. CLIP and ResNet50 both selected on `val_unseen_content`.
> Choosing the checkpoint on the split you then report is a model-selection
> leak, and it biases 0.9662 upward. See **Model selection, and an honest
> correction** below.

### Frozen CLIP ViT-B/16 + linear head (the shipped default)
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

### Robustness to degradation (bonus C)

Full grid in [report/robustness.md](report/robustness.md) — 9 conditions
(JPEG q90/q70/q50/q30, resize ½ and ¼, screenshot-then-share, light edit,
plus a clean reference) across both splits and both models, reproducible with
`python model/robustness.py`. Mean ROC-AUC change across the 8 degraded
conditions:

| model | test | val_unseen_generator | worst single condition |
|---|---|---|---|
| CLIP ViT-B/16 (primary) | −0.0342 | −0.0567 | JPEG q30 on unseen-generator, 0.9412 → 0.8218 |
| **ResNet50 (baseline)** | **−0.0037** | **−0.0080** | resize ¼ on unseen-generator, 0.9502 → 0.9229 |

**ResNet50 is roughly 8× more robust than the frozen-CLIP model** (−0.0059 vs
−0.0455 mean AUC averaged over every split and condition). Resizing is nearly
free for both; JPEG compression is what hurts, and it hurts CLIP far more.

**The more useful finding is *how* they break.** For both models the ranking
largely survives compression — it is the fixed operating point that drifts,
and the two drift in **opposite directions**:

- **CLIP drifts toward "likely real."** At JPEG q30 on the unseen-generator
  split, recall falls 0.716 → 0.131 while FPR falls to 0.004. Compressed AI
  images quietly stop being flagged.
- **ResNet50 drifts toward "likely AI-generated."** At the same setting, recall
  *rises* 0.769 → 0.872 but FPR rises 0.050 → 0.133 — it starts flagging
  compressed real photos, which §4.2 names as the costly error.

So the honest read is that PixelProof's robustness weakness is **calibration
drift under compression, not loss of discrimination**. The concrete mitigation
that follows — fitting the temperature and threshold on a degradation-mixed
`val` rather than a clean one, so the operating point is chosen under the
conditions images actually arrive in — is **not implemented in this build**;
it is what the measurement points at, stated as a next step rather than a
claim.

### Model selection, and an honest correction

The three models were **not selected under the same rule**, and that changes
what can be claimed:

| model | epoch selected on | `val_unseen_generator` AUC | comparable? |
|---|---|---|---|
| CLIP ViT-B/16 | `val_unseen_content` | 0.9410 | ✅ clean |
| ResNet50 | `val_unseen_content` | 0.9519 | ✅ clean |
| 3-stream | **`val_unseen_generator`** | 0.9662 | ❌ selected on the split it reports |

3-stream's training log picks epoch 8 precisely because it maximised
`val_unseen_generator` AUC (0.9662 — the per-epoch values climb 0.9009 → 0.9402
→ 0.9467 → 0.9292 → 0.9491 → 0.9642 → 0.9659 → 0.9661). Reporting that same
split as the headline result is a **model-selection leak**: the number is
optimistically biased and not comparable to the two models selected on a
different split.

**Applying the other models' rule to 3-stream:** its best `val_unseen_content`
epoch is **5** (0.9624), and at epoch 5 its `val_unseen_generator` AUC is
**0.9491**. Under a matched protocol the ranking is therefore:

| ranking by | 1st | 2nd | 3rd |
|---|---|---|---|
| `val_unseen_content` (proxy) | CLIP 0.9723 | ResNet50 0.9685 | 3-stream 0.9624 |
| `val_unseen_generator` (matched selection) | **ResNet50 0.9519** | 3-stream 0.9491 | CLIP 0.9410 |

So **3-stream does not actually beat the ResNet50 baseline once the selection
rule is matched** — it lands between the two. The frequency and noise streams
are not demonstrated to help; the apparent +0.0143 gain was an artifact of
choosing the checkpoint on the evaluation split. Settling this properly needs a
re-run with `--monitor-split val_unseen_content`; that has not been done, so the
honest position is *not demonstrated*, not *shown to fail*.

**What survives, and still matters:** the content-holdout proxy ranks CLIP
**first** and it comes **last** on a genuinely unseen generator — the proxy's
best model is the real test's worst. Generalising to unseen *content* and to
unseen *generators* are different problems, and optimising the first can
actively mislead about the second.

All three comfortably beat the 18–31% accuracy range the problem statement's
own reference material cites for naively-transferred detectors on unseen
generators.

## 5. Architecture, calibration, and limitations

**Architecture.** Three interchangeable backbones, selected with
`--backbone`, behind one head (`Dropout → Linear(→256) → GELU → Linear(→1)`):
- **`3-stream`** (`model/frequency_features.py`): three parallel
  feature extractors fused into one 2066-d vector, then the head.
  - *Spatial* (2048-d): a fine-tuned ResNet50 — content and texture.
  - *Frequency* (10-d): FFT and DCT statistics. Diffusion and GAN upsamplers
    leave periodic spectral signatures that are **generator-specific rather
    than content-specific**, which is the stated reason this stream exists and
    why it was expected to transfer across generators. §11 names
    frequency-domain features as a differentiator; this is that idea, built.
  - *Noise* (8-d): Laplacian and Gaussian-residual statistics — sensor noise
    and edge-coherence patterns that synthesis tends not to reproduce.
  The fusion is deliberately lopsided (2048 learned + 18 hand-built), the
  intent being that the hand-built streams act as a *bias correction* on a CNN
  that would otherwise key on content. **Whether they do is not established** —
  see "Model selection, and an honest correction" in §4. Because the ResNet50
  baseline *is* this model's spatial stream in isolation, a matched-protocol
  re-run would make this a clean ablation of the frequency and noise streams;
  that re-run has not happened, so no benefit is claimed.
- **Secondary:** `vit_base_patch16_clip_224.openai` via timm, **entirely frozen**
  (verified: 197,121 of 85,996,545 params trainable, 0.229%). Chosen because
  frozen-CLIP-features-plus-linear-probe is reported to generalise across
  generators better than end-to-end fine-tuning (Ojha et al. 2023,
  [arXiv:2302.10174](https://arxiv.org/abs/2302.10174)) — a frozen backbone also
  lets features be **precomputed once** (4 fixed augmentation variants per
  image: clean / JPEG-q40 / downscale-upscale / blur) so the head trains in
  under a second per epoch.
- **Baseline:** `resnet50` (torchvision, ImageNet weights), fine-tuned end to
  end, same head. This is also exactly the 3-stream model's spatial stream in
  isolation, which makes the comparison between them a clean ablation of the
  frequency and noise streams rather than a comparison of two unrelated models.

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
  explanation only ever states what it actually measured. **Partially
  addressed** by the native-resolution GenImage data added to `train`/`val`
  (§3), but the metrics reported in §4 predate that addition and no model has
  yet been scored on the held-out `test_native` split — so the resolution
  shortcut is *mitigated in the data*, not *demonstrated fixed in the numbers*.
- **The ProGAN holdout is small** (1,000 images) and single-generator; it is
  real evidence, not a comprehensive cross-generator benchmark.
- **Compression breaks the calibrated threshold, in opposite directions for
  the two models** — now measured rather than assumed; see the robustness
  grid above and [report/robustness.md](report/robustness.md). The grid
  applies one degradation at a time, so compounded handling (compressed *and*
  resized *and* re-saved) is still unmeasured, and it degrades images upscaled
  from 32×32 rather than natively high-resolution ones.
- Accuracy is expected to drop further on generators released after this
  build, and on content categories very unlike CIFAR-10's ten classes.
- Per the challenge's own scope rules, this system makes **no claim about
  people** in an image and is not evaluated on face-swap deepfakes.

## 6. Demo video & deployed app

- **Demo video (~3 min walkthrough):** **[TODO: add hosted link here]** —
  recorded locally but too large (~230 MB) to commit directly to GitHub
  (over the 100 MB per-file limit without Git LFS); upload to Drive/YouTube
  and drop the link in.
- **Full project report:** [Google Drive folder](https://drive.google.com/drive/folders/13LkQxfQDSFrQXZU1hseW03y0Buizw8TS?usp=sharing)
- **Deployed app:** [pixel-proof-plum.vercel.app](https://pixel-proof-plum.vercel.app/)
  — frontend on Vercel, backend (FastAPI + model) on Render. **Known issue:**
  the free-tier hosts don't have enough RAM/GPU to load the CLIP backbone and
  run inference reliably, so image upload and analysis don't fully work on the
  deployed version yet — use the local setup in §2 for a working end-to-end
  demo.

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
    fetch_genimage.py      # streams a capped native-resolution sample of
                           # GenImage (Midjourney + ImageNet reals) via HTTP
                           # range requests — pays for the images pulled, not
                           # the 7.7 GB archive
    normalise_native.py    # normalises the fetched native images
    make_native_testset.py # carves the held-out native split, physically
                           # moving it to data/holdout_native/ so it cannot
                           # leak into train/val → data/csv/test_native.csv
    rebalance_train.py     # oversamples the native data into train/val
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
    robustness.py             # degradation-vs-accuracy grid (bonus C):
                              # 9 conditions x splits x models, at 224px
                              # display scale. --self-check verifies the
                              # degradations; --rebuild-report rewrites the
                              # write-up without re-scoring
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
    robustness.json           # the degradation grid, every cell
    robustness.md             # the degradation-vs-accuracy analysis (bonus C)
    explanation_samples/       # sample /predict outputs + heat-maps
```
