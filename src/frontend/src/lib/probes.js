// Real PixelProof scores for the twenty probe images in public/intro/.
//
// Every one of these is machine-generated: they are the ProGAN samples from
// data/progan_native, the unseen-generator split, re-encoded as JPEG for the
// web. Ground truth is therefore AI for all twenty, which makes the strip a
// live read-out of the miss rate section 03 argues about rather than a
// showcase -- the scores below are whatever the shipped checkpoint returned,
// misses included.
//
// Regenerate with:
//   python model/predict.py --input src/frontend/public/intro --out preds.csv

export const PROBE_THRESHOLD = 0.7833248972892761

/** Calibrated P(AI-generated) per probe, from the frozen CLIP ViT-B/16 head. */
export const PROBES = [
  { src: '/intro/g01.jpg', prob: 0.9157, band: 'clear' },
  { src: '/intro/g02.jpg', prob: 0.2373, band: 'clear' },
  { src: '/intro/g03.jpg', prob: 0.766, band: 'clear' },
  { src: '/intro/g04.jpg', prob: 0.4315, band: 'inconclusive' },
  { src: '/intro/g05.jpg', prob: 0.9972, band: 'clear' },
  { src: '/intro/g06.jpg', prob: 0.9804, band: 'clear' },
  { src: '/intro/g07.jpg', prob: 0.7286, band: 'clear' },
  { src: '/intro/g08.jpg', prob: 0.9943, band: 'clear' },
  { src: '/intro/g09.jpg', prob: 0.1021, band: 'clear' },
  { src: '/intro/g10.jpg', prob: 0.9976, band: 'clear' },
  { src: '/intro/g11.jpg', prob: 0.9994, band: 'clear' },
  { src: '/intro/g12.jpg', prob: 0.0056, band: 'clear' },
  { src: '/intro/g13.jpg', prob: 0.8872, band: 'clear' },
  { src: '/intro/g14.jpg', prob: 0.002, band: 'clear' },
  { src: '/intro/g15.jpg', prob: 0.226, band: 'clear' },
  { src: '/intro/g16.jpg', prob: 0.0295, band: 'clear' },
  { src: '/intro/g17.jpg', prob: 0.9131, band: 'clear' },
  { src: '/intro/g18.jpg', prob: 0.0077, band: 'clear' },
  { src: '/intro/g19.jpg', prob: 0.0497, band: 'clear' },
  { src: '/intro/g20.jpg', prob: 0.6971, band: 'clear' },
]

/** Flagged at the calibrated threshold. Derived, never hand-counted. */
export const PROBE_CAUGHT = PROBES.filter((p) => p.prob >= PROBE_THRESHOLD).length
export const PROBE_TOTAL = PROBES.length
