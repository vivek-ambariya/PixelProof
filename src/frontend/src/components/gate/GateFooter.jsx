import s from './GateFooter.module.css'

const CLAIMS = [
  'PROBABILITIES REPORTED, NEVER VERDICTS',
  'NO IMAGES RETAINED AFTER SCORING',
  'THRESHOLD SET AT FPR ≤ 5%',
]

// Counts are image totals per split, from report/model_report.md.
const SPLITS = [
  { label: 'Train', count: '24k' },
  { label: 'Validation', count: '8k' },
  { label: 'Test', count: '20k' },
  { label: 'Unseen generator', count: '1k' },
]

const MODEL = [
  'Frozen CLIP ViT-B/16',
  'Linear head · 197,121 params',
  'ResNet50 baseline',
  'Temperature 0.7813',
  'Threshold 0.7833',
]

export default function GateFooter({ onNavigate, onEnter, onTop }) {
  const sections = [
    { label: 'How it works', id: 'gate-method' },
    { label: 'Proof', id: 'gate-proof' },
    { label: 'Position', id: 'gate-about' },
    { label: 'Questions', id: 'gate-faq' },
  ]

  return (
    <footer className={s.footer}>
      <div className={s.inner}>
        <div className={s.columns}>
          <div className={s.col}>
            <div className={s.colLabel}>EXPLORE</div>
            {sections.map((item) => (
              <button
                key={item.id}
                type="button"
                className={s.link}
                onClick={() => onNavigate(item.id)}
              >
                {item.label}
              </button>
            ))}
          </div>

          <div className={s.col}>
            <div className={s.colLabel}>THE TOOL</div>
            <button type="button" className={s.link} onClick={onEnter}>
              Open detector
            </button>
            <button type="button" className={s.link} onClick={onEnter}>
              Batch mode
            </button>
            <a className={s.link} href="/docs">API docs</a>
            <a className={s.link} href="/api/health">Health</a>
            <a
              className={s.link}
              href="https://github.com"
              target="_blank"
              rel="noreferrer noopener"
            >
              Source
            </a>
          </div>

          <div className={s.col}>
            <div className={s.colLabel}>MODEL</div>
            {MODEL.map((line) => (
              <span key={line} className={s.spec}>{line}</span>
            ))}
          </div>

          <div className={s.col}>
            <div className={s.colLabel}>EVALUATION</div>
            {SPLITS.map((split) => (
              <span key={split.label} className={s.spec}>
                {split.label} <em className={s.count}>{split.count}</em>
              </span>
            ))}
          </div>
        </div>

        <p className={s.note}>
          Calibration and decision threshold are derived from a validation split
          the model never trained on. Every score reports the backbone that
          produced it, and unseen-generator results are published alongside the
          in-distribution ones.
        </p>

        <div className={s.claims}>
          {CLAIMS.map((claim) => (
            <span key={claim}>{claim}</span>
          ))}
        </div>

        <div className={s.wordmarkWrap} aria-hidden="true">
          <span className={s.wordmark}>PIXELPROOF</span>
        </div>

        <div className={s.bottom}>
          <span>© 2026 PIXELPROOF · MEDIA FORENSICS · SIH 2026 · SIGNALSCOPE</span>
          <button type="button" className={s.top} onClick={onTop}>
            BACK TO TOP <span aria-hidden="true">&uarr;</span>
          </button>
        </div>
      </div>
    </footer>
  )
}
