import { useState } from 'react'
import { motion } from 'framer-motion'
import { useReducedMotion } from '../hooks/useReducedMotion.js'
import s from './Result.module.css'

/** Amber above the threshold, teal below, grey while inconclusive. */
function toneFor(result) {
  if (result.confidence_band === 'inconclusive') return s.unsure
  return result.verdict?.includes('AI') ? s.ai : s.real
}

function barColor(result) {
  if (result.confidence_band === 'inconclusive') return 'var(--pp-muted)'
  return result.verdict?.includes('AI') ? 'var(--pp-amber)' : 'var(--pp-teal)'
}

export default function Result({ result, preview, onReset }) {
  const [wipe, setWipe] = useState(50)
  const reduced = useReducedMotion()

  const pct = (result.probability * 100).toFixed(1)
  const tone = toneFor(result)
  const heat = result.heatmap_base64
    ? `data:image/png;base64,${result.heatmap_base64}`
    : null

  return (
    <div className={s.panel}>
      <div className={s.head}>
        <span className={s.filename}>{result.filename || 'upload'}</span>
        <span>
          {result.image?.width}&times;{result.image?.height}
          {result.elapsed_ms != null && ` · ${(result.elapsed_ms / 1000).toFixed(2)}s`}
        </span>
      </div>

      <div className={s.frame}>
        {preview && <img className={s.base} src={preview} alt="" />}
        <span className={`${s.cornerLabel} ${s.left}`}>ORIGINAL</span>
        {heat && (
          <div className={s.heatWrap} style={{ clipPath: `inset(0 0 0 ${wipe}%)` }}>
            <img className={s.heat} src={heat} alt="" />
            <span className={`${s.cornerLabel} ${s.rightLabel}`}>SALIENCY</span>
            <span className={s.seam} style={{ left: `${wipe}%` }} />
          </div>
        )}
      </div>

      {heat && (
        <div className={s.wipeRow}>
          <span className={s.wipeLabel}>WIPE</span>
          <input
            className={s.slider}
            type="range"
            min="0"
            max="100"
            value={wipe}
            aria-label="Compare the original with the saliency heat-map"
            onChange={(e) => setWipe(Number(e.target.value))}
          />
          <div className={s.legend}>
            <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
              <i className={s.swatch} style={{ background: 'var(--pp-amber)' }} />HIGH
            </span>
            <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
              <i className={s.swatch} style={{ background: 'var(--pp-heat-low)' }} />LOW
            </span>
          </div>
        </div>
      )}

      <div className={s.readout}>
        <div className={s.verdictRow}>
          <div className={`${s.verdict} ${tone}`}>{result.verdict}</div>
          <div className={`${s.pct} ${tone}`}>{pct}%</div>
        </div>

        {/* The confidence bar is the one place framer-motion earns its keep:
            the width animates from 0 so the number lands rather than appears. */}
        <div
          className={s.track}
          role="meter"
          aria-valuenow={Number(pct)}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-label="Probability that this image is AI-generated"
        >
          <motion.div
            className={s.fill}
            style={{ background: barColor(result) }}
            initial={{ width: reduced ? `${pct}%` : 0 }}
            animate={{ width: `${pct}%` }}
            transition={reduced
              ? { duration: 0 }
              : { duration: 1.4, ease: [0.2, 0.8, 0.25, 1] }}
          />
          <span
            className={s.tick}
            style={{ left: `${(result.threshold ?? 0.5) * 100}%` }}
            title={`decision threshold ${result.threshold}`}
          />
        </div>
        <div className={s.scale}>
          <span>0 · CAMERA-CONSISTENT</span>
          <span>{((result.threshold ?? 0.5) * 100).toFixed(0)}</span>
          <span>GENERATOR-CONSISTENT · 1</span>
        </div>

        <div className={s.explainBox}>
          <div className={s.explainLabel}>Explanation</div>
          <p className={s.explainText}>{result.explanation}</p>
        </div>

        {result.confidence_band === 'inconclusive' && (
          <div className={s.banner}>
            Inconclusive range — treat this as a reason to look further, not as an answer.
          </div>
        )}
        {result.calibrated === false && (
          <div className={s.banner}>
            Uncalibrated model — this score is a raw output, not a probability.
          </div>
        )}
        {result.heatmap_error && (
          <div className={s.banner}>
            Saliency map unavailable for this image; the probability still stands.
          </div>
        )}

        <div className={s.actions}>
          <button type="button" className={s.btn} onClick={onReset}>
            Analyse another
          </button>
          <a href="#batch" className={`${s.btn} ${s.btnLink}`}>Batch mode &darr;</a>
        </div>
      </div>
    </div>
  )
}
