import { motion } from 'framer-motion'
import Dropzone from './Dropzone.jsx'
import Result from './Result.jsx'
import { useReducedMotion } from '../hooks/useReducedMotion.js'
import s from './Hero.module.css'

const STEPS = [
  'decoding · normalising',
  'frozen backbone features',
  'detector head',
  'grad-cam saliency',
]

function Loading() {
  const reduced = useReducedMotion()
  return (
    <div className={s.loading}>
      {!reduced && <div className={s.sweep} aria-hidden="true" />}
      <div className={s.loadingLabel}>ANALYSING · POST /predict</div>
      <div className={s.steps} aria-live="polite">
        {STEPS.map((t, i) => (
          <div key={t} className={i === STEPS.length - 1 ? s.stepActive : undefined}>
            &rarr; {t}
          </div>
        ))}
      </div>
      <div className={s.bar}>
        <motion.div
          className={s.barFill}
          initial={{ width: '8%' }}
          animate={{ width: '92%' }}
          transition={reduced ? { duration: 0 } : { duration: 1.4, ease: 'easeOut' }}
        />
      </div>
    </div>
  )
}

function ErrorPanel({ error, onRetry, onReset }) {
  // A rejected file is the user's to fix; a failed request is ours to retry.
  const rejected = error.kind === 'rejected'
  return (
    <div className={s.error} role="alert">
      <div className={s.errorCode}>
        {rejected ? 'FILE NOT ACCEPTED' : `INFERENCE FAILED${error.status ? ` · ${error.status}` : ''}`}
      </div>
      <div className={s.errorTitle}>
        {rejected ? 'That file could not be read as an image.' : 'The model endpoint did not respond.'}
      </div>
      <p className={s.errorBody}>
        {error.message}
        {' '}Nothing was scored and nothing was stored. The image stayed in your browser.
      </p>
      <div className={s.errorActions}>
        {!rejected && (
          <button type="button" className={s.primary} onClick={onRetry}>Retry</button>
        )}
        <button type="button" className={s.ghost} onClick={onReset}>
          Choose another file
        </button>
      </div>
    </div>
  )
}

export default function Hero({
  phase, result, error, preview, onFile, onRetry, onReset,
}) {
  return (
    <section className={s.hero}>
      <div className={s.wrap}>
        {phase === 'idle' && <Dropzone onFile={onFile} />}
        {phase === 'loading' && <Loading />}
        {phase === 'error' && (
          <ErrorPanel error={error} onRetry={onRetry} onReset={onReset} />
        )}
        {phase === 'result' && result && (
          <Result result={result} preview={preview} onReset={onReset} />
        )}
      </div>
    </section>
  )
}
