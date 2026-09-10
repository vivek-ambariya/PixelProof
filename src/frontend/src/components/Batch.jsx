import { useCallback, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { predictImage, rejectReason } from '../lib/api.js'
import { useReducedMotion } from '../hooks/useReducedMotion.js'
import s from './Batch.module.css'

/**
 * Batch mode. Files are scored one request at a time against the same
 * POST /predict the single-image flow uses, so there is no second contract to
 * keep in sync. Rows appear immediately as "queued" and fill in as each
 * response lands, then sort by likelihood so the images worth a second look
 * sit at the top.
 */
export default function Batch() {
  const [rows, setRows] = useState([])
  const [running, setRunning] = useState(false)
  const inputRef = useRef(null)
  const reduced = useReducedMotion()

  const onPick = useCallback(async (e) => {
    const files = Array.from(e.target.files || [])
    e.target.value = ''
    if (!files.length) return

    const seeded = files.map((f, i) => ({
      id: `${Date.now()}-${i}-${f.name}`,
      name: f.name,
      status: rejectReason(f) ? 'skipped' : 'queued',
      note: rejectReason(f) || '',
      probability: null,
      verdict: null,
      width: null,
      height: null,
    }))
    setRows((prev) => [...prev, ...seeded])
    setRunning(true)

    for (let i = 0; i < files.length; i += 1) {
      const seed = seeded[i]
      if (seed.status === 'skipped') continue
      try {
        const data = await predictImage(files[i])
        setRows((prev) => prev.map((r) => (r.id === seed.id ? {
          ...r,
          status: 'done',
          probability: data.probability,
          verdict: data.verdict,
          band: data.confidence_band,
          width: data.image?.width,
          height: data.image?.height,
        } : r)))
      } catch (err) {
        setRows((prev) => prev.map((r) => (r.id === seed.id ? {
          ...r, status: 'skipped', note: err.message,
        } : r)))
      }
    }
    setRunning(false)
  }, [])

  // Scored rows first, highest probability at the top; skipped sink to the end.
  const sorted = [...rows].sort((a, b) => {
    if (a.probability == null && b.probability == null) return 0
    if (a.probability == null) return 1
    if (b.probability == null) return -1
    return b.probability - a.probability
  })

  const skipped = rows.filter((r) => r.status === 'skipped').length

  const toneFor = (r) => {
    if (r.status !== 'done') return s.skipped
    if (r.band === 'inconclusive') return s.unsure
    return r.verdict?.includes('AI') ? s.ai : s.real
  }
  const readFor = (r) => {
    if (r.status === 'skipped') return 'SKIPPED'
    if (r.status !== 'done') return 'SCORING…'
    if (r.band === 'inconclusive') return 'UNCERTAIN'
    return r.verdict?.includes('AI') ? 'LIKELY AI' : 'LIKELY REAL'
  }

  return (
    <section className="pp-section" id="batch">
      <div className={s.head}>
        <div>
          <div className="pp-kicker">04 · Batch</div>
          <h2 className="pp-h2">Batch mode</h2>
          <p className={s.intro}>
            Add several files to score a set at once. Sorted by likelihood, so
            the images worth a second look sit at the top.
          </p>
        </div>
        <div>
          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            multiple
            onChange={onPick}
            className="pp-visually-hidden"
            tabIndex={-1}
            aria-hidden="true"
          />
          <button
            type="button"
            className={s.add}
            disabled={running}
            onClick={() => inputRef.current?.click()}
          >
            {running ? 'Scoring…' : 'Add files'}
          </button>
        </div>
      </div>

      <div className={s.table}>
        <div className={s.header}>
          <span>File</span>
          <span>Dimensions</span>
          <span>Status</span>
          <span>P(generated)</span>
          <span className={s.right}>Read</span>
        </div>

        {sorted.length === 0 && (
          <div className={s.empty}>NO FILES YET — ADD SOME TO SCORE A SET</div>
        )}

        {/* Staggered entrance: each row is delayed by its index, so a batch
            resolves as a cascade rather than all at once. */}
        <AnimatePresence initial={false}>
          {sorted.map((r, i) => (
            <motion.div
              key={r.id}
              className={s.row}
              layout={!reduced}
              initial={reduced ? { opacity: 1 } : { opacity: 0, y: 8 }}
              animate={reduced
                ? { opacity: 1 }
                : { opacity: 1, y: 0, transition: { delay: Math.min(i * 0.05, 0.6) } }}
              exit={{ opacity: 0 }}
              transition={{ duration: reduced ? 0 : 0.35 }}
            >
              <span className={s.name} title={r.name}>{r.name}</span>
              <span className={s.dim}>
                {r.width ? `${r.width}×${r.height}` : '—'}
              </span>
              <span className={s.dim} title={r.note}>
                {r.status === 'done' ? 'scored'
                  : r.status === 'skipped' ? (r.note ? 'not scored' : 'skipped')
                  : 'queued'}
              </span>
              <span className={s.meter}>
                <i className={s.meterTrack}>
                  {r.probability != null && (
                    <motion.b
                      className={s.meterFill}
                      style={{
                        background: r.band === 'inconclusive'
                          ? 'var(--pp-muted)'
                          : r.verdict?.includes('AI')
                            ? 'var(--pp-amber)'
                            : 'var(--pp-teal)',
                      }}
                      initial={{ width: reduced ? `${r.probability * 100}%` : 0 }}
                      animate={{ width: `${r.probability * 100}%` }}
                      transition={{ duration: reduced ? 0 : 0.9, ease: 'easeOut' }}
                    />
                  )}
                </i>
                <em className={`${s.value} ${toneFor(r)}`}>
                  {r.probability != null ? r.probability.toFixed(2) : 'n/a'}
                </em>
              </span>
              <span className={`${s.right} ${toneFor(r)}`}>{readFor(r)}</span>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>

      {rows.length > 0 && (
        <div className={s.footNote}>
          {rows.length} FILE{rows.length === 1 ? '' : 'S'}
          {skipped > 0 && ` · ${skipped} SKIPPED`}
          {' '}· SCORES ARE LIKELIHOODS, NOT CLASSIFICATIONS
        </div>
      )}
    </section>
  )
}
