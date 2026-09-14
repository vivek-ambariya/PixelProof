import s from './GateHero.module.css'

const STATS = [
  { value: '0.985', label: 'ROC-AUC ON HELD-OUT TEST' },
  { value: '3', label: 'INDEPENDENT SIGNAL FAMILIES' },
  { value: '0', label: 'IMAGES RETAINED' },
]

export default function GateHero({ onEnter, onNavigate }) {
  return (
    <section className={s.hero}>
      <div className={s.inner}>
        <p className={s.kicker} data-hero-reveal>
          MEDIA FORENSICS · SIH 2026 BUILD
        </p>

        <h1 className={s.title} data-display data-hero-reveal>
          Was this made<br />
          by a camera,<br />
          or by a model?
        </h1>

        <p className={s.lede} data-hero-reveal>
          PixelProof estimates the likelihood that an image was synthetically
          generated, then shows you where in the frame that estimate comes from.
          It reports probabilities, not verdicts.
        </p>

        <div className={s.actions} data-hero-reveal>
          <button type="button" className={s.cta} onClick={onEnter}>
            Dive in <span aria-hidden="true">&rarr;</span>
          </button>
          <button
            type="button"
            className={s.secondary}
            onClick={() => onNavigate('gate-method')}
          >
            How it reads an image <span aria-hidden="true">&darr;</span>
          </button>
        </div>

        <dl className={s.stats} data-hero-reveal>
          {STATS.map((stat) => (
            <div key={stat.label} className={s.stat}>
              <dt className={s.statValue}>{stat.value}</dt>
              <dd className={s.statLabel}>{stat.label}</dd>
            </div>
          ))}
        </dl>
      </div>
    </section>
  )
}
