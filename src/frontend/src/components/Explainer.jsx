import { useEffect, useRef } from 'react'
import { gsap } from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'
import { useScrubEnabled } from '../hooks/useReducedMotion.js'
import s from './Explainer.module.css'

// Registered once at module scope. Registering inside the effect risks the
// tween being created before the plugin is known, in which case GSAP silently
// drops the scrollTrigger config and nothing pins -- exactly the failure the
// design mockup shipped with.
gsap.registerPlugin(ScrollTrigger)

const STEPS = [
  {
    kicker: 'STEP 01',
    title: 'Frequency artefacts',
    body: 'Generators build images by repeatedly upsampling a small latent grid. That process tends to leave faint, regularly spaced peaks in the frequency domain that a lens and sensor do not produce. The peaks are invisible to the eye and survive moderate resizing, though not heavy re-compression.',
    fig: 'FIG. 1 — LOG-MAGNITUDE SPECTRUM',
    caption: <>PERIODIC PEAKS OFF-AXIS &rarr; <span className={s.hot}>UPSAMPLING LATTICE</span></>,
  },
  {
    kicker: 'STEP 02',
    title: 'Texture and grain statistics',
    body: 'Real sensor noise scales with the amount of light hitting each photosite, so variance climbs with brightness and differs per colour channel. Synthesised grain is usually applied more evenly. PixelProof measures noise variance against local luminance and looks for that relationship to be missing.',
    fig: 'FIG. 2 — NOISE VARIANCE BY LUMINANCE',
    caption: <>&sigma;&sup2; INDEPENDENT OF LUMINANCE &rarr; <span className={s.hot}>SYNTHETIC TEXTURE</span></>,
  },
  {
    kicker: 'STEP 03',
    title: 'Lighting and geometry',
    body: 'Highlights, shadows and reflections in a photograph all resolve to the same light sources. Generated scenes often disagree by a few degrees between objects, and vanishing lines drift. This signal is the weakest of the three and the easiest to trip on a genuinely odd photograph, so it is weighted lowest.',
    fig: 'FIG. 3 — ESTIMATED LIGHT DIRECTION',
    caption: <>SHADING NORMALS DISAGREE &rarr; <span className={s.hot}>INCONSISTENT SCENE</span></>,
  },
]

function FigureBody({ index }) {
  if (index === 0) {
    return (
      <div className={s.canvas}>
        <div className={s.spectrum} />
        <div className={s.lattice} data-lattice />
        <div className={s.axisV} />
        <div className={s.axisH} />
      </div>
    )
  }
  if (index === 1) {
    return (
      <div className={s.canvas}>
        <svg className={s.noiseSvg} viewBox="0 0 300 180" preserveAspectRatio="none">
          <path d="M10,160 C70,150 140,120 290,30" fill="none"
                stroke="#4fd8c4" strokeWidth="1.5" data-sensor-line />
          <path d="M10,120 C90,118 180,116 290,113" fill="none"
                stroke="#e0a13a" strokeWidth="1.5" strokeDasharray="4 3" />
        </svg>
        <span className={`${s.legendLine} ${s.legendSensor}`}>SENSOR — RISES WITH SIGNAL</span>
        <span className={`${s.legendLine} ${s.legendGen}`}>GENERATED — NEARLY FLAT</span>
      </div>
    )
  }
  return (
    <div className={s.canvas}>
      <svg className={s.lightSvg} viewBox="0 0 300 180">
        <ellipse cx="95" cy="128" rx="34" ry="9" fill="#4fd8c4" opacity="0.28" />
        <ellipse cx="205" cy="128" rx="34" ry="9" fill="#e0a13a" opacity="0.28" />
        <circle cx="95" cy="104" r="26" fill="none" stroke="#8a807a" strokeWidth="1" />
        <circle cx="205" cy="104" r="26" fill="none" stroke="#8a807a" strokeWidth="1" />
        <line x1="95" y1="104" x2="48" y2="52" stroke="#4fd8c4" strokeWidth="1.5" />
        <line x1="205" y1="104" x2="242" y2="46" stroke="#e0a13a" strokeWidth="1.5" />
        <text x="150" y="40" fill="#e0a13a" fontSize="12"
              fontFamily="JetBrains Mono, monospace" textAnchor="middle">&#916; 24&deg;</text>
      </svg>
    </div>
  )
}

export default function Explainer() {
  const scrub = useScrubEnabled()
  const rootRef = useRef(null)
  const figureRefs = useRef([])
  const stepRefs = useRef([])

  useEffect(() => {
    if (!scrub) return
    const figures = figureRefs.current.filter(Boolean)
    const steps = stepRefs.current.filter(Boolean)
    if (figures.length !== STEPS.length || steps.length !== STEPS.length) return

    const ctx = gsap.context(() => {
      // Each step owns a trigger that cross-fades to its figure. gsap.set on
      // enter (rather than a tween per figure) keeps the states mutually
      // exclusive when the visitor scrolls fast or jumps via a hash link.
      const show = (active) => {
        figures.forEach((el, i) => {
          gsap.to(el, {
            opacity: i === active ? 1 : 0,
            y: i === active ? 0 : 12,
            duration: 0.4,
            overwrite: 'auto',
          })
        })
        const lattice = figures[0]?.querySelector('[data-lattice]')
        if (lattice) {
          gsap.to(lattice, { opacity: active === 0 ? 1 : 0, duration: 0.6 })
        }
      }

      steps.forEach((step, i) => {
        ScrollTrigger.create({
          trigger: step,
          start: 'top center',
          end: 'bottom center',
          onEnter: () => show(i),
          onEnterBack: () => show(i),
        })
      })
      show(0)
    }, rootRef)

    return () => ctx.revert()
  }, [scrub])

  return (
    <section className="pp-section" ref={rootRef}>
      <div className="pp-kicker">02 · Method</div>
      <h2 className="pp-h2" style={{ maxWidth: '20ch' }}>How it reads an image</h2>
      <p style={{
        margin: '18px 0 0', maxWidth: '52ch',
        color: 'var(--pp-text-3)', fontSize: 16, lineHeight: 1.6,
      }}>
        Three independent signal families are scored separately, then combined.
        Each is weak alone; the combination is what carries information.
      </p>

      <div className={s.grid}>
        <div className={s.stickyCol}>
          <div className={s.stage}>
            {STEPS.map((step, i) => (
              <div
                key={step.title}
                ref={(el) => { figureRefs.current[i] = el }}
                className={`${s.figure} ${i === 0 ? s.figureFirst : ''}`}
              >
                <div className={s.figLabel}>{step.fig}</div>
                <FigureBody index={i} />
                <div className={s.figCaption}>{step.caption}</div>
              </div>
            ))}
          </div>
        </div>

        <div>
          {STEPS.map((step, i) => (
            <div
              key={step.title}
              ref={(el) => { stepRefs.current[i] = el }}
              className={s.step}
            >
              {/* Inline copy of the figure, shown only under 768px. */}
              <div className={`${s.figure} ${s.inlineFigure}`}>
                <div className={s.figLabel}>{step.fig}</div>
                <FigureBody index={i} />
              </div>
              <div className={s.stepKicker}>{step.kicker}</div>
              <h3 className={s.stepTitle}>{step.title}</h3>
              <p className={s.stepBody}>{step.body}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
