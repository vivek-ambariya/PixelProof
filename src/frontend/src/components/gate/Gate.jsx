import { useCallback, useEffect, useRef } from 'react'
import Lenis from 'lenis'
import { ScrollTrigger } from 'gsap/ScrollTrigger'
import GateNav from './GateNav.jsx'
import GateHero from './GateHero.jsx'
import GateHowItWorks from './GateHowItWorks.jsx'
import GateProof from './GateProof.jsx'
import GateAbout from './GateAbout.jsx'
import GateFaq from './GateFaq.jsx'
import Footer from '../Footer.jsx'
import { useReducedMotion } from '../../hooks/useReducedMotion.js'
import { useGateReveal } from './useGateReveal.js'
import s from './Gate.module.css'

/**
 * The landing page shown before the tool itself.
 *
 * Owns the Lenis instance, which exists only while this component is mounted —
 * diving in destroys it so the app underneath scrolls natively.
 */
export default function Gate({ onEnter }) {
  const rootRef = useRef(null)
  const lenisRef = useRef(null)
  const reduced = useReducedMotion()

  useGateReveal(rootRef)

  // The app paints html and body dark; the gate is light, so the ground has to
  // change too or overscroll shows the wrong colour behind the page.
  useEffect(() => {
    document.documentElement.classList.add('pp-gate-active')
    return () => document.documentElement.classList.remove('pp-gate-active')
  }, [])

  useEffect(() => {
    if (reduced) return

    const lenis = new Lenis({ lerp: 0.085, smoothWheel: true })
    lenisRef.current = lenis
    // ScrollTrigger reads scroll position from the browser; Lenis moves it on
    // its own schedule, so it has to be told when to re-measure.
    lenis.on('scroll', ScrollTrigger.update)

    let frame = requestAnimationFrame(function raf(time) {
      lenis.raf(time)
      frame = requestAnimationFrame(raf)
    })

    return () => {
      cancelAnimationFrame(frame)
      lenis.destroy()
      lenisRef.current = null
    }
  }, [reduced])

  const scrollTo = useCallback((id) => {
    const el = document.getElementById(id)
    if (!el) return
    if (lenisRef.current) lenisRef.current.scrollTo(el, { offset: -72 })
    else el.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }, [])

  return (
    <div className={s.gate} ref={rootRef}>
      <GateNav onEnter={onEnter} onNavigate={scrollTo} />
      <GateHero onEnter={onEnter} onNavigate={scrollTo} />
      <GateHowItWorks />
      <GateProof />
      <GateAbout />
      <GateFaq />

      <section className={s.closing} data-reveal-group>
        <div className={s.closingInner}>
          <p className={s.closingKicker} data-reveal>READY WHEN YOU ARE</p>
          <h2 className={s.closingTitle} data-display data-reveal>
            Try it on an image<br />you are unsure about.
          </h2>
          <button type="button" className={s.closingCta} onClick={onEnter} data-reveal>
            Dive in <span aria-hidden="true">&rarr;</span>
          </button>
          <p className={s.closingNote} data-reveal>
            No account, no upload kept, no tracking.
          </p>
        </div>
      </section>

      <div className="pp-shell">
        <Footer />
      </div>
    </div>
  )
}
