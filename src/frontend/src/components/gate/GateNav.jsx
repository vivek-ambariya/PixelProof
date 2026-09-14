import { useEffect, useRef, useState } from 'react'
import s from './GateNav.module.css'

const LINKS = [
  { id: 'gate-method', label: 'How it works' },
  { id: 'gate-proof', label: 'Proof' },
  { id: 'gate-faq', label: 'FAQ' },
]

export default function GateNav({ onEnter, onNavigate }) {
  const sentinel = useRef(null)
  const [stuck, setStuck] = useState(false)

  // A sentinel at the very top, rather than a scroll listener: it costs nothing
  // per frame and reports the same thing under Lenis or native scrolling.
  useEffect(() => {
    const el = sentinel.current
    if (!el) return
    const io = new IntersectionObserver(
      ([entry]) => setStuck(!entry.isIntersecting),
      { threshold: 0 },
    )
    io.observe(el)
    return () => io.disconnect()
  }, [])

  return (
    <>
      <div ref={sentinel} className={s.sentinel} aria-hidden="true" />
      <header className={`${s.nav} ${stuck ? s.navStuck : ''}`}>
        <div className={s.inner}>
          <div className={s.brand}>
            <div className={s.mark} aria-hidden="true" />
            <span className={s.wordmark}>PIXELPROOF</span>
          </div>

          <nav className={s.links} aria-label="Page sections">
            {LINKS.map((link) => (
              <button
                key={link.id}
                type="button"
                className={s.link}
                onClick={() => onNavigate(link.id)}
              >
                {link.label}
              </button>
            ))}
          </nav>

          <button type="button" className={s.cta} onClick={onEnter}>
            Dive in <span aria-hidden="true">&rarr;</span>
          </button>
        </div>
      </header>
    </>
  )
}
