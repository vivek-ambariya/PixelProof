import { useEffect, useMemo, useRef, useState } from 'react'
import { motion, useMotionValue, useSpring } from 'framer-motion'
import s from './IntroSequence.module.css'

const TOTAL = 20
const CARD_W = 60
const CARD_H = 85

// Wheel delta needed to take the ring all the way out to the arc.
const MAX_SCROLL = 1400

// Samples from data/progan_native — the unseen-generator split. Every one of
// these was machine-generated, which is the point the copy makes.
const IMAGES = Array.from(
  { length: TOTAL },
  (_, i) => `/intro/g${String(i + 1).padStart(2, '0')}.jpg`,
)

// The entrance plays on its own; everything after the ring is scroll-driven.
const ENTRANCE = [
  { phase: 'line', at: 350 },
  { phase: 'circle', at: 1700 },
]

const lerp = (a, b, t) => a * (1 - t) + b * t
const clamp01 = (v) => Math.min(Math.max(v, 0), 1)

function useStageSize(ref) {
  const [size, setSize] = useState({ width: 0, height: 0 })
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const observer = new ResizeObserver(([entry]) => {
      setSize({ width: entry.contentRect.width, height: entry.contentRect.height })
    })
    observer.observe(el)
    setSize({ width: el.offsetWidth, height: el.offsetHeight })
    return () => observer.disconnect()
  }, [ref])
  return size
}

function circleAt(i, size) {
  const radius = Math.min(Math.min(size.width, size.height) * 0.35, 320)
  const angle = (i / TOTAL) * 360
  const rad = (angle * Math.PI) / 180
  return {
    x: Math.cos(rad) * radius,
    y: Math.sin(rad) * radius,
    rotate: angle + 90,
    scale: 1,
  }
}

// A wide, convex-up sweep across the lower half, the cards fanning out along
// its circumference. The spread stays shallow enough that the end cards only
// just bleed off the sides rather than flying out of frame.
function arcAt(i, size) {
  const isMobile = size.width < 768
  const radius = Math.min(size.width, size.height * 1.5) * (isMobile ? 1.0 : 0.84)
  const apexY = size.height * (isMobile ? 0.1 : 0.07)
  const spread = isMobile ? 78 : 92
  const angle = -90 - spread / 2 + i * (spread / (TOTAL - 1))
  const rad = (angle * Math.PI) / 180
  return {
    x: Math.cos(rad) * radius,
    y: Math.sin(rad) * radius + apexY + radius,
    rotate: angle + 90,
    scale: isMobile ? 1.2 : 1.5,
  }
}

function Card({ src, target, index }) {
  return (
    <motion.div
      className={s.card}
      animate={{
        x: target.x,
        y: target.y,
        rotate: target.rotate,
        scale: target.scale,
        opacity: target.opacity,
      }}
      transition={{ type: 'spring', stiffness: 40, damping: 15 }}
      style={{ width: CARD_W, height: CARD_H }}
    >
      <motion.div
        className={s.flipper}
        whileHover={{ rotateY: 180 }}
        transition={{ type: 'spring', stiffness: 260, damping: 20 }}
      >
        <div className={s.face}>
          <img src={src} alt="" className={s.img} loading="eager" />
        </div>
        <div className={`${s.face} ${s.back}`}>
          <span className={s.backLabel}>GENERATED</span>
          <span className={s.backIndex}>{String(index + 1).padStart(2, '0')}</span>
        </div>
      </motion.div>
    </motion.div>
  )
}

export default function IntroSequence({ onDone }) {
  const stageRef = useRef(null)
  const size = useStageSize(stageRef)
  const [phase, setPhase] = useState('scatter')
  const [morph, setMorph] = useState(0)
  const [fading, setFading] = useState(false)

  const scrolled = useRef(0)
  const morphValue = useMotionValue(0)
  const smoothMorph = useSpring(morphValue, { stiffness: 50, damping: 20 })

  const scatter = useMemo(
    () =>
      Array.from({ length: TOTAL }, () => ({
        x: (Math.random() - 0.5) * 1500,
        y: (Math.random() - 0.5) * 1000,
        rotate: (Math.random() - 0.5) * 180,
        scale: 0.6,
        opacity: 0,
      })),
    [],
  )

  useEffect(() => {
    const timers = ENTRANCE.map(({ phase: next, at }) =>
      setTimeout(() => setPhase(next), at),
    )
    return () => timers.forEach(clearTimeout)
  }, [])

  useEffect(() => smoothMorph.on('change', setMorph), [smoothMorph])

  // Scroll only starts driving once the ring has formed.
  const scrollable = phase === 'circle'

  useEffect(() => {
    const el = stageRef.current
    if (!el || !scrollable) return

    const push = (delta) => {
      scrolled.current = Math.min(Math.max(scrolled.current + delta, 0), MAX_SCROLL)
      morphValue.set(scrolled.current / MAX_SCROLL)
    }

    // preventDefault keeps the app mounted behind this overlay from scrolling.
    const onWheel = (e) => {
      e.preventDefault()
      push(e.deltaY)
    }

    let lastTouch = 0
    const onTouchStart = (e) => { lastTouch = e.touches[0].clientY }
    const onTouchMove = (e) => {
      e.preventDefault()
      const y = e.touches[0].clientY
      push((lastTouch - y) * 2.2)
      lastTouch = y
    }

    el.addEventListener('wheel', onWheel, { passive: false })
    el.addEventListener('touchstart', onTouchStart, { passive: false })
    el.addEventListener('touchmove', onTouchMove, { passive: false })
    return () => {
      el.removeEventListener('wheel', onWheel)
      el.removeEventListener('touchstart', onTouchStart)
      el.removeEventListener('touchmove', onTouchMove)
    }
  }, [scrollable, morphValue])

  // Scrolling the morph to the end is what opens the tool. The hand-off timer
  // is kept in its own effect: if it shared one with the `fading` check, that
  // effect's cleanup would clear the timeout the moment `fading` flipped.
  useEffect(() => {
    if (morph >= 0.98) setFading(true)
  }, [morph])

  useEffect(() => {
    if (!fading) return
    const timer = setTimeout(onDone, 620)
    return () => clearTimeout(timer)
  }, [fading, onDone])

  const dark = morph > 0.45

  const targetFor = (i) => {
    if (phase === 'scatter') return scatter[i]
    if (phase === 'line') {
      const spacing = 70
      return {
        x: i * spacing - (TOTAL * spacing) / 2,
        y: 0,
        rotate: 0,
        scale: 1,
        opacity: 1,
      }
    }
    const from = circleAt(i, size)
    const to = arcAt(i, size)
    return {
      x: lerp(from.x, to.x, morph),
      y: lerp(from.y, to.y, morph),
      rotate: lerp(from.rotate, to.rotate, morph),
      scale: lerp(from.scale, to.scale, morph),
      opacity: 1,
    }
  }

  return (
    <motion.div
      className={s.overlay}
      ref={stageRef}
      // The ground crosses from the landing page's cream to the app's black, so
      // the handoff reads as one move rather than two screens.
      animate={{
        backgroundColor: dark ? '#0b0908' : '#ebe9e4',
        opacity: fading ? 0 : 1,
      }}
      transition={{
        backgroundColor: { duration: 0.9, ease: [0.12, 0.23, 0.5, 1] },
        opacity: { duration: 0.55, ease: 'easeOut' },
      }}
    >
      <motion.button
        type="button"
        className={s.skip}
        onClick={onDone}
        animate={{ color: dark ? '#8a807a' : '#6a6a6a' }}
      >
        SKIP <span aria-hidden="true">&rarr;</span>
      </motion.button>

      <div className={s.stage}>
        {IMAGES.map((src, i) => (
          <Card key={src} src={src} index={i} target={targetFor(i)} />
        ))}
      </div>

      <motion.div
        className={s.centreCopy}
        animate={{
          opacity: phase === 'circle' ? clamp01(1 - morph * 3) : 0,
          y: phase === 'circle' ? 0 : 12,
        }}
        transition={{ duration: 0.5 }}
      >
        <p className={s.kicker}>UNSEEN-GENERATOR SPLIT · PROGAN</p>
        <h2 className={s.headline}>Every one of these was generated.</h2>
        <motion.p
          className={s.hint}
          animate={{ opacity: morph > 0.04 ? 0 : 1 }}
          transition={{ duration: 0.3 }}
        >
          SCROLL TO CONTINUE <span aria-hidden="true">&darr;</span>
        </motion.p>
      </motion.div>

      <motion.div
        className={s.topCopy}
        animate={{
          opacity: clamp01((morph - 0.38) / 0.3),
          y: lerp(16, 0, clamp01((morph - 0.38) / 0.3)),
        }}
        transition={{ duration: 0.3 }}
      >
        <h2 className={s.headlineLight}>Now tell them from the real ones.</h2>
        <p className={s.sub}>Keep scrolling to open the detector.</p>
      </motion.div>

      <div className={s.progress} aria-hidden="true">
        <motion.div className={s.progressFill} style={{ scaleX: smoothMorph }} />
      </div>
    </motion.div>
  )
}
