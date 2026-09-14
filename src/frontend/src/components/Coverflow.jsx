import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'
import { useReducedMotion } from '../hooks/useReducedMotion.js'
import s from './Coverflow.module.css'

const useIsoLayoutEffect = typeof window !== 'undefined' ? useLayoutEffect : useEffect

/**
 * A 3D coverflow ring.
 *
 * Slides are `{ src, alt }` at minimum; anything else on the object is handed
 * straight back to `renderCaption` so the caption stays this component's
 * caller's business rather than a fixed schema.
 *
 * The ring loops without cloning nodes: each card's offset from centre is
 * folded into the shorter way round, so a card that walks off the right edge
 * is the same DOM node that reappears on the left.
 */
export default function Coverflow({
  slides,
  /** Degrees the first neighbour tilts. */
  rotate = 44,
  /** How far the first neighbour recedes, as a fraction of card width. */
  depth = 0.6,
  /** Viewer distance as a multiple of card width - smaller is a wider lens. */
  perspective = 3,
  /** Exponent on distance. Below 1 the rake eases off as cards travel out. */
  falloff = 0.56,
  /** Opacity lost per step from the centre. */
  fade = 0.1,
  /** Any CSS length. Everything else derives from it, so the rake scales. */
  cardWidth = 'clamp(148px, 22vw, 260px)',
  /** Space between cards, as a fraction of card width. */
  gap = 0.05,
  loop = true,
  showPagination = true,
  showNavigation = true,
  /** Names the carousel for assistive tech. */
  label = 'Cover carousel',
  renderCaption,
  className = '',
}) {
  const count = slides.length
  const reduced = useReducedMotion()

  const frameRef = useRef(null)
  const cardRefs = useRef([])
  /** Fractional card index at the centre. The single source of truth. */
  const posRef = useRef(0)
  /** Where the current settle is headed. Stepping off `pos` instead would
      swallow a keypress that lands mid-flight, before the round-off moves. */
  const targetRef = useRef(0)
  const widthRef = useRef(0)
  const rafRef = useRef(null)
  const dragRef = useRef(null)

  const [selected, setSelected] = useState(0)

  /** Nearest whole card, folded back into 0..count-1. */
  const indexAt = useCallback(
    (pos) => ((Math.round(pos) % count) + count) % count,
    [count],
  )

  // Paint straight to the DOM. Sixty state updates a second would re-render
  // every card for numbers React never needs to see.
  const paint = useCallback(() => {
    const width = widthRef.current
    if (!width) return
    const pitch = width * (1 + gap)
    const pos = posRef.current

    cardRefs.current.forEach((card, index) => {
      if (!card) return

      // Fold the distance into the shorter way round the ring. This is the
      // whole looping mechanism - no cloned nodes, no shuffling the DOM.
      let offset = index - pos
      if (loop) {
        offset = ((offset % count) + count) % count
        if (offset > count / 2) offset -= count
      }

      const distance = Math.abs(offset)
      // Both the tilt and the recession ease off as cards travel out --
      // doubling the distance adds only about half again as much of each.
      // A linear ramp folds the second card shut; this keeps it readable.
      const ramp = Math.pow(distance, falloff)
      // Capped short of edge-on so a far card never turns its back.
      const tilt = Math.min(rotate * ramp, 82) * Math.sign(offset)

      card.style.transform =
        `translateX(calc(-50% + ${offset * pitch}px)) ` +
        `translateZ(${-depth * width * ramp}px) rotateY(${-tilt}deg)`

      // A card is teleported across the ring at exactly half a turn out, so it
      // has to be gone by then or the jump is visible.
      const edge = loop ? Math.min(1, Math.max(0, count / 2 - distance)) : 1
      card.style.opacity = String(Math.max(0, 1 - fade * distance) * edge)
      card.style.zIndex = String(100 - Math.round(distance))
    })
  }, [count, depth, fade, falloff, gap, loop, rotate])

  const settle = useCallback(
    (target) => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current)
      targetRef.current = target
      setSelected(indexAt(target))

      // Reduced motion gets the destination, not the journey.
      if (reduced) {
        posRef.current = target
        paint()
        rafRef.current = null
        return
      }

      const step = () => {
        const remaining = target - posRef.current
        if (Math.abs(remaining) < 0.0004) {
          posRef.current = target
          paint()
          rafRef.current = null
          return
        }
        // Exponential ease-out, not a spring. Swap in a spring only if the
        // settle ever needs overshoot.
        posRef.current += remaining * 0.16
        paint()
        rafRef.current = requestAnimationFrame(step)
      }
      rafRef.current = requestAnimationFrame(step)
    },
    [indexAt, paint, reduced],
  )

  const clamp = useCallback(
    (pos) => (loop ? pos : Math.max(0, Math.min(count - 1, pos))),
    [count, loop],
  )

  const goTo = useCallback(
    (index) => {
      // Take the shorter way round rather than unwinding the whole ring.
      const target = loop
        ? index + Math.round((targetRef.current - index) / count) * count
        : index
      settle(clamp(target))
    },
    [clamp, count, loop, settle],
  )

  const nudge = useCallback(
    (by) => settle(clamp(Math.round(targetRef.current) + by)),
    [clamp, settle],
  )

  const onPointerDown = (event) => {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current)
      rafRef.current = null
    }
    event.currentTarget.setPointerCapture(event.pointerId)
    targetRef.current = posRef.current
    dragRef.current = {
      id: event.pointerId,
      x: event.clientX,
      pos: posRef.current,
      v: 0,
      t: performance.now(),
    }
  }

  const onPointerMove = (event) => {
    const drag = dragRef.current
    if (!drag || drag.id !== event.pointerId) return

    const pitch = widthRef.current * (1 + gap)
    if (!pitch) return

    const now = performance.now()
    const previous = posRef.current
    posRef.current = clamp(drag.pos - (event.clientX - drag.x) / pitch)
    // Cards per second, for the throw.
    drag.v = ((posRef.current - previous) / Math.max(now - drag.t, 1)) * 1000
    drag.t = now

    const index = indexAt(posRef.current)
    if (index !== selected) setSelected(index)
    paint()
  }

  const endDrag = (event) => {
    const drag = dragRef.current
    if (!drag || drag.id !== event.pointerId) return
    dragRef.current = null
    // Let a flick carry, but never more than two cards.
    const carried = Math.max(-2, Math.min(2, drag.v * 0.18))
    settle(clamp(Math.round(posRef.current + carried)))
  }

  // Card width drives pitch, depth and perspective, so it is the only thing
  // worth measuring - and only when the box actually changes.
  useIsoLayoutEffect(() => {
    const frame = frameRef.current
    if (!frame) return

    const measure = () => {
      const card = cardRefs.current[0]
      if (!card) return
      widthRef.current = card.offsetWidth
      paint()
    }

    measure()
    const observer = new ResizeObserver(measure)
    observer.observe(frame)
    return () => observer.disconnect()
  }, [paint])

  useEffect(
    () => () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current)
    },
    [],
  )

  return (
    <div
      className={`${s.root} ${className}`}
      style={{ '--cf-card': cardWidth }}
      role="region"
      aria-roledescription="carousel"
      aria-label={label}
    >
      <div className={s.stage}>
        <div
          ref={frameRef}
          tabIndex={0}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={endDrag}
          onPointerCancel={endDrag}
          onKeyDown={(event) => {
            if (event.key === 'ArrowLeft') {
              event.preventDefault()
              nudge(-1)
            } else if (event.key === 'ArrowRight') {
              event.preventDefault()
              nudge(1)
            }
          }}
          className={s.frame}
          style={{ perspective: `calc(var(--cf-card) * ${perspective})` }}
        >
          <div className={s.ring}>
            {slides.map((slide, index) => (
              <div
                key={slide.src}
                ref={(node) => {
                  cardRefs.current[index] = node
                }}
                role="group"
                aria-roledescription="slide"
                aria-label={`${index + 1} of ${count}`}
                className={s.card}
                data-active={index === selected ? '' : undefined}
              >
                <img
                  src={slide.src}
                  alt={slide.alt}
                  draggable={false}
                  loading="lazy"
                  decoding="async"
                  className={s.img}
                />
              </div>
            ))}
          </div>
        </div>

        {showNavigation && (
          <>
            <button
              type="button"
              aria-label="Previous slide"
              onClick={() => nudge(-1)}
              className={`${s.nav} ${s.navPrev}`}
            >
              <span aria-hidden="true">&larr;</span>
            </button>
            <button
              type="button"
              aria-label="Next slide"
              onClick={() => nudge(1)}
              className={`${s.nav} ${s.navNext}`}
            >
              <span aria-hidden="true">&rarr;</span>
            </button>
          </>
        )}
      </div>

      {renderCaption && renderCaption(slides[selected], selected)}

      {showPagination && (
        <div className={s.dots}>
          {slides.map((slide, index) => (
            <button
              key={slide.src}
              type="button"
              aria-label={`Go to slide ${index + 1}`}
              aria-current={index === selected}
              onClick={() => goTo(index)}
              className={s.dot}
              data-on={index === selected ? '' : undefined}
            />
          ))}
        </div>
      )}
    </div>
  )
}
