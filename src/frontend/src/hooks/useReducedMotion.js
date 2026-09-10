import { useEffect, useState } from 'react'

/** True when the visitor has asked for reduced motion. Live-updating. */
export function useReducedMotion() {
  const [reduced, setReduced] = useState(
    () => typeof window !== 'undefined' &&
      window.matchMedia('(prefers-reduced-motion: reduce)').matches,
  )
  useEffect(() => {
    const mq = window.matchMedia('(prefers-reduced-motion: reduce)')
    const onChange = (e) => setReduced(e.matches)
    mq.addEventListener('change', onChange)
    return () => mq.removeEventListener('change', onChange)
  }, [])
  return reduced
}

/** True when the viewport is at least `min` wide. */
export function useMinWidth(min = 768) {
  const [wide, setWide] = useState(
    () => typeof window !== 'undefined' && window.innerWidth > min,
  )
  useEffect(() => {
    const mq = window.matchMedia(`(min-width: ${min + 1}px)`)
    const onChange = (e) => setWide(e.matches)
    mq.addEventListener('change', onChange)
    setWide(mq.matches)
    return () => mq.removeEventListener('change', onChange)
  }, [min])
  return wide
}

/**
 * Whether scroll-scrubbed motion should run at all.
 *
 * Two gates, per the brief: never under reduced-motion, and never below 768px
 * where pinned sections are replaced by simple fades.
 */
export function useScrubEnabled() {
  const reduced = useReducedMotion()
  const wide = useMinWidth(768)
  return !reduced && wide
}
