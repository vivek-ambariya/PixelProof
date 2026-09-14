import { useLayoutEffect } from 'react'
import { gsap } from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'
import { CustomEase } from 'gsap/CustomEase'
import { useReducedMotion } from '../../hooks/useReducedMotion.js'

gsap.registerPlugin(ScrollTrigger, CustomEase)

// The reference site's reveal curve: cubic-bezier(0.12, 0.23, 0.5, 1).
CustomEase.create('ppGate', 'M0,0 C0.12,0.23 0.5,1 1,1')

/**
 * Drives every reveal on the gate from one place.
 *
 * `[data-hero-reveal]` runs once on mount, staggered, for content that is
 * already on screen. `[data-reveal]` inside a `[data-reveal-group]` waits for
 * its group to scroll in.
 *
 * useLayoutEffect, not useEffect: `gsap.from` must set the hidden start state
 * before the browser paints, or the first frame shows the content in place and
 * then snaps it away.
 */
export function useGateReveal(rootRef) {
  const reduced = useReducedMotion()

  useLayoutEffect(() => {
    const root = rootRef.current
    if (!root || reduced) return

    const ctx = gsap.context(() => {
      const hero = gsap.utils.toArray('[data-hero-reveal]')
      if (hero.length) {
        gsap.from(hero, {
          opacity: 0,
          y: 20,
          duration: 0.5,
          ease: 'ppGate',
          stagger: 0.06,
          delay: 0.12,
        })
      }

      gsap.utils.toArray('[data-reveal-group]').forEach((group) => {
        const items = group.querySelectorAll('[data-reveal]')
        if (!items.length) return
        gsap.from(items, {
          opacity: 0,
          y: 20,
          duration: 0.5,
          ease: 'ppGate',
          stagger: 0.06,
          scrollTrigger: { trigger: group, start: 'top 82%', once: true },
        })
      })
    }, root)

    return () => ctx.revert()
  }, [reduced, rootRef])
}
