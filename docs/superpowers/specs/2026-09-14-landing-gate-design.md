# PixelProof Landing Gate — Design

**Date:** 2026-09-14
**Status:** Approved, ready for implementation plan

## Purpose

Put a marketing landing page in front of the existing PixelProof tool. Visitors
arrive at the gate, read what PixelProof is and how it works, and click
**"Dive in"** to reach the app that exists today. The gate sells the tool; the
app underneath is unchanged.

The visual and motion reference is `grigoletti.ch/en/`: oversized bold headline,
generous whitespace, staggered fade-up reveals on scroll, and a weighty,
inertia-damped scroll feel. What is borrowed is structure and motion only —
PixelProof keeps its own palette, typography and voice.

Explicitly out of scope: any carousel or slideshow. The reference site's project
slideshow is replaced with a static stat grid.

## Architecture

`App.jsx` gains one piece of state:

```js
const [entered, setEntered] = useState(false)
```

- `entered === false` → render `<Gate onEnter={() => setEntered(true)} />`.
- `entered === true` → render the current tree (`Nav`, `Hero`, `Explainer`,
  `HardPart`, `Batch`, `Footer`) exactly as it renders today.

No router, no URL change, no persistence — the gate is the front door on every
page load. All existing single-image and batch state stays where it is; the gate
never touches it.

### New files

```
src/frontend/src/components/gate/
  Gate.jsx / Gate.module.css              orchestrator, owns Lenis + ScrollTrigger setup
  GateNav.jsx / .module.css               wordmark, anchor links, sticky CTA
  GateHero.jsx / .module.css              kicker, display headline, lede, primary CTA
  GateHowItWorks.jsx / .module.css        three process cards
  GateProof.jsx / .module.css             static stat grid
  GateAbout.jsx / .module.css             positioning and honest limits
  GateFaq.jsx / .module.css               accordion question list
  useGateReveal.js                        shared GSAP reveal hook
```

`Footer.jsx` is reused as-is; it is not duplicated for the gate.

### Dependency

One addition: `lenis` (~4kb), installed into `src/frontend`. `gsap` and
`framer-motion` are already dependencies and need no changes.

## Motion

- **Lenis** is created in a `useEffect` inside `Gate.jsx` and `destroy()`d in the
  cleanup. It exists only while the gate is mounted, so the app underneath keeps
  native scrolling with no conflict after the user dives in.
- **GSAP ScrollTrigger** drives reveals. Each section's heading and children fade
  up (`opacity 0 → 1`, `y: 20 → 0`) on a stagger, using
  `cubic-bezier(0.12, 0.23, 0.5, 1)` to match the reference's snappy settle.
  Registration happens once in the shared `useGateReveal` hook.
- **Reduced motion:** the existing `useReducedMotion` hook gates both. When
  reduced, Lenis is never instantiated (native scroll) and GSAP tweens resolve
  instantly. This matches how the rest of the app already branches.

## Design language

The gate adopts the reference site's palette and typography. It does this by
**overriding the existing `--pp-*` token names inside the `.gate` scope** rather
than inventing a parallel token set — so every gate component, and the reused
`Footer` which reads the same tokens, re-themes from one block, and the app
reverts to its dark identity the moment the gate unmounts.

Palette (from grigoletti.ch):

- Ground `#ebe9e4` cream, panels `#e4e1da`, rules `#dbdbdb`
- Text `#111` primary, `#333` / `#555` secondary, `#6a6a6a`–`#aaa` muted
- Accent `#ff4c24` orange-red, mapped over both the teal and amber roles
- Primary buttons are solid `#111` with cream text; the accent is the hover

Typography mirrors the reference's three-tier system. Their faces (PP Neue Corp
Tight Ultrabold, PP Neue Montreal) are commercial and served from Framer's CDN,
so close free equivalents stand in:

- **Display** — Archivo loaded variable, set at `weight 900` / `font-stretch 76%`
  for a tight ultrabold condensed cut. Applied via a `data-display` attribute so
  each section stylesheet owns only its own size and rhythm.
- **Body** — Inter
- **Labels/kickers** — JetBrains Mono, as the app already uses

Two globals follow the gate: `html.pp-gate-active` repaints the document ground
cream so overscroll does not reveal the dark app behind it, and the film-grain
overlay renders only after entry, since it belongs to the app's darkroom look.

## Sections

All copy is drawn from the existing README, `report/model_report.md`, the
current `Footer.jsx`, and — for accepted file types and size limits — the actual
`rejectReason()` rules in `src/lib/api.js`. Nothing is invented, and no metric is
stated that the report does not support.

| Section | Content |
|---|---|
| Nav | `PIXELPROOF` wordmark, anchor links (How it works / Proof / FAQ), sticky "Dive in →" button |
| Hero | Kicker `MEDIA FORENSICS · HACKATHON BUILD`; headline "Was this image made by a camera or by a model?"; the existing lede about reporting probabilities, not verdicts; primary "Dive in →" CTA |
| How it works | Three cards — Upload (stays in your browser) → Frozen CLIP ViT-B/16 + detector head → Grad-CAM heat-map and a grounded, measurement-derived explanation |
| Proof | Static grid, no carousel: 0.9851 ROC-AUC in-distribution test · 0.9410 ROC-AUC on a genuinely unseen generator (ProGAN) · frozen backbone, 0.23% of params trained · 0 images retained |
| About | Honest positioning from the current footer: likelihoods not accusations, no claim about who made an image or who appears in one, accuracy drops on new generators and heavy compression, mid-range scores are inconclusive |
| FAQ | What files are accepted · How accurate is it, honestly · Are my images stored · What does the score actually mean · Can it tell who made the image · Is it open source · What are the known limits |
| Closing CTA | Final full-width "Dive in →" before the footer |
| Footer | Existing `Footer.jsx`, unchanged |

Every "Dive in" control — nav, hero, closing CTA — calls the same `onEnter`
prop. There is one entry path, not three.

## Data flow

The gate is presentational. It makes no network calls, so there is no loading,
empty or error state to design — `/api/health` and `/api/predict` are still only
touched by the app after entry. The single interaction is:

```
CTA click → onEnter() → App sets entered = true
          → Gate unmounts → Lenis.destroy() runs in cleanup
          → existing app tree mounts, window scrolled to top
```

Scroll position is reset to top on entry so the app's own hero is what greets
the user, not the scroll offset they left the gate at.

## Verification

Run `npm run dev` in `src/frontend` and confirm in the browser:

1. Gate renders on load; the app tree is not mounted behind it.
2. Scroll feels inertia-damped; sections reveal with a visible stagger.
3. All three "Dive in" controls enter the app, and the app works as before —
   upload, single result, batch.
4. After entry, scrolling is native and jank-free (Lenis destroyed).
5. Reloading returns to the gate.
6. With OS reduced-motion enabled, no Lenis and no animation; content is
   immediately visible.
7. At ~400px width, nothing overflows horizontally and the nav CTA stays usable.

## Decisions taken (and the alternatives rejected)

- **In-page state swap, not routes.** No router dependency; the trade-off is no
  shareable `/app` URL and back button does not return to the gate. Accepted.
- **Gate every visit, not once per session.** Simplest, and the landing page is
  meant to be the front door.
- **Lenis added.** Native scroll plus reveals alone did not reproduce the
  reference's physical feel; one small dependency was judged worth it.
- **Stat grid, not a portfolio slideshow.** Requested explicitly — no carousel
  anywhere on the page.
- **The reference's cream/black/orange palette and typography, not PixelProof's
  dark teal/amber.** The borrow is the whole visual design; only the content,
  facts and metrics are PixelProof's. Scoped to the gate, so the tool itself is
  untouched.
