import { useCallback, useEffect, useRef, useState } from 'react'
import Nav from './components/Nav.jsx'
import Hero from './components/Hero.jsx'
import Explainer from './components/Explainer.jsx'
import HardPart from './components/HardPart.jsx'
import Batch from './components/Batch.jsx'
import Footer from './components/Footer.jsx'
import Gate from './components/gate/Gate.jsx'
import IntroSequence from './components/intro/IntroSequence.jsx'
import { useReducedMotion } from './hooks/useReducedMotion.js'
import { fetchHealth, predictImage, rejectReason } from './lib/api.js'

/**
 * Owns the single-image flow: idle -> loading -> result | error.
 *
 * Batch mode keeps its own state, because its rows resolve independently and
 * mixing the two made the hero panel flicker between phases.
 */
export default function App() {
  // gate -> intro -> app. The intro is a timed interstitial, skipped wholesale
  // for visitors who have asked for reduced motion.
  const [stage, setStage] = useState('gate')
  const [phase, setPhase] = useState('idle')
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [preview, setPreview] = useState(null)
  const [fileMeta, setFileMeta] = useState(null)
  const [health, setHealth] = useState(null)

  const lastFile = useRef(null)
  const objectUrl = useRef(null)
  const abort = useRef(null)

  const reduced = useReducedMotion()

  // Fired once the visitor leaves the gate, so the intro doubles as cover for
  // the health round-trip and the status is settled when the tool appears.
  useEffect(() => {
    if (stage === 'gate') return
    fetchHealth().then(setHealth)
  }, [stage])

  // The gate leaves the page scrolled wherever the visitor stopped reading.
  useEffect(() => {
    if (stage === 'app') window.scrollTo(0, 0)
  }, [stage])

  const enter = useCallback(() => {
    setStage(reduced ? 'app' : 'intro')
  }, [reduced])

  const finishIntro = useCallback(() => setStage('app'), [])

  // Reset the scroll before the gate remounts, so Lenis initialises at the top
  // rather than adopting wherever the app happened to be scrolled to.
  const exitToGate = useCallback(() => {
    window.scrollTo(0, 0)
    setStage('gate')
  }, [])

  // Revoke the previous preview URL whenever it is replaced, and on unmount.
  useEffect(() => () => {
    if (objectUrl.current) URL.revokeObjectURL(objectUrl.current)
  }, [])

  const setPreviewFor = useCallback((file) => {
    if (objectUrl.current) URL.revokeObjectURL(objectUrl.current)
    objectUrl.current = URL.createObjectURL(file)
    setPreview(objectUrl.current)
  }, [])

  const analyse = useCallback(async (file) => {
    const reason = rejectReason(file)
    if (reason) {
      setError({ message: reason, kind: 'rejected' })
      setPhase('error')
      return
    }
    lastFile.current = file
    setFileMeta({ name: file.name, size: file.size })
    setPreviewFor(file)
    setError(null)
    setPhase('loading')

    abort.current?.abort()
    abort.current = new AbortController()
    try {
      const data = await predictImage(file, { signal: abort.current.signal })
      setResult(data)
      setPhase('result')
    } catch (e) {
      if (e.name === 'AbortError') return
      setError({ message: e.message, kind: 'failed', status: e.status })
      setPhase('error')
    }
  }, [setPreviewFor])

  const retry = useCallback(() => {
    if (lastFile.current) analyse(lastFile.current)
  }, [analyse])

  const reset = useCallback(() => {
    abort.current?.abort()
    if (objectUrl.current) {
      URL.revokeObjectURL(objectUrl.current)
      objectUrl.current = null
    }
    setPhase('idle')
    setResult(null)
    setError(null)
    setPreview(null)
    setFileMeta(null)
    lastFile.current = null
  }, [])

  return (
    <>
      {/* Grain belongs to the app's darkroom look; the gate is clean stock. */}
      {stage === 'app' && <div className="pp-grain" aria-hidden="true" />}
      {stage === 'intro' && <IntroSequence onDone={finishIntro} />}
      {stage === 'gate' ? (
        <Gate onEnter={enter} />
      ) : (
        <div className="pp-shell">
          <Nav health={health} onHome={exitToGate} />
          <Hero
            phase={phase}
            result={result}
            error={error}
            preview={preview}
            fileMeta={fileMeta}
            onFile={analyse}
            onRetry={retry}
            onReset={reset}
          />
          <Explainer />
          <HardPart />
          <Batch />
          <Footer />
        </div>
      )}
    </>
  )
}
