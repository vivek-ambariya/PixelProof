import { useCallback, useEffect, useRef, useState } from 'react'
import Nav from './components/Nav.jsx'
import Hero from './components/Hero.jsx'
import Explainer from './components/Explainer.jsx'
import HardPart from './components/HardPart.jsx'
import Batch from './components/Batch.jsx'
import Footer from './components/Footer.jsx'
import Gate from './components/gate/Gate.jsx'
import { fetchHealth, predictImage, rejectReason } from './lib/api.js'

/**
 * Owns the single-image flow: idle -> loading -> result | error.
 *
 * Batch mode keeps its own state, because its rows resolve independently and
 * mixing the two made the hero panel flicker between phases.
 */
export default function App() {
  const [entered, setEntered] = useState(false)
  const [phase, setPhase] = useState('idle')
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [preview, setPreview] = useState(null)
  const [fileMeta, setFileMeta] = useState(null)
  const [health, setHealth] = useState(null)

  const lastFile = useRef(null)
  const objectUrl = useRef(null)
  const abort = useRef(null)

  // Held until entry so the landing page itself makes no network calls, and
  // so the status in the header is fresh when the tool appears.
  useEffect(() => {
    if (!entered) return
    fetchHealth().then(setHealth)
  }, [entered])

  // The gate leaves the page scrolled wherever the visitor stopped reading.
  useEffect(() => {
    if (entered) window.scrollTo(0, 0)
  }, [entered])

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
      {entered && <div className="pp-grain" aria-hidden="true" />}
      {!entered ? (
        <Gate onEnter={() => setEntered(true)} />
      ) : (
        <div className="pp-shell">
          <Nav health={health} />
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
