import { useCallback, useRef, useState } from 'react'
import s from './Dropzone.module.css'

/**
 * Keyboard-accessible file dropzone.
 *
 * It is a real focusable control: Tab reaches it, Enter and Space open the
 * picker, and it carries a visible focus ring. A bare div with an onClick would
 * be unreachable without a mouse.
 *
 * Drag state uses a counter rather than a boolean, because dragenter/dragleave
 * also fire for child elements and a boolean flickers as the pointer moves
 * across them.
 */
export default function Dropzone({ onFile }) {
  const inputRef = useRef(null)
  const dragDepth = useRef(0)
  const [dragging, setDragging] = useState(false)

  const open = useCallback(() => inputRef.current?.click(), [])

  const onKeyDown = useCallback((e) => {
    if (e.key === 'Enter' || e.key === ' ' || e.key === 'Spacebar') {
      e.preventDefault()
      open()
    }
  }, [open])

  const onDragEnter = useCallback((e) => {
    e.preventDefault()
    dragDepth.current += 1
    setDragging(true)
  }, [])

  const onDragLeave = useCallback((e) => {
    e.preventDefault()
    dragDepth.current -= 1
    if (dragDepth.current <= 0) {
      dragDepth.current = 0
      setDragging(false)
    }
  }, [])

  const onDrop = useCallback((e) => {
    e.preventDefault()
    dragDepth.current = 0
    setDragging(false)
    const file = e.dataTransfer?.files?.[0]
    if (file) onFile(file)
  }, [onFile])

  const onChange = useCallback((e) => {
    const file = e.target.files?.[0]
    if (file) onFile(file)
    // Reset so picking the same file twice still fires a change event.
    e.target.value = ''
  }, [onFile])

  return (
    <>
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        onChange={onChange}
        className="pp-visually-hidden"
        tabIndex={-1}
        aria-hidden="true"
      />
      <div
        role="button"
        tabIndex={0}
        aria-label="Upload an image to analyse. Accepts JPG, PNG or WEBP up to 12 megabytes."
        className={`${s.zone} ${dragging ? s.dragging : ''}`}
        onClick={open}
        onKeyDown={onKeyDown}
        onDragOver={(e) => e.preventDefault()}
        onDragEnter={onDragEnter}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
      >
        <div className={s.plus} aria-hidden="true" />
        <div className={s.label}>Drop an image, or click to browse</div>
        <div className={s.hint}>
          JPG &middot; PNG &middot; WEBP &nbsp;/&nbsp; MAX 12 MB
          <br />
          PROCESSED IN-REQUEST, NOT STORED
        </div>
      </div>
    </>
  )
}
