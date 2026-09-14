import s from './Nav.module.css'

/** Header bar. Model name and status come from /api/health, not hard-coded. */
export default function Nav({ health, onHome }) {
  const model = health?.model?.backbone
  const loaded = health?.model_loaded
  // Until /api/health answers, say nothing rather than claim a state.
  const statusText = health === null ? 'CHECKING' : loaded ? 'ONLINE' : 'NO MODEL'
  const statusClass = health === null ? '' : loaded ? s.online : s.offline

  return (
    <header className={s.nav}>
      <button
        type="button"
        className={s.brand}
        onClick={onHome}
        aria-label="PixelProof — back to the landing page"
      >
        <span className={s.mark} aria-hidden="true" />
        <span className={s.wordmark}>PIXELPROOF</span>
      </button>
      <div className={s.meta}>
        <span className={s.hideSm}>
          MODEL&nbsp;<span className={s.value}>{model || 'pp-detect'}</span>
        </span>
        <span className={s.hideSm}>
          STATUS&nbsp;<span className={statusClass}>&#9679; {statusText}</span>
        </span>
        <a
          className={s.link}
          href="https://github.com"
          target="_blank"
          rel="noreferrer noopener"
        >
          GITHUB &#8599;
        </a>
      </div>
    </header>
  )
}
