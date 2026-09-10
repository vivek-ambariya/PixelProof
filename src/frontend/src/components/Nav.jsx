import s from './Nav.module.css'

/** Header bar. Model name and status come from /api/health, not hard-coded. */
export default function Nav({ health }) {
  const model = health?.model?.backbone
  const loaded = health?.model_loaded
  // Until /api/health answers, say nothing rather than claim a state.
  const statusText = health === null ? 'CHECKING' : loaded ? 'ONLINE' : 'NO MODEL'
  const statusClass = health === null ? '' : loaded ? s.online : s.offline

  return (
    <header className={s.nav}>
      <div className={s.brand}>
        <div className={s.mark} aria-hidden="true" />
        <span className={s.wordmark}>PIXELPROOF</span>
      </div>
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
