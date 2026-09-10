import s from './Footer.module.css'

export default function Footer() {
  return (
    <footer className={s.footer}>
      <div className={s.wrap}>
        <div>
          <div className={s.brand}>
            <div className={s.mark} aria-hidden="true" />
            <span className={s.wordmark}>PIXELPROOF</span>
          </div>
          <p className={s.blurb}>
            A hackathon build for media forensics. Open source, no accounts, and
            images are discarded as soon as the response is sent.
          </p>
          <div className={s.links}>
            <a href="https://github.com" target="_blank" rel="noreferrer noopener">GitHub &#8599;</a>
            <a href="/docs">API docs &#8599;</a>
            <a href="/api/health">Health &#8599;</a>
          </div>
        </div>
        <div className={s.cards}>
          <div className={`${s.card} ${s.cardAmber}`}>
            <div className={`${s.cardLabel} ${s.amber}`}>Limitations</div>
            <p className={s.cardBody}>
              Accuracy drops sharply on generators released after training, on
              heavily compressed or resized images, on screenshots, and on
              photographs with unusual noise profiles. Scores in the middle of
              the range should be treated as inconclusive.
            </p>
          </div>
          <div className={`${s.card} ${s.cardTeal}`}>
            <div className={`${s.cardLabel} ${s.teal}`}>What this is not</div>
            <p className={`${s.cardBody} ${s.strong}`}>
              PixelProof reports likelihoods, not accusations. It makes no claim
              about who produced an image, and no claim about any real person
              appearing in one. A high score is a reason to check further, never
              a conclusion on its own.
            </p>
          </div>
        </div>
      </div>
      <div className={s.rule}>
        <span>PP-DETECT · FROZEN CLIP VIT-B/16 + LINEAR HEAD</span>
        <span>NO IMAGES RETAINED · NO TRACKING</span>
      </div>
    </footer>
  )
}
