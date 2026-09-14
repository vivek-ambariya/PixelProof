import s from './GateAbout.module.css'

const BLOCKS = [
  {
    tone: 'teal',
    label: 'What this is',
    body: 'A likelihood, with its working shown. PixelProof reports a calibrated probability and the regions that produced it, so the estimate can be checked instead of trusted. A high score is a reason to look closer, never a conclusion on its own.',
  },
  {
    tone: 'plain',
    label: 'What this is not',
    body: 'It makes no claim about who produced an image, and no claim about any real person appearing in one. It is not a detector of intent, authorship or truth — only of the statistical traces that generation tends to leave behind.',
  },
  {
    tone: 'amber',
    label: 'Where it gets weaker',
    body: 'Accuracy drops on generators released after training, on heavily compressed or resized images, on screenshots, and on photographs with unusual noise profiles. Scores in the middle of the range should be treated as inconclusive.',
  },
]

export default function GateAbout() {
  return (
    <section className={s.section} id="gate-about" data-reveal-group>
      <div className={s.inner}>
        <p className={s.kicker} data-reveal>03 · POSITION</p>
        <h2 className={s.heading} data-display data-reveal>
          Honest about what<br />it cannot tell you.
        </h2>

        <div className={s.grid}>
          {BLOCKS.map((block) => (
            <div key={block.label} className={`${s.block} ${s[block.tone]}`} data-reveal>
              <div className={s.label}>{block.label}</div>
              <p className={s.body}>{block.body}</p>
            </div>
          ))}
        </div>

        <div className={s.finding} data-reveal>
          <div className={s.findingLabel}>THE FINDING WE DID NOT BURY</div>
          <p className={s.findingBody}>
            On a genuinely unseen generator, the fine-tuned ResNet50 baseline
            generalises slightly <em>better</em> than the frozen-CLIP primary
            model — 0.9519 against 0.9410 ROC-AUC — the reverse of what the
            literature and our own content-holdout proxy predicted. Both still
            clear the 18–31% accuracy range the problem statement cites for
            naively transferred detectors. It is reported here because a result
            that contradicts the design choice is the one worth publishing.
          </p>
        </div>
      </div>
    </section>
  )
}
