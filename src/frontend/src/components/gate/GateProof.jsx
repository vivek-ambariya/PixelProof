import s from './GateProof.module.css'

const TILES = [
  {
    value: '0.9851',
    label: 'ROC-AUC · HELD-OUT TEST',
    note: '20,000 images, in-distribution. Macro-F1 0.9362.',
  },
  {
    value: '0.9410',
    label: 'ROC-AUC · UNSEEN GENERATOR',
    note: 'ProGAN, a generator family never seen during training.',
    accent: true,
  },
  {
    value: '0.23%',
    label: 'OF PARAMETERS TRAINED',
    note: '197,121 of 86.0M. The CLIP backbone stays frozen.',
  },
  {
    value: '≤ 5%',
    label: 'FALSE-POSITIVE RATE',
    note: 'Where the threshold sits. Flagging a real photo is the costly error.',
  },
]

export default function GateProof() {
  return (
    <section className={s.section} id="gate-proof" data-reveal-group>
      <div className={s.inner}>
        <p className={s.kicker} data-reveal>02 · MEASURED</p>
        <h2 className={s.heading} data-display data-reveal>
          Numbers from the<br />evaluation, not the pitch.
        </h2>

        <div className={s.grid}>
          {TILES.map((tile) => (
            <div
              key={tile.label}
              className={`${s.tile} ${tile.accent ? s.tileAccent : ''}`}
              data-reveal
            >
              <div className={s.value}>{tile.value}</div>
              <div className={s.label}>{tile.label}</div>
              <p className={s.note}>{tile.note}</p>
            </div>
          ))}
        </div>

        <p className={s.footnote} data-reveal>
          Primary model: frozen CLIP ViT-B/16 with a linear head, trained on
          CIFAKE. Confusion matrices, calibration and the fine-tuned ResNet50
          baseline are all in <code>report/model_report.md</code>.
        </p>
      </div>
    </section>
  )
}
