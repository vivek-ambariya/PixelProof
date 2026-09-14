import s from './GateHowItWorks.module.css'

const CARDS = [
  {
    index: '01',
    title: 'Drop an image',
    body: 'Anything the browser reads as an image, up to 12 MB. The file is checked before it leaves the page, scored, and discarded as soon as the response is sent. No account, no queue, no copy kept.',
    points: ['Client-side checks', '12 MB limit', 'Nothing stored'],
  },
  {
    index: '02',
    title: 'Read the signal',
    body: 'A frozen CLIP ViT-B/16 backbone turns the image into features and a small trained head scores them. Three independent signal families — frequency artefacts, grain statistics, lighting geometry — are weak alone and informative together.',
    points: ['Frozen backbone', 'Calibrated probability', 'Threshold at FPR ≤ 5%'],
  },
  {
    index: '03',
    title: 'See where it came from',
    body: 'Every score arrives with a Grad-CAM heat-map showing which regions moved it, and an explanation derived from those measurements rather than written as free-form prose.',
    points: ['Grad-CAM overlay', 'Measured evidence', 'Probability, not verdict'],
  },
]

export default function GateHowItWorks() {
  return (
    <section className={s.section} id="gate-method" data-reveal-group>
      <div className={s.inner}>
        <p className={s.kicker} data-reveal>01 · METHOD</p>
        <h2 className={s.heading} data-display data-reveal>
          Three steps, and<br />nothing hidden in between.
        </h2>

        <div className={s.grid}>
          {CARDS.map((card) => (
            <article key={card.index} className={s.card} data-reveal>
              <div className={s.cardIndex}>{card.index}</div>
              <h3 className={s.cardTitle}>{card.title}</h3>
              <p className={s.cardBody}>{card.body}</p>
              <ul className={s.points}>
                {card.points.map((point) => (
                  <li key={point}>{point}</li>
                ))}
              </ul>
            </article>
          ))}
        </div>
      </div>
    </section>
  )
}
