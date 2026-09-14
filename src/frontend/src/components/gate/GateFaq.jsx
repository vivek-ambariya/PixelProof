import { useState } from 'react'
import s from './GateFaq.module.css'

const QUESTIONS = [
  {
    q: 'What files can I test?',
    a: 'Anything the browser reads as an image — JPEG, PNG, WebP and the rest — up to 12 MB. Files that are not images are rejected in the page before anything is sent, with the same limit the server would quote back.',
  },
  {
    q: 'Are my images stored?',
    a: 'No. The image is scored and discarded as soon as the response is sent. There are no accounts, no tracking, and nothing is written to disk for later.',
  },
  {
    q: 'How accurate is it, honestly?',
    a: '0.9851 ROC-AUC on the held-out in-distribution test set of 20,000 images, and 0.9410 against ProGAN — a generator family it never trained on. Accuracy falls on newer generators, heavy re-compression, screenshots and unusual noise profiles.',
  },
  {
    q: 'What does the score actually mean?',
    a: 'A calibrated probability that the image was synthetically generated. The decision threshold is set where the false-positive rate stays at or below 5%, because wrongly flagging a real photograph is the more costly error. Scores in the middle of the range are inconclusive by design, not by accident.',
  },
  {
    q: 'Can it tell me who made the image?',
    a: 'No. It makes no claim about who produced an image, which tool produced it, or about any real person appearing in one. Generator attribution was not attempted in this build.',
  },
  {
    q: 'Why does it show a heat-map?',
    a: 'So the estimate can be checked rather than trusted. Grad-CAM marks the regions that moved the score, and the text alongside it is derived from those measurements rather than written as free-form prose.',
  },
  {
    q: 'What is it built on?',
    a: 'A frozen CLIP ViT-B/16 backbone with a 197,121-parameter trained head, plus a fine-tuned ResNet50 baseline kept for comparison. Training data is CIFAKE; evaluation uses held-out, unseen-content and unseen-generator splits that calibration never touched.',
  },
]

export default function GateFaq() {
  const [open, setOpen] = useState(null)

  return (
    <section className={s.section} id="gate-faq" data-reveal-group>
      <div className={s.inner}>
        <p className={s.kicker} data-reveal>04 · QUESTIONS</p>
        <h2 className={s.heading} data-display data-reveal>
          The things worth<br />asking first.
        </h2>

        <div className={s.list}>
          {QUESTIONS.map((item, i) => {
            const isOpen = open === i
            return (
              <div
                key={item.q}
                className={s.item}
                data-reveal
                // Pointing at a question opens it. It stays open once the
                // pointer leaves, so moving down to read the answer cannot
                // collapse the thing you were reading.
                onMouseEnter={() => setOpen(i)}
              >
                <button
                  type="button"
                  className={s.summary}
                  aria-expanded={isOpen}
                  aria-controls={`gate-faq-a${i}`}
                  onClick={() => setOpen(isOpen ? null : i)}
                  onFocus={() => setOpen(i)}
                >
                  <span>{item.q}</span>
                  <span
                    className={`${s.icon} ${isOpen ? s.iconOpen : ''}`}
                    aria-hidden="true"
                  />
                </button>
                <div
                  id={`gate-faq-a${i}`}
                  className={`${s.answerWrap} ${isOpen ? s.answerOpen : ''}`}
                  role="region"
                >
                  <div className={s.answerInner}>
                    <p className={s.answer}>{item.a}</p>
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      </div>
    </section>
  )
}
