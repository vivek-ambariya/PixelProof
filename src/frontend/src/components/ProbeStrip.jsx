import Coverflow from './Coverflow.jsx'
import { PROBES, PROBE_CAUGHT, PROBE_THRESHOLD, PROBE_TOTAL } from '../lib/probes.js'
import s from './ProbeStrip.module.css'

// Ground truth is AI for every probe, so the alt text says so rather than
// describing twenty near-identical 256px ProGAN crops.
const SLIDES = PROBES.map((probe, i) => ({
  ...probe,
  alt: `Probe ${i + 1} of ${PROBE_TOTAL}: a ProGAN-generated image, scored ${probe.prob.toFixed(2)}`,
}))

const pct = (p) => `${(p * 100).toFixed(1)}%`

function Caption(slide, index) {
  const flagged = slide.prob >= PROBE_THRESHOLD
  return (
    <div className={s.caption} key={index}>
      <div className={s.verdictRow}>
        <span
          className={s.verdict}
          data-tone={flagged ? 'caught' : 'missed'}
        >
          {flagged ? 'FLAGGED AI' : 'MISSED'}
        </span>
        <span className={s.prob} data-tone={flagged ? 'caught' : 'missed'}>
          {pct(slide.prob)}
        </span>
      </div>

      <div className={s.meter} aria-hidden="true">
        <div
          className={s.meterFill}
          data-tone={flagged ? 'caught' : 'missed'}
          style={{ width: pct(slide.prob) }}
        />
        <div
          className={s.meterMark}
          style={{ left: pct(PROBE_THRESHOLD) }}
        />
      </div>

      <dl className={s.meta}>
        <div className={s.metaRow}>
          <dt>Probe</dt>
          <dd>{String(index + 1).padStart(2, '0')} / {PROBE_TOTAL}</dd>
        </div>
        <div className={s.metaRow}>
          <dt>Ground truth</dt>
          <dd className={s.truth}>AI · ProGAN</dd>
        </div>
        <div className={s.metaRow}>
          <dt>Band</dt>
          <dd>{slide.band}</dd>
        </div>
      </dl>
    </div>
  )
}

/**
 * The probe strip: twenty known-synthetic images the detector has to call.
 *
 * Every card is machine-generated, so the section is a miss-rate read-out, not
 * a highlight reel -- it is the claim section 03 makes, shown on actual images
 * with the actual shipped checkpoint.
 */
export default function ProbeStrip() {
  const missed = PROBE_TOTAL - PROBE_CAUGHT

  return (
    <section className="pp-section" aria-labelledby="probe-strip-title">
      <div className={s.head}>
        <div>
          <div className="pp-kicker">05 · Probes</div>
          <h2 className="pp-h2" id="probe-strip-title" style={{ maxWidth: '16ch' }}>
            Twenty it has to call
          </h2>
        </div>
        <p className={s.lede}>
          Every image below is machine-generated &mdash; ProGAN samples from the
          unseen-generator split, re-encoded as JPEG the way anything on the web
          would be. Ground truth is the same for all twenty, so the only thing
          that varies is what PixelProof made of them. Drag the strip.
        </p>
      </div>

      <Coverflow
        slides={SLIDES}
        autoplay={3500}
        label="Probe images and their scores"
        renderCaption={Caption}
      />

      <div className={s.tally}>
        <div className={s.tallyStat}>
          <span className={s.tallyNum} data-tone="caught">{PROBE_CAUGHT}</span>
          <span className={s.tallyLabel}>FLAGGED AT THRESHOLD {PROBE_THRESHOLD.toFixed(4)}</span>
        </div>
        <div className={s.tallyStat}>
          <span className={s.tallyNum} data-tone="missed">{missed}</span>
          <span className={s.tallyLabel}>SCORED BELOW IT, AND SO MISSED</span>
        </div>
        <p className={s.tallyNote}>
          Published rather than hidden. Recompression and an unseen generator
          are the two conditions the model is weakest under, and this strip
          stacks both. Treat it as the floor, not the headline number.
        </p>
      </div>
    </section>
  )
}
