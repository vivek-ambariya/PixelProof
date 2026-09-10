import { motion } from 'framer-motion'
import { useReducedMotion } from '../hooks/useReducedMotion.js'
import s from './HardPart.module.css'

export default function HardPart() {
  const reduced = useReducedMotion()
  return (
    <section className="pp-section">
      <div className={s.wrap}>
        <div>
          <div className="pp-kicker">03 · Limits</div>
          <h2 className="pp-h2" style={{ maxWidth: '18ch' }}>
            The hard part isn&rsquo;t detection
          </h2>
          <p className={s.body}>
            On the generators a detector was trained against, accuracy above 95%
            is routine. Point the same detector at a model released afterwards
            and it collapses. The cues are generator-specific: change the
            architecture or the upsampling scheme and the artefacts move.
          </p>
          <p className={s.body}>
            So the useful output isn&rsquo;t a label. It&rsquo;s a probability
            with the evidence attached, so a person can weigh it against
            everything else they know about the image.
          </p>
        </div>
        <div>
          <motion.div
            className={s.stat}
            initial={reduced ? {} : { opacity: 0, y: 16 }}
            whileInView={reduced ? {} : { opacity: 1, y: 0 }}
            viewport={{ once: true, amount: 0.25 }}
            transition={{ duration: 0.6, ease: [0.2, 0.7, 0.3, 1] }}
          >
            18&ndash;31<span className={s.pctSign}>%</span>
          </motion.div>
          <div className={s.statNote}>
            AVERAGE ACCURACY OF PUBLISHED DETECTORS ON GENERATORS THEY WERE{' '}
            <span className={s.emph}>NOT</span> TRAINED ON.
            <br />
            WORSE THAN A COIN FLIP AT THE LOW END.
          </div>
        </div>
      </div>
    </section>
  )
}
