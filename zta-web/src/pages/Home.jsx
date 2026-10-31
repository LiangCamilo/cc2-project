import { useApp } from '../state.jsx'
import { MODES, formatInt } from '../lib/features.js'
import { ErrorBox } from '../components/Ui.jsx'

const STEPS = [
  { n: '1', title: 'Arma tu perfil', text: 'Mueve nueve sliders (bailabilidad, energía, acústica…) o elige canciones que te gusten. El perfil es el centro de tu gusto en el espacio acústico.', href: '#/perfil', cta: 'Ir a Perfil' },
  { n: '2', title: 'Elige hasta dónde salir', text: 'Familiar se queda cerca de tu gusto; Discovery y Exploration se alejan por bandas de distancia calibradas con el catálogo.', href: '#/recomendar', cta: 'Ir a Recomendar' },
  { n: '3', title: 'Descubre y compara', text: 'ZTA ordena las canciones de cada zona y KNN busca las más cercanas. Compara ambos y revisa qué géneros predicen cuatro modelos entrenados.', href: '#/modelos', cta: 'Ver modelos' },
]

export default function Home() {
  const { health, meta, loadError, reload } = useApp()
  const stats = [
    { value: health && health !== 'down' ? formatInt(health.songs) : '–', label: 'canciones únicas' },
    { value: health && health !== 'down' ? formatInt(health.genres) : '–', label: 'géneros' },
    { value: meta ? meta.features.length : 9, label: 'características acústicas' },
    { value: meta ? meta.models.length : 4, label: 'modelos entrenados' },
  ]
  return (
    <>
      <section className="hero">
        <span className="eyebrow">Demo académica · Universidad Surcolombiana</span>
        <h1>Música que suena como <span className="green">tú</span>.</h1>
        <p className="lead">
          Un recomendador basado en el contenido del audio. No necesita historial ni cuenta: le dices cómo quieres que suene y
          las zonas de tolerancia acústica (ZTA) te llevan de lo familiar a lo desconocido.
        </p>
        <div className="hero-actions">
          <a className="btn btn-green btn-lg" href="#/perfil">Armar mi perfil</a>
          <a className="btn btn-ghost btn-lg" href="#/laboratorio">Probar la API</a>
        </div>
      </section>

      <ErrorBox message={loadError} onRetry={reload} />

      <section className="stats">
        {stats.map((s) => (
          <div key={s.label} className="stat"><strong>{s.value}</strong><span>{s.label}</span></div>
        ))}
      </section>

      <h2 className="section-title">¿Cómo funciona?</h2>
      <section className="grid-3">
        {STEPS.map((s) => (
          <article key={s.n} className="card step">
            <span className="step-n">{s.n}</span>
            <h3>{s.title}</h3>
            <p className="muted">{s.text}</p>
            <a className="link" href={s.href}>{s.cta} →</a>
          </article>
        ))}
      </section>

      <h2 className="section-title">Los tres modos de escucha</h2>
      <section className="grid-3">
        {Object.entries(MODES).map(([key, m]) => (
          <article key={key} className="card mode">
            <h3>{m.label}</h3>
            <p className="muted">{m.text}</p>
          </article>
        ))}
      </section>

      <section className="card about">
        <h3>Qué hay detrás</h3>
        <p className="muted">
          Cada canción se describe con nueve características acústicas normalizadas. El modelo matemático ZTA mide la distancia
          (euclídea o de Mahalanobis) entre tu perfil y cada canción. Además, cuatro modelos entrenados con scikit-learn
          (regresión lineal, KNN, Random Forest y SVM) estiman los géneros y la popularidad asociados a un perfil. Esta
          aplicación es una demostración que consume la API en Python; sus resultados son orientativos y no miden la
          satisfacción de un usuario real.
        </p>
      </section>
    </>
  )
}
