import { useState } from 'react'
import { Bar } from 'react-chartjs-2'
import { GRAY, GREEN } from '../lib/charts.js'
import { GENRE_TEST, MODEL_CARDS, POPULARITY_TEST, ZTA_TEST } from '../lib/metrics.js'
import { Card, PageHead, Segmented } from '../components/Ui.jsx'

const colors = (rows) => rows.map((r, i) => (r.baseline ? GRAY : i === 0 ? GREEN : 'rgba(29,185,84,0.45)'))
const hbar = (rows, key, label) => ({
  data: { labels: rows.map((r) => r.model), datasets: [{ label, data: rows.map((r) => r[key]), backgroundColor: colors(rows), borderRadius: 4 }] },
  options: { indexAxis: 'y', maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { beginAtZero: true } } },
})

function ChartCard({ title, subtitle, cfg }) {
  return (
    <Card title={title} subtitle={subtitle}>
      <div className="chart-box"><Bar data={cfg.data} options={cfg.options} /></div>
    </Card>
  )
}

export default function ModelsPage() {
  const [task, setTask] = useState('genres')
  const zta = {
    data: {
      labels: ZTA_TEST.map((r) => r.mode),
      datasets: [
        { label: 'Precision@10 (%)', data: ZTA_TEST.map((r) => r.precision), backgroundColor: GREEN, borderRadius: 4 },
        { label: 'Hit Rate@10 (%)', data: ZTA_TEST.map((r) => r.hit), backgroundColor: 'rgba(255,255,255,0.75)', borderRadius: 4 },
      ],
    },
    options: { maintainAspectRatio: false, plugins: { legend: { position: 'bottom' } }, scales: { y: { beginAtZero: true } } },
  }
  return (
    <>
      <PageHead
        eyebrow="Evidencia"
        title="Modelos y métricas"
        text="Resultados oficiales del informe sobre canciones que los modelos nunca vieron (conjunto de prueba de 13.438 canciones). Cada modelo se compara con una referencia que siempre predice la media."
      />

      <section className="grid-4 model-cards">
        {MODEL_CARDS.map((m) => (
          <article key={m.key} className="card">
            <span className="badge">{m.role}</span>
            <h3>{m.name}</h3>
            <p className="muted small">{m.text}</p>
          </article>
        ))}
      </section>

      <div className="tabs-row">
        <Segmented label="Tarea" value={task} onChange={setTask} options={[{ value: 'genres', label: 'Predicción de géneros' }, { value: 'popularity', label: 'Estimación de popularidad' }]} />
      </div>

      {task === 'genres' ? (
        <>
          <div className="grid-2">
            <ChartCard title="LRAP (mayor es mejor)" subtitle="Qué tan arriba pone cada modelo los géneros correctos" cfg={hbar(GENRE_TEST, 'lrap', 'LRAP')} />
            <ChartCard title="top5_hit (%)" subtitle="Canciones con algún género real entre los 5 primeros" cfg={hbar(GENRE_TEST, 'top5', 'top5_hit')} />
          </div>
          <Card title="Tabla de resultados · géneros (114 etiquetas)">
            <table className="table">
              <thead><tr><th>Modelo</th><th className="num">LRAP</th><th className="num">AP macro</th><th className="num">top1_hit %</th><th className="num">top5_hit %</th></tr></thead>
              <tbody>
                {GENRE_TEST.map((r) => (
                  <tr key={r.model} className={r.baseline ? 'baseline' : ''}>
                    <td>{r.model}</td><td className="num">{r.lrap.toFixed(4)}</td><td className="num">{r.apMacro.toFixed(4)}</td><td className="num">{r.top1.toFixed(2)}</td><td className="num">{r.top5.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="muted small">
              Cómo leerlo: Random Forest incluye un género real entre sus cinco primeras opciones en el 56,81 % de las canciones y acierta el primero en el 31,87 %,
              frente a ≈1 % del azar con 114 géneros. LRAP no es un porcentaje de exactitud. Un valor cercano al 95 % en esta tarea sería señal de error o de filtración de datos.
            </p>
          </Card>
        </>
      ) : (
        <>
          <div className="grid-2">
            <ChartCard title="RMSE (menor es mejor)" subtitle="Error típico en puntos de popularidad (escala 0-100)" cfg={hbar(POPULARITY_TEST, 'rmse', 'RMSE')} />
            <ChartCard title="R² (mayor es mejor)" subtitle="Fracción de la variación que explica el modelo" cfg={hbar(POPULARITY_TEST, 'r2', 'R²')} />
          </div>
          <Card title="Tabla de resultados · popularidad">
            <table className="table">
              <thead><tr><th>Modelo</th><th className="num">RMSE</th><th className="num">MAE</th><th className="num">R²</th></tr></thead>
              <tbody>
                {POPULARITY_TEST.map((r) => (
                  <tr key={r.model} className={r.baseline ? 'baseline' : ''}>
                    <td>{r.model}</td><td className="num">{r.rmse.toFixed(4)}</td><td className="num">{r.mae.toFixed(4)}</td><td className="num">{r.r2.toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="muted small">
              Random Forest reduce el RMSE un 11,89 % frente a la referencia, pero el error sigue siendo alto (≈18 puntos de 100): el sonido explica solo una parte de la popularidad.
              KNN logra el menor MAE; Random Forest se eligió por RMSE, el criterio definido antes de usar la prueba.
            </p>
          </Card>
        </>
      )}

      <h2 className="section-title">Calidad de las recomendaciones ZTA</h2>
      <div className="grid-2">
        <Card title="Recuperación del género objetivo" subtitle="570 perfiles sintéticos del conjunto de prueba">
          <div className="chart-box"><Bar data={zta.data} options={zta.options} /></div>
        </Card>
        <Card title="Diversidad acústica interna (ILD)">
          <table className="table">
            <thead><tr><th>Modo</th><th className="num">Precision@10 %</th><th className="num">Recall@10 %</th><th className="num">ILD</th></tr></thead>
            <tbody>
              {ZTA_TEST.map((r) => (
                <tr key={r.mode}><td>{r.mode}</td><td className="num">{r.precision.toFixed(2)}</td><td className="num">{r.recall.toFixed(2)}</td><td className="num">{r.ild.toFixed(4)}</td></tr>
              ))}
            </tbody>
          </table>
          <p className="muted small">
            Familiar recupera más canciones del género objetivo; Discovery y Exploration se alejan por diseño y ganan diversidad (ILD). Estos números describen listas generadas
            para perfiles sintéticos, no la satisfacción de personas reales.
          </p>
        </Card>
      </div>
    </>
  )
}
