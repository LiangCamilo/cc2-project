import { Radar } from 'react-chartjs-2'
import '../lib/charts.js'
import { GREEN, GREEN_SOFT } from '../lib/charts.js'
import { FEATURES, MODEL_LABELS, MODEL_ORDER, formatReal } from '../lib/features.js'
import { Card } from './Ui.jsx'

function Gauge({ value }) {
  const v = Math.max(0, Math.min(100, value))
  return (
    <div className="gauge" role="img" aria-label={`Popularidad esperada ${Math.round(v)} de 100`}>
      <div className="gauge-track">
        <div className="gauge-fill" style={{ width: `${v}%` }} />
        <div className="gauge-marker" style={{ left: `${v}%` }} />
      </div>
      <div className="gauge-scale"><span>0 · nicho</span><span>100 · éxito masivo</span></div>
    </div>
  )
}

function GenreBars({ items }) {
  const max = Math.max(...items.map((i) => i.score))
  return (
    <ol className="genre-bars">
      {items.map((item, idx) => {
        const width = max > 0 ? Math.max(6, (item.score / max) * 100) : Math.max(6, 100 - idx * 12)
        return (
          <li key={item.genre}>
            <div className="genre-line"><span>{item.genre}</span><span className="muted">{item.score.toFixed(3)}</span></div>
            <div className="bar"><div style={{ width: `${width}%` }} /></div>
          </li>
        )
      })}
    </ol>
  )
}

// Géneros que aparecen en el top de al menos dos modelos.
function consensus(genres) {
  const tally = {}
  MODEL_ORDER.forEach((m) => (genres[m] || []).forEach((g, idx) => {
    tally[g.genre] = tally[g.genre] || { genre: g.genre, count: 0, rank: 0 }
    tally[g.genre].count += 1
    tally[g.genre].rank += idx
  }))
  return Object.values(tally)
    .filter((t) => t.count >= 2)
    .sort((a, b) => b.count - a.count || a.rank - b.rank)
}

export default function ProfileResult({ data, meta }) {
  const labels = data.profile.map((p) => FEATURES[p.feature]?.label ?? p.feature)
  const average = data.profile.map((p) => meta.features.find((f) => f.name === p.feature)?.default_slider ?? 0.5)
  const common = consensus(data.genres)

  const radar = {
    labels,
    datasets: [
      { label: 'Tu perfil', data: data.profile.map((p) => p.slider), borderColor: GREEN, backgroundColor: GREEN_SOFT, pointBackgroundColor: GREEN, borderWidth: 2 },
      { label: 'Promedio del catálogo', data: average, borderColor: '#b3b3b3', backgroundColor: 'rgba(255,255,255,0.04)', borderDash: [5, 5], pointRadius: 0, borderWidth: 1.5 },
    ],
  }
  const radarOptions = {
    maintainAspectRatio: false,
    scales: { r: { min: 0, max: 1, ticks: { display: false, stepSize: 0.25 }, grid: { color: 'rgba(255,255,255,0.1)' }, angleLines: { color: 'rgba(255,255,255,0.1)' }, pointLabels: { color: '#e5e5e5', font: { size: 12 } } } },
    plugins: { legend: { position: 'bottom' } },
  }

  return (
    <div className="result-stack">
      <div className="grid-2">
        <Card title="Tu perfil acústico" subtitle="Posición de cada característica dentro del rango del catálogo">
          <div className="chart-box tall"><Radar data={radar} options={radarOptions} /></div>
        </Card>
        <Card title="Valores en unidades reales">
          <table className="table">
            <thead><tr><th>Característica</th><th className="num">Slider</th><th className="num">Valor real</th></tr></thead>
            <tbody>
              {data.profile.map((p) => (
                <tr key={p.feature}>
                  <td>{FEATURES[p.feature]?.label}</td>
                  <td className="num">{Math.round(p.slider * 100)}%</td>
                  <td className="num">{formatReal(p.feature, p.real_value)} <span className="muted">{FEATURES[p.feature]?.unit}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
          <h4 className="mt">Popularidad esperada: <span className="green">{Math.round(data.expected_popularity)}/100</span></h4>
          <Gauge value={data.expected_popularity} />
          <p className="muted small">Estimada por Random Forest (R² ≈ 0,22). Es orientativa: la popularidad depende de factores ajenos al audio.</p>
        </Card>
      </div>

      <Card title="Géneros que encajan con tu perfil" subtitle="Cuatro modelos entrenados puntúan los 114 géneros; aquí se muestran los primeros de cada uno">
        <div className="consensus">
          <strong>Consenso entre modelos:</strong>
          {common.length ? common.map((c) => <span key={c.genre} className="chip chip-on">{c.genre} · {c.count}/4</span>) : <span className="muted"> ningún género coincide en dos o más modelos.</span>}
        </div>
        <div className="grid-4">
          {MODEL_ORDER.filter((m) => data.genres[m]).map((m) => (
            <div key={m} className={`model-col ${m === 'rf' ? 'best' : ''}`}>
              <h4>{MODEL_LABELS[m]} {m === 'rf' && <span className="badge">mejor validado</span>}</h4>
              <GenreBars items={data.genres[m]} />
            </div>
          ))}
        </div>
        <p className="muted small">Los puntajes solo se comparan dentro de un mismo modelo: son probabilidades aproximadas en Random Forest y KNN, márgenes en SVM y valores sin acotar en la regresión lineal. Léelos como rankings.</p>
      </Card>
    </div>
  )
}
