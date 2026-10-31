import { useMemo, useState } from 'react'
import { useApp } from '../state.jsx'
import { api } from '../lib/api.js'
import { FEATURES, MODES } from '../lib/features.js'
import SongCard from '../components/SongCard.jsx'
import SongsChart from '../components/SongsChart.jsx'
import { Card, ErrorBox, PageHead, Segmented, Spinner, Switch } from '../components/Ui.jsx'

const METRIC_LABEL = { mahalanobis: 'Mahalanobis', euclidean: 'Euclídea' }

function ProfileSummary() {
  const { mode, enabled, values, seeds } = useApp()
  const active = Object.entries(enabled).filter(([, on]) => on)
  return (
    <div className="summary">
      {mode === 'seeds' ? (
        seeds.length ? (
          <div className="chips">{seeds.map((s) => <span key={s.track_id} className="chip chip-on">{s.track_name}</span>)}</div>
        ) : <p className="muted">No has elegido canciones.</p>
      ) : active.length ? (
        <div className="chips">
          {active.map(([name]) => <span key={name} className="chip chip-on">{FEATURES[name].label} {Math.round((values[name] ?? 0) * 100)}%</span>)}
        </div>
      ) : <p className="muted">Sin preferencias: se usará el promedio del catálogo.</p>}
      <a className="link small" href="#/perfil">Editar perfil →</a>
    </div>
  )
}

function GenreFilter({ genres, selected, setSelected }) {
  const [query, setQuery] = useState('')
  const visible = useMemo(() => {
    const q = query.trim().toLowerCase()
    const pool = genres.filter((g) => !selected.includes(g) && (!q || g.includes(q)))
    return pool.slice(0, 16)
  }, [genres, selected, query])
  return (
    <div className="genre-filter">
      <input className="input" placeholder="Filtrar por género (opcional)…" value={query} onChange={(e) => setQuery(e.target.value)} aria-label="Buscar género" />
      {selected.length > 0 && (
        <div className="chips">
          {selected.map((g) => <button key={g} className="chip chip-on" onClick={() => setSelected(selected.filter((x) => x !== g))}>{g} ✕</button>)}
        </div>
      )}
      <div className="chips scroll">
        {visible.map((g) => <button key={g} className="chip chip-btn" onClick={() => { setSelected([...selected, g]); setQuery('') }}>{g}</button>)}
        {visible.length === 0 && <span className="muted small">Sin coincidencias.</span>}
      </div>
    </div>
  )
}

function ResultList({ title, subtitle, data, distanceLabel, shared, compact = false }) {
  return (
    <Card title={title} subtitle={subtitle}>
      {data.songs.length === 0 ? (
        <p className="muted">Esta zona no tiene canciones con esas condiciones. Prueba con otro modo o quita el filtro de géneros.</p>
      ) : (
        <>
          <ol className={`songs ${compact ? 'compact' : ''}`}>
            {data.songs.map((s) => <SongCard key={s.track_id} song={s} distanceLabel={distanceLabel} highlight={shared?.has(s.track_id)} />)}
          </ol>
          <SongsChart songs={data.songs} distanceLabel={distanceLabel} />
        </>
      )}
    </Card>
  )
}

export default function RecommendPage() {
  const { meta, payload, ready, loadError, reload } = useApp()
  const [method, setMethod] = useState('zta')
  const [zone, setZone] = useState('discovery')
  const [metric, setMetric] = useState('mahalanobis')
  const [k, setK] = useState(10)
  const [lambda, setLambda] = useState(0)
  const [dedupe, setDedupe] = useState(true)
  const [genres, setGenres] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [result, setResult] = useState(null)

  const run = async () => {
    setLoading(true)
    setError(null)
    const base = { ...payload, k, dedupe, genres: genres.length ? genres : undefined }
    const zta = () => api.recommend({ ...base, method: 'zta', mode: zone, metric, popularity_lambda: lambda > 0 ? lambda : undefined })
    const knn = () => api.recommend({ ...base, method: 'knn' })
    try {
      if (method === 'both') {
        const [z, n] = await Promise.all([zta(), knn()])
        setResult({ kind: 'both', zta: z, knn: n, zone, metric })
      } else {
        const data = await (method === 'zta' ? zta() : knn())
        setResult({ kind: method, data, zone, metric })
      }
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const usesZta = method !== 'knn'
  const overlap = result?.kind === 'both'
    ? new Set(result.zta.songs.map((s) => s.track_id).filter((id) => result.knn.songs.some((s) => s.track_id === id)))
    : null

  return (
    <>
      <PageHead
        eyebrow="Función 2"
        title="Recomendaciones"
        text="Usa tu perfil para obtener canciones. ZTA las ordena dentro de una zona de tolerancia; KNN devuelve las más cercanas. También puedes compararlos."
      />
      <ErrorBox message={loadError} onRetry={reload} />
      <div className="layout-side">
        <aside>
          <Card title="Tu perfil"><ProfileSummary /></Card>
          <Card title="Opciones" className="mt">
            <div className="option">
              <span className="option-label">Método</span>
              <Segmented label="Método" value={method} onChange={setMethod} options={[{ value: 'zta', label: 'ZTA' }, { value: 'knn', label: 'KNN' }, { value: 'both', label: 'Comparar' }]} />
            </div>
            {usesZta && (
              <>
                <div className="option">
                  <span className="option-label">Modo de escucha</span>
                  <div className="mode-picker">
                    {Object.entries(MODES).map(([key, m]) => (
                      <button key={key} className={`mode-opt ${zone === key ? 'active' : ''}`} onClick={() => setZone(key)} aria-pressed={zone === key}>
                        <span>{m.emoji}</span><strong>{m.label}</strong><small>{m.text}</small>
                      </button>
                    ))}
                  </div>
                </div>
                <div className="option">
                  <span className="option-label">Distancia</span>
                  <Segmented label="Distancia" value={metric} onChange={setMetric} options={[{ value: 'mahalanobis', label: 'Mahalanobis' }, { value: 'euclidean', label: 'Euclídea' }]} />
                </div>
                <div className="option">
                  <span className="option-label">Penalizar populares (λ): <strong>{lambda.toFixed(2)}</strong></span>
                  <input type="range" min="0" max="5" step="0.25" value={lambda} style={{ '--fill': `${(lambda / 5) * 100}%` }} onChange={(e) => setLambda(parseFloat(e.target.value))} aria-label="Penalización por popularidad" />
                  <span className="muted small">0 = sin penalización. Valores altos empujan hacia canciones menos conocidas.</span>
                </div>
              </>
            )}
            <div className="option">
              <span className="option-label">Cantidad: <strong>{k}</strong></span>
              <input type="range" min="1" max="30" step="1" value={k} style={{ '--fill': `${((k - 1) / 29) * 100}%` }} onChange={(e) => setK(parseInt(e.target.value, 10))} aria-label="Cantidad de canciones" />
            </div>
            <div className="option">
              <span className="option-label">Géneros</span>
              {meta ? <GenreFilter genres={meta.genres} selected={genres} setSelected={setGenres} /> : <Spinner />}
              <span className="muted small">El filtro de géneros cambia las condiciones del experimento del informe: úsalo solo como función de la aplicación.</span>
            </div>
            <div className="option row">
              <span>Ocultar repetidas<br /><small className="muted">Colapsa reediciones del mismo título y artista</small></span>
              <Switch checked={dedupe} onChange={setDedupe} label="Ocultar repetidas" />
            </div>
            <button className="btn btn-green btn-lg block" disabled={!ready || loading || !meta} onClick={run}>
              {loading ? 'Buscando…' : 'Recomendar canciones'}
            </button>
            {!ready && <p className="muted small">Añade canciones en «Perfil» para continuar.</p>}
          </Card>
        </aside>

        <div>
          <ErrorBox message={error} />
          {loading && <Spinner label="Calculando distancias…" />}
          {!result && !loading && (
            <div className="empty"><h3>Aún no hay recomendaciones</h3><p className="muted">Elige un método y pulsa «Recomendar canciones».</p></div>
          )}
          {result?.kind === 'zta' && (
            <ResultList title={`ZTA · ${MODES[result.zone].label}`} subtitle={`Distancia ${METRIC_LABEL[result.metric]} · ${result.data.count} canciones`} data={result.data} distanceLabel={METRIC_LABEL[result.metric]} />
          )}
          {result?.kind === 'knn' && (
            <ResultList title="KNN · vecinos más cercanos" subtitle={`Distancia euclídea · ${result.data.count} canciones`} data={result.data} distanceLabel="Euclídea" />
          )}
          {result?.kind === 'both' && (
            <>
              <div className="notice">{overlap.size === 1 ? 'Coincide' : 'Coinciden'} <strong>{overlap.size}</strong> {overlap.size === 1 ? 'canción' : 'canciones'} entre ZTA y KNN (resaltadas en verde). Es normal que difieran: ZTA filtra por zonas de distancia y KNN solo toma las más cercanas.</div>
              <div className="grid-2">
                <ResultList title={`ZTA · ${MODES[result.zone].label}`} subtitle={`${METRIC_LABEL[result.metric]} · ${result.zta.count} canciones`} data={result.zta} distanceLabel={METRIC_LABEL[result.metric]} shared={overlap} compact />
                <ResultList title="KNN · vecinos" subtitle={`Euclídea · ${result.knn.count} canciones`} data={result.knn} distanceLabel="Euclídea" shared={overlap} compact />
              </div>
            </>
          )}
          {result && (result.kind === 'zta' || result.kind === 'both') && result.zone !== 'familiar' && (
            <p className="muted small">En Discovery y Exploration los primeros resultados quedan en el borde interno de la zona, por diseño; pueden parecer variados.</p>
          )}
        </div>
      </div>
    </>
  )
}
