import { useEffect, useState } from 'react'
import { useApp } from '../state.jsx'
import { api } from '../lib/api.js'
import { FEATURE_ORDER, PRESETS } from '../lib/features.js'
import SliderControl from './SliderControl.jsx'
import { ErrorBox, Segmented, Spinner } from './Ui.jsx'

function SeedSearch() {
  const { seeds, addSeed, removeSeed } = useApp()
  const [query, setQuery] = useState('')
  const [results, setResults] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (query.trim().length < 2) {
      setResults([])
      setError(null)
      return undefined
    }
    let cancelled = false
    const timer = setTimeout(async () => {
      setLoading(true)
      try {
        const data = await api.search(query.trim(), 8)
        if (!cancelled) {
          setResults(data.songs)
          setError(null)
        }
      } catch (e) {
        if (!cancelled) setError(e.message)
      } finally {
        if (!cancelled) setLoading(false)
      }
    }, 350)
    return () => {
      cancelled = true
      clearTimeout(timer)
    }
  }, [query])

  const isSeed = (id) => seeds.some((s) => s.track_id === id)

  return (
    <div className="seed-search">
      <p className="muted small">
        Busca canciones que te gusten. El perfil será el promedio de su sonido, y ni ellas ni sus reediciones se recomendarán.
      </p>
      <input
        className="input"
        placeholder="Busca por título o artista…"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        aria-label="Buscar canciones"
      />
      {seeds.length > 0 && (
        <div className="chips" aria-label="Canciones elegidas">
          {seeds.map((s) => (
            <button key={s.track_id} className="chip chip-on" onClick={() => removeSeed(s.track_id)} title="Quitar">
              {s.track_name} · {s.artists} ✕
            </button>
          ))}
        </div>
      )}
      {loading && <Spinner label="Buscando…" />}
      <ErrorBox message={error} />
      {!loading && query.trim().length >= 2 && !error && results.length === 0 && <p className="muted">Sin resultados.</p>}
      <ul className="result-list">
        {results.map((s) => (
          <li key={s.track_id}>
            <div className="result-main">
              <strong>{s.track_name}</strong>
              <span className="muted">{s.artists}</span>
              <div className="chips small-chips">
                {s.genres.slice(0, 3).map((g) => (
                  <span key={g} className="chip">{g}</span>
                ))}
              </div>
            </div>
            <button
              className={`btn btn-sm ${isSeed(s.track_id) ? 'btn-ghost' : 'btn-green'}`}
              onClick={() => (isSeed(s.track_id) ? removeSeed(s.track_id) : addSeed(s))}
            >
              {isSeed(s.track_id) ? 'Quitar' : '+ Añadir'}
            </button>
          </li>
        ))}
      </ul>
    </div>
  )
}

export default function ProfileBuilder({ compact = false }) {
  const { meta, loadError, mode, setMode, values, setValue, enabled, toggle, applyPreset } = useApp()
  if (!meta) return loadError ? null : <Spinner label="Cargando controles…" />
  const info = Object.fromEntries(meta.features.map((f) => [f.name, f]))
  return (
    <div className="builder">
      <Segmented
        label="Cómo definir tu perfil"
        value={mode}
        onChange={setMode}
        options={[
          { value: 'sliders', label: 'Con sliders' },
          { value: 'seeds', label: 'Con canciones' },
        ]}
      />
      {mode === 'sliders' ? (
        <>
          <div className="presets" aria-label="Perfiles de ejemplo">
            <span className="muted small">Atajos:</span>
            {PRESETS.map((p) => (
              <button key={p.id} className="chip chip-btn" onClick={() => applyPreset(p)}>
                {p.label}
              </button>
            ))}
          </div>
          <p className="muted small">
            Activa el interruptor de las características que te importan. Las que dejes apagadas se toman como el promedio del catálogo.
          </p>
          <div className={`sliders ${compact ? 'compact' : ''}`}>
            {FEATURE_ORDER.filter((n) => info[n]).map((name) => (
              <SliderControl
                key={name}
                feature={name}
                info={info[name]}
                value={values[name] ?? info[name].default_slider}
                enabled={!!enabled[name]}
                onValue={(v) => setValue(name, v)}
                onToggle={(on) => toggle(name, on)}
              />
            ))}
          </div>
        </>
      ) : (
        <SeedSearch />
      )}
    </div>
  )
}
