import { useState } from 'react'
import { useApp } from '../state.jsx'
import { api } from '../lib/api.js'
import ProfileBuilder from '../components/ProfileBuilder.jsx'
import ProfileResult from '../components/ProfileResult.jsx'
import { Card, ErrorBox, PageHead, Spinner } from '../components/Ui.jsx'

export default function ProfilePage() {
  const { meta, payload, ready, topGenres, setTopGenres, profileResult, setProfileResult, loadError, reload, mode } = useApp()
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const calculate = async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await api.profile({ ...payload, top_genres: topGenres })
      setProfileResult({ data, payload })
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const stale = profileResult && JSON.stringify(profileResult.payload) !== JSON.stringify(payload)

  return (
    <>
      <PageHead
        eyebrow="Función 1"
        title="Tu perfil acústico"
        text="Define cómo quieres que suene y calcula tu perfil: verás qué géneros encajan, según cuatro modelos, y qué popularidad suele tener un sonido así."
      />
      <ErrorBox message={loadError} onRetry={reload} />
      <div className="layout-side">
        <aside>
          <Card title="Define tu gusto" className="sticky">
            <ProfileBuilder />
            <div className="builder-foot">
              <label className="field">
                <span className="muted small">Géneros por modelo</span>
                <select className="input" value={topGenres} onChange={(e) => setTopGenres(Number(e.target.value))}>
                  {[3, 5, 8, 10].map((n) => <option key={n} value={n}>{n}</option>)}
                </select>
              </label>
              <button className="btn btn-green btn-lg" disabled={!ready || loading || !meta} onClick={calculate}>
                {loading ? 'Calculando…' : 'Calcular perfil'}
              </button>
            </div>
            {!ready && mode === 'seeds' && <p className="muted small">Añade al menos una canción para calcular el perfil.</p>}
          </Card>
        </aside>
        <div>
          <ErrorBox message={error} />
          {loading && <Spinner label="Consultando los modelos…" />}
          {stale && !loading && <div className="notice">Cambiaste el perfil. Pulsa «Calcular perfil» para actualizar los resultados.</div>}
          {profileResult && meta ? (
            <ProfileResult data={profileResult.data} meta={meta} />
          ) : (
            !loading && (
              <div className="empty">
                <h3>Todavía no hay perfil</h3>
                <p className="muted">Ajusta los sliders (o elige un atajo) y pulsa «Calcular perfil».</p>
              </div>
            )
          )}
        </div>
      </div>
    </>
  )
}
