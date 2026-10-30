import { useEffect, useState } from 'react'
import { getLog, request, subscribeLog } from '../lib/api.js'
import { Card, PageHead } from '../components/Ui.jsx'

const ENDPOINTS = [
  { id: 'health', method: 'GET', path: '/health', text: 'Estado de la API, número de canciones y de géneros.' },
  { id: 'meta', method: 'GET', path: '/meta', text: 'Rango de cada slider, géneros, modos, métricas y modelos.' },
  { id: 'search', method: 'GET', path: '/songs/search?q=love&limit=5', text: 'Busca canciones por título o artista.' },
  {
    id: 'profile', method: 'POST', path: '/profile', text: 'Perfil → géneros por modelo y popularidad esperada.',
    body: { sliders: { danceability: 0.8, energy: 0.7, acousticness: 0.1 }, top_genres: 3 },
  },
  {
    id: 'recommend', method: 'POST', path: '/recommend', text: 'Perfil → canciones (ZTA o KNN).',
    body: { sliders: { energy: 0.2, acousticness: 0.9 }, method: 'zta', mode: 'familiar', metric: 'mahalanobis', k: 5, dedupe: true },
  },
]

const pretty = (v) => JSON.stringify(v, null, 2)
const statusClass = (s) => (s >= 200 && s < 300 ? 'ok' : 'bad')

export default function LabPage() {
  const [selected, setSelected] = useState(ENDPOINTS[3])
  const [path, setPath] = useState(ENDPOINTS[3].path)
  const [body, setBody] = useState(pretty(ENDPOINTS[3].body))
  const [log, setLog] = useState(getLog())
  const [view, setView] = useState(null) // entrada del registro que se muestra
  const [sending, setSending] = useState(false)
  const [formError, setFormError] = useState(null)

  useEffect(() => subscribeLog(setLog), [])

  const choose = (ep) => {
    setSelected(ep)
    setPath(ep.path)
    setBody(ep.body ? pretty(ep.body) : '')
    setFormError(null)
  }

  const send = async () => {
    setFormError(null)
    let parsed
    if (selected.method === 'POST') {
      try {
        parsed = JSON.parse(body || '{}')
      } catch {
        setFormError('El cuerpo no es un JSON válido.')
        return
      }
    }
    setSending(true)
    try {
      await request(selected.method, path, parsed)
    } catch {
      /* el error queda registrado con su respuesta */
    } finally {
      setSending(false)
      setView(getLog()[0])
    }
  }

  return (
    <>
      <PageHead
        eyebrow="Para desarrolladores"
        title="Laboratorio de la API"
        text="Prueba cada endpoint con tus propios datos, mira la respuesta cruda y el tiempo que tarda. Todas las llamadas de la aplicación quedan registradas aquí."
      />
      <div className="grid-lab">
        <Card title="Endpoints">
          <ul className="endpoints">
            {ENDPOINTS.map((ep) => (
              <li key={ep.id}>
                <button className={selected.id === ep.id ? 'active' : ''} onClick={() => choose(ep)}>
                  <span className={`method ${ep.method}`}>{ep.method}</span>
                  <span><strong>{ep.path.split('?')[0]}</strong><small className="muted">{ep.text}</small></span>
                </button>
              </li>
            ))}
          </ul>
        </Card>

        <Card title="Petición">
          <label className="field"><span className="muted small">{selected.method} · ruta (editable)</span>
            <input className="input mono" value={path} onChange={(e) => setPath(e.target.value)} aria-label="Ruta" />
          </label>
          {selected.method === 'POST' && (
            <label className="field"><span className="muted small">Cuerpo JSON</span>
              <textarea className="input mono" rows={11} value={body} onChange={(e) => setBody(e.target.value)} spellCheck={false} aria-label="Cuerpo JSON" />
            </label>
          )}
          {formError && <p className="error-text">{formError}</p>}
          <button className="btn btn-green" onClick={send} disabled={sending}>{sending ? 'Enviando…' : 'Enviar petición'}</button>
        </Card>

        <Card title="Respuesta">
          {view ? (
            <>
              <div className="resp-head">
                <span className={`status-badge ${statusClass(view.status)}`}>{view.status || 'sin conexión'}</span>
                <span className="muted">{view.method} {view.path}</span>
                <span className="muted">· {view.ms} ms</span>
              </div>
              <pre className="json">{pretty(view.response)}</pre>
            </>
          ) : <p className="muted">Envía una petición para ver la respuesta.</p>}
        </Card>
      </div>

      <Card title="Historial de llamadas" subtitle="Incluye las que hace el resto de la aplicación. Haz clic en una para ver su respuesta.">
        {log.length === 0 ? <p className="muted">Sin llamadas todavía.</p> : (
          <table className="table clickable">
            <thead><tr><th>Hora</th><th>Método</th><th>Ruta</th><th className="num">Estado</th><th className="num">ms</th></tr></thead>
            <tbody>
              {log.map((e) => (
                <tr key={e.id} onClick={() => setView(e)} className={view?.id === e.id ? 'selected' : ''}>
                  <td>{e.at.toLocaleTimeString('es-CO')}</td>
                  <td><span className={`method ${e.method}`}>{e.method}</span></td>
                  <td className="mono">{e.path}</td>
                  <td className="num"><span className={`status-badge ${statusClass(e.status)}`}>{e.status || '–'}</span></td>
                  <td className="num">{e.ms}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>
    </>
  )
}
