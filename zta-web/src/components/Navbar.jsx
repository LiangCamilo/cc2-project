import { useApp } from '../state.jsx'
import { formatInt } from '../lib/features.js'

export const ROUTES = [
  { key: '', label: 'Inicio' },
  { key: 'perfil', label: 'Perfil' },
  { key: 'recomendar', label: 'Recomendar' },
  { key: 'modelos', label: 'Modelos y métricas' },
  { key: 'laboratorio', label: 'Laboratorio API' },
]

const Logo = () => (
  <svg viewBox="0 0 32 32" width="34" height="34" aria-hidden="true">
    <circle cx="16" cy="16" r="16" fill="#1DB954" />
    <g fill="#000">
      <rect x="7" y="13" width="3" height="6" rx="1.5" />
      <rect x="12" y="9" width="3" height="14" rx="1.5" />
      <rect x="17" y="6" width="3" height="20" rx="1.5" />
      <rect x="22" y="11" width="3" height="10" rx="1.5" />
    </g>
  </svg>
)

export default function Navbar({ route }) {
  const { health } = useApp()
  const down = health === 'down'
  return (
    <header className="navbar">
      <a className="brand" href="#/">
        <Logo />
        <span>ZTA&nbsp;Music</span>
      </a>
      <nav aria-label="Principal">
        {ROUTES.map((r) => (
          <a key={r.key} href={`#/${r.key}`} className={route === r.key ? 'active' : ''}>
            {r.label}
          </a>
        ))}
      </nav>
      <div className={`status ${down ? 'down' : health ? 'up' : ''}`} title="Estado de la API">
        <span className="dot" />
        {down ? 'API sin conexión' : health ? `API conectada · ${formatInt(health.songs)} canciones` : 'Conectando…'}
      </div>
    </header>
  )
}
