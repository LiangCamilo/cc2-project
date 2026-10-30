export const Spinner = ({ label }) => (
  <div className="spinner-wrap" role="status">
    <span className="spinner" />
    {label && <span>{label}</span>}
  </div>
)

export const ErrorBox = ({ message, onRetry }) =>
  message ? (
    <div className="error-box" role="alert">
      <span>
        <strong>Algo salió mal. </strong>
        {message}
      </span>
      {onRetry && (
        <button className="btn btn-ghost btn-sm" onClick={onRetry}>
          Reintentar
        </button>
      )}
    </div>
  ) : null

export function Switch({ checked, onChange, label }) {
  return (
    <label className="switch" title={label}>
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} aria-label={label} />
      <span className="switch-track" />
    </label>
  )
}

export function Segmented({ options, value, onChange, label }) {
  return (
    <div className="segmented" role="tablist" aria-label={label}>
      {options.map((o) => (
        <button
          key={o.value}
          role="tab"
          aria-selected={value === o.value}
          className={value === o.value ? 'active' : ''}
          onClick={() => onChange(o.value)}
        >
          {o.label}
        </button>
      ))}
    </div>
  )
}

export const Card = ({ title, subtitle, action, children, className = '' }) => (
  <section className={`card ${className}`}>
    {(title || action) && (
      <header className="card-head">
        <div>
          {title && <h3>{title}</h3>}
          {subtitle && <p className="muted small">{subtitle}</p>}
        </div>
        {action}
      </header>
    )}
    {children}
  </section>
)

export const PageHead = ({ eyebrow, title, text }) => (
  <div className="page-head">
    {eyebrow && <span className="eyebrow">{eyebrow}</span>}
    <h1>{title}</h1>
    {text && <p className="lead">{text}</p>}
  </div>
)
