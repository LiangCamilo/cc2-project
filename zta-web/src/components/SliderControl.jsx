import { FEATURES, formatReal, realValue } from '../lib/features.js'
import { Switch } from './Ui.jsx'

export default function SliderControl({ feature, info, value, enabled, onValue, onToggle }) {
  const f = FEATURES[feature]
  const real = info ? realValue(info, value) : null
  return (
    <div className={`slider-row ${enabled ? '' : 'is-off'}`}>
      <div className="slider-head">
        <div className="slider-title">
          <span className="slider-label">{f.label}</span>
          <span className="slider-hint">{f.hint}</span>
        </div>
        <div className="slider-values">
          {enabled ? (
            <>
              <strong>{real != null ? formatReal(feature, real) : '–'}</strong>
              {f.unit && <span className="muted"> {f.unit}</span>}
              <span className="pill">{Math.round(value * 100)}%</span>
            </>
          ) : (
            <span className="muted">Sin preferencia</span>
          )}
        </div>
        <Switch checked={enabled} onChange={onToggle} label={`Usar ${f.label}`} />
      </div>
      <input
        type="range"
        min="0"
        max="1"
        step="0.01"
        value={value}
        disabled={!enabled}
        style={{ '--fill': `${value * 100}%` }}
        onChange={(e) => onValue(parseFloat(e.target.value))}
        aria-label={f.label}
      />
    </div>
  )
}
