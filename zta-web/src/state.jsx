import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react'
import { api } from './lib/api.js'
import { PRESETS } from './lib/features.js'

const Ctx = createContext(null)
export const useApp = () => useContext(Ctx)

const round = (n) => Math.round(n * 1000) / 1000

export function AppProvider({ children }) {
  const [health, setHealth] = useState(null) // { status, songs, genres } | 'down' | null (cargando)
  const [meta, setMeta] = useState(null)
  const [loadError, setLoadError] = useState(null)

  // Perfil en construcción (compartido entre "Perfil" y "Recomendar")
  const [mode, setMode] = useState('sliders') // 'sliders' | 'seeds'
  const [values, setValues] = useState({})
  const [enabled, setEnabled] = useState({})
  const [seeds, setSeeds] = useState([])
  const [topGenres, setTopGenres] = useState(5)
  const [profileResult, setProfileResult] = useState(null)

  const defaults = useMemo(
    () => (meta ? Object.fromEntries(meta.features.map((f) => [f.name, f.default_slider])) : {}),
    [meta],
  )

  const applyPreset = useCallback(
    (preset, base = defaults) => {
      setMode('sliders')
      setValues({ ...base, ...preset.values })
      setEnabled(Object.fromEntries(Object.keys(preset.values).map((k) => [k, true])))
    },
    [defaults],
  )

  const load = useCallback(async () => {
    setLoadError(null)
    try {
      const [h, m] = await Promise.all([api.health(), api.meta()])
      if (
        !Array.isArray(m.features) ||
        m.features.length === 0 ||
        m.features.some((f) => !f || typeof f.name !== 'string' || !Number.isFinite(f.default_slider))
      ) {
        throw new Error('La API /meta respondió con metadatos inválidos: falta una lista válida de características.')
      }
      setHealth(h)
      setMeta(m)
      const base = Object.fromEntries(m.features.map((f) => [f.name, f.default_slider]))
      setValues((prev) => (Object.keys(prev).length ? prev : { ...base, ...PRESETS[0].values }))
      setEnabled((prev) =>
        Object.keys(prev).length ? prev : Object.fromEntries(Object.keys(PRESETS[0].values).map((k) => [k, true])),
      )
    } catch (error) {
      setHealth('down')
      setLoadError(error.message)
    }
  }, [])

  useEffect(() => {
    load()
  }, [load])

  const setValue = (name, v) => setValues((prev) => ({ ...prev, [name]: v }))
  const toggle = (name, on) => setEnabled((prev) => ({ ...prev, [name]: on }))
  const addSeed = (song) => setSeeds((prev) => (prev.some((s) => s.track_id === song.track_id) ? prev : [...prev, song]))
  const removeSeed = (id) => setSeeds((prev) => prev.filter((s) => s.track_id !== id))

  // Cuerpo que la API espera: exactamente uno de `sliders` o `seed_ids`.
  const payload = useMemo(() => {
    if (mode === 'seeds') return { seed_ids: seeds.map((s) => s.track_id) }
    const sliders = {}
    for (const [name, on] of Object.entries(enabled)) if (on && values[name] != null) sliders[name] = round(values[name])
    return { sliders }
  }, [mode, seeds, enabled, values])

  const ready = mode === 'seeds' ? seeds.length > 0 : true

  const value = {
    health, meta, loadError, reload: load,
    mode, setMode, values, setValue, enabled, toggle, seeds, addSeed, removeSeed,
    topGenres, setTopGenres, payload, ready, defaults, applyPreset,
    profileResult, setProfileResult,
  }
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}
