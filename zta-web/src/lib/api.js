// Cliente de la API de Python (FastAPI). Registra cada llamada para el "Laboratorio".
const BASE = (import.meta.env.VITE_API_URL ?? '/api').replace(/\/$/, '')

let log = []
let counter = 0
const listeners = new Set()

export const getLog = () => log
export const subscribeLog = (fn) => {
  listeners.add(fn)
  return () => listeners.delete(fn)
}
const record = (entry) => {
  log = [{ id: ++counter, at: new Date(), ...entry }, ...log].slice(0, 30)
  listeners.forEach((fn) => fn(log))
}

export class ApiError extends Error {
  constructor(message, status) {
    super(message)
    this.status = status
  }
}

// FastAPI devuelve `detail` como texto (reglas de negocio) o como lista (validación del esquema).
function describe(detail) {
  if (Array.isArray(detail)) {
    return detail
      .map((d) => `${(d.loc || []).filter((p) => p !== 'body').join('.')}: ${d.msg}`)
      .join(' · ')
  }
  return typeof detail === 'string' ? detail : detail ? JSON.stringify(detail) : ''
}

export async function request(method, path, body) {
  const started = performance.now()
  let status = 0
  let data = null
  try {
    const res = await fetch(`${BASE}${path}`, {
      method,
      headers: body !== undefined ? { 'Content-Type': 'application/json' } : undefined,
      body: body !== undefined ? JSON.stringify(body) : undefined,
    })
    status = res.status
    data = await res.json().catch(() => null)
    if (!res.ok) throw new ApiError(describe(data?.detail) || `Error ${res.status}`, res.status)
    if (!data || typeof data !== 'object' || Array.isArray(data)) {
      throw new ApiError('La API devolvió una respuesta vacía o JSON inválido.', res.status)
    }
    return data
  } catch (error) {
    if (error instanceof ApiError) throw error
    throw new ApiError('No se pudo conectar con la API. ¿Está corriendo `python -m zta.api --port 8000`?', 0)
  } finally {
    record({ method, path, status, ms: Math.round(performance.now() - started), request: body ?? null, response: data })
  }
}

export const api = {
  health: () => request('GET', '/health'),
  meta: () => request('GET', '/meta'),
  search: (q, limit = 8) => request('GET', `/songs/search?q=${encodeURIComponent(q)}&limit=${limit}`),
  profile: (body) => request('POST', '/profile', body),
  recommend: (body) => request('POST', '/recommend', body),
}
