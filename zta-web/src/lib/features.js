// Metadatos de presentación de las nueve características acústicas y utilidades de formato.
export const FEATURES = {
  danceability: { label: 'Bailabilidad', hint: 'Qué tan fácil es bailarla', unit: '', decimals: 2 },
  energy: { label: 'Energía', hint: 'Intensidad y actividad', unit: '', decimals: 2 },
  valence: { label: 'Positividad', hint: 'Qué tan alegre suena', unit: '', decimals: 2 },
  acousticness: { label: 'Acústica', hint: 'Sonido sin amplificar ni electrónica', unit: '', decimals: 2 },
  instrumentalness: { label: 'Instrumental', hint: 'Probabilidad de no tener voz', unit: '', decimals: 2 },
  liveness: { label: 'En vivo', hint: 'Presencia de público', unit: '', decimals: 2 },
  loudness: { label: 'Volumen', hint: 'Sonoridad media', unit: 'dB', decimals: 1 },
  tempo: { label: 'Tempo', hint: 'Velocidad de la pieza', unit: 'BPM', decimals: 0 },
  speechiness: { label: 'Palabra hablada', hint: 'Presencia de voz hablada', unit: '', decimals: 2 },
}

export const FEATURE_ORDER = Object.keys(FEATURES)

export const PRESETS = [
  { id: 'fiesta', label: 'Fiesta', values: { danceability: 0.85, energy: 0.8, valence: 0.8, acousticness: 0.1 } },
  { id: 'gym', label: 'Gimnasio', values: { energy: 0.92, loudness: 0.85, tempo: 0.7, danceability: 0.65 } },
  { id: 'estudio', label: 'Estudio', values: { energy: 0.25, instrumentalness: 0.7, acousticness: 0.6, speechiness: 0.05 } },
  { id: 'acustico', label: 'Acústico', values: { acousticness: 0.9, energy: 0.25, loudness: 0.4, valence: 0.45 } },
  { id: 'melancolia', label: 'Melancolía', values: { valence: 0.1, energy: 0.3, tempo: 0.3, acousticness: 0.7 } },
]

export const MODES = {
  familiar: {
    label: 'Familiar',
    text: 'Lo más parecido a tu perfil. Para quedarte en tu zona de confort.',
  },
  discovery: {
    label: 'Discovery',
    text: 'Un paso fuera de lo habitual: cercanía intermedia al perfil.',
  },
  exploration: {
    label: 'Exploration',
    text: 'Lejos de tu perfil, para salir del radar y descubrir sonidos nuevos.',
  },
}

export const MODEL_LABELS = { rf: 'Random Forest', knn: 'KNN', svm: 'SVM lineal', linear: 'Regresión lineal' }
export const MODEL_ORDER = ['rf', 'knn', 'svm', 'linear']

// Posición del slider (0-1) → valor real, usando el rango del catálogo que entrega /meta.
export const realValue = (info, position) => info.min + position * (info.max - info.min)

export const formatReal = (feature, value) => Number(value).toFixed(FEATURES[feature]?.decimals ?? 2)

export const formatInt = (n) => (n == null ? '–' : new Intl.NumberFormat('es-CO').format(n))
