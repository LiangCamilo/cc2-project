// Resultados oficiales del informe (sección 7.3, conjunto de prueba de 13.438 canciones; sección 6.15 para ZTA).
// Son datos fijos: si se vuelve a entrenar, actualízalos con reports/models/*/test_metrics.csv.
export const GENRE_TEST = [
  { model: 'Random Forest', lrap: 0.4298, apMacro: 0.3028, top1: 31.87, top5: 56.81 },
  { model: 'KNN', lrap: 0.3804, apMacro: 0.2941, top1: 27.64, top5: 50.83 },
  { model: 'SVM lineal', lrap: 0.2775, apMacro: 0.1036, top1: 15.69, top5: 41.06 },
  { model: 'Regresión lineal', lrap: 0.2273, apMacro: 0.0857, top1: 10.49, top5: 35.21 },
  { model: 'Referencia (media)', lrap: 0.0454, apMacro: 0.0112, top1: 0.92, top5: 4.73, baseline: true },
]

export const POPULARITY_TEST = [
  { model: 'Random Forest', rmse: 18.2538, mae: 14.2974, r2: 0.2237 },
  { model: 'KNN', rmse: 18.5048, mae: 14.0837, r2: 0.2022 },
  { model: 'SVM (SVR RBF)', rmse: 20.0355, mae: 15.9164, r2: 0.0648 },
  { model: 'Regresión lineal', rmse: 20.3911, mae: 16.8159, r2: 0.0313 },
  { model: 'Referencia (media)', rmse: 20.7181, mae: 17.3582, r2: -0.0001, baseline: true },
]

export const ZTA_TEST = [
  { mode: 'Familiar', precision: 9.8596, recall: 0.7012, hit: 48.7719, ild: 0.0833 },
  { mode: 'Discovery', precision: 1.2632, recall: 0.0866, hit: 10.8772, ild: 0.2207 },
  { mode: 'Exploration', precision: 0.614, recall: 0.0413, hit: 5.614, ild: 0.2847 },
]

export const MODEL_CARDS = [
  {
    key: 'linear',
    name: 'Regresión lineal',
    role: 'Referencia interpretable',
    text: 'Mide cuánto explica una relación lineal entre el sonido y el objetivo. Si un modelo más complejo no la supera, no se justifica.',
  },
  {
    key: 'knn',
    name: 'KNN',
    role: 'Predicción por similitud',
    text: 'Estima géneros y popularidad promediando a las canciones más cercanas en el espacio acústico, la misma idea de distancia que usa ZTA.',
  },
  {
    key: 'rf',
    name: 'Random Forest',
    role: 'Mejor desempeño en ambas tareas',
    text: 'Un conjunto de árboles que captura relaciones no lineales. Es el modelo recomendado para mostrar los géneros que encajan con un perfil.',
  },
  {
    key: 'svm',
    name: 'SVM',
    role: 'Método de margen',
    text: 'Un clasificador lineal uno contra el resto para géneros y una regresión SVR con kernel RBF para popularidad.',
  },
]
