# Modelos de aprendizaje automático y API del recomendador

Esta carpeta añade, **sin modificar nada de lo que ya existía**, los cuatro modelos
supervisados del proyecto (Regresión Lineal, KNN, Random Forest y SVM), la capa que los
combina con el modelo matemático ZTA y una API HTTP para que el servidor web los consuma.

## 1. Qué se añadió

| Archivo | Para qué sirve |
|---|---|
| `src/zta/models/targets.py` | Construye lo que se predice: matriz 0/1 de géneros y la popularidad. |
| `src/zta/models/registry.py` | Define los 4 modelos, el baseline y las variantes que se prueban de cada uno. |
| `src/zta/models/evaluation.py` | Métricas de cada tarea. |
| `src/zta/models/runner.py` | Entrenamiento: validación cruzada y ajuste final con evaluación en prueba. |
| `src/zta/models/__main__.py` | Comando `python -m zta.models` para entrenar. |
| `src/zta/models/inference.py` | Carga un modelo guardado y puntúa canciones nuevas. |
| `src/zta/models/app.py` | `MusicRecommender`: sliders → perfil → géneros y canciones recomendadas. |
| `src/zta/models/demo.py` | Prueba rápida desde la terminal. |
| `src/zta/api/` | API FastAPI (`schemas.py`, `main.py`, `__main__.py`). |

## 2. Qué se entrena y con qué datos

**Entrada (X), igual para los cuatro modelos:** las 9 variables acústicas (`danceability`, `energy`,
`valence`, `acousticness`, `instrumentalness`, `liveness`, `loudness`, `tempo`, `speechiness`)
normalizadas con el `MinMaxNormalizer` del equipo, ajustado **solo con las canciones de entrenamiento**.

**Salida (y).** No existen valoraciones de usuarios en el dataset, por lo que solo se usan etiquetas
reales que ya estaban en él (no se inventó ninguna):

| Tarea | Qué predice | Para qué se usa |
|---|---|---|
| `genre` | Un puntaje por cada uno de los 114 géneros | Función 1: qué géneros encajan con el perfil del usuario |
| `popularity` | Popularidad de 0 a 100 | Indicar qué tan popular suele ser un sonido así |

| Modelo | Implementación | Variantes probadas |
|---|---|---|
| Regresión Lineal | `LinearRegression` | – |
| KNN | `KNeighborsRegressor` con ponderación por distancia | `n_neighbors` = 15, 50, 100 |
| Random Forest | `RandomForestRegressor` | `min_samples_leaf` = 5, 20 |
| SVM | Géneros: `OneVsRest(LinearSVC)`. Popularidad: `SVR` RBF sobre una submuestra fija de 15 000 filas | `C` |

KNN y Random Forest se usan como regresores sobre la matriz 0/1 de géneros: el promedio de vecinos u
hojas equivale a una probabilidad por género y permite manejar los 114 géneros a la vez. El SVM de
géneros es lineal porque un SVC con kernel RBF sobre ~61 000 filas × 114 géneros no es viable.
Siempre se entrena también un **baseline** (predice la media) como punto de referencia.

## 3. Preparar los datos
Basado en lo que hay en el README.md principal del proyecto

```bash
python src/download_data.py
python -m clean.export_dataset
python -m zta.data --save-partitions
```

Esto deja `data/processed/tracks.csv`, `track_genres.csv` y las particiones
(15 % de las canciones reservado como prueba; el 85 % restante, "desarrollo", se divide en 5 folds con 80 % para entrenar y 20 % para validar en cada uno; semilla 42).

## 4. Entrenar los modelos

```bash
# 1) Prueba rápida de que todo está conectado (un fold, pocas filas)
python -m zta.models --task genre --mode cv --folds 1 --max-train-rows 5000

# 2) Validación cruzada completa: compara modelos y elige la mejor variante de cada uno
python -m zta.models --task genre --mode cv
python -m zta.models --task popularity --mode cv

# 3) Ajuste final con todas las canciones de desarrollo y evaluación en prueba (una sola vez)
python -m zta.models --task genre --mode final --confirm-test
python -m zta.models --task popularity --mode final --confirm-test
```

| Opción | Efecto |
|---|---|
| `--task genre\|popularity` | Qué se predice. |
| `--mode cv\|final` | Validación cruzada, o ajuste final + prueba. |
| `--models linear knn rf svm` | Entrena solo algunos (por defecto, los cuatro). |
| `--folds 1 2 …` | Solo esos folds (por defecto, los 5). |
| `--max-train-rows N` | Submuestrea el entrenamiento; **solo para pruebas rápidas** en modo `cv`. |
| `--confirm-test` | Obligatorio en `final`: confirma que el conjunto de prueba se usa una sola vez. |

`--mode final` exige haber corrido antes `--mode cv` (lee `best_configs.json`): la prueba nunca se usa
para elegir. Cada ejecución **sobrescribe** los resultados anteriores; con la misma semilla el resultado
es reproducible.

## 5. Usar los modelos

### 5.1 Desde la terminal

```bash
python -m zta.models.demo --danceability 0.8 --energy 0.7 --acousticness 0.1 --mode discovery --k 5
python -m zta.models.demo --energy 0.2 --acousticness 0.9 --mode familiar --genres jazz blues
python -m zta.models.demo --seeds <track_id> <track_id>        # perfil a partir de canciones que gustan
```

Cada slider va de 0 a 1 (posición dentro del rango del dataset). Los que no se indican toman el promedio
del dataset ("sin preferencia"). Muestra el perfil (en unidades reales), los géneros que encajan según
cada modelo, las canciones de ZTA y las de KNN.

### 5.2 API HTTP

```bash
python -m zta.api --port 8000
```

El servidor carga el catálogo y los modelos una vez al arrancar (tarda unos segundos). Documentación
interactiva en <http://127.0.0.1:8000/docs> y contrato en `/openapi.json`.

| Método y ruta | Función |
|---|---|
| `GET /health` | Estado, número de canciones y de géneros. |
| `GET /meta` | Lo que necesita el front: rango y valor neutro de cada slider, lista de géneros, modos, métricas y modelos. |
| `GET /songs/search?q=…&limit=10` | Busca canciones por título o artista (una fila por canción distinta). |
| `POST /profile` | Función 1: perfil + géneros que encajan por modelo + popularidad esperada. |
| `POST /recommend` | Función 2: canciones recomendadas con `method` `zta` o `knn`. |

El cuerpo de `/profile` y `/recommend` describe al usuario con **`sliders` o `seed_ids` (exactamente uno)**.

```bash
curl -X POST http://127.0.0.1:8000/profile -H 'content-type: application/json' \
  -d '{"sliders": {"danceability": 0.8, "energy": 0.7, "acousticness": 0.1}, "top_genres": 3}'

curl -X POST http://127.0.0.1:8000/recommend -H 'content-type: application/json' \
  -d '{"sliders": {"energy": 0.2, "acousticness": 0.9}, "method": "zta", "mode": "familiar", "k": 5, "genres": ["jazz", "blues"]}'

curl -X POST http://127.0.0.1:8000/recommend -H 'content-type: application/json' \
  -d '{"seed_ids": ["<track_id>"], "method": "knn", "k": 5}'
```

Parámetros de `/recommend`: `method` (`zta`|`knn`), `mode` (`familiar`|`discovery`|`exploration`, solo ZTA),
`metric` (`euclidean`|`mahalanobis`, solo ZTA), `k` (1–50), `genres`, `popularity_lambda` (≥ 0, solo ZTA;
0 = sin penalización por popularidad) y `dedupe`.

Los datos inválidos devuelven **422** con un campo `detail` (una lista si falla la validación del esquema,
un texto si falla una regla de negocio, por ejemplo un género o canción inexistente).
