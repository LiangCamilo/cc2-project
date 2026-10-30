# Modelos de aprendizaje automático y API del recomendador

Este módulo implementa cuatro familias supervisadas (regresión lineal, KNN,
Random Forest y SVM), su validación cruzada y su ajuste final. La capa de
aplicación utiliza sus predicciones como información auxiliar del perfil y genera
listas mediante ZTA o búsqueda de vecinos. La API HTTP expone estas operaciones.

La validación completa y el ajuste final de géneros y popularidad ya fueron
ejecutados sobre el catálogo real, conservando las selecciones de desarrollo.
La ejecución completa de la aplicación con esos artefactos permanece pendiente.
La preparación completa del entorno y
del dataset se describe en el [README principal](../../../README.md).

## 1. Qué se añadió

| Archivo | Para qué sirve |
|---|---|
| `src/zta/models/targets.py` | Construye lo que se predice: matriz 0/1 de géneros y la popularidad. |
| `src/zta/models/registry.py` | Define los 4 modelos, el baseline y las variantes que se prueban de cada uno. |
| `src/zta/models/evaluation.py` | Métricas de cada tarea. |
| `src/zta/models/runner.py` | Entrenamiento: validación cruzada y ajuste final con evaluación en prueba. |
| `src/zta/models/__main__.py` | Comando `python -m zta.models` para entrenar. |
| `src/zta/models/artifacts.py` | Valida esquema, normalizador, características, etiquetas, catálogo, particiones y compatibilidad de versiones. |
| `src/zta/models/inference.py` | Carga un modelo guardado compatible y puntúa canciones nuevas. |
| `src/zta/models/app.py` | `MusicRecommender`: sliders → perfil → géneros y canciones recomendadas. |
| `src/zta/models/demo.py` | Prueba rápida desde la terminal. |
| `src/zta/api/` | API FastAPI (`schemas.py`, `main.py`, `__main__.py`). |

## 2. Qué se entrena y con qué datos

**Entrada (X), igual para los cuatro modelos:** las 9 variables acústicas (`danceability`, `energy`,
`valence`, `acousticness`, `instrumentalness`, `liveness`, `loudness`, `tempo`, `speechiness`)
normalizadas con el `MinMaxNormalizer` del equipo, ajustado **solo con las canciones de entrenamiento**.

**Objetivos (y).** El dataset no contiene valoraciones ni interacciones de usuarios.
Se utilizan los géneros registrados y la popularidad consolidada durante la limpieza:

| Tarea | Qué predice | Para qué se usa |
|---|---|---|
| `genre` | Una puntuación por cada uno de los 114 géneros; una canción puede tener varias etiquetas. | Ordenar géneros asociados a las características del perfil, como estimación auxiliar. |
| `popularity` | Popularidad consolidada por canción, calculada como media de los valores distintos observados. | Proporcionar una estimación orientativa, presentada entre 0 y 100 en la aplicación. |

| Modelo | Implementación | Variantes probadas |
|---|---|---|
| Regresión Lineal | `LinearRegression` | – |
| KNN | `KNeighborsRegressor` con ponderación por distancia | `n_neighbors` = 15, 50, 100 |
| Random Forest | `RandomForestRegressor` | `min_samples_leaf` = 5, 20 |
| SVM | Géneros: `OneVsRest(LinearSVC)`. Popularidad: `SVR` RBF con hasta 15.000 filas de entrenamiento | Géneros: `C` = 0.1, 1. Popularidad: `C` = 1, 10; `epsilon` = 1. |

KNN y Random Forest se utilizan como regresores sobre la matriz 0/1 de géneros:
sus salidas son promedios ponderados de etiquetas y permiten ordenar los géneros.
No se deben presentar como probabilidades calibradas. SVM utiliza márgenes de
decisión y la regresión lineal puede producir valores fuera de `[0,1]`; las
puntuaciones se deben interpretar dentro de cada modelo.

La referencia predice la media del entrenamiento: prevalencia por etiqueta para
géneros y popularidad media para regresión. SVR ajusta el estimador con una
submuestra reproducible de hasta 15.000 filas; utiliza el normalizador del
entrenamiento completo y las mismas canciones de validación que las otras familias.

Los indicadores supervisados se calculan sobre canciones. Su desempeño sobre
centroides de perfiles y su utilidad para preferencias personales requieren una
evaluación específica. La generación de listas ZTA no convierte automáticamente
estas predicciones en etiquetas de relevancia del usuario.

## 3. Preparar los datos
Se debe activar el entorno e instalar las dependencias siguiendo el README
principal. Si el CSV original no está disponible, se puede obtener mediante
`python src/download_data.py`. Si ya existe, se debe conservar y continuar con:

```bash
python -m clean.export_dataset
python -m zta.data --save-partitions
python -m zta.data --check-partitions
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
```

La etapa final se debe ejecutar después de cerrar las decisiones de desarrollo:
ajusta con todo desarrollo y evalúa sobre prueba. Ya se ejecutó para los
resultados de referencia de esta guía:

```bash
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
| `--confirm-test` | Obligatorio en `final`: autoriza explícitamente evaluar sobre el conjunto de prueba reservado. |

Una ejecución con todos los folds y sin límite de filas guarda la selección en
`reports/models/<tarea>`. Una ejecución con folds parciales o límite de filas se
guarda en `reports/models/<tarea>/smoke` y no sustituye la selección completa.

`--mode final` verifica una validación completa compatible con el catálogo,
particiones, configuraciones y versión de scikit-learn actuales, además de las
huellas de `best_configs.json`, `cv_folds.csv` y `cv_summary.csv`. Una ejecución
reducida no habilita el ajuste final.

Repetir una validación en su mismo destino sustituye sus reportes. La etapa
final exige `--confirm-test` y rechaza un destino con `final_manifest.json` ya
existente. Se deben conservar también los reportes de validación utilizados
para la selección y no escoger nuevas configuraciones a partir de prueba.

### Resultados comprobados de validación

Cada tarea evaluó nueve alternativas incluyendo la referencia en cinco folds,
para un total de 45 resultados por tarea. La siguiente tabla presenta la mejor
configuración de cada familia, con media y desviación estándar entre folds en
los indicadores principales:

| Modelo | Géneros: LRAP | Popularidad: RMSE | Popularidad: MAE medio |
| --- | ---: | ---: | ---: |
| Random Forest | **0.4157 ± 0.0024** | **18.1981 ± 0.1493** | 14.3395 |
| KNN | 0.3702 ± 0.0030 | 18.3951 ± 0.1726 | **14.1322** |
| SVM | 0.2773 ± 0.0045 | 19.8454 ± 0.1643 | 15.7200 |
| Regresión lineal | 0.2301 ± 0.0032 | 20.2452 ± 0.1582 | 16.6749 |
| Referencia | 0.0459 ± 0.0018 | 20.5627 ± 0.1542 | 17.2253 |

Fuente: `reports/models/{genre,popularity}/cv_summary.csv`.

Se maximiza LRAP en géneros y se minimiza RMSE en popularidad. Random Forest
obtuvo los mejores indicadores principales; KNN obtuvo menor MAE. El R² medio de
Random Forest para popularidad fue 0.2168. LRAP no corresponde a un porcentaje de
exactitud ni a Precision@10 de recomendaciones.

Las configuraciones seleccionadas dentro de cada familia fueron:

| Familia | Géneros | Popularidad |
| --- | --- | --- |
| Regresión lineal | Configuración única. | Configuración única. |
| KNN | 100 vecinos, ponderación por distancia. | 100 vecinos, ponderación por distancia. |
| Random Forest | 100 árboles; mínimo 5 observaciones por hoja; `max_samples=0.5`. | 200 árboles; mínimo 5 observaciones por hoja. |
| SVM | `C=1`, SVM lineal uno contra el resto. | `C=10`, `epsilon=1`, kernel RBF y hasta 15.000 filas. |

Los resultados por fold se guardan en `cv_folds.csv`, el resumen en
`cv_summary.csv`, la selección por familia en `best_configs.json` y su contexto
de integridad en `cv_manifest.json`. Estos archivos permiten revisar la selección;
todavía no son artefactos de inferencia ni resultados finales sobre prueba.

La evaluación de listas se ejecuta separadamente mediante `python -m zta.evaluation`
y se guarda en `reports/recommendations`. Comparó 2.850 perfiles y seleccionó
Mahalanobis por Precision@10 de Familiar (10.02 % frente a 9.10 % de Euclidiana).
El género de relevancia se utiliza solo en el evaluador y no filtra candidatas.

### Resultados finales sobre prueba

Se ajustaron los modelos con 76.145 canciones de desarrollo y se evaluaron sobre
13.438 canciones de prueba. Se mantuvieron las configuraciones seleccionadas.
Se evaluó además la referencia basada en medias de desarrollo.

| Modelo | Géneros: LRAP | Popularidad: RMSE | Popularidad: MAE | Popularidad: R² |
| --- | ---: | ---: | ---: | ---: |
| Random Forest | 0.4298 | 18.2538 | 14.2974 | 0.2237 |
| KNN | 0.3804 | 18.5048 | 14.0837 | 0.2022 |
| SVM | 0.2775 | 20.0355 | 15.9164 | 0.0648 |
| Regresión lineal | 0.2273 | 20.3911 | 16.8159 | 0.0313 |
| Referencia | 0.0454 | 20.7181 | 17.3582 | −0.0001 |

Fuente: `reports/models/{genre,popularity}/test_metrics.csv`. Son resultados de
un único conjunto de prueba, no medias entre folds. La referencia no se exporta
como modelo de inferencia. Cada tarea conserva el normalizador, sus reportes de
rangos y `final_manifest.json`, con huellas de selección, artefactos y métricas.

La evaluación final ZTA se ejecutó con
`python -m zta.evaluation --mode final --confirm-test` y quedó guardada en
`reports/recommendations/final`: Mahalanobis obtuvo Precision@10 de 9.86 % en
Familiar sobre 570 perfiles. No se volvió a seleccionar la distancia con prueba.

## 5. Usar los modelos

Antes de iniciar la demostración o la API se debe completar la etapa final de
géneros y popularidad. `MusicRecommender` carga los siguientes archivos:

```text
data/processed/models/
  genre/linear.joblib
  genre/knn.joblib
  genre/rf.joblib
  genre/svm.joblib
  popularity/rf.joblib
```

El ajuste final de popularidad genera también las otras familias solicitadas,
pero la capa de aplicación utiliza Random Forest para esa estimación. Se
comprueba la compatibilidad de los artefactos, incluido el normalizador ajustado
con desarrollo. La validación cruzada por sí sola no genera estos archivos.

### 5.1 Desde la terminal

```bash
python -m zta.models.demo --danceability 0.8 --energy 0.7 --acousticness 0.1 --mode discovery --k 5
python -m zta.models.demo --energy 0.2 --acousticness 0.9 --mode familiar --genres jazz blues
```

También se pueden indicar canciones mediante `--seeds`, seguido de uno o más
identificadores reales del catálogo, en lugar de los controles acústicos.

Cada control utiliza valores de 0 a 1 dentro del espacio normalizado con
desarrollo. Los controles omitidos toman la media normalizada de desarrollo.
La demostración muestra el perfil, las puntuaciones de géneros, la popularidad
estimada y las recomendaciones de ZTA y de búsqueda de vecinos. Esta búsqueda
de canciones utiliza `NearestNeighbors`; es una operación diferente del
`KNeighborsRegressor` empleado en las tareas supervisadas.

### 5.2 API HTTP

```bash
python -m zta.api --port 8000
```

El servidor carga el catálogo y los modelos al iniciar. La documentación
interactiva queda disponible en <http://127.0.0.1:8000/docs> y el contrato en
`/openapi.json`. La ejecución completa con los artefactos finales reales todavía
debe verificarse; los resultados de validación anteriores no constituyen esa
verificación de la aplicación.

| Método y ruta | Función |
|---|---|
| `GET /health` | Estado, número de canciones y de géneros. |
| `GET /meta` | Lo que necesita el front: rango y valor neutro de cada slider, lista de géneros, modos, métricas y modelos. |
| `GET /songs/search?q=…&limit=10` | Busca canciones por título o artista (una fila por canción distinta). |
| `POST /profile` | Función 1: perfil + géneros que encajan por modelo + popularidad esperada. |
| `POST /recommend` | Función 2: canciones recomendadas con `method` `zta` o `knn`. |

El cuerpo de `/profile` y `/recommend` define el perfil mediante
**`sliders` o `seed_ids` (exactamente uno)**.

```bash
curl -X POST http://127.0.0.1:8000/profile -H 'content-type: application/json' \
  -d '{"sliders": {"danceability": 0.8, "energy": 0.7, "acousticness": 0.1}, "top_genres": 3}'

curl -X POST http://127.0.0.1:8000/recommend -H 'content-type: application/json' \
  -d '{"sliders": {"energy": 0.2, "acousticness": 0.9}, "method": "zta", "mode": "familiar", "k": 5, "genres": ["jazz", "blues"]}'

curl -X POST http://127.0.0.1:8000/recommend -H 'content-type: application/json' \
  -d '{"seed_ids": ["<track_id>"], "method": "knn", "k": 5}'
```

En el último ejemplo se debe sustituir `<track_id>` por un identificador real
del catálogo.

Parámetros de `/recommend`: `method` (`zta`|`knn`), `mode` (`familiar`|`discovery`|`exploration`, solo ZTA),
`metric` (`euclidean`|`mahalanobis`, solo ZTA), `k` (1–50), `genres`, `popularity_lambda` (≥ 0, solo ZTA;
0 = sin penalización por popularidad) y `dedupe`.

Los datos inválidos devuelven **422** con un campo `detail` (una lista si falla la validación del esquema,
un texto si falla una regla de negocio, por ejemplo un género o canción inexistente).

La API admite filtros de género y deduplicación por título/artista sobre el
catálogo completo. La validación ZTA documentada utiliza únicamente canciones
del fold de validación, sin filtro de género y con identidad por `track_id`.
Se deben distinguir estos escenarios al presentar resultados de la aplicación.
