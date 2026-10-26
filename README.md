# Proyecto CC2: recomendación musical mediante ZTA

Proyecto de Ciencias de la Computación II para generar recomendaciones musicales
controladas mediante Zonas de Tolerancia Acústica (ZTA). A partir de canciones
semilla se construye un perfil acústico y se obtienen recomendaciones en tres
modos: Familiar, Descubrimiento y Exploración.

Actualmente se pueden reproducir la auditoría y limpieza, las particiones, el
modelo matemático ZTA, la validación cruzada de cuatro familias de modelos
supervisados y la evaluación de recomendaciones mediante perfiles sintéticos.
Las ejecuciones completas y los resultados de referencia se describen más abajo.

También están implementadas una capa de aplicación, una demostración por consola
y una API HTTP. Su uso con el catálogo real requiere generar los modelos finales,
etapa todavía pendiente junto con la evaluación final sobre prueba y la
verificación completa de la aplicación. Las interfaces Streamlit disponibles
corresponden a auditoría y limpieza.

## 1. Preparación del entorno

Los comandos se deben ejecutar desde la raíz del repositorio, la carpeta `cc2-project`.
El entorno de referencia utiliza Python 3.14.8. Las dependencias fijadas en
`requirements.txt` requieren al menos Python 3.12 por la versión de NumPy,
aunque `pyproject.toml` declare un mínimo de Python 3.10 para el paquete.

En Linux o macOS:

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

En Windows PowerShell, se debe crear el entorno con `python -m venv .venv` y
activarlo con `.\.venv\Scripts\Activate.ps1`. Posteriormente, se deben ejecutar
los mismos comandos de instalación con `python -m pip`.

La instalación editable registra los paquetes de `src`, incluido `zta`, y refleja
los cambios al código sin reinstalar. Si el entorno ya está preparado, únicamente
se requiere activarlo. Se debe repetir la instalación cuando se cree nuevamente
el entorno o cambien las dependencias.
`requirements.txt` fija dependencias del entorno y las aplicaciones;
`pip install -e .` instala el proyecto y resuelve también las dependencias
declaradas en `pyproject.toml`.

Para reproducir las versiones de las bibliotecas de entrenamiento utilizadas en
los resultados de referencia, se debe ejecutar después de la instalación:

```bash
python -m pip install scikit-learn==1.9.1 scipy==1.18.1 joblib==1.6.0 threadpoolctl==3.7.0
```

Estas versiones no están fijadas en `requirements.txt`. La ejecución de referencia
utilizó, además, NumPy 2.5.2, pandas 3.0.5 y Matplotlib 3.11.1, fijados en ese archivo.
Las versiones y condiciones efectivamente utilizadas quedan registradas en los
manifiestos y, para los modelos supervisados, en `execution_environment.json`.

## 2. Obtención del dataset original

Los datos y resultados generados están excluidos de Git. Después de clonar el
repositorio, se debe disponer del CSV original en `data/raw/dataset.csv`.
Si el archivo no se encuentra disponible, se debe ejecutar:

```bash
python src/download_data.py
```

El script utiliza `kagglehub` para descargar
`maharshipandya/-spotify-tracks-dataset` en `data/raw`. Necesita Internet y fuerza
una nueva descarga. Si ya se dispone del CSV correcto, se debe omitir este paso.
También se puede colocar manualmente en esa ubicación el CSV original del mismo
dataset, conservando su columna de índice de origen.

La versión utilizada tiene 114.000 filas. Los conteos de esta guía corresponden
a esa entrada y a las reglas actuales; otra versión puede producir resultados
distintos. El manifiesto de limpieza registra la huella SHA-256 del CSV utilizado.

## 3. Reproducción del recorrido completo

Con el entorno activo y el CSV disponible, se deben ejecutar los siguientes
comandos en el orden indicado:

```bash
python -m clean.export_dataset
python -m zta.data --save-partitions
python -m zta.data --check-partitions
python -m zta.example --fold 1 --k 10 --save
python -m zta.models --task genre --mode cv
python -m zta.models --task popularity --mode cv
python -m zta.evaluation
```

| Comando | Resultado |
| --- | --- |
| `clean.export_dataset` | Genera el catálogo depurado, sus géneros y reportes de calidad en `data/processed`. |
| `zta.data --save-partitions` | Valida las nueve características y guarda los identificadores de desarrollo, prueba y cinco folds. |
| `zta.data --check-partitions` | Comprueba que las particiones corresponden al catálogo y la configuración actuales. |
| `zta.example --fold 1 --k 10 --save` | Ejecuta el modelo completo y exporta seis listas de recomendaciones, tablas, parámetros y tres gráficos en PNG y PDF. |
| `zta.models --task genre --mode cv` | Compara las cuatro familias y una referencia para predicción multietiqueta de géneros mediante los cinco folds. |
| `zta.models --task popularity --mode cv` | Compara las cuatro familias y una referencia para estimación de popularidad mediante los cinco folds. |
| `zta.evaluation` | Compara las dos distancias de ZTA con perfiles sintéticos compartidos, calcula métricas de listas y guarda la selección de distancia. |

El ejemplo necesita las tablas depuradas y las particiones guardadas. Ajusta
la normalización y la covarianza con entrenamiento del fold seleccionado y
consulta sus canciones de validación. No necesita exportaciones previas de
normalización, perfiles o recomendaciones: integra esas operaciones y guarda
sus propios resultados en `data/processed/worked_example/fold_1`.

Si ya se dispone del catálogo y de particiones válidas, se puede comenzar con
`python -m zta.data --check-partitions` y ejecutar el ejemplo sin repetir las
etapas anteriores.

Los comandos de validación supervisada y evaluación ZTA necesitan el catálogo y
las particiones, pero son independientes del ejemplo y de las exportaciones
intermedias. Este recorrido utiliza entrenamiento y validación de desarrollo;
no ejecuta el ajuste final ni la evaluación sobre prueba.

### Resultados de referencia

| Elemento | Cantidad |
| --- | ---: |
| Canciones únicas depuradas | 89.583 |
| Asociaciones canción–género | 113.392 |
| Géneros | 114 |
| Canciones de desarrollo | 76.145 |
| Canciones de prueba reservadas | 13.438 |
| Entrenamiento del fold 1 | 60.916 |
| Validación del fold 1 | 15.229 |
| Semillas del ejemplo predeterminado | 5 |
| Candidatas del ejemplo, excluyendo semillas | 15.224 |

Las semillas predeterminadas son las primeras cinco canciones de validación
ordenadas por `track_id`. Es una elección determinista para ilustrar el cálculo,
no un perfil observado de un usuario ni una selección por afinidad de género.
Con `k=10`, este ejemplo obtiene diez canciones en cada una de las seis listas:
tres modos por dos medidas de distancia.

## 4. Ejecución de las etapas por separado

Para inspeccionar las salidas intermedias, primero se debe exportar el catálogo
y guardar las particiones. Posteriormente, se deben ejecutar los siguientes
comandos en orden:

```bash
python -m zta.data
python -m zta.preprocessing --save
python -m zta.domain --fold 1 --save
python -m zta.recommendation --fold 1 --k 10 --save
```

| Etapa | Función y salida |
| --- | --- |
| `zta.data` | Verifica el catálogo y resume la matriz acústica de nueve columnas; no guarda una copia de la matriz. |
| `zta.preprocessing --save` | Ajusta un normalizador por fold con entrenamiento. Guarda cinco archivos de parámetros, reporte de rangos y manifiesto en `data/processed/normalization`. |
| `zta.domain --fold 1 --save` | Reutiliza el normalizador guardado, construye el perfil y calcula distancias. Exporta a `data/processed/profile_distances/fold_1`. |
| `zta.recommendation --fold 1 --k 10 --save` | Lee el perfil y las distancias guardadas, calibra las zonas y exporta seis listas, umbrales y manifiesto a `data/processed/recommendations/fold_1`. |

`zta.domain` exige la exportación de normalización. `zta.recommendation` exige
la exportación de perfil y distancias del mismo fold. El ejemplo integrado es
una alternativa a esta cadena; no rellena sus directorios intermedios.

En los módulos que ofrecen `--save`, omitir esa opción permite calcular y revisar
el resumen JSON sin guardar resultados. En `zta.data`, guardar particiones
requiere `--save-partitions` y verificarlas requiere `--check-partitions`;
esas dos opciones son excluyentes. Las opciones disponibles se pueden consultar
mediante el siguiente comando:

```bash
python -m zta.example --help
```

### Modificación de la consulta

Para ejecutar otro fold y solicitar una cantidad diferente de recomendaciones,
se puede utilizar el siguiente comando:

```bash
python -m zta.example --fold 2 --k 5 --save
```

Para seleccionar semillas explícitas, se deben utilizar identificadores distintos
pertenecientes a validación del fold elegido. Las dos canciones del siguiente
comando pertenecen a validación del fold 1 con las particiones actuales:

```bash
python -m zta.example --fold 1 --k 10 --seeds \
  000Iz0K615UepwSJ5z2RE5 \
  0017XiMkqbTfF2AUOzlhj6
```

Para modificar la penalización de popularidad, se puede ejecutar:

```bash
python -m zta.example --fold 1 --k 10 --popularity-lambda 0.2
```

`--seeds` también está disponible en `zta.domain`, y `--popularity-lambda` en
`zta.recommendation`. Las dos consultas anteriores no guardan archivos porque
omiten `--save`. Si se incorpora esa opción, los resultados sustituyen la
exportación anterior del mismo módulo y fold. Para comparar varias consultas,
se debe conservar una copia de los resultados de cada ejecución.

## 5. Funcionamiento del modelo matemático

La configuración central está en `src/zta/config.py`.

1. **Características:** utiliza, en este orden, `danceability`, `energy`,
   `valence`, `acousticness`, `instrumentalness`, `liveness`, `loudness`, `tempo`
   y `speechiness`. Popularidad, títulos, identificadores, géneros y marcas de
   revisión quedan fuera del vector acústico.
2. **Particiones:** reserva el 15 % para prueba y divide el 85 % de desarrollo
   en cinco folds de validación. La semilla aleatoria es 42. Se divide por
   identificador de canción, sin estratificación por género.
3. **Normalización:** aprende mínimos y máximos solo con entrenamiento de cada
   fold. Transforma a `[0,1]`, recorta valores fuera de esos límites y transforma
   las características constantes a cero. Conserva el catálogo original.
4. **Perfil:** promedia las nueve componentes normalizadas de las semillas.
5. **Distancias:** calcula Euclídea y Mahalanobis. Esta última utiliza covarianza
   muestral de entrenamiento más `1e-6 * I` como regularización inicial.
6. **Umbrales:** obtiene los percentiles 25, 60 y 90 de las distancias entre el
   perfil y entrenamiento. Las candidatas no se utilizan para estimarlos.
7. **Zonas:** filtra las candidatas mediante las bandas siguientes.
8. **Ranking:** ordena dentro de la zona por puntuación descendente; los empates
   se resuelven por `track_id` ascendente.

| Modo | Banda admitida |
| --- | --- |
| Familiar | `0 <= distancia <= Q25` |
| Descubrimiento | `Q25 < distancia <= Q60` |
| Exploración | `Q60 < distancia <= Q90` |

Las canciones por encima de `Q90` quedan fuera. Las semillas se excluyen por ID;
otra canción con distancia cero sigue siendo elegible en Familiar. Si una banda
está vacía o tiene menos de `k` candidatas, se devuelven solo las disponibles y
se informa el estado de la lista.

La puntuación es:

```text
score = -distance - popularity_lambda * (popularity / 100)
```

Con `popularity_lambda=0`, el ranking depende solo de la distancia. Un valor
positivo penaliza popularidad después de filtrar por zona; no cambia los
umbrales ni la pertenencia a las bandas. Familiar expresa cercanía acústica,
sin acreditar escucha previa; Exploración tampoco acredita novedad subjetiva.
Las dos distancias son variantes de ZTA. Las cuatro familias supervisadas se
comparan separadamente mediante tareas de predicción de géneros y popularidad.

## 6. Validación de los modelos supervisados

Se comparan regresión lineal, KNN, Random Forest y SVM, además de una referencia
basada en la media de los objetivos de entrenamiento. Las entradas son las nueve
características acústicas. En cada fold, el normalizador se ajusta exclusivamente
con entrenamiento.

Cada tarea evalúa nueve alternativas, incluyendo la referencia, en cinco folds:
45 resultados por tarea y 90 entre ambas. Se elige la mejor configuración de
cada familia por LRAP medio en géneros y RMSE medio en popularidad.

### Resultados comprobados de desarrollo

La tabla presenta la mejor configuración de cada familia. LRAP y RMSE se expresan
como media y desviación estándar muestral entre los cinco folds; MAE es la media.

| Modelo | Géneros: LRAP, mayor es mejor | Popularidad: RMSE, menor es mejor | Popularidad: MAE |
| --- | ---: | ---: | ---: |
| Random Forest | **0.4157 ± 0.0024** | **18.1981 ± 0.1493** | 14.3395 |
| KNN | 0.3702 ± 0.0030 | 18.3951 ± 0.1726 | **14.1322** |
| SVM | 0.2773 ± 0.0045 | 19.8454 ± 0.1643 | 15.7200 |
| Regresión lineal | 0.2301 ± 0.0032 | 20.2452 ± 0.1582 | 16.6749 |
| Referencia | 0.0459 ± 0.0018 | 20.5627 ± 0.1542 | 17.2253 |

Fuente: `reports/models/{genre,popularity}/cv_summary.csv`. Cada carpeta conserva
también los resultados por fold y `best_configs.json`.

Random Forest obtuvo el mejor indicador principal en ambas tareas. KNN obtuvo
menor MAE en popularidad; la elección por RMSE conserva el criterio definido.
El R² medio de Random Forest para popularidad fue 0.2168. Estos indicadores
describen predicción de atributos de canciones y no precisión de recomendaciones
ni satisfacción de usuarios. La validación sobre canciones tampoco demuestra
por sí sola el desempeño de estos predictores sobre centroides de perfiles.

Las configuraciones elegidas dentro de cada familia, los límites de ajuste de SVR
y los comandos de la API se detallan en [la guía de modelos](src/zta/models/README.md).

### Comprobaciones reducidas y ajuste final pendiente

Para comprobar la conexión de entrenamiento con menos datos, se puede ejecutar:

```bash
python -m zta.models --task genre --mode cv --folds 1 --max-train-rows 5000
```

Las ejecuciones con folds parciales o límite de filas se guardan en
`reports/models/<tarea>/smoke`. No sustituyen los resultados completos ni sirven
como selección para el ajuste final.

Una vez cerradas las decisiones de desarrollo, el módulo supervisado permite
ajustar las configuraciones seleccionadas con las 76.145 canciones de desarrollo
y evaluarlas sobre las 13.438 canciones de prueba:

```bash
python -m zta.models --task genre --mode final --confirm-test
python -m zta.models --task popularity --mode final --confirm-test
```

Estos dos comandos corresponden a una etapa pendiente y se deben ejecutar cuando
se haya decidido realizar la evaluación final. Exigen una validación completa
compatible con el catálogo, particiones, configuraciones y versión de
scikit-learn actuales. Generan modelos `.joblib` en `data/processed/models` y
`test_metrics.csv` en los reportes de cada tarea.

`--confirm-test` exige una decisión explícita, pero no bloquea una segunda
ejecución. Se deben conservar los resultados y evitar utilizar prueba para
escoger nuevas configuraciones.

## 7. Evaluación de las recomendaciones ZTA

La evaluación completa se ejecuta mediante:

```bash
python -m zta.evaluation
```

El comando genera cinco perfiles por género elegible y por fold. Cada perfil
utiliza cinco semillas del género y considera relevantes todas las restantes
canciones de validación que compartan esa etiqueta. Las candidatas son todas las
canciones de validación excepto las semillas. El género objetivo se utiliza
exclusivamente en el evaluador; no se entrega a ZTA como filtro.

Las dos distancias y los tres modos utilizan los mismos perfiles y objetivos.
Los percentiles son 25, 60 y 90, la regularización de Mahalanobis es `1e-6` y la
penalización de popularidad predeterminada es cero. Los umbrales se calculan con
la referencia de entrenamiento para cada perfil.

Se calculan Precision@K, Recall@K, Hit Rate@K, diversidad interna (ILD), indicador
inverso de popularidad, cobertura de géneros y completitud. Precision@K conserva
el denominador K incluso con listas incompletas. ILD utiliza la misma distancia
euclidiana normalizada, dividida por la raíz del número de características, para
comparar ambas configuraciones. Las métricas no definidas permanecen vacías en
los CSV y se acompañan de recuentos de observaciones evaluables.

### Resultados comprobados de desarrollo

Los 114 géneros fueron elegibles en los cinco folds. Se evaluaron 2.850 perfiles,
17.100 listas y 171.000 recomendaciones. La tabla presenta medias de los folds;
Precision@10 incluye su desviación estándar, expresada en puntos porcentuales.

| Distancia | Modo | Precision@10 (%) | Hit Rate@10 (%) | ILD |
| --- | --- | ---: | ---: | ---: |
| Euclidiana | Familiar | 9.10 ± 0.47 | 46.49 | 0.0688 |
| Euclidiana | Descubrimiento | 1.31 ± 0.08 | 11.86 | 0.2036 |
| Euclidiana | Exploración | 0.53 ± 0.08 | 4.88 | 0.2613 |
| Mahalanobis | Familiar | **10.02 ± 0.74** | 49.19 | 0.0810 |
| Mahalanobis | Descubrimiento | 1.18 ± 0.13 | 10.88 | 0.2198 |
| Mahalanobis | Exploración | 0.66 ± 0.06 | 6.35 | 0.2872 |

Fuente: `reports/recommendations/cv_summary.csv`.

Se seleccionó Mahalanobis por mayor Precision@10 medio en Familiar. La regla de
desempate compara la proporción de listas completas en Familiar y, si persiste
la igualdad, prefiere Euclidiana. La decisión se guarda en `selection.json`;
este archivo no cambia automáticamente los valores predeterminados de otros
comandos ni ajusta modelos finales.

Todas las listas obtuvieron diez resultados. El 100 % de completitud describe
disponibilidad de canciones; Precision@10 mide afinidad sintética por género.
Estos resultados no demuestran precisión superior al 95 % ni satisfacción de
usuarios. La desviación estándar es descriptiva y no constituye una prueba de
significancia estadística. El conjunto de prueba permanece reservado.

Se observó mayor ILD media en los modos más alejados, junto con menor recuperación
del género objetivo. La popularidad y la cobertura de géneros no siguieron una
tendencia monótona entre los tres modos; no se debe suponer que Exploración
garantiza menor popularidad o mayor cobertura.

Para una comprobación reducida, se puede ejecutar:

```bash
python -m zta.evaluation --folds 1 --genres acoustic --profiles-per-genre 1
```

Los protocolos reducidos se guardan en `reports/recommendations/smoke` y se marcan
como no elegibles para la selección final. Cambiar folds, géneros, cantidad de
réplicas, semillas o K respecto al protocolo completo produce una ejecución
reducida. Las opciones se pueden consultar con `python -m zta.evaluation --help`.

`--reports-dir` permite conservar ejecuciones en directorios distintos. Repetir
un protocolo completo en el mismo directorio sustituye sus reportes. Si se
modifica `--popularity-lambda` o los percentiles de `Settings`, se debe utilizar
un directorio diferente para conservar la comparación original.

Este módulo implementa validación de desarrollo. La evaluación final de listas
ZTA sobre prueba todavía debe implementarse y ejecutarse tras cerrar la selección.

## 8. Resultados generados y trazabilidad

```text
data/
  raw/dataset.csv
  processed/
    tracks.csv
    track_genres.csv
    manifest.json                   # Limpieza y reportes adicionales
    partitions/                     # Asignaciones, géneros y manifiesto
    normalization/                  # Parámetros de los cinco folds y rangos
    profile_distances/fold_1/        # Perfil, modelo y distancias
    recommendations/fold_1/         # Seis listas y umbrales
    worked_example/fold_1/          # Ejemplo integrado, tablas y figuras
    models/                         # Se crea al ejecutar el ajuste final pendiente
reports/
  models/genre/                     # Validación supervisada de géneros
  models/popularity/                # Validación supervisada de popularidad
  recommendations/                  # Evaluación de listas ZTA y selección
```

Cada carpeta se crea con su comando correspondiente. La limpieza genera también
`data/processed/README.md` con la descripción de sus tablas. Las tareas que allí
se describen como posteriores a la limpieza se implementan actualmente en `zta`.

En `worked_example/fold_1` se generan los siguientes archivos:

| Archivo | Contenido |
| --- | --- |
| `raw_seeds.csv`, `normalized_seeds.csv` | Semillas originales y sus nueve características transformadas. |
| `feature_parameters.csv` | Mínimos, máximos y centroide por característica. |
| `validation_clipping.csv` | Conteos de valores fuera del rango de entrenamiento. |
| `profile.json`, `distance_model.json` | Perfil y parámetros del espacio y las distancias. |
| `reference_distances.csv`, `candidate_distances.csv` | Las dos distancias por canción. |
| `thresholds.json`, `zone_counts.csv` | Umbrales y conteos, incluyendo el conjunto exterior. |
| `euclidean_*.csv`, `mahalanobis_*.csv` | Una lista por modo para cada distancia. |
| `figures/acoustic_profile.png` y `.pdf` | Centroide de las nueve características. |
| `figures/distance_zones.png` y `.pdf` | Distribuciones de distancias de entrenamiento y sus bandas. |
| `figures/candidate_counts.png` y `.pdf` | Cantidad de candidatas por zona y medida. |
| `manifest.json` | Configuración, IDs, conteos, versiones y huellas de entrada, código y resultados. |

Las listas incluyen posición, identificador, título, artistas, géneros,
popularidad, distancia, puntuación, modo y límites de la banda. Los géneros se
exportan como arreglos JSON dentro de las celdas CSV.

Los manifiestos registran huellas SHA-256 para comprobar la correspondencia de
los archivos. Si cambia el catálogo o las particiones, se deben regenerar las
etapas dependientes en orden. La ruta por etapas vincula también los manifiestos por
sus huellas: volver a exportar una etapa puede exigir regenerar las siguientes
aunque sus valores calculados no hayan cambiado.

Con las mismas entradas, configuración y versiones se espera reproducir los
resultados numéricos. Las fechas de ejecución y sus manifiestos pueden cambiar;
reproducibilidad no implica igualdad de todos los archivos byte por byte.

### Reportes de entrenamiento y evaluación

| Directorio o archivo | Contenido |
| --- | --- |
| `reports/models/<tarea>/cv_folds.csv` | Resultados por modelo, configuración y fold. |
| `reports/models/<tarea>/cv_summary.csv` | Medias y desviaciones estándar por configuración. |
| `reports/models/<tarea>/best_configs.json` | Mejor configuración de cada familia para esa tarea. |
| `reports/models/<tarea>/cv_manifest.json` | Tipo de ejecución, contexto de entrenamiento, versión de scikit-learn y huellas de resultados. |
| `reports/recommendations/profiles.jsonl` | Perfiles, semillas y todos sus objetivos relevantes. |
| `reports/recommendations/fold_candidates.csv` | IDs del conjunto de evaluación por fold; se excluyen las semillas de cada perfil. |
| `reports/recommendations/recommendations.csv` | Listas con IDs, posiciones, distancias, puntuaciones y bandas. |
| `reports/recommendations/profile_metrics.csv` | Métricas y tiempos por perfil, distancia y modo. |
| `reports/recommendations/fold_metrics.csv`, `cv_summary.csv` | Resultados por fold y resumen entre folds, con recuentos evaluables. |
| `reports/recommendations/genre_eligibility.csv`, `selection.json` | Géneros elegibles y criterio de selección de distancia. |
| `reports/recommendations/fold_*_distance_model.json` | Normalización, covarianza y regularización ajustadas con entrenamiento. |
| `reports/recommendations/fold_*_normalization_ranges.csv` | Rangos y recortes al normalizar entrenamiento y validación. |
| `reports/recommendations/manifest.json` | Configuración, versiones, cantidades y huellas de entradas lógicas, código y resultados. |
| `cv_comparison.png` | Gráfico de cada ejecución de validación; ZTA exporta además PDF. |

Las revisiones de referencia conservaron, además, `execution.log` y
`execution_environment.json` en cada tarea supervisada, así como
`reports/models/cv_comparison_review.png` y `.pdf`. Estos archivos adicionales
proceden de la revisión de la ejecución; los comandos ordinarios de entrenamiento
no los generan. Los reportes principales de las tablas anteriores permiten
reproducir y consultar los resultados sin depender de esos archivos adicionales.

Los datos, reportes y modelos generados permanecen excluidos de Git. Se deben
conservar junto a sus manifiestos cuando se necesite compartir la evidencia de
una ejecución; clonar el código no recupera automáticamente esos archivos.

## 9. Aplicaciones y API

### Auditoría y limpieza

Se debe ejecutar una aplicación a la vez:

```bash
python -m streamlit run src/audit/dataset_audit.py
```

```bash
python -m streamlit run src/clean/dataset_cleaning.py
```

Ambas leen `data/raw/dataset.csv`. La aplicación de limpieza permite revisar las
reglas y guardar los resultados desde el botón final. La exportación por consola
aplica las mismas reglas sin abrir Streamlit. Un control `fail` bloquea la
exportación; las condiciones `review` se conservan y documentan. Los extremos
estadísticos se marcan sin eliminarlos por esa condición, y los géneros múltiples
se conservan sin remuestrear ni asumir balance perfecto.

### Demostración de recomendaciones y API HTTP

Están implementados `MusicRecommender`, la demostración por terminal y la API
FastAPI. Para utilizarlos se deben generar primero los cuatro modelos finales
de géneros y el Random Forest final de popularidad mediante la etapa final de
la sección 6. La validación cruzada por sí sola no exporta estos modelos.

Después de disponer de esos artefactos compatibles, se pueden ejecutar:

```bash
python -m zta.models.demo --energy 0.7 --acousticness 0.1 --mode familiar --metric mahalanobis --k 10
python -m zta.api --host 127.0.0.1 --port 8000
```

La documentación HTTP queda disponible en `http://127.0.0.1:8000/docs`.
La [guía de modelos y API](src/zta/models/README.md) describe perfiles por semillas
o controles acústicos, filtros, búsqueda de canciones y peticiones HTTP.

La API permite escenarios con filtros de género, deduplicación de título/artista
y consultas al catálogo completo. Su funcionamiento con los artefactos finales
debe verificarse por separado: la evaluación de la sección 7 utiliza únicamente
candidatas de validación sin filtro de género y distingue canciones por ID.

## 10. Estado y trabajo pendiente

| Componente | Estado |
| --- | --- |
| Auditoría, limpieza y catálogo con géneros múltiples | Implementado. |
| Nueve características y particiones reproducibles | Implementado. |
| Normalización, perfil, distancias, bandas y ranking ZTA | Implementado. |
| Ejemplo completo con tablas, parámetros y gráficos | Implementado. |
| Cuatro familias supervisadas y referencia | Implementadas; validación completa ejecutada para géneros y popularidad. |
| Simulación por afinidad de género y métricas de listas | Implementada; 2.850 perfiles y 17.100 listas evaluadas. |
| Selección de configuraciones supervisadas en validación | Completada por familia; Random Forest obtuvo los mejores indicadores principales. |
| Comparación de distancias ZTA en validación | Completada; Mahalanobis seleccionada por Precision@10 en Familiar. |
| Ajuste final de modelos supervisados y evaluación sobre prueba | Implementado en el módulo; pendiente de ejecución y generación de artefactos reales. |
| Evaluación final de listas ZTA sobre prueba | Pendiente de implementación y ejecución. |
| Capa de aplicación, demostración y API | Implementadas; pendientes de ejecución completa con artefactos finales del catálogo real. |
| Verificación de las diez funcionalidades y entrega de la aplicación | Pendiente. |

La formulación matemática, sus ejemplos y la evaluación de desarrollo disponen
de resultados comprobados. La validación y selección no sustituyen la evaluación
final sobre prueba ni la verificación de la aplicación. El conjunto de prueba
permanece reservado durante el recorrido de desarrollo de esta guía.

## Problemas frecuentes

- **`ModuleNotFoundError: zta`, `clean` o `common`:** se debe activar el entorno y ejecutar
  `python -m pip install -e .` desde la raíz del repositorio.
- **Falta `dataset.csv`:** se debe descargar o colocar el original en `data/raw`
  antes de abrir las aplicaciones o ejecutar la limpieza.
- **Faltan `tracks.csv` o `track_genres.csv`:** se debe ejecutar
  `python -m clean.export_dataset`.
- **Faltan particiones:** se debe ejecutar `python -m zta.data --save-partitions`.
- **Faltan parámetros o distancias:** se debe seguir el orden de la ruta por
  etapas o utilizar `zta.example` después de guardar las particiones.
- **Error de integridad o correspondencia:** se deben regenerar los resultados
  desde la primera etapa cuya entrada cambió. Los manifiestos se deben conservar
  junto a sus tablas y parámetros.
- **Semillas desconocidas en un ejemplo:** se deben utilizar IDs de validación
  del fold elegido. Estos se pueden consultar en
  `data/processed/partitions/assignments.csv`, seleccionando `split=development`
  y el número de fold correspondiente.
- **El ajuste final rechaza la selección:** se debe ejecutar validación completa
  de la tarea con todos los folds, sin límite de filas y con el catálogo,
  particiones y versión de scikit-learn actuales. Una ejecución `smoke` no sirve
  para esta selección.
- **Faltan archivos `.joblib` al iniciar la demostración o la API:** la validación
  cruzada no guarda modelos finales. Se debe completar la etapa de ajuste final
  cuando se cierre la selección de desarrollo.
- **Modelos guardados incompatibles:** se deben conservar juntos modelos,
  etiquetas, características, normalizador y contexto de entrenamiento de una
  misma ejecución compatible. No se deben mezclar artefactos de catálogos o
  versiones de scikit-learn diferentes.
