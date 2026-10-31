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
y una API HTTP. Ya se generaron los modelos finales y se ejecutaron las
evaluaciones supervisadas y ZTA sobre prueba. Permanece pendiente la
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

Las versiones de las bibliotecas de entrenamiento utilizadas en los resultados de
referencia están fijadas en `requirements.txt`: scikit-learn 1.9.1, SciPy 1.18.1,
joblib 1.6.0 y threadpoolctl 3.7.0. Se instalan con el mismo comando de preparación
del entorno, junto con NumPy 2.5.2, pandas 3.0.5 y Matplotlib 3.11.1.
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

### Comprobaciones reducidas y ajuste final

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

Estos comandos se ejecutaron después de cerrar las decisiones de desarrollo.
Para una reproducción desde cero, exigen una validación completa
compatible con el catálogo, particiones, configuraciones y versión de
scikit-learn actuales. Generan modelos `.joblib` en `data/processed/models` y
`test_metrics.csv` en los reportes de cada tarea.

`--confirm-test` exige una decisión explícita. Si ya existe `final_manifest.json`,
se rechaza otra ejecución en ese destino para conservar los resultados finales.
Se deben mantener también los archivos de validación que fundamentaron la
selección y evitar escoger nuevas configuraciones mediante prueba.

### Resultados comprobados sobre prueba

Se ajustaron las configuraciones seleccionadas con las 76.145 canciones de
desarrollo y se evaluaron sobre las 13.438 canciones reservadas. Se incluyó la
referencia basada en medias de desarrollo. La tabla contiene resultados del
único conjunto de prueba; no son medias ni desviaciones entre folds.

| Modelo | Géneros: LRAP | Popularidad: RMSE | Popularidad: MAE | Popularidad: R² |
| --- | ---: | ---: | ---: | ---: |
| Random Forest | 0.4298 | 18.2538 | 14.2974 | 0.2237 |
| KNN | 0.3804 | 18.5048 | 14.0837 | 0.2022 |
| SVM | 0.2775 | 20.0355 | 15.9164 | 0.0648 |
| Regresión lineal | 0.2273 | 20.3911 | 16.8159 | 0.0313 |
| Referencia | 0.0454 | 20.7181 | 17.3582 | −0.0001 |

Fuente: `reports/models/{genre,popularity}/test_metrics.csv`. Se conservaron las
configuraciones de validación; estos resultados no se utilizaron para una nueva
selección. Los ocho artefactos `.joblib` se guardaron con el mismo normalizador
ajustado exclusivamente con desarrollo. La referencia se evaluó sin exportar un
artefacto de inferencia.

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
significancia estadística. Esta validación no utilizó el conjunto de prueba;
la evaluación final se presenta a continuación.

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

### Evaluación final sobre prueba

Después de disponer de la selección completa de desarrollo, se puede ejecutar
una evaluación final mediante:

```bash
python -m zta.evaluation --mode final --confirm-test
```

Este comando verifica la selección y sus huellas, ajusta la normalización y la
covarianza con desarrollo y utiliza únicamente la distancia elegida. Los perfiles,
las semillas y las candidatas se toman del conjunto de prueba. Se mantienen las
cinco semillas, cinco réplicas por género y diez recomendaciones por modo. El
muestreo utiliza semilla 42 con un espacio de identificadores propio de prueba;
prueba no se trata como un sexto fold de validación.

Se evaluaron 570 perfiles, 1.710 listas y 17.100 recomendaciones con Mahalanobis.
Los tres modos obtuvieron listas completas, sin zonas vacías.

| Modo | Precision@10 (%) | Recall@10 (%) | Hit Rate@10 (%) | ILD |
| --- | ---: | ---: | ---: | ---: |
| Familiar | 9.86 | 0.7012 | 48.77 | 0.0833 |
| Descubrimiento | 1.26 | 0.0866 | 10.88 | 0.2207 |
| Exploración | 0.61 | 0.0413 | 5.61 | 0.2847 |

Fuente: `reports/recommendations/final/test_metrics.csv`. Las cifras son medias
de perfiles en un único conjunto de prueba. Las columnas `*_profile_std` del CSV
describen dispersión entre perfiles sintéticos que pueden compartir canciones;
no representan variación entre folds ni intervalos de confianza.

Los resultados finales, las listas, el modelo de distancia y la comparación con
validación se guardan en `reports/recommendations/final`. El comando rechaza
opciones de protocolo reducido, selecciones incompatibles y un destino con
`manifest.json` final ya existente. No se comparan de nuevo las distancias sobre
prueba para escoger una ganadora.

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
    models/                         # Ocho artefactos finales supervisados
reports/
  models/genre/                     # Validación supervisada de géneros
  models/popularity/                # Validación supervisada de popularidad
  recommendations/                  # Evaluación de listas ZTA y selección
    final/                          # Evaluación final de la distancia seleccionada
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
| `reports/models/<tarea>/test_metrics.csv` | Métricas finales de las cuatro familias y la referencia. |
| `reports/models/<tarea>/final_normalizer.json`, `final_normalization_ranges.csv` | Normalizador de desarrollo y comprobaciones de rangos en desarrollo y prueba. |
| `reports/models/<tarea>/final_manifest.json` | Selección utilizada, huellas de reportes y modelos, versiones y contexto final. |
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
| `reports/recommendations/final/test_metrics.csv` | Medias, dispersión descriptiva y cantidades evaluables sobre prueba. |
| `reports/recommendations/final/validation_test_comparison.csv`, `test_comparison.png` y `.pdf` | Comparación de la configuración seleccionada entre validación y prueba. |
| `reports/recommendations/final/manifest.json` | Selección congelada, protocolo, separación desarrollo/prueba y huellas finales. |

Las revisiones de referencia conservaron, además, `execution.log` y
`execution_environment.json` en cada tarea supervisada, así como
`reports/models/cv_comparison_review.png` y `.pdf`. Estos archivos adicionales
proceden de la revisión de la ejecución; los comandos ordinarios de entrenamiento
no los generan. Los reportes principales de las tablas anteriores permiten
reproducir y consultar los resultados sin depender de esos archivos adicionales.

La revisión final conservó también `reports/final_evaluation_plan.json`, registros
`final_execution.log` y `final_execution_environment.json` y comparaciones
supervisadas `validation_test_comparison.csv`, representadas en
`reports/models/final_comparison.png` y `.pdf`. Estos archivos adicionales se
generaron durante la revisión de la ejecución. Los manifiestos ordinarios de los
módulos conservan las huellas de selección, modelos y reportes principales.

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
FastAPI. Para utilizarlos se debe disponer de los cuatro modelos finales
de géneros y el Random Forest final de popularidad, ya generados en esta
ejecución. En una reproducción desde cero se deben obtener mediante la etapa
final de la sección 6; la validación cruzada por sí sola no los exporta.

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
candidatas del conjunto evaluado (validación o prueba) sin filtro de género y
distingue canciones por ID.

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
| Ajuste final de modelos supervisados y evaluación sobre prueba | Completado; ocho artefactos guardados y métricas de las cuatro familias y la referencia en ambas tareas. |
| Evaluación final de listas ZTA sobre prueba | Completada con Mahalanobis: 570 perfiles y 1.710 listas. |
| Capa de aplicación, demostración y API | Implementadas; pendientes de ejecución completa con artefactos finales del catálogo real. |
| Verificación de las diez funcionalidades y entrega de la aplicación | Pendiente. |

La formulación matemática, sus ejemplos y las evaluaciones de desarrollo y
prueba disponen de resultados comprobados. Prueba se utilizó después de cerrar
las decisiones de desarrollo. Continúan pendientes la incorporación de los
resultados finales al entregable y la verificación completa de la aplicación.

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
