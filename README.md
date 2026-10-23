# Proyecto CC2: recomendación musical mediante ZTA

Proyecto de Ciencias de la Computación II para generar recomendaciones musicales
controladas mediante Zonas de Tolerancia Acústica (ZTA). A partir de canciones
semilla se construye un perfil acústico y se obtienen recomendaciones en tres
modos: Familiar, Descubrimiento y Exploración.

Actualmente se pueden reproducir la auditoría y limpieza del dataset, las
particiones y el modelo matemático completo: normalización, centroide,
distancias, zonas y ranking. El ejemplo integrado exporta tablas, parámetros
y gráficos. Quedan pendientes los cuatro modelos de ML, la evaluación
comparativa y la aplicación de recomendación. Las interfaces Streamlit actuales
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
```

| Comando | Resultado |
| --- | --- |
| `clean.export_dataset` | Genera el catálogo depurado, sus géneros y reportes de calidad en `data/processed`. |
| `zta.data --save-partitions` | Valida las nueve características y guarda los identificadores de desarrollo, prueba y cinco folds. |
| `zta.data --check-partitions` | Comprueba que las particiones corresponden al catálogo y la configuración actuales. |
| `zta.example --fold 1 --k 10 --save` | Ejecuta el modelo completo y exporta seis listas de recomendaciones, tablas, parámetros y tres gráficos en PNG y PDF. |

El ejemplo necesita las tablas depuradas y las particiones guardadas. Ajusta
la normalización y la covarianza con entrenamiento del fold seleccionado y
consulta sus canciones de validación. No necesita exportaciones previas de
normalización, perfiles o recomendaciones: integra esas operaciones y guarda
sus propios resultados en `data/processed/worked_example/fold_1`.

Si ya se dispone del catálogo y de particiones válidas, se puede comenzar con
`python -m zta.data --check-partitions` y ejecutar el ejemplo sin repetir las
etapas anteriores.

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
Las dos distancias son variantes de ZTA y no se contabilizan como dos de los
cuatro modelos de ML pendientes.

## 6. Resultados generados y trazabilidad

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

## 7. Aplicaciones de auditoría y limpieza

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

## 8. Estado y trabajo pendiente

| Componente | Estado |
| --- | --- |
| Auditoría, limpieza y catálogo con géneros múltiples | Implementado. |
| Nueve características y particiones reproducibles | Implementado. |
| Normalización, perfil, distancias, bandas y ranking ZTA | Implementado. |
| Ejemplo completo con tablas, parámetros y gráficos | Implementado. |
| Cuatro modelos de ML | Pendiente. |
| Simulación por afinidad de género y métricas | Pendiente. |
| Comparación y selección de configuraciones en validación | Pendiente. |
| Evaluación final sobre prueba | Pendiente. |
| Aplicación de recomendación y sus diez funcionalidades | Pendiente. |

La base matemática ejecutable está construida. Para completar su análisis se
necesita evaluar el comportamiento bajo el protocolo acordado y discutir sus
resultados. Reproducir listas o verificar la pertenencia a bandas no demuestra
precisión superior al 95 %. El conjunto de prueba permanece reservado durante
los recorridos de desarrollo descritos aquí.

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
