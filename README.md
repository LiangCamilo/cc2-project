# Proyecto CC2

Ejecuta estos comandos desde la carpeta `cc2-project`.

## Preparar el entorno

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

Si ya se tiene el entorno y sus dependencias, basta con activarlo y ejecutar
`python -m pip install -e .` una vez. La instalacion editable permite importar
los paquetes de `src` por su nombre y refleja los cambios al codigo sin
reinstalar. Repite la instalacion si recreas el entorno virtual.

## Ejecutar las aplicaciones

```bash
python -m streamlit run src/audit/dataset_audit.py
python -m streamlit run src/clean/dataset_cleaning.py
```

## Modulos compartidos

Los paquetes `common`, `audit` y `clean` tienen un archivo `__init__.py`.
La configuracion de `pyproject.toml` indica que se encuentran dentro de `src`.
Desde cualquiera de ellos puedes usar imports absolutos:

```python
from common.data_access import load_dataset, compute_audit
from common.audit_checks import audit_dataset
```

`common/data_access.py` conserva la ruta al dataset en `data/raw/dataset.csv`.
`__pycache__` contiene cache de Python y no configura los imports.

## Generar el dataset depurado

```bash
python -m clean.export_dataset
```

La ejecución aplica las mismas reglas que la aplicación de limpieza y guarda en
`data/processed` las canciones, sus asociaciones de géneros, los registros
apartados y los reportes de calidad. No modifica `data/raw/dataset.csv`.
En Streamlit, el botón final permite guardar nuevamente los resultados.

`validation.csv` distingue errores que bloquean la exportación de alertas
documentadas que conservamos. La popularidad consolidada admite decimales.
Los valores estadísticamente extremos se marcan y conservan. La representación
por género se informa sin remuestrear ni asumir balance perfecto.

`data/processed/README.md` explica cada archivo; `manifest.json` registra las
decisiones, conteos, tipos y huellas SHA-256 del origen, resultados y código.
La selección de características y el escalado para el modelo quedan pendientes.
