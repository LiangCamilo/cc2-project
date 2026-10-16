"""Export data and evidence reports only when integrity checks pass."""

import argparse
import hashlib
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from clean.pipeline import run_cleaning


PROJECT_DIR = Path(__file__).resolve().parents[2]


def export_result(result, output_dir):
    failures = result.tables["validation"].loc[lambda frame: frame["status"].eq("fail")]
    if not failures.empty:
        raise ValueError("Export blocked by failed checks: " + ", ".join(failures["check"]))
    output_dir = Path(output_dir).resolve()
    if output_dir == (PROJECT_DIR / "data/raw").resolve():
        raise ValueError("Exports must not overwrite the raw dataset directory.")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    tracks = result.tables["tracks"]
    relations = result.tables["track_genres"]
    genres = result.tables["genre_representation"]
    flags = result.tables["outlier_flags"]
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": result.source_sha256,
        "source_rows": int(result.tables["cleaning_log"].iloc[0]["input_rows"]),
        "retained_tracks": len(tracks), "genre_associations": len(relations),
        "retained_genres": int(relations["track_genre"].nunique()),
        "excluded_tracks": len(result.tables["excluded_tracks"]),
        "time_signature_review": int(tracks["time_signature_review"].sum()),
        "loudness_review": int(tracks["loudness_review"].sum()),
        "iqr_review_tracks": int(flags.drop(columns="track_id").any(axis=1).sum()),
        "genre_counts": {
            "min": int(genres["tracks_after"].min()),
            "max": int(genres["tracks_after"].max()),
            "max_min_ratio": float(genres["tracks_after"].max() / genres["tracks_after"].min())
            if genres["tracks_after"].min() > 0 else None,
            "lost": genres.loc[genres["tracks_after"].eq(0), "track_genre"].tolist(),
        },
        "decisions": {
            "popularity": "Arithmetic mean of distinct observed scores per track; decimals preserved. Not a verified current Spotify score.",
            "tempo": "Exclude nonpositive estimates from the first catalog requiring tempo; retain exclusions separately.",
            "time_signature": "Preserve original values, flag codes outside 3..7; omit from initial distance features.",
            "loudness": "Preserve finite values outside typical -60..0 dB; no clipping.",
            "outliers": "Screen numeric distributions with 1.5*IQR; do not delete or clip statistical extremes.",
            "genres": "Multi-label relationships preserved. No oversampling, undersampling or claim of exact balance.",
        },
        "model_preparation_pending": [
            "Choose acoustic features; exclude identifiers, text, popularity as an acoustic distance feature, category codes without encoding, and all review flags.",
            "Fit scalers on the training/reference partition only after defining the evaluation protocol.",
        ],
        "code_sha256": {
            str(path.relative_to(PROJECT_DIR)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(__file__).parent.rglob("*.py"))
        },
        "tables": {},
    }
    with TemporaryDirectory(prefix=".clean-export-", dir=output_dir.parent) as temporary:
        staging = Path(temporary)
        for name, table in result.tables.items():
            path = staging / (name + ".csv")
            table.to_csv(path, index=False, encoding="utf-8")
            manifest["tables"][path.name] = {
                "rows": len(table), "columns": table.columns.tolist(),
                "dtypes": {column: str(dtype) for column, dtype in table.dtypes.items()},
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        report = f"""# Dataset musical depurado

El catálogo contiene {len(tracks):,} canciones y {relations['track_genre'].nunique()} géneros,
representados mediante {len(relations):,} asociaciones. Las tablas se exportan sin índice.
El CSV original permanece intacto. Su huella SHA-256 está en `manifest.json`.

## Tablas principales

- `tracks.csv`: una fila por `track_id`; metadatos, características y marcas de revisión.
- `track_genres.csv`: una fila por pareja `track_id` y `track_genre`. Una canción puede tener varios géneros.
- `excluded_tracks.csv`: registros apartados, con el motivo en `exclusion_reason`.
- `excluded_track_genres.csv`: géneros de los registros apartados.

## Evidencia

- `cleaning_log.csv`: cantidades antes y después de cada regla; las filas consolidadas no son canciones descartadas.
- `text_changes.csv`: cambios de formato con índice de origen y textos representados entre comillas.
- `popularity_conflicts.csv`: valores mínimos y máximos observados y media de puntuaciones distintas.
- `validation.csv`: `pass` para comprobaciones superadas; `review` para discrepancias conservadas de forma explícita. Un `fail` bloquea la exportación.
- `genre_representation.csv`: canciones únicas por género antes y después, pérdidas y retención.
- `outlier_summary.csv`: cuantiles y límites IQR calculados sobre canciones únicas.
- `outlier_flags.csv`: marcas estadísticas por canción y variable; no implican eliminación.
- `manifest.json`: decisiones, tipos de columnas, conteos y huellas de los datos y del código.

## Interpretación y límites

La popularidad consolidada puede tener decimales: es un resumen calculado, no una
observación verificada de popularidad actual. `time_signature_review`,
`loudness_review`, `loudness_invalid` y las marcas IQR son metadatos de calidad;
no deben entrar automáticamente en las distancias acústicas. Se conservan
{manifest['time_signature_review']} discrepancias de compás en esta versión;
los conteos efectivos están en el manifiesto. Los valores atípicos finitos no
se recortan ni se eliminan por una regla estadística.

Los porcentajes de catálogo por género pueden sumar más del 100 %, porque una
canción admite varios géneros. La conservación de géneros no demuestra por sí
sola balance ni representatividad de toda la música. No se aplicó remuestreo.
La selección de características, codificación y escalado corresponden a la
siguiente etapa del modelo; no se entrenó un modelo en este proceso.

## Reproducir

Desde `cc2-project`, con su entorno virtual activo:

```bash
python -m clean.export_dataset
```

En Streamlit, la sección final permite guardar de nuevo las tablas y reportes.
"""
        (staging / "README.md").write_text(report, encoding="utf-8")
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        # All files are prepared first; the manifest is published last.
        paths = sorted(path for path in staging.iterdir() if path.name != "manifest.json")
        paths.append(staging / "manifest.json")
        for path in paths:
            os.replace(path, output_dir / path.name)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=PROJECT_DIR / "data/raw/dataset.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "data/processed")
    args = parser.parse_args()
    payload = args.source.read_bytes()
    fingerprint = hashlib.sha256(payload).hexdigest()
    data = pd.read_csv(io.BytesIO(payload), index_col=0)
    result = run_cleaning(data, fingerprint)
    manifest = export_result(result, args.output_dir)
    print(json.dumps({
        "output_dir": str(args.output_dir.resolve()),
        "tracks": manifest["retained_tracks"], "genres": manifest["retained_genres"],
        "associations": manifest["genre_associations"], "excluded": manifest["excluded_tracks"],
        "failed_checks": 0,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
