"""Export report figures from the same values used by the worked example."""

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from .workflow import WorkedExample

COLORS = ("#267A6C", "#3E73B6", "#B77922", "#A3A8AF")
ZONES = ("Familiar", "Descubrimiento", "Exploración", "Fuera de las zonas")
METRICS = {"euclidean": "Euclídea", "mahalanobis": "Mahalanobis"}


def _save(figure: Figure, directory: Path, name: str) -> None:
    FigureCanvasAgg(figure)
    figure.savefig(directory / f"{name}.png", dpi=250, bbox_inches="tight")
    figure.savefig(
        directory / f"{name}.pdf", bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    figure.clear()


def export_figures(example: WorkedExample, directory: Path) -> None:
    """Use independent axes for the two distance scales; no quality metrics."""
    directory.mkdir(parents=True, exist_ok=True)
    fig = Figure(figsize=(9, 5.4), layout="constrained")
    ax = fig.subplots()
    y = np.arange(len(example.profile.features))
    values = np.array(example.profile.centroid)
    ax.barh(y, values, color=COLORS[0], height=0.65)
    ax.set_yticks(y, example.profile.features)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Valor normalizado del centroide (media de las semillas)")
    ax.set_title(f"Perfil acústico · fold {example.fold} · {len(example.profile.seed_ids)} semillas")
    ax.bar_label(ax.containers[0], labels=[f"{value:.3f}" for value in values], padding=5)
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    _save(fig, directory, "acoustic_profile")

    fig = Figure(figsize=(12, 5.2), layout="constrained")
    axes = fig.subplots(1, 2)
    for ax, (metric, zones) in zip(axes, example.thresholds.items()):
        distances = example.reference_distances[metric]
        limit = max(float(distances.max()), zones.cutoffs[-1], 0.01) * 1.03
        edges = (0.0, *zones.cutoffs, limit)
        for left, right, color in zip(edges, edges[1:], COLORS):
            ax.axvspan(left, right, color=color, alpha=0.2)
        ax.hist(distances, bins=60, color="#394B59", alpha=0.7)
        for percentile, cutoff in zip(zones.percentiles, zones.cutoffs):
            ax.axvline(cutoff, color="#263238", linewidth=1, linestyle="--")
        # A separate text row avoids collisions when cutoff values are close.
        ax.set_title(METRICS[metric] + "\n" + "   ".join(
            f"Q{p:g} = {q:.3f}" for p, q in zip(zones.percentiles, zones.cutoffs)
        ), fontsize=10)
        ax.set_xlim(0, limit)
        ax.set_xlabel("Distancia al perfil")
        ax.set_ylabel("Canciones de entrenamiento")
        ax.grid(axis="y", alpha=0.15)
    fig.suptitle(f"Distribución de referencia y zonas · fold {example.fold} · n = {len(example.reference_distances):,}")
    fig.legend(
        handles=[Patch(facecolor=color, alpha=0.5, label=label) for color, label in zip(COLORS, ZONES)],
        loc="outside lower center", ncols=4, frameon=False,
    )
    _save(fig, directory, "distance_zones")

    fig = Figure(figsize=(10, 5.2), layout="constrained")
    ax = fig.subplots()
    x = np.arange(4)
    for offset, (metric, label) in zip((-0.19, 0.19), METRICS.items()):
        counts = example.zone_counts.loc[example.zone_counts["metric"].eq(metric), "candidates"].to_numpy()
        bars = ax.bar(x + offset, counts, width=0.36, label=label)
        ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_xticks(x, ZONES)
    ax.set_ylabel("Canciones candidatas de validación")
    ax.set_title(
        f"Candidatos por zona · fold {example.fold}\n"
        f"{len(example.candidate_distances):,} canciones; semillas excluidas"
    )
    ax.set_ylim(0, max(1, example.zone_counts["candidates"].max()) * 1.18)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    _save(fig, directory, "candidate_counts")
