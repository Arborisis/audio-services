"""
Utilitaires I/O, plots et helpers.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ensure_dir(path: str) -> str:
    """Crée le répertoire s'il n'existe pas."""
    os.makedirs(path, exist_ok=True)
    return path


def save_figure(fig: plt.Figure, output_path: str, dpi: int = 150, fmt: str = "png") -> str:
    """
    Sauvegarde une figure matplotlib.

    Args:
        fig: Figure matplotlib
        output_path: Chemin de sortie (sans extension si fmt spécifié)
        dpi: Résolution
        fmt: Format ('png', 'jpg', 'svg')

    Returns:
        Chemin final du fichier
    """
    ensure_dir(os.path.dirname(output_path))

    if not output_path.lower().endswith(f".{fmt}"):
        output_path = f"{output_path}.{fmt}"

    fig.savefig(output_path, dpi=dpi, format=fmt, bbox_inches="tight", facecolor="auto")
    plt.close(fig)
    return output_path


def load_json(path: str) -> dict:
    """Charge un fichier JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, path: str) -> str:
    """Sauvegarde un dictionnaire en JSON."""
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


def setup_plot_style():
    """Configure le style matplotlib pour Arborisis."""
    plt.rcParams.update({
        "figure.facecolor": "#0B1220",
        "axes.facecolor": "#111827",
        "axes.edgecolor": "#2a3142",
        "axes.labelcolor": "#8FA68E",
        "text.color": "#F3F0E7",
        "xtick.color": "#8FA68E",
        "ytick.color": "#8FA68E",
        "grid.color": "#2a3142",
        "grid.alpha": 0.3,
        "figure.dpi": 100,
    })
