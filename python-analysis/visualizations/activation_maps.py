from typing import Optional
"""
Cartes d'activation Deep Learning — placeholder.
"""

import numpy as np
import matplotlib.pyplot as plt

from core.utils import save_figure, setup_plot_style


def plot_activation_placeholder(input_shape: tuple, output_path: str = None, dpi: int = 150) -> Optional[str]:
    """
    Génère un placeholder pour les futures cartes d'activation DL.

    Args:
        input_shape: Forme de l'entrée (ex: (128, 128))
        output_path: Chemin de sortie
        dpi: Résolution

    Returns:
        Chemin du fichier généré
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    # Génère une heatmap aléatoire comme placeholder
    data = np.random.rand(*input_shape)
    im = ax.imshow(data, cmap="viridis", aspect="auto")
    ax.set_title("Carte d'activation (placeholder)", fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Activation")

    # Ajoute un texte indicatif
    ax.text(
        0.5, 0.5, "DL Activation Map\n(placeholder)",
        transform=ax.transAxes,
        ha="center", va="center",
        fontsize=16, color="white", alpha=0.5,
    )

    if output_path:
        return save_figure(fig, output_path, dpi=dpi)
    plt.show()
    return None
