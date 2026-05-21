from typing import Optional
"""
Heatmaps : corrélation features, matrices de confusion.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from core.utils import save_figure, setup_plot_style


def plot_feature_correlation(features_df: pd.DataFrame, output_path: str = None, dpi: int = 150) -> Optional[str]:
    """
    Génère une heatmap de corrélation entre features.

    Args:
        features_df: DataFrame avec une feature par colonne
        output_path: Chemin de sortie
        dpi: Résolution

    Returns:
        Chemin du fichier généré
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 8))

    corr = features_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="magma",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Corrélation"},
        annot_kws={"color": "white", "fontsize": 8},
    )

    ax.set_title("Matrice de corrélation des features", fontsize=14, fontweight="bold")

    if output_path:
        return save_figure(fig, output_path, dpi=dpi)
    plt.show()
    return None


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str],
    output_path: str = None,
    normalize: bool = True,
    dpi: int = 150,
) -> Optional[str]:
    """
    Génère une heatmap de matrice de confusion.

    Args:
        cm: Matrice de confusion (2D array)
        labels: Noms des classes
        output_path: Chemin de sortie
        normalize: Normaliser les valeurs
        dpi: Résolution
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="magma",
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        ax=ax,
        annot_kws={"color": "white"},
    )

    ax.set_title("Matrice de confusion", fontsize=14, fontweight="bold")
    ax.set_xlabel("Prédiction")
    ax.set_ylabel("Réel")

    if output_path:
        return save_figure(fig, output_path, dpi=dpi)
    plt.show()
    return None
