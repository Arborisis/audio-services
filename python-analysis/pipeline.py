"""
Pipeline end-to-end d'analyse audio.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import merge_config, MAX_DURATION_SECONDS, DEFAULT_OUTPUT_DIR
from core.loader import load_audio, get_audio_info
from core.preprocessing import apply_filter, normalize, trim_silence, detect_voice_activity
from core.segmentation import segment_frames
from core.utils import ensure_dir, save_json, setup_plot_style
from features.temporal import extract_temporal_features
from features.spectral import extract_spectral_features
from features.cepstral import extract_cepstral_features, extract_mfcc
from features.biodiversity import extract_acoustic_events, compute_acoustic_diversity_index
from visualizations.spectrograms import plot_combined_spectrograms
from visualizations.heatmaps import plot_feature_correlation


def run_analysis(audio_path: str, output_dir: str = None, config_dict: dict = None) -> dict:
    """
    Pipeline complet d'analyse audio.

    Args:
        audio_path: Chemin du fichier audio
        output_dir: Répertoire de sortie (défaut: ./output)
        config_dict: Configuration utilisateur (merge avec defaults)

    Returns:
        Dict avec features, chemins visualisations, métadonnées
    """
    config = merge_config(config_dict)
    output_dir = output_dir or DEFAULT_OUTPUT_DIR
    ensure_dir(output_dir)

    setup_plot_style()

    # 1. Chargement
    info = get_audio_info(audio_path)
    if info["duration"] > MAX_DURATION_SECONDS:
        raise ValueError(
            f"Durée trop longue: {info['duration']:.1f}s (max {MAX_DURATION_SECONDS}s)"
        )

    y, sr = load_audio(audio_path, sr=config["sample_rate"], mono=True)

    # 2. Prétraitement
    preprocessing = config.get("preprocessing", {})

    if preprocessing.get("filter"):
        filt = preprocessing["filter"]
        y = apply_filter(y, sr, filt["type"], filt["cutoff"])

    if preprocessing.get("normalize", True):
        y = normalize(y, method=preprocessing.get("normalize_method", "peak"))

    if preprocessing.get("trim_silence", True):
        y, _ = trim_silence(y, sr, top_db=preprocessing.get("top_db", 20))

    if preprocessing.get("vad", False):
        vad_mask = detect_voice_activity(
            y, sr,
            energy_threshold=preprocessing.get("vad_threshold", 0.01),
            use_zcr=preprocessing.get("vad_use_zcr", False),
        )
        # Applique le masque au signal (garde seulement les frames actives)
        hop_length = config.get("hop_length", 512)
        active_samples = []
        for i, is_active in enumerate(vad_mask):
            if is_active:
                start = i * hop_length
                end = start + hop_length
                active_samples.extend(y[start:end])
        if len(active_samples) > 0:
            y = np.array(active_samples)

    # 3. Segmentation (pour info, pas utilisée directement dans l'extraction librosa)
    frames = segment_frames(y, sr, frame_length=config.get("n_fft", 2048), hop_length=config["hop_length"])

    # 4. Extraction features
    n_fft = config.get("n_fft", 2048)
    hop_length = config.get("hop_length", 512)
    n_mels = config.get("n_mels", 128)
    n_mfcc = config.get("n_mfcc", 13)

    temporal = extract_temporal_features(y, sr)
    spectral = extract_spectral_features(y, sr, n_fft, hop_length)
    cepstral = extract_cepstral_features(y, sr, n_mfcc, n_fft, hop_length)

    # 4b. Biodiversity features
    biodiversity = extract_acoustic_events(y, sr, hop_length=hop_length)
    adi = compute_acoustic_diversity_index(y, sr, n_bands=10)

    features = {
        "temporal": temporal,
        "spectral": spectral,
        "cepstral": cepstral,
        "biodiversity": biodiversity,
        "acoustic_diversity_index": adi,
    }

    # 5. Visualisations
    viz_config = config.get("visualizations", {})
    viz_paths = {}

    if viz_config.get("spectrograms"):
        spec_paths = plot_combined_spectrograms(
            y, sr,
            config={
                "n_fft": n_fft,
                "hop_length": hop_length,
                "n_mels": n_mels,
                "n_mfcc": n_mfcc,
                "stft_scale": config.get("frequency_scale", "linear"),
                "colormap": viz_config.get("colormap", "viridis"),
                "dpi": viz_config.get("dpi", 150),
            },
            output_dir=output_dir,
        )
        viz_paths.update(spec_paths)

    # Heatmap de corrélation
    if "feature_correlation" in viz_config.get("heatmaps", []):
        # Construit un DataFrame avec les features frame-level
        n_frames = min(
            len(temporal["zcr"]["values"]),
            len(temporal["rms"]["values"]),
            len(spectral["centroid"]["values"]),
            len(spectral["bandwidth"]["values"]),
            len(spectral["rolloff"]["values"]),
            len(spectral["flatness"]["values"]),
        )
        df = pd.DataFrame({
            "zcr": temporal["zcr"]["values"][:n_frames],
            "rms": temporal["rms"]["values"][:n_frames],
            "centroid": spectral["centroid"]["values"][:n_frames],
            "bandwidth": spectral["bandwidth"]["values"][:n_frames],
            "rolloff": spectral["rolloff"]["values"][:n_frames],
            "flatness": spectral["flatness"]["values"][:n_frames],
        })
        corr_path = plot_feature_correlation(df, output_path=f"{output_dir}/feature_correlation")
        if corr_path:
            viz_paths["feature_correlation"] = corr_path

    # 6. Export JSON
    features_path = os.path.join(output_dir, "features.json")
    save_json(features, features_path)

    result = {
        "success": True,
        "audio_info": info,
        "features": features,
        "visualizations": viz_paths,
        "features_json_path": features_path,
        "config_used": config,
    }

    return result
