import json
import os
from typing import Any

from app.core.logger import get_logger
from app.services.environment_classifier import EnvironmentClassifier
from app.services.r2_storage import R2Storage

logger = get_logger(__name__)

# BirdNET taxonomy → tags
TAG_MAP = {
    "bird": ["Aves", "Passeriformes", "Turdus", "Parus", "Troglodytes", "Sylvia",
             "Phylloscopus", "Erithacus", "Luscinia", "Turdus", "Fringilla"],
    "insect": ["Orthoptera", "Cicadidae", "Gryllidae", "Tettigoniidae", "Acrididae"],
    "mammal": ["Mammalia", "Chiroptera", "Lepus", "Vulpes", "Sus", "Capreolus"],
    "amphibian": ["Amphibia", "Anura", "Ranidae", "Bufonidae"],
    "water": ["water", "stream", "river", "lake", "pond", "ripple", "splash"],
    "forest": ["forest", "wood", "tree", "canopy", "undergrowth"],
    "morning": ["dawn", "morning", "daybreak", "sunrise"],
    "night": ["night", "nocturnal", "evening", "dusk", "twilight"],
}


def _heuristic_tags(features: dict[str, Any]) -> set[str]:
    """Suggère des tags d'ambiance sonore basés sur les features acoustiques."""
    tags = set()

    centroid = features.get("spectral_centroid", 3000)
    flatness = features.get("spectral_flatness", 0.1)
    entropy = features.get("spectral_entropy", 5.0)
    zcr = features.get("zero_crossing_rate", 0.05)
    zcr_std = features.get("zcr_std", 0.02)
    event_density = features.get("event_density", 0.0)
    low_ratio = features.get("low_freq_ratio", 0.33)
    mid_ratio = features.get("mid_freq_ratio", 0.33)
    high_ratio = features.get("high_freq_ratio", 0.33)
    rms_db = features.get("rms_db", -40)
    silence_ratio = features.get("silence_ratio", 0.0)
    periodicity = features.get("periodicity", 0.0)
    harmonic_ratio = features.get("harmonic_ratio", 0.5)
    percussive_ratio = features.get("percussive_ratio", 0.0)
    tempo = features.get("tempo_bpm", 0)

    # ── Water (stream, river, rain, waves) ──
    # Bruit large bande, basse-fréquence dominante, ZCR très bas, plat stable
    if low_ratio > 0.5 and flatness > 0.15 and zcr < 0.03 and event_density < 0.5:
        tags.add("water")
        if entropy > 6.5:
            tags.add("stream")
        elif entropy > 5.5:
            tags.add("rain")

    # ── Wind ──
    # Large bande, forte variance temporelle, spectral centroid modéré-élevé
    if flatness > 0.2 and high_ratio > 0.3 and zcr_std > 0.03 and event_density < 0.3:
        tags.add("wind")

    # ── Birds ──
    # Événements intermittents haute-fréquence, harmonicité, ZCR modéré
    if event_density > 0.3 and high_ratio > 0.2 and harmonic_ratio > 0.4 and zcr > 0.03:
        tags.add("birds")
        if centroid > 5000:
            tags.add("songbirds")

    # ── Insects ──
    # Tonalité très haute fréquence, ZCR élevé, bruit continu
    if centroid > 6000 and zcr > 0.08 and event_density < 1.0 and flatness < 0.3:
        tags.add("insects")
        if centroid > 9000:
            tags.add("cicadas")

    # ── Mammals ──
    # Fréquences basses-moyennes, patterns répétitifs, périodicité
    if low_ratio > 0.4 and periodicity > 0.3 and event_density < 0.5 and centroid < 3000:
        tags.add("mammals")

    # ── Urban / Human ──
    # Mix harmonique + percussif, tempo régulier
    if percussive_ratio > 0.3 and tempo > 60 and event_density > 0.5:
        tags.add("rhythmic")
        if harmonic_ratio < 0.4:
            tags.add("mechanical")

    # ── Silence / Quiet ──
    if rms_db < -50 or silence_ratio > 0.3:
        tags.add("quiet")
    if silence_ratio > 0.6:
        tags.add("silent")

    # ── Dense / Complex ──
    if entropy > 7.0 and event_density > 1.0:
        tags.add("dense")

    # ── Open / Spacious ──
    if harmonic_ratio > 0.6 and event_density < 0.3 and rms_db < -35:
        tags.add("spacious")

    return tags


class SummaryBuilder:
    def __init__(self, storage: R2Storage) -> None:
        self.storage = storage

    def build_and_upload(
        self,
        sound_id: int,
        duration: float,
        detections: list[dict[str, Any]],
        quality: dict[str, Any],
        features: dict[str, Any],
    ) -> str:
        main_species = []
        seen_species = set()
        for detection in sorted(detections, key=lambda x: x.get("aggregate_confidence", x["confidence"]), reverse=True):
            species_key = detection["common_name"].lower()
            if species_key in seen_species:
                continue
            seen_species.add(species_key)
            main_species.append({
                "name": detection["common_name"],
                "scientific_name": detection.get("scientific_name"),
                "confidence": detection.get("aggregate_confidence", detection["confidence"]),
                "segments": detection.get("species_detection_count", 1),
            })
            if len(main_species) >= 5:
                break

        environment_classifier = EnvironmentClassifier()
        acoustic_environments = environment_classifier.classify(features, detections)
        suggested_tags = self._suggest_tags(detections, features, environment_classifier)

        # Acoustic profile
        acoustic_profile = {
            "dominant_frequency_range": self._freq_range_label(features.get("spectral_centroid", 3000)),
            "texture": self._texture_label(features),
            "tempo_description": self._tempo_label(features.get("tempo_bpm", 0)),
            "space_description": self._space_label(features),
        }

        summary = {
            "sound_id": sound_id,
            "duration_seconds": round(duration, 2),
            "main_detected_species": main_species,
            "suggested_tags": suggested_tags,
            "acoustic_environments": acoustic_environments,
            "acoustic_profile": acoustic_profile,
            "quality": {
                "noise_level": self._noise_level_label(quality.get("noise_floor_db", -60)),
                "clipping_detected": quality.get("clipping_detected", False),
                "usable_for_analysis": quality.get("usable_for_analysis", True),
            },
            "feature_highlights": {
                "tempo_bpm": features.get("tempo_bpm"),
                "event_density": features.get("event_density"),
                "spectral_centroid_hz": features.get("spectral_centroid"),
                "harmonic_ratio": features.get("harmonic_ratio"),
                "dynamic_range_db": features.get("dynamic_range_db"),
            },
        }

        temp_json = f"/tmp/analyzer/{sound_id}_summary.json"
        os.makedirs(os.path.dirname(temp_json), exist_ok=True)
        with open(temp_json, "w") as f:
            json.dump(summary, f, indent=2)

        r2_key = f"sounds/analysis/{sound_id}/summary.json"
        self.storage.upload(temp_json, r2_key, content_type="application/json")
        os.remove(temp_json)

        logger.info("summary_built", sound_id=sound_id, r2_key=r2_key, tags=suggested_tags)
        return r2_key

    def _suggest_tags(
        self,
        detections: list[dict[str, Any]],
        features: dict[str, Any],
        environment_classifier: EnvironmentClassifier | None = None,
    ) -> list[str]:
        tags = set()

        # Tags from BirdNET detections
        for d in detections:
            sci = d.get("scientific_name", "").lower()
            com = d.get("common_name", "").lower()
            for tag, keywords in TAG_MAP.items():
                if any(kw.lower() in sci or kw.lower() in com for kw in keywords):
                    tags.add(tag)

        # Tags from acoustic environment scoring
        classifier = environment_classifier or EnvironmentClassifier()
        tags.update(classifier.tags(features, detections))

        # Fallback
        if not tags:
            tags.add("nature")

        # Frequency-based tags
        centroid = features.get("spectral_centroid", 3000)
        if centroid > 8000:
            tags.add("high-frequency")
        elif centroid < 1000:
            tags.add("low-frequency")

        return sorted(list(tags))

    def _noise_level_label(self, noise_floor_db: float) -> str:
        if noise_floor_db > -30:
            return "high"
        if noise_floor_db > -45:
            return "medium"
        return "low"

    def _freq_range_label(self, centroid: float) -> str:
        if centroid > 6000:
            return "high"
        if centroid > 2500:
            return "mid-high"
        if centroid > 800:
            return "mid"
        return "low"

    def _texture_label(self, features: dict[str, Any]) -> str:
        event_density = features.get("event_density", 0)
        flatness = features.get("spectral_flatness", 0)
        if event_density < 0.2 and flatness > 0.2:
            return "continuous"
        if event_density > 1.0:
            return "eventful"
        if flatness > 0.3:
            return "noisy"
        return "tonal"

    def _tempo_label(self, tempo: float) -> str:
        if tempo == 0:
            return "aperiodic"
        if tempo < 40:
            return "very slow"
        if tempo < 70:
            return "slow"
        if tempo < 120:
            return "moderate"
        return "fast"

    def _space_label(self, features: dict[str, Any]) -> str:
        silence_ratio = features.get("silence_ratio", 0)
        dynamic_range = features.get("dynamic_range_db", 0)
        if silence_ratio > 0.2 and dynamic_range > 30:
            return "open"
        if silence_ratio < 0.05 and dynamic_range < 15:
            return "dense"
        return "mixed"
