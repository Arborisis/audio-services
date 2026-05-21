from typing import Any


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _ratio(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0

    return _clamp((value - low) / (high - low))


class EnvironmentClassifier:
    """Score broad acoustic environments from aggregate audio features."""

    def classify(self, features: dict[str, Any], detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        bird_detection_score = self._bird_detection_score(detections)
        scores = {
            "birds": self._birds_score(features, bird_detection_score),
            "water": self._water_score(features),
            "rain": self._rain_score(features),
            "wind": self._wind_score(features),
            "insects": self._insects_score(features),
            "quiet": self._quiet_score(features),
            "open_space": self._open_space_score(features),
            "dense_habitat": self._dense_habitat_score(features),
            "human_activity": self._human_activity_score(features),
        }

        ranked = [
            {"label": label, "confidence": round(score, 3)}
            for label, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
            if score >= 0.35
        ]

        return ranked[:5] or [{"label": "nature", "confidence": 0.35}]

    def tags(self, features: dict[str, Any], detections: list[dict[str, Any]]) -> set[str]:
        environments = self.classify(features, detections)
        tags = {environment["label"] for environment in environments}

        for environment in environments:
            label = environment["label"]
            confidence = float(environment["confidence"])
            if label == "birds" and confidence >= 0.62:
                tags.add("songbirds")
            if label == "water" and confidence >= 0.58:
                tags.add("stream")
            if label == "insects" and features.get("spectral_centroid", 0) > 8000:
                tags.add("cicadas")
            if label == "quiet" and features.get("silence_ratio", 0) > 0.6:
                tags.add("silent")

        return tags

    def _bird_detection_score(self, detections: list[dict[str, Any]]) -> float:
        if not detections:
            return 0.0

        confidences = [float(d.get("aggregate_confidence", d.get("confidence", 0.0))) for d in detections]
        unique_species = {
            f"{d.get('scientific_name', '').lower()}|{d.get('common_name', '').lower()}"
            for d in detections
        }

        return _clamp(max(confidences) * 0.80 + _ratio(len(unique_species), 0, 4) * 0.20)

    def _birds_score(self, features: dict[str, Any], detection_score: float) -> float:
        bird_band = features.get("bird_band_ratio", features.get("high_freq_ratio", 0.0))
        flux = features.get("spectral_flux", 0.0)
        event_density = features.get("event_density", 0.0)
        harmonic = features.get("harmonic_ratio", 0.0)
        flatness = features.get("spectral_flatness", 0.0)

        acoustic = (
            _ratio(bird_band, 0.25, 0.62) * 0.28
            + _ratio(flux, 0.03, 0.12) * 0.22
            + _ratio(event_density, 0.15, 1.1) * 0.18
            + _ratio(harmonic, 0.35, 0.75) * 0.17
            + (1.0 - _ratio(flatness, 0.18, 0.45)) * 0.15
        )

        return _clamp(detection_score * 0.65 + acoustic * 0.35)

    def _water_score(self, features: dict[str, Any]) -> float:
        flatness = features.get("spectral_flatness", 0.0)
        entropy = features.get("spectral_entropy", 0.0)
        low_ratio = features.get("low_freq_ratio", 0.0)
        flux_std = features.get("spectral_flux_std", 0.0)
        event_density = features.get("event_density", 0.0)

        return _clamp(
            _ratio(flatness, 0.12, 0.35) * 0.28
            + _ratio(entropy, 5.3, 7.5) * 0.24
            + _ratio(low_ratio, 0.38, 0.72) * 0.22
            + (1.0 - _ratio(event_density, 0.25, 0.9)) * 0.16
            + _ratio(flux_std, 0.02, 0.12) * 0.10
        )

    def _rain_score(self, features: dict[str, Any]) -> float:
        return _clamp(
            _ratio(features.get("spectral_flatness", 0.0), 0.18, 0.45) * 0.35
            + _ratio(features.get("high_freq_ratio", 0.0), 0.24, 0.55) * 0.25
            + _ratio(features.get("spectral_entropy", 0.0), 6.0, 7.8) * 0.25
            + (1.0 - _ratio(features.get("event_density", 0.0), 0.35, 1.1)) * 0.15
        )

    def _wind_score(self, features: dict[str, Any]) -> float:
        return _clamp(
            _ratio(features.get("spectral_flatness", 0.0), 0.15, 0.42) * 0.30
            + _ratio(features.get("spectral_flux_std", 0.0), 0.04, 0.18) * 0.28
            + _ratio(features.get("low_freq_ratio", 0.0), 0.35, 0.75) * 0.22
            + (1.0 - _ratio(features.get("harmonic_ratio", 0.0), 0.35, 0.75)) * 0.20
        )

    def _insects_score(self, features: dict[str, Any]) -> float:
        return _clamp(
            _ratio(features.get("insect_band_ratio", features.get("high_freq_ratio", 0.0)), 0.18, 0.56) * 0.35
            + _ratio(features.get("spectral_centroid", 0.0), 5200, 9500) * 0.24
            + _ratio(features.get("zero_crossing_rate", 0.0), 0.055, 0.14) * 0.22
            + (1.0 - _ratio(features.get("event_density", 0.0), 0.8, 2.0)) * 0.19
        )

    def _quiet_score(self, features: dict[str, Any]) -> float:
        return _clamp(
            _ratio(-float(features.get("rms_db", -40)), 38, 62) * 0.45
            + _ratio(features.get("silence_ratio", 0.0), 0.2, 0.75) * 0.40
            + (1.0 - _ratio(features.get("event_density", 0.0), 0.1, 0.8)) * 0.15
        )

    def _open_space_score(self, features: dict[str, Any]) -> float:
        return _clamp(
            _ratio(features.get("dynamic_range_db", 0.0), 22, 45) * 0.35
            + _ratio(features.get("silence_ratio", 0.0), 0.08, 0.35) * 0.25
            + _ratio(features.get("harmonic_ratio", 0.0), 0.45, 0.8) * 0.20
            + (1.0 - _ratio(features.get("event_density", 0.0), 0.4, 1.4)) * 0.20
        )

    def _dense_habitat_score(self, features: dict[str, Any]) -> float:
        return _clamp(
            _ratio(features.get("spectral_entropy", 0.0), 6.2, 7.9) * 0.35
            + _ratio(features.get("event_density", 0.0), 0.8, 2.2) * 0.30
            + _ratio(features.get("mid_freq_ratio", 0.0), 0.35, 0.65) * 0.20
            + (1.0 - _ratio(features.get("silence_ratio", 0.0), 0.05, 0.25)) * 0.15
        )

    def _human_activity_score(self, features: dict[str, Any]) -> float:
        tempo = features.get("tempo_bpm", 0.0)

        return _clamp(
            _ratio(features.get("percussive_ratio", 0.0), 0.28, 0.65) * 0.32
            + _ratio(features.get("low_mid_ratio", 0.0), 0.24, 0.58) * 0.25
            + _ratio(features.get("event_density", 0.0), 0.5, 1.8) * 0.22
            + (0.21 if 55 <= tempo <= 150 else 0.0)
        )
