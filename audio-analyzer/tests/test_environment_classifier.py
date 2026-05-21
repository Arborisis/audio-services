from app.services.environment_classifier import EnvironmentClassifier


def test_environment_classifier_boosts_birds_with_birdnet_context():
    features = {
        "bird_band_ratio": 0.42,
        "spectral_flux": 0.08,
        "event_density": 0.6,
        "harmonic_ratio": 0.62,
        "spectral_flatness": 0.12,
    }
    detections = [
        {
            "scientific_name": "Erithacus rubecula",
            "common_name": "European Robin",
            "confidence": 0.61,
            "aggregate_confidence": 0.64,
        }
    ]

    environments = EnvironmentClassifier().classify(features, detections)

    assert environments[0]["label"] == "birds"
    assert environments[0]["confidence"] >= 0.55


def test_environment_classifier_detects_water_texture():
    features = {
        "spectral_flatness": 0.27,
        "spectral_entropy": 7.1,
        "low_freq_ratio": 0.58,
        "spectral_flux_std": 0.06,
        "event_density": 0.12,
    }

    tags = EnvironmentClassifier().tags(features, [])

    assert "water" in tags
    assert "stream" in tags
