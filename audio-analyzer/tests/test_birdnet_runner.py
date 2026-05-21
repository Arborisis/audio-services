from app.services.birdnet_runner import BirdnetRunner


def test_postprocess_drops_single_low_confidence_detection():
    runner = BirdnetRunner(storage=None)

    detections = runner._postprocess_detections([
        {
            "scientific_name": "Turdus merula",
            "common_name": "Common Blackbird",
            "confidence": 0.31,
            "start_time": 0.0,
            "end_time": 3.0,
        }
    ])

    assert detections == []


def test_postprocess_keeps_repeated_moderate_species():
    runner = BirdnetRunner(storage=None)

    detections = runner._postprocess_detections([
        {
            "scientific_name": "Turdus merula",
            "common_name": "Common Blackbird",
            "confidence": 0.36,
            "start_time": 0.0,
            "end_time": 3.0,
        },
        {
            "scientific_name": "Turdus merula",
            "common_name": "Common Blackbird",
            "confidence": 0.34,
            "start_time": 3.0,
            "end_time": 6.0,
        },
    ])

    assert len(detections) == 2
    assert detections[0]["aggregate_confidence"] >= 0.34
    assert detections[0]["species_detection_count"] == 2


def test_csv_parser_handles_commas_in_common_name(tmp_path):
    csv_path = tmp_path / "birdnet.csv"
    csv_path.write_text(
        "Start (s),End (s),Scientific name,Common name,Confidence\n"
        '0.0,3.0,Turdus merula,"Blackbird, Common",0.72\n'
    )

    detections = BirdnetRunner(storage=None)._parse_csv(str(csv_path))

    assert detections == [
        {
            "scientific_name": "Turdus merula",
            "common_name": "Blackbird, Common",
            "confidence": 0.72,
            "start_time": 0.0,
            "end_time": 3.0,
        }
    ]


def test_week_uses_recording_date():
    runner = BirdnetRunner(storage=None)

    assert runner._week_from_recorded_at("2026-04-10T06:30:00+02:00") == 15
