import csv
import json
import os
import shutil
import subprocess
from collections import defaultdict
from datetime import datetime
from typing import Any

from app.config import settings
from app.core.logger import get_logger
from app.services.r2_storage import R2Storage

logger = get_logger(__name__)

MIN_DURATION_FOR_BIRDNET = 3.0
DEFAULT_CONFIDENCE = 0.3  # lowered from 0.5 to catch more detections
DEFAULT_OVERLAP = 1.5
DEFAULT_SENSITIVITY = 1.25
DEFAULT_PUBLISH_CONFIDENCE = 0.45
DEFAULT_REPEATED_CONFIDENCE = 0.32


class BirdnetRunner:
    def __init__(self, storage: R2Storage) -> None:
        self.storage = storage

    def analyze_and_upload(
        self,
        local_path: str,
        sound_id: int,
        duration: float,
        usable: bool,
        lat: float | None = None,
        lon: float | None = None,
        recorded_at: str | None = None,
    ) -> tuple[str | None, list[dict[str, Any]]]:
        if duration < MIN_DURATION_FOR_BIRDNET or not usable:
            logger.info("birdnet_skipped", sound_id=sound_id, duration=duration, usable=usable)
            return None, []

        result_dir = local_path.replace(os.path.splitext(local_path)[1], "_birdnet")

        cmd = [
            "python3", "-m", "birdnet_analyzer.analyze",
            local_path,
            "-o", result_dir,
            "--rtype", "csv",
            "--min_conf", str(settings.birdnet_confidence_threshold or DEFAULT_CONFIDENCE),
            "--overlap", str(DEFAULT_OVERLAP),
            "--sensitivity", str(DEFAULT_SENSITIVITY),
        ]

        # Geo-filters improve accuracy when available
        if lat is not None:
            cmd.extend(["--lat", str(lat)])
        if lon is not None:
            cmd.extend(["--lon", str(lon)])

        # BirdNET uses the week as an occurrence prior. Prefer the recording date
        # over the processing date so old uploads are classified in the right season.
        week = self._week_from_recorded_at(recorded_at)
        cmd.extend(["--week", str(week)])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=300)
            logger.info("birdnet_stdout", sound_id=sound_id, stdout=result.stdout[:500])
        except subprocess.CalledProcessError as e:
            logger.error("birdnet_failed", sound_id=sound_id, error=str(e), stderr=e.stderr[:500])
            return None, []
        except FileNotFoundError as e:
            logger.error("birdnet_not_found", sound_id=sound_id, error=str(e))
            return None, []

        result_csv = self._find_csv(result_dir)
        if not result_csv:
            logger.warning("birdnet_no_csv_output", sound_id=sound_id, output_dir=result_dir)
            return None, []

        raw_detections = self._parse_csv(result_csv)
        detections = self._postprocess_detections(raw_detections)

        if os.path.exists(result_dir):
            shutil.rmtree(result_dir, ignore_errors=True)

        birdnet_data = {
            "model": "BirdNET Analyzer V2.4",
            "confidence_threshold": settings.birdnet_confidence_threshold or DEFAULT_CONFIDENCE,
            "publish_confidence_threshold": settings.birdnet_publish_confidence_threshold or DEFAULT_PUBLISH_CONFIDENCE,
            "repeated_confidence_threshold": settings.birdnet_repeated_confidence_threshold or DEFAULT_REPEATED_CONFIDENCE,
            "overlap_seconds": DEFAULT_OVERLAP,
            "sensitivity": DEFAULT_SENSITIVITY,
            "latitude": lat,
            "longitude": lon,
            "week_of_year": week,
            "raw_detection_count": len(raw_detections),
            "detections": detections,
        }

        temp_json = local_path.replace(os.path.splitext(local_path)[1], "_birdnet.json")
        with open(temp_json, "w") as f:
            json.dump(birdnet_data, f, indent=2)

        r2_key = f"sounds/analysis/{sound_id}/birdnet.json"
        self.storage.upload(temp_json, r2_key, content_type="application/json")
        os.remove(temp_json)

        logger.info("birdnet_completed", sound_id=sound_id, r2_key=r2_key, detections=len(detections))
        return r2_key, detections

    def _find_csv(self, output_dir: str) -> str | None:
        for root, _, files in os.walk(output_dir):
            for filename in files:
                if filename.lower().endswith(".csv"):
                    return os.path.join(root, filename)
        return None

    def _parse_csv(self, csv_path: str) -> list[dict[str, Any]]:
        if not os.path.exists(csv_path):
            return []

        detections = []
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                detection = self._parse_csv_row(row)
                if detection is not None:
                    detections.append(detection)

        return detections

    def _parse_csv_row(self, row: dict[str, str]) -> dict[str, Any] | None:
        try:
            start = row.get("Start (s)") or row.get("Start") or row.get("start_time")
            end = row.get("End (s)") or row.get("End") or row.get("end_time")
            scientific_name = row.get("Scientific name") or row.get("Scientific Name") or row.get("scientific_name")
            common_name = row.get("Common name") or row.get("Common Name") or row.get("common_name")
            confidence = row.get("Confidence") or row.get("confidence")

            if not start or not end or not scientific_name or not common_name or not confidence:
                return None

            return {
                "scientific_name": scientific_name.strip(),
                "common_name": common_name.strip(),
                "confidence": round(float(confidence.strip()), 3),
                "start_time": round(float(start.strip()), 2),
                "end_time": round(float(end.strip()), 2),
            }
        except (ValueError, AttributeError):
            return None

    def _postprocess_detections(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not detections:
            return []

        publish_threshold = settings.birdnet_publish_confidence_threshold or DEFAULT_PUBLISH_CONFIDENCE
        repeated_threshold = settings.birdnet_repeated_confidence_threshold or DEFAULT_REPEATED_CONFIDENCE
        max_detections = max(1, settings.birdnet_max_detections)

        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for detection in detections:
            key = f"{detection['scientific_name'].lower()}|{detection['common_name'].lower()}"
            grouped[key].append(detection)

        accepted: list[dict[str, Any]] = []
        for species_detections in grouped.values():
            species_detections = sorted(species_detections, key=lambda d: d["confidence"], reverse=True)
            confidence_values = [float(d["confidence"]) for d in species_detections]
            max_confidence = max(confidence_values)
            mean_confidence = sum(confidence_values) / len(confidence_values)
            detection_count = len(species_detections)

            keep_species = max_confidence >= publish_threshold or (
                detection_count >= 2 and mean_confidence >= repeated_threshold
            )

            if not keep_species:
                continue

            aggregate_confidence = round((max_confidence * 0.7) + (mean_confidence * 0.3), 3)
            segment_floor = min(publish_threshold, max(repeated_threshold, mean_confidence * 0.9))

            for detection in species_detections:
                if detection["confidence"] < segment_floor:
                    continue
                enriched = {
                    **detection,
                    "aggregate_confidence": aggregate_confidence,
                    "species_detection_count": detection_count,
                    "source": "birdnet",
                }
                accepted.append(enriched)

        accepted.sort(
            key=lambda d: (
                d.get("aggregate_confidence", d["confidence"]),
                d["confidence"],
                d.get("species_detection_count", 1),
            ),
            reverse=True,
        )

        return accepted[:max_detections]

    def _week_from_recorded_at(self, recorded_at: str | None) -> int:
        if not recorded_at:
            return datetime.now().isocalendar().week

        normalized = recorded_at.strip()
        if not normalized:
            return datetime.now().isocalendar().week

        if normalized.endswith("Z"):
            normalized = normalized[:-1] + "+00:00"

        try:
            return datetime.fromisoformat(normalized).isocalendar().week
        except ValueError:
            try:
                return datetime.strptime(normalized[:10], "%Y-%m-%d").isocalendar().week
            except ValueError:
                logger.warning("birdnet_invalid_recorded_at", recorded_at=recorded_at)
                return datetime.now().isocalendar().week
