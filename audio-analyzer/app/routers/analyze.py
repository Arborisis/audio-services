import asyncio
import os
import shutil
import uuid
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends

from app.config import settings
from app.core.exceptions import AnalysisError
from app.core.logger import get_logger
from app.core.security import verify_token
from app.models.requests import AnalyzeRequest
from app.models.responses import AnalyzeResponse
from app.services.audio_downloader import AudioDownloader
from app.services.birdnet_runner import BirdnetRunner
from app.services.feature_extractor import FeatureExtractor
from app.services.laravel_callback import LaravelCallback
from app.services.metadata_extractor import MetadataExtractor
from app.services.preview_generator import PreviewGenerator
from app.services.quality_analyzer import QualityAnalyzer
from app.services.r2_storage import R2Storage
from app.services.spectrogram_generator import SpectrogramGenerator
from app.services.summary_builder import SummaryBuilder
from app.services.waveform_generator import WaveformGenerator

router = APIRouter()
logger = get_logger(__name__)

# In-memory tracking (replace with Redis in production for multi-replica)
_analysis_lock: set[str] = set()


def _get_storage() -> R2Storage:
    return R2Storage()


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    request: AnalyzeRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token),
) -> AnalyzeResponse:
    analysis_id = str(uuid.uuid4())
    lock_key = f"{request.sound_id}:{request.original_r2_key}"

    if lock_key in _analysis_lock and not request.force:
        return AnalyzeResponse(
            status="already_running",
            sound_id=request.sound_id,
            analysis_id=analysis_id,
            message="Analysis already in progress for this sound.",
        )

    storage = _get_storage()
    if not request.force and storage.key_exists(f"sounds/analysis/{request.sound_id}/summary.json"):
        return AnalyzeResponse(
            status="already_exists",
            sound_id=request.sound_id,
            analysis_id=analysis_id,
            message="Analysis already completed for this sound. Use force=true to rerun.",
        )

    _analysis_lock.add(lock_key)

    background_tasks.add_task(
        _run_analysis,
        sound_id=request.sound_id,
        original_r2_key=request.original_r2_key,
        analysis_id=analysis_id,
        lock_key=lock_key,
        lat=request.lat,
        lon=request.lon,
        recorded_at=request.recorded_at,
        force=request.force,
    )

    return AnalyzeResponse(
        status="accepted",
        sound_id=request.sound_id,
        analysis_id=analysis_id,
        message="Analysis started.",
    )


async def _run_analysis(sound_id: int, original_r2_key: str, analysis_id: str, lock_key: str, lat: float | None = None, lon: float | None = None, recorded_at: str | None = None, force: bool = False) -> None:
    # Run the blocking analysis pipeline in a thread pool so the event loop stays responsive
    await asyncio.to_thread(_run_analysis_sync, sound_id, original_r2_key, analysis_id, lock_key, lat, lon, recorded_at, force)


def _run_analysis_sync(sound_id: int, original_r2_key: str, analysis_id: str, lock_key: str, lat: float | None = None, lon: float | None = None, recorded_at: str | None = None, force: bool = False) -> None:
    temp_dir = f"/tmp/analyzer/{analysis_id}"
    local_path: str | None = None

    try:
        os.makedirs(temp_dir, exist_ok=True)
        storage = R2Storage()

        # 1. Download
        logger.info("analysis_start", sound_id=sound_id, analysis_id=analysis_id, r2_key=original_r2_key)
        downloader = AudioDownloader(storage)
        local_path = downloader.download(original_r2_key)

        # 2. Metadata
        meta = MetadataExtractor().extract(local_path)
        duration = meta["duration_seconds"]

        if duration > settings.max_duration_seconds:
            raise AnalysisError(f"Duration {duration}s exceeds max {settings.max_duration_seconds}s.")

        # 3. Preview
        preview_r2_key = PreviewGenerator(storage).generate_and_upload(local_path, sound_id)

        # 4. Waveform
        waveform_r2_key = WaveformGenerator(storage).generate_and_upload(local_path, sound_id, duration)

        # 5. Spectrogram
        spectrogram_r2_key = SpectrogramGenerator(storage).generate_and_upload(local_path, sound_id)

        # 6. Features
        feature_extractor = FeatureExtractor(storage)
        features, features_r2_key = feature_extractor.extract_and_upload(local_path, sound_id, meta)

        # 7. Quality
        import librosa
        y, sr = librosa.load(local_path, sr=None, mono=True)
        quality = QualityAnalyzer().analyze(y, sr, features)

        # 8. BirdNET
        birdnet_r2_key, detections = BirdnetRunner(storage).analyze_and_upload(
            local_path, sound_id, duration, quality["usable_for_analysis"], lat=lat, lon=lon, recorded_at=recorded_at
        )

        # 9. Summary
        summary_r2_key = SummaryBuilder(storage).build_and_upload(
            sound_id, duration, detections, quality, features
        )

        # 10. Callback to Laravel
        callback_payload = {
            "sound_id": sound_id,
            "status": "completed",
            "force": force,
            "results": {
                "original_r2_key": original_r2_key,
                "duration_seconds": features["duration_seconds"],
                "sample_rate": features["sample_rate"],
                "channels": features["channels"],
                "bitrate": features["bitrate"],
                "format": features["format"],
                "loudness_lufs": features["loudness_lufs"],
                "peak_db": features["peak_db"],
                "rms_db": features["rms_db"],
                "noise_floor_db": features["noise_floor_db"],
                "spectral_centroid": features["spectral_centroid"],
                "spectral_rolloff": features["spectral_rolloff"],
                "zero_crossing_rate": features["zero_crossing_rate"],
                "features_json": features,
                "waveform_r2_key": waveform_r2_key,
                "spectrogram_r2_key": spectrogram_r2_key,
                "features_r2_key": features_r2_key,
                "birdnet_r2_key": birdnet_r2_key,
                "summary_r2_key": summary_r2_key,
                "preview_r2_key": preview_r2_key,
                "quality_label": quality["quality_label"],
                "quality_json": quality,
                "birdnet_detections": detections,
            },
        }
        LaravelCallback()._send_sync(callback_payload)

        logger.info("analysis_completed", sound_id=sound_id, analysis_id=analysis_id)

    except Exception as e:
        logger.error("analysis_failed", sound_id=sound_id, analysis_id=analysis_id, error=str(e))
        error_payload = {
            "sound_id": sound_id,
            "status": "failed",
            "force": force,
            "error_message": str(e),
        }
        LaravelCallback()._send_sync(error_payload)
    finally:
        _analysis_lock.discard(lock_key)
        if local_path and os.path.exists(local_path):
            shutil.rmtree(os.path.dirname(local_path), ignore_errors=True)
