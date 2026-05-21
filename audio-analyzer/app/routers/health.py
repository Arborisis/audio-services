import importlib.util
import shutil

from fastapi import APIRouter

from app.core.logger import get_logger
from app.models.responses import HealthResponse

router = APIRouter()
logger = get_logger(__name__)

# Cache healthcheck results at module level to avoid repeated import overhead on every request
_ffmpeg_ok: bool | None = None
_librosa_ok: bool | None = None
_birdnet_ok: bool | None = None


def _command_exists(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def _check_ffmpeg() -> bool:
    global _ffmpeg_ok
    if _ffmpeg_ok is None:
        _ffmpeg_ok = _command_exists("ffmpeg")
    return _ffmpeg_ok


def _check_librosa() -> bool:
    global _librosa_ok
    if _librosa_ok is None:
        try:
            import librosa
            _librosa_ok = True
        except ImportError:
            _librosa_ok = False
    return _librosa_ok


def _check_birdnet() -> bool:
    global _birdnet_ok
    if _birdnet_ok is None:
        _birdnet_ok = importlib.util.find_spec("birdnet_analyzer") is not None
    return _birdnet_ok


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        birdnet_available=_check_birdnet(),
        ffmpeg_available=_check_ffmpeg(),
        librosa_available=_check_librosa(),
    )
