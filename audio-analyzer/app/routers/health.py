from fastapi import APIRouter

from app.core.logger import get_logger
from app.models.responses import HealthResponse

router = APIRouter()
logger = get_logger(__name__)


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        birdnet_available=True,
        ffmpeg_available=True,
        librosa_available=True,
    )
