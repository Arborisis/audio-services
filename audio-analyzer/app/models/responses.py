from pydantic import BaseModel
from typing import Optional


class AnalyzeResponse(BaseModel):
    status: str
    sound_id: int
    analysis_id: str
    message: str


class HealthResponse(BaseModel):
    status: str
    birdnet_available: bool
    ffmpeg_available: bool
    librosa_available: bool
    version: str = "1.0.0"
