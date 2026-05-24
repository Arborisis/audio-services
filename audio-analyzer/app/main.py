from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.core.logger import configure_logging
from app.routers import analyze, health

configure_logging(settings.log_level)

app = FastAPI(
    title="Arborisis Audio Analyzer",
    description="Audio analysis service for Arborisis — waveform, spectrogram, features, BirdNET.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["health"])
app.include_router(analyze.router, tags=["analysis"])
