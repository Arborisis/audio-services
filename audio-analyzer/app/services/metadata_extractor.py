import json
import subprocess
from typing import Any

from app.core.logger import get_logger

logger = get_logger(__name__)


class MetadataExtractor:
    def extract(self, local_path: str) -> dict[str, Any]:
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            local_path,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
            data = json.loads(result.stdout)
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            logger.error("ffprobe_failed", path=local_path, error=str(e))
            raise RuntimeError(f"ffprobe failed: {e}")

        stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), {})
        fmt = data.get("format", {})

        duration = self._parse_float(fmt.get("duration")) or self._parse_float(stream.get("duration")) or 0.0
        sample_rate = self._parse_int(stream.get("sample_rate"))
        channels = self._parse_int(stream.get("channels"))
        bitrate = self._parse_int(fmt.get("bit_rate"))
        format_name = fmt.get("format_name", "").split(",")[0]

        return {
            "duration_seconds": round(duration, 2),
            "sample_rate": sample_rate,
            "channels": channels,
            "bitrate": bitrate,
            "format": format_name,
        }

    @staticmethod
    def _parse_float(value: Any) -> float | None:
        try:
            return float(value) if value is not None else None
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _parse_int(value: Any) -> int | None:
        try:
            return int(value) if value is not None else None
        except (ValueError, TypeError):
            return None
