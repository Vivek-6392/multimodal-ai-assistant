import tempfile
import wave
from io import BytesIO
from pathlib import Path
from typing import Any

from app.providers.groq_client import GroqClient


class AudioTranscriptionService:
    def __init__(self, groq_client: GroqClient):
        self.groq_client = groq_client

    def transcribe(self, content: bytes, suffix: str) -> tuple[str, dict[str, Any], list[str]]:
        warnings: list[str] = []
        metadata = self._duration_metadata(content, suffix, warnings)
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(content)
                temp_path = Path(temp_file.name)
            return self.groq_client.transcribe_audio(str(temp_path)).strip(), metadata, warnings
        except Exception as exc:
            return "", metadata, warnings + [f"Audio transcription failed: {exc}"]
        finally:
            if temp_path and temp_path.exists():
                temp_path.unlink(missing_ok=True)

    @staticmethod
    def _duration_metadata(content: bytes, suffix: str, warnings: list[str]) -> dict[str, Any]:
        duration = AudioTranscriptionService._wav_duration(content)
        if duration is None:
            duration = AudioTranscriptionService._mutagen_duration(content, suffix, warnings)
        if duration is None:
            return {}
        return {"duration_seconds": round(duration, 2)}

    @staticmethod
    def _wav_duration(content: bytes) -> float | None:
        try:
            with wave.open(BytesIO(content), "rb") as audio:
                frame_rate = audio.getframerate()
                if frame_rate <= 0:
                    return None
                return audio.getnframes() / float(frame_rate)
        except (wave.Error, EOFError):
            return None

    @staticmethod
    def _mutagen_duration(content: bytes, suffix: str, warnings: list[str]) -> float | None:
        try:
            from mutagen import File as MutagenFile
        except ImportError:
            warnings.append("Audio duration metadata unavailable. Install mutagen for non-WAV duration detection.")
            return None

        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(content)
                temp_path = Path(temp_file.name)
            audio = MutagenFile(str(temp_path))
            if not audio or not getattr(audio, "info", None):
                return None
            length = getattr(audio.info, "length", None)
            return float(length) if length else None
        except Exception as exc:
            warnings.append(f"Audio duration detection failed: {exc}")
            return None
        finally:
            if temp_path and temp_path.exists():
                temp_path.unlink(missing_ok=True)
