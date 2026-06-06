from pathlib import Path
from typing import Any

from fastapi import UploadFile

from app.models.schemas import ArtifactType, ExtractedArtifact
from app.services.audio_service import AudioTranscriptionService
from app.services.ocr_service import OCRService
from app.services.pdf_service import PDFService
from app.services.url_service import detect_urls


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm", ".mp4"}
TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".py", ".js", ".ts", ".tsx", ".html", ".css", ".log"}


class FileProcessor:
    def __init__(
        self,
        ocr_service: OCRService,
        pdf_service: PDFService,
        audio_service: AudioTranscriptionService,
    ):
        self.ocr_service = ocr_service
        self.pdf_service = pdf_service
        self.audio_service = audio_service

    async def process_uploads(self, files: list[UploadFile] | None) -> list[ExtractedArtifact]:
        artifacts: list[ExtractedArtifact] = []
        for upload in files or []:
            content = await upload.read()
            artifacts.append(self.process_file(upload.filename or "uploaded_file", content, upload.content_type))
        return artifacts

    def process_file(self, file_name: str, content: bytes, content_type: str | None = None) -> ExtractedArtifact:
        suffix = Path(file_name).suffix.lower()
        artifact_type = self._detect_type(suffix, content_type)
        text = ""
        metadata = {"content_type": content_type, "size_bytes": len(content)}
        warnings: list[str] = []
        ocr_used = False
        ocr_confidence: float | None = None

        if artifact_type == ArtifactType.image:
            text, ocr_confidence, warnings = self._normalize_ocr_result(self.ocr_service.image_to_text(content))
            ocr_used = True
            if ocr_confidence is not None:
                metadata["ocr_confidence"] = ocr_confidence
        elif artifact_type == ArtifactType.pdf:
            text, ocr_used, pdf_metadata, warnings = self.pdf_service.extract_text(content)
            metadata.update(pdf_metadata)
            ocr_confidence = metadata.get("ocr_confidence")
        elif artifact_type == ArtifactType.audio:
            text, audio_metadata, warnings = self._normalize_audio_result(
                self.audio_service.transcribe(content, suffix or ".audio")
            )
            metadata.update(audio_metadata)
        elif artifact_type == ArtifactType.text:
            text, warnings = self._decode_text(content)
        else:
            text, warnings = self._decode_text(content)
            if not text:
                warnings.append("Unsupported file type. No extractor could read this file.")

        return ExtractedArtifact(
            file_name=file_name,
            artifact_type=artifact_type,
            text=text,
            urls=detect_urls(text),
            metadata=metadata,
            ocr_used=ocr_used,
            ocr_confidence=ocr_confidence,
            warnings=warnings,
        )

    def _detect_type(self, suffix: str, content_type: str | None) -> ArtifactType:
        content_type = content_type or ""
        if suffix in IMAGE_EXTENSIONS or content_type.startswith("image/"):
            return ArtifactType.image
        if suffix == ".pdf" or content_type == "application/pdf":
            return ArtifactType.pdf
        if suffix in AUDIO_EXTENSIONS or content_type.startswith("audio/"):
            return ArtifactType.audio
        if suffix in TEXT_EXTENSIONS or content_type.startswith("text/"):
            return ArtifactType.text
        return ArtifactType.unknown

    @staticmethod
    def _decode_text(content: bytes) -> tuple[str, list[str]]:
        warnings: list[str] = []
        for encoding in ("utf-8", "utf-16", "latin-1"):
            try:
                return content.decode(encoding).strip(), warnings
            except UnicodeDecodeError:
                continue
        return "", ["Text decoding failed for utf-8, utf-16, and latin-1."]

    @staticmethod
    def _normalize_ocr_result(result: tuple) -> tuple[str, float | None, list[str]]:
        if len(result) == 3:
            text, confidence, warnings = result
            return text, confidence, warnings
        text, warnings = result
        return text, None, warnings

    @staticmethod
    def _normalize_audio_result(result: tuple) -> tuple[str, dict[str, Any], list[str]]:
        if len(result) == 3:
            text, metadata, warnings = result
            return text, metadata, warnings
        text, warnings = result
        return text, {}, warnings
