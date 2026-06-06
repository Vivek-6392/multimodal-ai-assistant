from app.models.schemas import ArtifactType
from app.services.file_processor import FileProcessor


class FakeOCR:
    def image_to_text(self, content: bytes):
        return "image text", 0.91, []


class FakePDF:
    def extract_text(self, content: bytes):
        return "pdf text", False, {"pages": 1}, []


class FakeAudio:
    def transcribe(self, content: bytes, suffix: str):
        return "audio text", {"duration_seconds": 3.5}, []


def make_processor() -> FileProcessor:
    return FileProcessor(FakeOCR(), FakePDF(), FakeAudio())


def test_process_text_file_extracts_urls():
    artifact = make_processor().process_file(
        "note.txt",
        b"Hello https://example.com",
        "text/plain",
    )

    assert artifact.artifact_type == ArtifactType.text
    assert artifact.text == "Hello https://example.com"
    assert artifact.urls == ["https://example.com"]


def test_process_pdf_uses_pdf_service():
    artifact = make_processor().process_file("report.pdf", b"%PDF", "application/pdf")

    assert artifact.artifact_type == ArtifactType.pdf
    assert artifact.text == "pdf text"
    assert artifact.metadata["pages"] == 1


def test_process_image_includes_ocr_confidence():
    artifact = make_processor().process_file("screenshot.png", b"image", "image/png")

    assert artifact.artifact_type == ArtifactType.image
    assert artifact.ocr_used is True
    assert artifact.ocr_confidence == 0.91
    assert artifact.metadata["ocr_confidence"] == 0.91


def test_process_audio_includes_duration_metadata():
    artifact = make_processor().process_file("lecture.mp3", b"audio", "audio/mpeg")

    assert artifact.artifact_type == ArtifactType.audio
    assert artifact.text == "audio text"
    assert artifact.metadata["duration_seconds"] == 3.5
