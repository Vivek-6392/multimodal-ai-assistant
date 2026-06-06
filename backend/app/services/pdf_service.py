from app.core.config import Settings
from app.services.ocr_service import OCRService


class PDFService:
    def __init__(self, settings: Settings, ocr_service: OCRService):
        self.settings = settings
        self.ocr_service = ocr_service

    def extract_text(self, content: bytes) -> tuple[str, bool, dict, list[str]]:
        warnings: list[str] = []
        metadata: dict = {}
        parsed_text = self._extract_with_pypdf(content, warnings, metadata)
        if parsed_text.strip():
            return parsed_text.strip(), False, metadata, warnings

        ocr_text, ocr_confidence, ocr_warnings = self._ocr_pdf_pages(content)
        warnings.extend(ocr_warnings)
        if ocr_confidence is not None:
            metadata["ocr_confidence"] = ocr_confidence
        return ocr_text.strip(), bool(ocr_text.strip()), metadata, warnings

    def _extract_with_pypdf(self, content: bytes, warnings: list[str], metadata: dict) -> str:
        try:
            from pypdf import PdfReader
        except ImportError:
            warnings.append("PDF text parser dependency is missing. Install pypdf.")
            return ""

        try:
            import io

            reader = PdfReader(io.BytesIO(content))
            metadata["pages"] = len(reader.pages)
            page_text: list[str] = []
            for page in reader.pages:
                page_text.append(page.extract_text() or "")
            return "\n\n".join(page_text)
        except Exception as exc:
            warnings.append(f"PDF text extraction failed: {exc}")
            return ""

    def _ocr_pdf_pages(self, content: bytes) -> tuple[str, float | None, list[str]]:
        warnings: list[str] = []
        try:
            import fitz
            from PIL import Image
        except ImportError:
            return "", None, ["PDF OCR fallback dependencies are missing. Install pymupdf, pillow, and pytesseract."]

        try:
            import io

            document = fitz.open(stream=content, filetype="pdf")
            page_count = min(len(document), self.settings.pdf_ocr_max_pages)
            text_parts: list[str] = []
            confidences: list[float] = []
            for page_index in range(page_count):
                page = document.load_page(page_index)
                pixmap = page.get_pixmap(dpi=200)
                image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
                image_buffer = io.BytesIO()
                image.save(image_buffer, format="PNG")
                text, confidence, image_warnings = self.ocr_service.image_to_text(image_buffer.getvalue())
                text_parts.append(text)
                warnings.extend(image_warnings)
                if confidence is not None:
                    confidences.append(confidence)
            if len(document) > page_count:
                warnings.append(f"OCR fallback limited to first {page_count} pages.")
            average_confidence = round(sum(confidences) / len(confidences), 3) if confidences else None
            return "\n\n".join(text_parts), average_confidence, warnings
        except Exception as exc:
            return "", None, [f"PDF OCR fallback failed: {exc}"]
