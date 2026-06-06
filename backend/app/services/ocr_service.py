from io import BytesIO

from app.core.config import Settings


class OCRService:
    def __init__(self, settings: Settings):
        self.settings = settings

    def image_to_text(self, content: bytes) -> tuple[str, float | None, list[str]]:
        warnings: list[str] = []
        try:
            from PIL import Image
            import pytesseract
        except ImportError:
            return "", None, ["Image OCR dependencies are missing. Install pillow and pytesseract."]

        try:
            image = Image.open(BytesIO(content))
            text = pytesseract.image_to_string(image, lang=self.settings.ocr_language)
            confidence = self._average_confidence(image, pytesseract, warnings)
            return text.strip(), confidence, warnings
        except Exception as exc:
            return "", None, [f"Image OCR failed: {exc}"]

    def _average_confidence(self, image, pytesseract, warnings: list[str]) -> float | None:
        try:
            data = pytesseract.image_to_data(
                image,
                lang=self.settings.ocr_language,
                output_type=pytesseract.Output.DICT,
            )
        except Exception as exc:
            warnings.append(f"OCR confidence extraction failed: {exc}")
            return None

        scores: list[float] = []
        for raw_score in data.get("conf", []):
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                continue
            if score >= 0:
                scores.append(score / 100)
        if not scores:
            return None
        return round(sum(scores) / len(scores), 3)
