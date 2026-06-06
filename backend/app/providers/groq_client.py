import json
import re
from typing import Any

from groq import Groq

from app.core.config import Settings


class GroqClient:
    def __init__(self, settings: Settings):
        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY is required.")
        self.settings = settings
        self.client = Groq(api_key=settings.groq_api_key)

    def chat(self, system: str, user: str, *, max_tokens: int = 1400) -> str:
        response = self.client.chat.completions.create(
            model=self.settings.groq_chat_model,
            temperature=self.settings.groq_temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content or ""

    def chat_json(self, system: str, user: str, *, max_tokens: int = 900) -> dict[str, Any]:
        raw = self.chat(system, user, max_tokens=max_tokens)
        return self._parse_json(raw)

    def transcribe_audio(self, file_path: str) -> str:
        with open(file_path, "rb") as audio_file:
            response = self.client.audio.transcriptions.create(
                file=audio_file,
                model=self.settings.groq_audio_model,
                response_format="text",
            )
        return str(response)

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            if not match:
                return {}
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return {}
