from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Agentic Multimodal AI"
    environment: str = "development"
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    groq_api_key: str | None = None
    groq_chat_model: str = "llama-3.3-70b-versatile"
    groq_audio_model: str = "whisper-large-v3-turbo"
    groq_temperature: float = 0.2

    max_upload_mb: int = 50
    ocr_language: str = "eng"
    pdf_ocr_max_pages: int = 12

    groq_input_cost_per_1m: float = 0.59
    groq_output_cost_per_1m: float = 0.79

    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
