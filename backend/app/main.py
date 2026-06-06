import json
from collections.abc import AsyncGenerator

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.agent.orchestrator import AgentOrchestrator
from app.core.config import Settings, get_settings
from app.models.schemas import AgentResponse, HealthResponse
from app.providers.groq_client import GroqClient
from app.services.audio_service import AudioTranscriptionService
from app.services.file_processor import FileProcessor
from app.services.ocr_service import OCRService
from app.services.pdf_service import PDFService


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", app_name=settings.app_name, environment=settings.environment)

    @app.post("/api/chat", response_model=AgentResponse)
    async def chat(
        message: str = Form(default=""),
        files: list[UploadFile] | None = File(default=None),
        deps: tuple[Settings, FileProcessor, AgentOrchestrator] = Depends(build_dependencies),
    ) -> AgentResponse:
        settings, file_processor, orchestrator = deps
        await enforce_upload_limit(files, settings)
        artifacts = await file_processor.process_uploads(files)
        return orchestrator.run(message, artifacts)

    @app.post("/api/chat/stream")
    async def chat_stream(
        message: str = Form(default=""),
        files: list[UploadFile] | None = File(default=None),
        deps: tuple[Settings, FileProcessor, AgentOrchestrator] = Depends(build_dependencies),
    ) -> StreamingResponse:
        async def events() -> AsyncGenerator[str, None]:
            settings, file_processor, orchestrator = deps
            await enforce_upload_limit(files, settings)
            yield sse("status", {"message": "Extracting uploaded content"})
            artifacts = await file_processor.process_uploads(files)
            yield sse("status", {"message": "Planning and executing tools"})
            response = orchestrator.run(message, artifacts)
            yield sse("final", response.model_dump(mode="json"))

        return StreamingResponse(events(), media_type="text/event-stream")

    return app


def build_dependencies(settings: Settings = Depends(get_settings)) -> tuple[Settings, FileProcessor, AgentOrchestrator]:
    try:
        groq_client = GroqClient(settings)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    ocr_service = OCRService(settings)
    pdf_service = PDFService(settings, ocr_service)
    audio_service = AudioTranscriptionService(groq_client)
    file_processor = FileProcessor(ocr_service, pdf_service, audio_service)
    orchestrator = AgentOrchestrator(settings, groq_client)
    return settings, file_processor, orchestrator


async def enforce_upload_limit(files: list[UploadFile] | None, settings: Settings) -> None:
    max_bytes = settings.max_upload_mb * 1024 * 1024
    for upload in files or []:
        size = getattr(upload, "size", None)
        if size and size > max_bytes:
            raise HTTPException(status_code=413, detail=f"{upload.filename} exceeds {settings.max_upload_mb} MB.")


def sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


app = create_app()
