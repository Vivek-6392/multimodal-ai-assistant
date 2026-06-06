from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class ArtifactType(str, Enum):
    text = "text"
    image = "image"
    pdf = "pdf"
    audio = "audio"
    unknown = "unknown"


class ExtractedArtifact(BaseModel):
    file_name: str
    artifact_type: ArtifactType
    text: str = ""
    urls: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    ocr_used: bool = False
    ocr_confidence: float | None = None
    warnings: list[str] = Field(default_factory=list)


class ToolTraceStep(BaseModel):
    step: int
    tool: str
    input_summary: str
    output_summary: str
    status: str = "completed"


class ToolGraphNode(BaseModel):
    id: str
    label: str
    status: str = "completed"


class ToolGraphEdge(BaseModel):
    source: str
    target: str


class ToolExecutionGraph(BaseModel):
    nodes: list[ToolGraphNode] = Field(default_factory=list)
    edges: list[ToolGraphEdge] = Field(default_factory=list)


class CostEstimate(BaseModel):
    input_tokens_estimate: int
    output_tokens_estimate: int
    estimated_usd: float


class AgentResponse(BaseModel):
    final_answer: str
    extracted_text: list[ExtractedArtifact]
    tool_execution_trace: list[ToolTraceStep]
    intent: str
    cost_estimate: CostEstimate
    tool_graph: ToolExecutionGraph


class HealthResponse(BaseModel):
    status: str
    app_name: str
    environment: str
