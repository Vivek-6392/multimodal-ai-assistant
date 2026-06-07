from app.agent.intent_classifier import IntentClassifier
from app.agent.planner import ReasoningPlanner
from app.core.config import Settings
from app.models.schemas import (
    AgentResponse,
    CostEstimate,
    ExtractedArtifact,
    ToolExecutionGraph,
    ToolGraphEdge,
    ToolGraphNode,
    ToolTraceStep,
)
from app.providers.groq_client import GroqClient
from app.services.url_service import detect_urls
from app.tools.registry import ToolRegistry, ToolResult


class AgentOrchestrator:
    def __init__(self, settings: Settings, groq_client: GroqClient):
        self.settings = settings
        self.classifier = IntentClassifier(groq_client)
        self.planner = ReasoningPlanner()
        self.tools = ToolRegistry(groq_client)

    def run(self, message: str, artifacts: list[ExtractedArtifact]) -> AgentResponse:
        trace: list[ToolTraceStep] = []
        combined_context = self._combine_context(message, artifacts)
        urls = sorted(set(detect_urls(message or "") + [url for artifact in artifacts for url in artifact.urls]))
        print("DETECTED URLS:", urls)
        has_audio = any(artifact.artifact_type.value == "audio" for artifact in artifacts)
        has_pdf = any(artifact.artifact_type.value == "pdf" for artifact in artifacts)

        extraction_summary = self._extraction_summary(artifacts)
        trace.append(
            ToolTraceStep(
                step=1,
                tool="extract_inputs",
                input_summary=f"{len(artifacts)} uploaded file(s).",
                output_summary=extraction_summary,
            )
        )
        trace.append(
            ToolTraceStep(
                step=2,
                tool="detect_urls",
                input_summary="Extracted text and user message.",
                output_summary=f"Detected {len(urls)} URL(s).",
            )
        )

        intent_data = self.classifier.classify(
            message,
            combined_context,
            urls,
            has_audio=has_audio,
            has_pdf=has_pdf,
        )
        intent = intent_data["intent"]
        trace.append(
            ToolTraceStep(
                step=3,
                tool="classify_intent",
                input_summary="User request plus content preview.",
                output_summary=f"Intent: {intent}. {intent_data.get('rationale', '')}".strip(),
            )
        )

        if intent_data.get("needs_clarification"):
            final_answer = intent_data.get("clarification_question") or "Could you clarify what outcome you want?"
            return self._response(final_answer, artifacts, trace, intent, combined_context)

        plan = self.planner.create_plan(intent, urls, has_audio=has_audio, has_pdf=has_pdf)
        working_context = combined_context
        tool_outputs: list[ToolResult] = []

        for tool_name in plan:
            if tool_name in {"extract_inputs", "detect_urls", "classify_intent", "synthesize_final_answer"}:
                continue
            result = self.tools.run(
                tool_name,
                message=message,
                context=working_context,
                urls=urls,
                summary_style=intent_data.get("summary_style", "three_bullets"),
            )
            tool_outputs.append(result)
            if tool_name == "youtube_transcript" and result.answer:
                working_context = f"{working_context}\n\n{result.answer}"
            trace.append(
                ToolTraceStep(
                    step=len(trace) + 1,
                    tool=tool_name,
                    input_summary=self._input_summary(tool_name, working_context, urls),
                    output_summary=result.output_summary,
                )
            )

        final_answer = self._final_answer(
            intent,
            tool_outputs,
            message,
            working_context,
            trace,
            artifacts
        )
        trace.append(
            ToolTraceStep(
                step=len(trace) + 1,
                tool="synthesize_final_answer",
                input_summary=f"{len(tool_outputs)} tool result(s).",
                output_summary="Produced clean final answer.",
            )
        )
        return self._response(final_answer, artifacts, trace, intent, f"{working_context}\n\n{final_answer}")

    def _final_answer(
        self,
        intent,
        tool_outputs,
        message,
        context,
        trace=None,
        artifacts=None,
    ):

        transcript_text = self._combined_transcript(tool_outputs)
        youtube_failure = self._youtube_failure(tool_outputs)

        print("=" * 60)
        print("TRANSCRIPT LENGTH:", len(transcript_text))
        print("TRANSCRIPT PREVIEW:")
        print(transcript_text[:500])
        print("=" * 60)

        if youtube_failure and self._has_substantive_video_context(context):

            system = (
                "You are a concise summarization assistant. "
                "Using ONLY the provided PDF content, produce a brief summary."
            )

            user = (
                f"User request:\n{message or 'Summarize the referenced video.'}\n\n"
                f"Extracted content:\n{context[:18000]}"
            )

            answer = self.tools.groq_client.chat(system, user)

        elif youtube_failure:

            answer = (
                "I found a YouTube URL in the PDF, but I could not retrieve "
                "the video's transcript."
            )

        elif transcript_text:

            # IMPORTANT FIX
            summary_result = self.tools.run(
                "summarize",
                message=message,
                context=transcript_text,
            )

            answer = summary_result.answer

        elif not tool_outputs:

            result = self.tools.run(
                "qa",
                message=message,
                context=context,
                urls=[]
            )

            answer = result.answer

        else:

            answer = tool_outputs[-1].answer

        return answer

    @staticmethod
    def _combined_transcript(tool_outputs: list[ToolResult]) -> str:
        transcript_sections: list[str] = []
        for output in tool_outputs:
            transcripts = output.metadata.get("transcripts") if output.metadata else None
            if isinstance(transcripts, dict):
                transcript_sections.extend(text for text in transcripts.values() if text)
        return "\n\n".join(transcript_sections).strip()

    @staticmethod
    def _youtube_failure(tool_outputs: list[ToolResult]) -> bool:
        """
        True only when transcript retrieval completely failed.
        """

        found_transcript = False
        found_failure = False

        for output in tool_outputs:
            if not output.metadata:
                continue

            transcripts = output.metadata.get("transcripts", {})
            failures = output.metadata.get("failures", {})

            if transcripts:
                found_transcript = True

            if failures:
                found_failure = True

        return found_failure and not found_transcript
    @staticmethod
    def _has_substantive_video_context(context: str) -> bool:
        import re

        without_urls = re.sub(r"https?://\S+", " ", context or "")
        words = re.findall(r"[A-Za-z0-9]{3,}", without_urls)
        return len(words) >= 20

    def _response(
        self,
        final_answer: str,
        artifacts: list[ExtractedArtifact],
        trace: list[ToolTraceStep],
        intent: str,
        cost_basis: str,
    ) -> AgentResponse:
        return AgentResponse(
            final_answer=final_answer.strip(),
            extracted_text=artifacts,
            tool_execution_trace=trace,
            intent=intent,
            cost_estimate=self._estimate_cost(cost_basis, final_answer),
            tool_graph=self._graph(trace),
        )

    def _combine_context(self, message: str, artifacts: list[ExtractedArtifact]) -> str:
        sections: list[str] = []
        if message:
            sections.append(f"User message:\n{message}")
        for artifact in artifacts:
            text = artifact.text or ""
            metadata_lines: list[str] = []
            if artifact.metadata:
                metadata_lines.append(f"Metadata: {artifact.metadata}")
            if artifact.ocr_confidence is not None:
                metadata_lines.append(f"OCR confidence: {artifact.ocr_confidence:.3f}")
            metadata = "\n".join(metadata_lines)
            sections.append(f"File: {artifact.file_name} ({artifact.artifact_type})\n{metadata}\n{text}".strip())
        return "\n\n---\n\n".join(sections).strip()

    @staticmethod
    def _extraction_summary(artifacts: list[ExtractedArtifact]) -> str:
        if not artifacts:
            return "No files uploaded; using user message only."
        readable = sum(1 for artifact in artifacts if artifact.text)
        ocr_count = sum(1 for artifact in artifacts if artifact.ocr_used)
        warning_count = sum(len(artifact.warnings) for artifact in artifacts)
        return f"Read text from {readable}/{len(artifacts)} file(s); OCR used for {ocr_count}; warnings: {warning_count}."

    @staticmethod
    def _input_summary(tool_name: str, context: str, urls: list[str]) -> str:
        if tool_name == "youtube_transcript":
            return f"{len(urls)} detected URL(s)."
        return f"{max(1, len(context) // 4)} estimated input tokens."

    def _estimate_cost(self, input_text: str, output_text: str) -> CostEstimate:
        input_tokens = max(1, len(input_text) // 4)
        output_tokens = max(1, len(output_text) // 4)
        usd = (
            input_tokens * self.settings.groq_input_cost_per_1m
            + output_tokens * self.settings.groq_output_cost_per_1m
        ) / 1_000_000
        return CostEstimate(
            input_tokens_estimate=input_tokens,
            output_tokens_estimate=output_tokens,
            estimated_usd=round(usd, 6),
        )

    @staticmethod
    def _graph(trace: list[ToolTraceStep]) -> ToolExecutionGraph:
        nodes = [ToolGraphNode(id=str(step.step), label=step.tool, status=step.status) for step in trace]
        edges = [ToolGraphEdge(source=str(trace[i].step), target=str(trace[i + 1].step)) for i in range(len(trace) - 1)]
        return ToolExecutionGraph(nodes=nodes, edges=edges)
