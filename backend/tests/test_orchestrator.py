from app.agent.orchestrator import AgentOrchestrator
from app.core.config import Settings
from app.tools.registry import ToolResult


class FakeGroq:
    def __init__(self):
        self.last_system = ""
        self.last_user = ""

    def chat(self, system: str, user: str, *, max_tokens: int = 1400):
        self.last_system = system
        self.last_user = user
        return "transcript summary"

    def chat_json(self, system: str, user: str, *, max_tokens: int = 900):
        return {
            "intent": "youtube_transcript",
            "summary_style": "none",
            "needs_clarification": False,
            "clarification_question": "",
            "rationale": "fake",
        }


class FakeSettings(Settings):
    groq_api_key: str | None = "test"


def test_final_answer_prefers_youtube_transcript_content():
    orchestrator = AgentOrchestrator(FakeSettings(), FakeGroq())
    outputs = [
        ToolResult(
            answer="",
            output_summary="Fetched 1 transcript.",
            metadata={"transcripts": {"https://youtu.be/abc123xyz": "First line of the transcript."}},
        )
    ]

    answer = orchestrator._final_answer("youtube_transcript", outputs, "Summarize it", "context")

    assert "transcript summary" in answer


def test_final_answer_reports_missing_youtube_transcript():
    orchestrator = AgentOrchestrator(FakeSettings(), FakeGroq())
    outputs = [
        ToolResult(
            answer="",
            output_summary="Fetched 0 transcript(s).",
            metadata={
                "transcripts": {},
                "failures": {
                    "https://youtu.be/abc123xyz": "Subtitles are disabled for this video."
                },
            },
        )
    ]

    answer = orchestrator._final_answer("youtube_transcript", outputs, "Summarize it", "File: link.pdf\nhttps://youtu.be/abc123xyz")

    assert "could not retrieve the video's transcript" in answer.lower()
    assert "subtitles are disabled" not in answer.lower()


def test_final_answer_summarizes_substantive_pdf_video_context():
    groq = FakeGroq()
    orchestrator = AgentOrchestrator(FakeSettings(), groq)
    outputs = [
        ToolResult(
            answer="",
            output_summary="Fetched 0 transcript(s).",
            metadata={
                "transcripts": {},
                "failures": {
                    "https://youtu.be/kcW4ABcY3zI": "Subtitles are disabled for this video."
                },
            },
        )
    ]
    context = (
        "File: youtube_url.pdf (pdf)\n"
        "Additional Video Reference\n"
        "The following YouTube video has been added as an additional learning resource.\n"
        "Video URL: https://youtu.be/kcW4ABcY3zI?si=jrP4Ogp0p699hphk\n"
        "Suggested description: This video serves as a supplementary educational resource."
    )

    answer = orchestrator._final_answer("youtube_transcript", outputs, "Summarize it", context)

    assert answer.strip() == "transcript summary"
    assert "available pdf text" in groq.last_system.lower()
