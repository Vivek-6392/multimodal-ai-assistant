from app.tools.registry import ToolRegistry


class FakeGroq:
    def __init__(self):
        self.system = ""
        self.user = ""

    def chat(self, system: str, user: str, *, max_tokens: int = 1400):
        self.system = system
        self.user = user
        return "summary"


def test_summarize_requests_all_required_formats():
    groq = FakeGroq()
    result = ToolRegistry(groq).summarize("summarize", "content")

    assert "1-Line Summary" in groq.system
    assert "3 Bullets" in groq.system
    assert "5-Sentence Summary" in groq.system
    assert result.output_summary == "Generated 1-line, 3-bullet, and 5-sentence summaries."


def test_tool_registry_registers_audio_transcription():
    registry = ToolRegistry(FakeGroq())

    result = registry.run("audio_transcription", context="audio text")

    assert result.answer == "audio text"
    assert result.output_summary == "Audio transcription retrieved."


class FakeTranscript:
    def __init__(self, language_code: str, text: str, translatable: bool = False):
        self.language_code = language_code
        self.text = text
        self.is_translatable = translatable

    def fetch(self):
        return [{"text": self.text, "start": 0.0, "duration": 1.0}]

    def translate(self, language_code: str):
        assert language_code == "en"
        return FakeTranscript("en", f"translated:{self.text}", translatable=False)


def test_ordered_transcripts_prefers_english_tracks():
    transcripts = ToolRegistry._ordered_transcripts([
        FakeTranscript("hi", "hindi", translatable=True),
        FakeTranscript("en", "english"),
    ])

    assert transcripts[0].language_code == "en"


def test_ordered_transcripts_keeps_hindi_track_available():
    transcripts = ToolRegistry._ordered_transcripts([
        FakeTranscript("hi", "hindi", translatable=True),
        FakeTranscript("es", "spanish"),
    ])

    assert transcripts[0].language_code == "hi"
    assert transcripts[1].language_code == "es"
