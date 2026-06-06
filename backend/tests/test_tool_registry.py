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
