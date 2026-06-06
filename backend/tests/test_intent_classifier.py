from app.agent.intent_classifier import IntentClassifier


class FakeGroq:
    def chat_json(self, system: str, user: str):
        return {
            "intent": "summarize",
            "summary_style": "none",
            "needs_clarification": False,
            "clarification_question": "",
            "rationale": "fake",
        }


def test_classifier_asks_for_clarification_without_goal():
    result = IntentClassifier(FakeGroq()).classify("", "File: note.txt\nSome extracted text.", [])

    assert result["needs_clarification"] is True
    assert "What would you like me to do" in result["clarification_question"]


def test_heuristic_prefers_summary_when_youtube_summary_requested():
    result = IntentClassifier(FakeGroq())._heuristic(
        "Hit the YouTube URL and give me a summary",
        ["https://youtu.be/abc123xyz"],
    )

    assert result["intent"] == "summarize"


def test_classifier_short_circuits_explicit_audio_pdf_comparison():
    classifier = IntentClassifier(FakeGroq())

    result = classifier.classify(
        "Do the audio and the pdf document discuss the same topic?",
        "File: audio.mp3 (audio)\nTranscript text\n\nFile: report.pdf (pdf)\nPDF text",
        [],
        has_audio=True,
        has_pdf=True,
    )

    assert result["intent"] == "compare_content"
    assert result["needs_clarification"] is False
