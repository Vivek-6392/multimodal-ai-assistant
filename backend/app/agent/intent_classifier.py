from typing import Any

from app.providers.groq_client import GroqClient


VALID_INTENTS = {"summarize", "sentiment", "code_analysis", "youtube_transcript", "compare_content", "qa"}


class IntentClassifier:
    def __init__(self, groq_client: GroqClient):
        self.groq_client = groq_client

    def classify(
        self,
        message: str,
        context_preview: str,
        urls: list[str],
        has_audio: bool = False,
        has_pdf: bool = False,
    ) -> dict[str, Any]:
        explicit_compare = self._explicit_compare_request(message)
        if explicit_compare and has_audio and has_pdf:
            return {
                "intent": "compare_content",
                "summary_style": "none",
                "needs_clarification": False,
                "clarification_question": "",
                "rationale": "Explicit comparison request for audio and PDF inputs.",
            }

        unclear = self._clarification_if_unclear(message, context_preview)
        if unclear:
            return unclear

        system = (
            "You classify user intent for an autonomous multimodal AI agent. "
            "Return strict JSON with keys: intent, summary_style, needs_clarification, clarification_question, rationale. "
            "intent must be one of summarize, sentiment, code_analysis, youtube_transcript, compare_content, qa. "
            "summary_style must be one_line, three_bullets, five_sentences, or none."
        )
        user = f"""
User request:
{message or "(no explicit request)"}

Detected URLs:
{urls}

Extracted content preview:
{context_preview[:3500]}
"""
        data = self.groq_client.chat_json(system, user)
        intent = str(data.get("intent") or "").strip()
        if intent not in VALID_INTENTS:
            return self._heuristic(message, urls)
        data["intent"] = intent
        data["summary_style"] = data.get("summary_style") or "none"
        data["needs_clarification"] = bool(data.get("needs_clarification", False))
        data["clarification_question"] = data.get("clarification_question") or ""
        return data

    @staticmethod
    def _explicit_compare_request(message: str) -> bool:
        lowered = (message or "").lower()
        return any(
            phrase in lowered
            for phrase in (
                "same topic",
                "compare",
                "comparison",
                "both",
                "difference",
                "different",
                "match",
                "similar",
            )
        )

    def _heuristic(self, message: str, urls: list[str]) -> dict[str, Any]:
        lowered = (message or "").lower()
        intent = "qa"
        if "sentiment" in lowered:
            intent = "sentiment"
        elif any(word in lowered for word in ("bug", "complexity", "explain code", "code")):
            intent = "code_analysis"
        elif any(word in lowered for word in ("same topic", "compare", "comparison", "both", "difference", "different")):
            intent = "compare_content"
        elif any(word in lowered for word in ("summarize", "summary", "tl;dr", "tldr")):
            intent = "summarize"
        elif "youtube" in lowered or "transcript" in lowered or any("youtu" in url for url in urls):
            intent = "youtube_transcript"

        style = "none"
        if "one line" in lowered or "1-line" in lowered:
            style = "one_line"
        elif "3 bullet" in lowered or "three bullet" in lowered:
            style = "three_bullets"
        elif "5 sentence" in lowered or "five sentence" in lowered:
            style = "five_sentences"

        return {
            "intent": intent,
            "summary_style": style,
            "needs_clarification": False,
            "clarification_question": "",
            "rationale": "Heuristic fallback classification.",
        }

    @staticmethod
    def _clarification_if_unclear(message: str, context_preview: str) -> dict[str, Any] | None:
        cleaned = (message or "").strip()
        has_context = bool(context_preview.strip())
        ambiguous_prompts = {
                "analyze",
                "process",
                "review",
                "check",
                "see attached",
                "look at this",
                "do this",
                "explain",
            }

        # If there's no explicit message, we need clarification.
        if not cleaned:
            if has_context:
                question = (
                    "What would you like me to do with the uploaded content?\n"
                    "1. Summarize\n"
                    "2. Sentiment Analysis\n"
                    "3. Code Explanation\n"
                    "4. Question Answering"
                )
            else:
                question = "Could you share a question or upload content for me to analyze?"
            return {
                "intent": "qa",
                "summary_style": "none",
                "needs_clarification": True,
                "clarification_question": question,
                "rationale": "No explicit user goal was provided.",
            }

        # If user provided a short/ambiguous instruction and we already have uploaded content,
        # ask a follow-up. Examples: "summarize", "explain", "analyze".
        cleaned_l = cleaned.lower()
        if has_context and (cleaned_l in ambiguous_prompts or len(cleaned.split()) <= 2):
            question = "What would you like me to do with the uploaded content?"
            return {
                "intent": "qa",
                "summary_style": "none",
                "needs_clarification": True,
                "clarification_question": question,
                "rationale": "User instruction was too short or ambiguous given uploaded content.",
            }

        # Otherwise assume the user's message is actionable and does not need clarification.
        return None
