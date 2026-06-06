from app.services.url_service import is_youtube_url


class ReasoningPlanner:
    def create_plan(
        self,
        intent: str,
        urls: list[str],
        has_audio: bool = False,
        has_pdf: bool = False,
    ) -> list[str]:

        plan = ["extract_inputs", "detect_urls", "classify_intent"]

        youtube_urls = [u for u in urls if is_youtube_url(u)]

        if youtube_urls:
            plan.append("youtube_transcript")

        if has_audio and intent == "summarize":
            plan.append("audio_transcription")
            plan.append("summarize")

        elif has_audio and has_pdf:
            plan.append("compare_content")

        elif intent == "summarize":
            plan.append("summarize")

        elif intent == "sentiment":
            plan.append("sentiment")

        elif intent == "code_analysis":
            plan.append("code_analysis")

        elif intent == "youtube_transcript":
            if youtube_urls:
                plan.append("summarize")
            else:
                plan.append("qa")

        else:
            plan.append("qa")

        plan.append("synthesize_final_answer")
        return plan