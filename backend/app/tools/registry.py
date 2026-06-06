from dataclasses import dataclass, field
from typing import Any, Callable

from app.providers.groq_client import GroqClient
from app.services.url_service import is_youtube_url


@dataclass
class ToolResult:
    answer: str
    output_summary: str
    metadata: dict[str, Any] = field(default_factory=dict)


class ToolRegistry:
    def __init__(self, groq_client: GroqClient):
        self.groq_client = groq_client
        self._tools: dict[str, Callable[..., ToolResult]] = {
            "summarize": self.summarize,
            "sentiment": self.sentiment,
            "code_analysis": self.code_analysis,
            "youtube_transcript": self.youtube_transcript,
            "qa": self.qa,
            "compare_content": self.compare_content,
        }

    def run(self, tool_name: str, **kwargs: Any) -> ToolResult:
        if tool_name not in self._tools:
            raise KeyError(f"Unknown tool: {tool_name}")
        return self._tools[tool_name](**kwargs)

    def summarize(self, message: str, context: str, **_) -> ToolResult:

            system = """
        Return EXACTLY:

        1-Line Summary:
        <one sentence>

        3 Bullets:
        - bullet 1
        - bullet 2
        - bullet 3

        5-Sentence Summary:
        1. ...
        2. ...
        3. ...
        4. ...
        5. ...

        If duration exists, append:

        Duration: <value>
        """

            answer = self.groq_client.chat(
                system,
                f"Summarize:\n\n{context[:18000]}"
            )

            return ToolResult(
                answer=answer,
                output_summary="Generated 1-line, 3-bullet, and 5-sentence summaries."
            )
    def sentiment(self, message: str, context: str, **_) -> ToolResult:

        system = """
    Return EXACTLY:

    Sentiment: <Positive|Negative|Neutral>

    Confidence: <0-1>

    Justification:
    <one sentence>
    """

        answer = self.groq_client.chat(
            system,
            context[:15000]
        )

        return ToolResult(
            answer=answer,
            output_summary="Sentiment analysis completed."
        )

    def code_analysis(self, message: str, context: str, **_) -> ToolResult:

        system = """
    Return EXACTLY:

    Language:
    <language>

    Explanation:
    ...

    Potential Bugs:
    - ...
    - ...

    Time Complexity:
    ...

    Space Complexity:
    ...
    """

        answer = self.groq_client.chat(
            system,
            context[:18000]
        )

        return ToolResult(
            answer=answer,
            output_summary="Code analysis completed."
        )

    def youtube_transcript(self, urls: list[str], **_: Any) -> ToolResult:
        youtube_urls = [url for url in urls if is_youtube_url(url)]
        if not youtube_urls:
            return ToolResult(answer="", output_summary="No YouTube URL found.", metadata={"transcripts": {}})

        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            from youtube_transcript_api.formatters import TextFormatter
        except ImportError:
            return ToolResult(
                answer="",
                output_summary="youtube-transcript-api is not installed.",
                metadata={"transcripts": {}, "warning": "Install youtube-transcript-api."},
            )

        formatter = TextFormatter()
        transcripts: dict[str, str] = {}
        failures: dict[str, str] = {}
        for url in youtube_urls:
            video_id = self._youtube_id(url)
            if not video_id:
                failures[url] = "Could not parse video id."
                continue
            try:
                transcript = YouTubeTranscriptApi.get_transcript(video_id)
                transcripts[url] = formatter.format_transcript(transcript)
            except Exception as exc:
                failures[url] = str(exc)

        joined = "\n\n".join(f"Transcript for {url}:\n{text}" for url, text in transcripts.items())
        summary = f"Fetched {len(transcripts)} YouTube transcript(s)."
        if failures:
            summary += f" {len(failures)} URL(s) failed."
        return ToolResult(answer=joined, output_summary=summary, metadata={"transcripts": transcripts, "failures": failures})

    def qa(self, message: str, context: str, **_: Any) -> ToolResult:
        system = (
            "You are an autonomous multimodal assistant. Answer the user using the extracted content first. "
            "If the content is insufficient, say what is missing and then answer from general knowledge if appropriate."
        )
        user = f"User request:\n{message or 'Answer based on the provided content.'}\n\nExtracted content:\n{context[:18000]}"
        answer = self.groq_client.chat(system, user)
        return ToolResult(answer=answer, output_summary="Answered using cross-input reasoning.")

    def compare_content(self, message: str, context: str, **_) -> ToolResult:

        system = """
    Compare two sources.

    Return EXACTLY:

    Topic Match:
    <Yes/No>

    Similarity Score:
    <0-1>

    Evidence:
    - ...
    - ...

    Conclusion:
    ...
    """

        answer = self.groq_client.chat(
            system,
            context[:18000]
        )

        return ToolResult(
            answer=answer,
            output_summary="Cross-input comparison completed."
        )

    @staticmethod
    def _youtube_id(url: str) -> str | None:
        import re

        patterns = [
            r"youtu\.be/([A-Za-z0-9_-]{6,})",
            r"[?&]v=([A-Za-z0-9_-]{6,})",
            r"youtube\.com/shorts/([A-Za-z0-9_-]{6,})",
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
