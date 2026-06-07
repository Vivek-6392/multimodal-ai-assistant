import os
import tempfile
import subprocess
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
            "audio_transcription": self.audio_transcription,
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

    def _fetch_transcript_with_ytdlp(self, video_url: str) -> str:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:

                cmd = [
                    "python", "-m", "yt_dlp",  # Use yt_dlp as a module to ensure the correct version is used
                    "--skip-download",
                    "--write-auto-subs",
                    "--write-subs",
                    "--sub-langs",
                    "en.*",
                    "-o",
                    os.path.join(temp_dir, "%(id)s"),
                    video_url,
                ]

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                )

                if result.returncode != 0:
                    print("yt-dlp error:")
                    print(result.stderr)
                    return ""

                subtitle_files = [
                    f for f in os.listdir(temp_dir)
                    if f.endswith(".vtt")
                ]

                if not subtitle_files:
                    print("No subtitle files found")
                    return ""

                subtitle_path = os.path.join(
                    temp_dir,
                    subtitle_files[0]
                )

                transcript_lines = []

                with open(
                    subtitle_path,
                    "r",
                    encoding="utf-8",
                    errors="ignore",
                ) as f:

                    for line in f:
                        line = line.strip()

                        if (
                            not line
                            or line.startswith("WEBVTT")
                            or "-->" in line
                        ):
                            continue

                        transcript_lines.append(line)

                return "\n".join(transcript_lines)

        except Exception as e:
            print("yt-dlp fallback failed:")
            print(e)
            return ""

    def _fetch_transcript_supadata(self, video_id: str) -> str:
        """Cloud-friendly transcript API. Set SUPADATA_API_KEY in HF Secrets."""
        api_key = os.environ.get("SUPADATA_API_KEY", "")
        if not api_key:
            print("Supadata: SUPADATA_API_KEY not set, skipping")
            return ""

        try:
            import requests

            video_url = f"https://www.youtube.com/watch?v={video_id}"
            print(f"Supadata: requesting transcript for {video_id}")

            resp = requests.get(
                "https://api.supadata.ai/v1/transcript",
                params={"url": video_url},
                headers={
                    "x-api-key": api_key,
                    "Accept": "application/json",
                    "User-Agent": "Mozilla/5.0 (compatible; DatasmithAssign/1.0)",
                },
                timeout=20,
            )

            print(f"Supadata: status {resp.status_code}")

            if not resp.ok:
                print(f"Supadata error body: {resp.text[:300]}")
                return ""

            data = resp.json()
            content = data.get("content", "")

            if isinstance(content, str):
                text = content.strip()
            else:
                # list of {lang, text} objects
                text = " ".join(seg["text"] for seg in content if seg.get("text"))

            if not text:
                print("Supadata: empty content in response")
                return ""

            print(f"Supadata: fetched {len(text)} chars")
            return text

        except Exception as e:
            print(f"Supadata failed: {e}")
            return ""
    

    def youtube_transcript(self, urls: list[str], **_: Any) -> ToolResult:
        print("=" * 80)
        print("youtube_transcript called")
        print("urls =", urls)
        print("=" * 80)

        youtube_urls = [url for url in urls if is_youtube_url(url)]

        if not youtube_urls:
            return ToolResult(
                answer="",
                output_summary="No YouTube URL found.",
                metadata={
                    "transcripts": {},
                    "failures": {},
                },
            )

        try:
            from youtube_transcript_api import YouTubeTranscriptApi
        except ImportError:
            return ToolResult(
                answer="",
                output_summary="youtube-transcript-api is not installed.",
                metadata={
                    "transcripts": {},
                    "failures": {
                        "import": "Install youtube-transcript-api"
                    },
                },
            )

        transcripts: dict[str, str] = {}
        failures: dict[str, str] = {}

        for url in youtube_urls:

            print("\n" + "=" * 80)
            print("PROCESSING URL:", url)

            video_id = self._youtube_id(url)

            print("VIDEO ID:", video_id)

            if not video_id:
                failures[url] = "Could not parse video id."
                continue

            try:
                from youtube_transcript_api import YouTubeTranscriptApi

                api = YouTubeTranscriptApi()

                transcript_data = api.fetch(video_id)

                transcript_text = "\n".join(
                    snippet.text
                    for snippet in transcript_data
                    if getattr(snippet, "text", "")
                ).strip()

                if not transcript_text:
                    failures[url] = "Transcript empty."
                    continue

                transcripts[url] = transcript_text

                print("SUCCESS")
                print("TRANSCRIPT LENGTH:", len(transcript_text))
                print("PREVIEW:")
                print(transcript_text[:500])

            except Exception as exc:
                print("youtube-transcript-api failed")
                print(type(exc).__name__)
                print(str(exc))

                # Try Supadata first (works on cloud IPs)
                transcript_text = self._fetch_transcript_supadata(video_id)

                # Fall back to yt-dlp (works in local dev)
                if not transcript_text:
                    transcript_text = self._fetch_transcript_with_ytdlp(url)

                if transcript_text:
                    transcripts[url] = transcript_text
                    print("Recovered transcript, length:", len(transcript_text))
                else:
                    failures[url] = f"{type(exc).__name__}: {str(exc)}"

        print("\n" + "=" * 80)
        print("FINAL RESULT")
        print("TRANSCRIPTS:", len(transcripts))
        print("FAILURES:", len(failures))
        print(failures)
        print("=" * 80)

        joined = "\n\n".join(
            text
            for text in transcripts.values()
            if text
        )

        return ToolResult(
            answer=joined,
            output_summary=(
                f"Fetched {len(transcripts)} transcript(s). "
                f"{len(failures)} failure(s)."
            ),
            metadata={
                "transcripts": transcripts,
                "failures": failures,
            },
        )

    def _make_ytt_api(self):
        """Return a YouTubeTranscriptApi instance, proxied if credentials exist."""
        from youtube_transcript_api import YouTubeTranscriptApi

        proxy_user = os.environ.get("WEBSHARE_USER")
        proxy_pass = os.environ.get("WEBSHARE_PASS")

        if proxy_user and proxy_pass:
            try:
                from youtube_transcript_api.proxies import WebshareProxyConfig
                print("YouTube: using Webshare proxy")
                return YouTubeTranscriptApi(
                    proxy_config=WebshareProxyConfig(
                        proxy_username=proxy_user,
                        proxy_password=proxy_pass,
                    )
                )
            except ImportError:
                print("WebshareProxyConfig unavailable — run: pip install youtube-transcript-api --upgrade")

        print("YouTube: no proxy configured, trying direct")
        return YouTubeTranscriptApi()

    def qa(self, message: str, context: str, **_: Any) -> ToolResult:
        system = (
            "You are an autonomous multimodal assistant. Answer the user using the extracted content first. "
            "If the content is insufficient, say what is missing and then answer from general knowledge if appropriate."
        )
        user = f"User request:\n{message or 'Answer based on the provided content.'}\n\nExtracted content:\n{context[:18000]}"
        answer = self.groq_client.chat(system, user)
        return ToolResult(answer=answer, output_summary="Answered using cross-input reasoning.")

    def audio_transcription(self, context: str, **_: Any) -> ToolResult:
        return ToolResult(
            answer=context,
            output_summary="Audio transcription retrieved.",
            metadata={"source": "groq_whisper"}
        )

    @staticmethod
    def _ordered_transcripts(transcript_list: Any) -> list[Any]:
        transcripts = list(transcript_list)
        if not transcripts:
            return []

        english: list[Any] = []
        hindi: list[Any] = []
        fallback: list[Any] = []
        for transcript in transcripts:
            language_code = str(getattr(transcript, "language_code", "")).lower()
            if language_code.startswith("en"):
                english.append(transcript)
            elif language_code == "hi":
                hindi.append(transcript)
            else:
                fallback.append(transcript)

        return english + hindi + fallback

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
        from urllib.parse import urlparse, parse_qs

        try:

            if "youtu.be/" in url:
                return urlparse(url).path.lstrip("/")

            if "youtube.com/watch" in url:
                return parse_qs(
                    urlparse(url).query
                ).get("v", [None])[0]

            if "/shorts/" in url:
                return (
                    url.split("/shorts/")[1]
                    .split("?")[0]
                )

        except Exception:
            pass

        return None