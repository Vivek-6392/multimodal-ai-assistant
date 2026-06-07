import os
import re
import json
import tempfile
import subprocess
import pdfplumber
from dotenv import load_dotenv
from groq import Groq
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
)

# =====================================================
# Configuration
# =====================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env")

client = Groq(api_key=GROQ_API_KEY)

# =====================================================
# PDF Processing
# =====================================================

def extract_pdf_text(pdf_path: str) -> str:
    text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    return text


# =====================================================
# URL Extraction
# =====================================================

def extract_youtube_urls(text: str):
    pattern = (
        r'https?://(?:www\.)?'
        r'(?:youtube\.com/watch\?v=[\w\-]+[^\s]*|'
        r'youtu\.be/[\w\-]+[^\s]*)'
    )

    urls = re.findall(pattern, text)

    unique_urls = []
    seen = set()

    for url in urls:
        if url not in seen:
            unique_urls.append(url)
            seen.add(url)

    return unique_urls


# =====================================================
# Video ID Extraction
# =====================================================

def get_video_id(url: str):
    patterns = [
        r"youtu\.be/([^?&]+)",
        r"youtube\.com/watch\?v=([^&]+)"
    ]

    for pattern in patterns:
        match = re.search(pattern, url)

        if match:
            return match.group(1)

    return None


# =====================================================
# Method 1: youtube-transcript-api
# =====================================================

def transcript_api_method(video_id: str):

    try:
        api = YouTubeTranscriptApi()

        transcript_list = api.list(video_id)

        try:
            transcript = transcript_list.find_manually_created_transcript(
                ['en', 'en-US', 'en-GB']
            )
        except Exception:
            transcript = transcript_list.find_generated_transcript(
                ['en', 'en-US', 'en-GB']
            )

        transcript_data = transcript.fetch()

        return " ".join(
            item.text
            for item in transcript_data
        )

    except Exception as e:
        print(f"[youtube-transcript-api] Failed: {e}")
        return ""


# =====================================================
# Method 2: yt-dlp subtitles fallback
# =====================================================

def ytdlp_method(video_url: str):

    try:

        with tempfile.TemporaryDirectory() as temp_dir:

            command = [
                "yt-dlp",
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
                command,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                print("[yt-dlp] Error:", result.stderr)
                return ""

            subtitle_files = [
                f
                for f in os.listdir(temp_dir)
                if f.endswith(".vtt")
            ]

            if not subtitle_files:
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
                errors="ignore"
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

            return " ".join(transcript_lines)

    except Exception as e:
        print(f"[yt-dlp] Failed: {e}")
        return ""


# =====================================================
# Transcript Fetcher
# =====================================================

def fetch_transcript(video_url: str, video_id: str):

    print("Trying youtube-transcript-api...")

    transcript = transcript_api_method(video_id)

    if transcript:
        print("Success via youtube-transcript-api")
        return transcript

    print("Trying yt-dlp fallback...")

    transcript = ytdlp_method(video_url)

    if transcript:
        print("Success via yt-dlp")
        return transcript

    return ""


# =====================================================
# Summarization
# =====================================================

def summarize_text(transcript: str):

    if not transcript.strip():
        return "Transcript unavailable."

    transcript = transcript[:15000]

    prompt = f"""
Summarize this YouTube transcript.

Provide:

1. Executive Summary
2. Key Concepts
3. Important Takeaways
4. Action Items (if any)

Transcript:

{transcript}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content


# =====================================================
# Main Processing
# =====================================================

def process_pdf(pdf_path: str):

    print(f"Reading PDF: {pdf_path}")

    pdf_text = extract_pdf_text(pdf_path)

    urls = extract_youtube_urls(pdf_text)

    if not urls:
        print("No YouTube URLs found.")
        return

    print(f"\nFound {len(urls)} URL(s)\n")

    all_results = []

    for index, url in enumerate(urls, start=1):

        print("=" * 60)
        print(f"Processing Video {index}")
        print(url)
        print("=" * 60)

        video_id = get_video_id(url)

        if not video_id:
            print("Invalid YouTube URL")
            continue

        transcript = fetch_transcript(
            url,
            video_id
        )

        if transcript:
            print(
                f"Transcript Length: "
                f"{len(transcript)} chars"
            )

            summary = summarize_text(transcript)

        else:
            summary = (
                "Transcript unavailable "
                "(API and yt-dlp failed)"
            )

        result = f"""
==================================================
VIDEO {index}
==================================================

URL:
{url}

SUMMARY:

{summary}

"""

        all_results.append(result)

    output_file = "youtube_summaries.txt"

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write("\n".join(all_results))

    print("\nDone!")
    print(f"Saved to: {output_file}")


# =====================================================
# Entry Point
# =====================================================

if __name__ == "__main__":

    pdf_path = "video_link_addition.pdf"

    process_pdf(pdf_path)