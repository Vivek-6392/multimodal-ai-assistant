import re


URL_PATTERN = re.compile(r"https?://[^\s<>()\"']+", flags=re.IGNORECASE)


def detect_urls(text: str) -> list[str]:
    urls = [url.rstrip(".,;:)]}") for url in URL_PATTERN.findall(text or "")]
    return sorted(set(urls))


def is_youtube_url(url: str) -> bool:
    lowered = url.lower()
    return "youtube.com/watch" in lowered or "youtu.be/" in lowered or "youtube.com/shorts/" in lowered
