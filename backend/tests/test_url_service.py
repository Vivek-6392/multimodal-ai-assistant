from app.services.url_service import detect_urls, is_youtube_url


def test_detect_urls_deduplicates_and_trims_punctuation():
    text = "See https://example.com/a, and https://youtu.be/abc123xyz."

    assert detect_urls(text) == ["https://example.com/a", "https://youtu.be/abc123xyz"]


def test_is_youtube_url():
    assert is_youtube_url("https://www.youtube.com/watch?v=abc123xyz")
    assert is_youtube_url("https://youtu.be/abc123xyz")
    assert not is_youtube_url("https://example.com/watch?v=abc123xyz")
