from unittest.mock import patch
from verifier.verdict import get_verdict
from verifier.news_fetcher import NewsFetchError


@patch("verifier.verdict.fetch_articles")
def test_strong_match_gives_real_verdict(mock_fetch):
    """When a trusted source's article closely matches the claim's wording,
    the verdict should be 'Real' with reasonably high confidence."""
    mock_fetch.return_value = [
        {
            "title": "NEET paper leak sparks protests across India",
            "description": "Students demand re-examination after leak allegations.",
            "url": "https://example.com/a1",
            "source": "The Hindu",
        }
    ]
    result = get_verdict("NEET paper leak sparks protests across India")
    assert result["verdict"] == "Real"
    assert result["confidence"] > 0
    assert len(result["top_matches"]) == 1


@patch("verifier.verdict.fetch_articles")
def test_no_articles_found_gives_unverified(mock_fetch):
    """If trusted sources have zero coverage, the correct verdict is
    'Unverified' - NOT 'Fake'. Absence of evidence isn't evidence of
    fabrication; it might just be too recent or too niche."""
    mock_fetch.return_value = []
    result = get_verdict("Some extremely obscure local claim")
    assert result["verdict"] == "Unverified"
    assert result["confidence"] == 0.0


@patch("verifier.verdict.fetch_articles")
def test_weak_similarity_gives_likely_fake(mock_fetch):
    """When articles exist but none of them are actually about the claim
    (low similarity), the system should lean toward 'Likely Fake' rather
    than falsely reporting 'Real' just because *some* article was returned."""
    mock_fetch.return_value = [
        {
            "title": "Completely unrelated topic about cricket scores",
            "description": "India wins the match by 6 wickets.",
            "url": "https://example.com/a2",
            "source": "BBC News",
        }
    ]
    result = get_verdict("Aliens landed in Mumbai yesterday")
    assert result["verdict"] == "Likely Fake"


@patch("verifier.verdict.fetch_articles")
def test_api_failure_returns_error_verdict_not_crash(mock_fetch):
    """If the NewsAPI call fails (bad key, rate limit, network issue), the
    app must degrade gracefully with an 'Error' verdict, never crash the
    whole app. This is critical for a production-quality tool."""
    mock_fetch.side_effect = NewsFetchError("Invalid API key")
    result = get_verdict("Any claim")
    assert result["verdict"] == "Error"
    assert "Invalid API key" in result["message"]


@patch("verifier.verdict.fetch_articles")
def test_top_matches_are_sorted_and_limited_to_three(mock_fetch):
    """Even if many articles are returned, only the top 3 most relevant
    should be surfaced to the user - keeps the UI focused and readable."""
    mock_fetch.return_value = [
        {"title": f"Article about topic {i}", "description": "desc", "url": f"u{i}", "source": "Reuters"}
        for i in range(5)
    ]
    result = get_verdict("Article about topic")
    assert len(result["top_matches"]) <= 3
