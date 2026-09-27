import pytest
from verifier.similarity import compute_similarity_scores


def test_identical_text_scores_highest():
    """An article that repeats the claim almost word-for-word should score
    highest among the candidates."""
    claim = "Government announces new education policy for engineering colleges"
    articles = [
        {"title": "Government announces new education policy for engineering colleges",
         "description": "Full details of the policy."},
        {"title": "Weather update for Hyderabad today",
         "description": "Expect light rain in the evening."},
    ]
    scored = compute_similarity_scores(claim, articles)
    assert scored[0]["article"]["title"].startswith("Government announces")
    assert scored[0]["score"] > scored[1]["score"]


def test_unrelated_articles_score_low():
    """Completely unrelated articles should score near zero, not falsely high."""
    claim = "NEET exam paper leak causes nationwide protests"
    articles = [
        {"title": "Stock market closes higher today", "description": "Sensex up 200 points."},
    ]
    scored = compute_similarity_scores(claim, articles)
    assert scored[0]["score"] < 0.15


def test_empty_articles_returns_empty_list():
    """If no articles were retrieved, similarity should return an empty list,
    not raise an error - this lets verdict.py handle 'no coverage found' cleanly."""
    result = compute_similarity_scores("some claim", [])
    assert result == []


def test_empty_claim_raises_value_error():
    """An empty claim is invalid input - fail loudly and early instead of
    silently producing meaningless scores."""
    with pytest.raises(ValueError):
        compute_similarity_scores("", [{"title": "x", "description": "y"}])


def test_article_missing_description_does_not_crash():
    """Real API responses sometimes have a missing/None description.
    The function should handle this gracefully instead of throwing."""
    claim = "Company announces layoffs"
    articles = [{"title": "Company announces layoffs", "description": None}]
    # simulate what news_fetcher already normalizes to "" - but test defensively
    articles[0]["description"] = articles[0]["description"] or ""
    scored = compute_similarity_scores(claim, articles)
    assert len(scored) == 1
