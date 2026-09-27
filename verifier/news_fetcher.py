"""
news_fetcher.py

Fetches recent articles related to a claim/headline from NewsAPI.org,
restricted to a curated list of trusted sources.
"""

import os
import re
import requests
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Curated list of trusted source IDs (as recognized by NewsAPI).
# You can expand this list - these are generally considered reliable,
# editorially-reviewed outlets.
TRUSTED_SOURCES = [
    "reuters",
    "associated-press",
    "the-hindu",
    "bbc-news",
    "the-times-of-india",
]

NEWSAPI_URL = "https://newsapi.org/v2/everything"

# Max number of keywords to keep when shortening a long headline into a
# search query. NewsAPI's search behaves more like exact-phrase matching
# than semantic search, so long full sentences return zero results even
# when highly relevant articles exist. Empirically, ~6 keywords works well.
MAX_QUERY_KEYWORDS = 6


class NewsFetchError(Exception):
    """Raised when the news API call fails (bad key, rate limit, network error)."""
    pass


def build_search_query(claim: str, max_keywords: int = MAX_QUERY_KEYWORDS) -> str:
    """
    Convert a long claim/headline into a short keyword-style search query.

    NewsAPI's /everything search performs poorly on long, highly specific
    sentences (it behaves closer to phrase matching than semantic search),
    so a full headline as-is often returns zero results even when relevant
    coverage exists. We strip punctuation and common stopwords, then keep
    the first `max_keywords` remaining words, preserving their original
    order (order matters less for search, but this keeps behavior
    predictable and easy to reason about/test).

    Note: this shortened query is used ONLY for the API search step. The
    full original claim is still used later for similarity scoring, so we
    don't lose precision there - we only make retrieval more forgiving.
    """
    if not claim or not claim.strip():
        raise ValueError("claim must be a non-empty string")

    # Strip punctuation (keep letters, numbers, spaces, hyphens)
    cleaned = re.sub(r"[^\w\s-]", " ", claim)
    words = cleaned.split()

    keywords = [
        w for w in words
        if w.lower() not in ENGLISH_STOP_WORDS and len(w) > 1
    ]

    if not keywords:
        # Fallback: if everything got filtered out (e.g. claim was only
        # stopwords), just use the original words so we still search
        # with something rather than an empty string.
        keywords = words

    return " ".join(keywords[:max_keywords])


def fetch_articles(query: str, api_key: str = None, page_size: int = 5) -> list:
    """
    Fetch articles matching `query` from trusted sources only.

    Args:
        query: the headline or claim text to search for
        api_key: NewsAPI key. Falls back to NEWSAPI_KEY env var if not given.
        page_size: max number of articles to retrieve

    Returns:
        List of dicts: [{"title": ..., "description": ..., "url": ..., "source": ...}, ...]
        Returns an empty list if nothing is found (NOT an error - just no matches).

    Raises:
        NewsFetchError: if the request itself fails (bad key, network issue, rate limit)
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")

    key = api_key or os.environ.get("NEWSAPI_KEY")
    if not key:
        raise NewsFetchError(
            "No API key provided. Set the NEWSAPI_KEY environment variable "
            "or pass api_key explicitly."
        )

    # Shorten long headlines/claims into an effective keyword search query
    # (see build_search_query docstring for why this is necessary).
    search_query = build_search_query(query)

    params = {
        "q": search_query,
        "sources": ",".join(TRUSTED_SOURCES),
        "sortBy": "relevancy",
        "pageSize": page_size,
        "apiKey": key,
    }

    try:
        response = requests.get(NEWSAPI_URL, params=params, timeout=10)
    except requests.exceptions.RequestException as e:
        raise NewsFetchError(f"Network error while contacting NewsAPI: {e}")

    if response.status_code == 401:
        raise NewsFetchError("Invalid NewsAPI key.")
    if response.status_code == 429:
        raise NewsFetchError("NewsAPI rate limit exceeded. Try again later.")
    if response.status_code != 200:
        raise NewsFetchError(f"NewsAPI returned status {response.status_code}: {response.text}")

    data = response.json()
    articles = data.get("articles", [])

    results = []
    for a in articles:
        results.append({
            "title": a.get("title") or "",
            "description": a.get("description") or "",
            "url": a.get("url") or "",
            "source": (a.get("source") or {}).get("name", "Unknown"),
        })
    return results