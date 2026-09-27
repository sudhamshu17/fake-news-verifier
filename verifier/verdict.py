"""
verdict.py

Combines news_fetcher + similarity to produce a final verdict on a claim.

IMPORTANT DESIGN NOTE (say this in interviews - it shows judgment):
This system does NOT claim to know objective truth. It reports whether
trusted sources appear to corroborate a claim, based on text similarity.
"Unverified" is used instead of "Fake" when there's simply no matching
coverage yet (e.g. very recent news) - conflating "not found" with "false"
is a common and dangerous mistake in naive fake-news tools.
"""

from verifier.news_fetcher import fetch_articles, NewsFetchError
from verifier.similarity import compute_similarity_scores

# Thresholds - tunable. Chosen conservatively to avoid false confidence.
STRONG_MATCH_THRESHOLD = 0.20   # confidently "Real"
WEAK_MATCH_THRESHOLD = 0.08     # some relevance, but not confident

def get_verdict(claim: str, api_key: str = None) -> dict:
    """
    Determine a verdict for a claim by checking trusted sources.

    Returns a dict:
        {
            "verdict": "Real" | "Likely Fake" | "Unverified" | "Error",
            "confidence": float,           # top similarity score (0 if none)
            "top_matches": [ {title, url, source, score}, ... ],
            "message": str                 # human-readable explanation
        }
    """
    try:
        articles = fetch_articles(claim, api_key=api_key)
    except NewsFetchError as e:
        return {
            "verdict": "Error",
            "confidence": 0.0,
            "top_matches": [],
            "message": f"Could not check sources: {e}",
        }

    if not articles:
        return {
            "verdict": "Unverified",
            "confidence": 0.0,
            "top_matches": [],
            "message": (
                "No coverage found from trusted sources. This could mean the "
                "claim is false, too recent to be reported yet, or too niche "
                "for these sources. Treat with caution."
            ),
        }

    scored = compute_similarity_scores(claim, articles)
    top_score = scored[0]["score"] if scored else 0.0

    top_matches = [
        {
            "title": s["article"]["title"],
            "url": s["article"]["url"],
            "source": s["article"]["source"],
            "score": round(s["score"], 3),
        }
        for s in scored[:3]
    ]

    if top_score >= STRONG_MATCH_THRESHOLD:
        verdict = "Real"
        message = "Trusted sources published closely matching coverage of this claim."
    elif top_score >= WEAK_MATCH_THRESHOLD:
        verdict = "Unverified"
        message = (
            "Related coverage was found, but it doesn't closely match the claim's "
            "specific wording. Review the sources below yourself."
        )
    else:
        verdict = "Likely Fake"
        message = (
            "No trusted source closely matches this claim, despite related "
            "articles existing. Treat this claim with skepticism."
        )

    return {
        "verdict": verdict,
        "confidence": round(top_score, 3),
        "top_matches": top_matches,
        "message": message,
    }
