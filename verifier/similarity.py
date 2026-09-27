"""
similarity.py

Uses TF-IDF + cosine similarity to measure how closely a claim/headline
matches articles retrieved from trusted sources.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def compute_similarity_scores(claim: str, articles: list) -> list:
    """
    Compare `claim` against each article's title+description using TF-IDF
    cosine similarity.

    Args:
        claim: the input headline/claim text
        articles: list of dicts with "title" and "description" keys
                  (as returned by news_fetcher.fetch_articles)

    Returns:
        List of dicts: [{"article": <article dict>, "score": float}, ...]
        sorted by score descending. Empty list if `articles` is empty.
    """
    if not claim or not claim.strip():
        raise ValueError("claim must be a non-empty string")

    if not articles:
        return []

    # Build the corpus: claim first, then each article's combined text
    article_texts = [
        f"{a.get('title', '')} {a.get('description', '')}".strip()
        for a in articles
    ]
    corpus = [claim] + article_texts

    # Some articles might have empty text (missing description) - TF-IDF
    # handles this fine, it'll just get a near-zero score.
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus)

    claim_vector = tfidf_matrix[0:1]
    article_vectors = tfidf_matrix[1:]

    scores = cosine_similarity(claim_vector, article_vectors)[0]

    scored = [
        {"article": articles[i], "score": float(scores[i])}
        for i in range(len(articles))
    ]
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored
