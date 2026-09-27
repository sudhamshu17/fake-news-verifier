# Real-Time Fake News Verifier

A retrieval-based claim verification tool that checks headlines/claims against
trusted news sources in real time, instead of relying on a static ML classifier
trained on old data.

## Why this approach, not a trained classifier?

A traditional "fake news detector" trains a model (e.g., TF-IDF + Logistic
Regression) on a labeled dataset to recognize *writing style patterns* —
sensational language, phrasing, structure. That approach has a fundamental
limitation: it has no factual knowledge and cannot judge claims about events
outside its training data — especially fast-moving, real-world stories.

This project instead:
1. Takes a claim/headline as input
2. Searches trusted sources (Reuters, AP, BBC, The Hindu, Times of India) in
   real time via NewsAPI
3. Uses TF-IDF + cosine similarity to check how closely retrieved coverage
   matches the claim
4. Returns a verdict: **Real**, **Likely Fake**, or **Unverified** — along
   with the actual source links, so users can verify themselves rather than
   blindly trusting a label

**Important limitation, by design:** "Unverified" is used (not "Fake") when
no matching coverage is found — because absence of evidence isn't evidence
of fabrication. A claim might simply be too recent for trusted sources to
have published on yet.

## Architecture

```
User input (claim)
        │
        ▼
news_fetcher.py  →  queries NewsAPI, restricted to trusted sources
        │
        ▼
similarity.py    →  TF-IDF + cosine similarity vs. retrieved articles
        │
        ▼
verdict.py       →  decision logic → Real / Likely Fake / Unverified / Error
        │
        ▼
app.py (Streamlit)  →  displays verdict + source links to user
```

## Testing

The project includes a pytest suite (`tests/`) covering:
- Similarity scoring correctness (matching vs. unrelated text)
- Verdict logic across all paths (strong match, weak match, no results, API failure)
- Graceful degradation — the app never crashes, even when NewsAPI is down or rate-limited

External API calls are mocked in tests, so the suite runs fast, deterministically,
and without needing a real API key — a GitHub Actions workflow (`.github/workflows/ci.yml`)
runs this suite automatically on every push.

## Running locally

```bash
# 1. Clone and install dependencies
pip install -r requirements.txt

# 2. Get a free API key from https://newsapi.org/register

# 3. Set your key
cp .env.example .env
# then edit .env and paste your real key

# 4. Run the tests
pytest tests/ -v

# 5. Run the app
streamlit run app.py
```

## Tech stack

Python, scikit-learn (TF-IDF, cosine similarity), NewsAPI, Streamlit, pytest,
GitHub Actions.

## A real bug found during development

Early testing surfaced a genuine issue: searching NewsAPI with a full, long
headline (e.g. *"NEET paper leak case: CBI charge sheet cites evidence from
access control app of NTA expert's society, phone records"*) returned **zero**
articles — even though relevant coverage existed. NewsAPI's search behaves
closer to phrase matching than semantic search, so long, highly specific
sentences rarely match anything.

**Fix:** `news_fetcher.build_search_query()` strips punctuation and common
stopwords from the input claim and keeps the top keywords, using this
shortened version only for the API search step. The full original claim is
still used for similarity scoring, so precision isn't lost — only retrieval
was made more forgiving. This also surfaced a second tuning decision: the
initial similarity threshold for a "Real" verdict (0.35) was too strict for
real-world headline rewordings, and was lowered to 0.20 based on testing
against this real NEET case.

## Known limitations

- **Free NewsAPI tier constraints:** 100 requests/day, and articles are
  delayed by ~24 hours before being indexed — so genuinely breaking news from
  the last day may show "Unverified" simply because it isn't searchable yet,
  not because it's false
- **Source coverage is narrow by design:** limited to 5 trusted outlets
  (Reuters, AP, BBC, The Hindu, Times of India). This favors precision over
  recall — hyper-local stories, regional-language claims, and niche topics
  will often show "Unverified" even when true, simply because none of these
  outlets cover them
- **English-language only:** claims in Hindi, Telugu, or other languages
  won't match well even when the underlying story exists in trusted sources
- **Similarity is text-based, not true semantic understanding** — heavily
  reworded true claims may still score lower than expected