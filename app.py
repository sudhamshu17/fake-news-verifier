"""
app.py

Streamlit demo UI for the retrieval-based fake news verifier.
Run with: streamlit run app.py
"""

import os
import streamlit as st
from dotenv import load_dotenv
from verifier.verdict import get_verdict

load_dotenv()  # loads NEWSAPI_KEY from a local .env file if present

st.set_page_config(page_title="Fake News Verifier", page_icon="🔍")

st.title("🔍 Real-Time Fake News Verifier")
st.caption(
    "Checks a claim or headline against trusted news sources (Reuters, AP, "
    "BBC, The Hindu, Times of India) using retrieval + text similarity — "
    "not a static pattern-matching classifier."
)

api_key = os.environ.get("NEWSAPI_KEY")
if not api_key:
    st.warning(
        "No NEWSAPI_KEY found in environment. Set it before running, e.g.:\n\n"
        "`export NEWSAPI_KEY=your_key_here` (Mac/Linux) or "
        "`set NEWSAPI_KEY=your_key_here` (Windows)."
    )

claim = st.text_input(
    "Enter a headline or claim to verify:",
    placeholder="e.g. NEET exam paper leak causes nationwide protests",
)

if st.button("Verify", type="primary") and claim:
    with st.spinner("Checking trusted sources..."):
        result = get_verdict(claim, api_key=api_key)

    verdict = result["verdict"]
    color = {
        "Real": "green",
        "Unverified": "orange",
        "Likely Fake": "red",
        "Error": "gray",
    }.get(verdict, "gray")

    st.markdown(f"### Verdict: :{color}[{verdict}]")
    st.write(f"**Confidence score:** {result['confidence']}")
    st.write(result["message"])

    if result["top_matches"]:
        st.subheader("Sources checked:")
        for match in result["top_matches"]:
            st.markdown(
                f"- **[{match['title']}]({match['url']})** "
                f"— *{match['source']}* (similarity: {match['score']})"
            )

    st.divider()
    st.caption(
        "⚠️ This tool checks whether trusted sources corroborate a claim's "
        "wording. It does not independently verify facts, and may miss very "
        "recent stories that trusted sources haven't published yet."
    )
