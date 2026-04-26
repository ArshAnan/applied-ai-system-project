"""
Streamlit UI for VibeFinder.

Run with:
    streamlit run src/app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st

from src.recommender import load_songs, recommend_songs
from src.profile_extractor import extract_profile
from src.critic import critique_recommendations
from src.logger import log_session

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="VibeFinder",
    page_icon="🎵",
    layout="wide",
)

st.title("🎵 VibeFinder")
st.caption("Describe what you're in the mood for and get personalized song picks.")

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

DATA_PATH = Path(__file__).parent.parent / "data" / "songs.csv"


@st.cache_resource
def get_songs():
    return load_songs(str(DATA_PATH))


songs = get_songs()

if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------------------------------------------------------
# Sidebar — song catalog
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("🎧 Song Catalog")
    st.caption(f"{len(songs)} songs available")

    catalog_df = pd.DataFrame([
        {
            "Title": s["title"],
            "Artist": s["artist"],
            "Genre": s["genre"],
            "Mood": s["mood"],
            "Energy": s["energy"],
        }
        for s in songs
    ])

    genre_options = ["All"] + sorted(catalog_df["Genre"].unique().tolist())
    selected_genre = st.selectbox("Filter by genre", genre_options)

    if selected_genre != "All":
        catalog_df = catalog_df[catalog_df["Genre"] == selected_genre]

    st.dataframe(catalog_df, hide_index=True, use_container_width=True)

# ---------------------------------------------------------------------------
# Result renderer
# ---------------------------------------------------------------------------

def render_result(entry: dict) -> None:
    prefs = entry["prefs"]

    # Extracted preferences as metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Genre", prefs.get("genre", "—"))
    col2.metric("Mood", prefs.get("mood", "—"))
    col3.metric("Energy", f"{prefs.get('energy', 0):.0%}")
    col4.metric("Acoustic", "Yes" if prefs.get("likes_acoustic") else "No")

    # Recommendations table
    rows = [
        {
            "Rank": i + 1,
            "Title": song["title"],
            "Artist": song["artist"],
            "Genre": song["genre"],
            "Mood": song["mood"],
            "Score": round(score, 2),
        }
        for i, (song, score, _) in enumerate(entry["recommendations"])
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # Critic verdict + confidence
    critique   = entry["critique"]
    verdict    = critique.get("verdict", "unknown")
    confidence = critique.get("confidence")
    explanation = critique.get("explanation", "")
    flags = critique.get("flags", [])

    conf_str = f"  Confidence: {confidence:.0%}" if confidence is not None else ""

    if verdict == "good":
        st.success(f"**AI Verdict: GOOD**{conf_str} — {explanation}")
    elif verdict == "mixed":
        st.warning(f"**AI Verdict: MIXED**{conf_str} — {explanation}")
    elif verdict == "poor":
        st.error(f"**AI Verdict: POOR**{conf_str} — {explanation}")
    else:
        st.info(f"**AI Verdict: {verdict.upper()}**{conf_str} — {explanation}")

    if flags:
        with st.expander(f"See {len(flags)} flag(s)"):
            for flag in flags:
                st.markdown(f"• {flag}")

# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------

for entry in st.session_state.history:
    with st.chat_message("user"):
        st.write(entry["input"])
    with st.chat_message("assistant"):
        render_result(entry)

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

user_input = st.chat_input("What are you in the mood for?")

if user_input and user_input.strip():
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Parsing your request..."):
            try:
                user_prefs = extract_profile(user_input)
            except (ValueError, KeyError) as exc:
                st.error(f"Could not parse your request: {exc}")
                st.stop()

        recommendations = recommend_songs(user_prefs, songs, k=5)

        with st.spinner("Reviewing recommendations..."):
            try:
                critique = critique_recommendations(user_input, user_prefs, recommendations)
            except (ValueError, KeyError) as exc:
                st.warning(f"Critic unavailable: {exc}")
                critique = {"verdict": "unknown", "flags": [], "explanation": ""}

        log_session(user_input, user_prefs, recommendations, critique)

        entry = {
            "input": user_input,
            "prefs": user_prefs,
            "recommendations": recommendations,
            "critique": critique,
        }
        st.session_state.history.append(entry)
        render_result(entry)
