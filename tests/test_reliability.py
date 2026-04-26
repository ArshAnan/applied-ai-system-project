"""
Reliability test suite for VibeFinder.

These tests do NOT call the Gemini API — they verify the scoring engine
stays consistent and handles edge/adversarial cases correctly.
"""

import pytest
from src.recommender import load_songs, recommend_songs, score_song


@pytest.fixture(scope="module")
def songs():
    return load_songs("data/songs.csv")


# ---------------------------------------------------------------------------
# Consistency tests
# ---------------------------------------------------------------------------

def test_same_profile_same_results(songs):
    """Identical inputs must produce identical outputs every time."""
    prefs = {"genre": "lofi", "mood": "chill", "energy": 0.4, "valence": 0.6, "likes_acoustic": True}
    first  = recommend_songs(prefs, songs, k=5)
    second = recommend_songs(prefs, songs, k=5)
    assert [s["title"] for s, _, _ in first] == [s["title"] for s, _, _ in second]


def test_results_are_sorted_descending(songs):
    """Scores in the returned list must be non-increasing."""
    prefs = {"genre": "pop", "mood": "happy", "energy": 0.85, "valence": 0.82, "likes_acoustic": False}
    results = recommend_songs(prefs, songs, k=5)
    scores = [score for _, score, _ in results]
    assert scores == sorted(scores, reverse=True)


def test_k_limits_results(songs):
    """recommend_songs must return exactly k results when catalog is larger than k."""
    prefs = {"genre": "rock", "mood": "intense", "energy": 0.9, "valence": 0.45, "likes_acoustic": False}
    for k in (1, 3, 5):
        results = recommend_songs(prefs, songs, k=k)
        assert len(results) == k


def test_scores_are_non_negative(songs):
    """Every possible score must be >= 0."""
    prefs = {"genre": "lofi", "mood": "chill", "energy": 0.4, "valence": 0.6, "likes_acoustic": True}
    for song in songs:
        score, _ = score_song(prefs, song)
        assert score >= 0.0


# ---------------------------------------------------------------------------
# Adversarial tests
# ---------------------------------------------------------------------------

def test_unknown_genre_does_not_crash(songs):
    """A genre absent from the catalog should degrade gracefully, not raise."""
    prefs = {"genre": "metal", "mood": "intense", "energy": 0.95, "valence": 0.25, "likes_acoustic": False}
    results = recommend_songs(prefs, songs, k=5)
    assert len(results) == 5
    # No song in the catalog has genre "metal" — none of the top-5 should either
    assert all(song["genre"] != "metal" for song, _, _ in results)


def test_conflicting_prefs_still_returns_results(songs):
    """High energy + sad folk — a contradictory profile — must still return 5 songs."""
    prefs = {"genre": "folk", "mood": "sad", "energy": 0.92, "valence": 0.30, "likes_acoustic": True}
    results = recommend_songs(prefs, songs, k=5)
    assert len(results) == 5


def test_extreme_energy_values(songs):
    """Energy at boundary values 0.0 and 1.0 must not crash or produce NaN scores."""
    for energy in (0.0, 1.0):
        prefs = {"genre": "pop", "mood": "happy", "energy": energy, "valence": 0.5, "likes_acoustic": False}
        results = recommend_songs(prefs, songs, k=5)
        for _, score, _ in results:
            assert score == score  # NaN check: NaN != NaN


# ---------------------------------------------------------------------------
# Score formula unit tests
# ---------------------------------------------------------------------------

def test_genre_match_adds_two_points():
    song  = {"genre": "pop", "mood": "sad", "energy": 0.5, "valence": 0.5, "acousticness": 0.1}
    prefs = {"genre": "pop", "mood": "happy", "energy": 0.5, "valence": 0.5, "likes_acoustic": False}
    score_match, _    = score_song(prefs, song)
    song_no_match     = {**song, "genre": "rock"}
    score_no_match, _ = score_song(prefs, song_no_match)
    assert abs((score_match - score_no_match) - 2.0) < 1e-9


def test_acoustic_bonus_requires_high_acousticness():
    prefs = {"genre": "folk", "mood": "chill", "energy": 0.4, "valence": 0.5, "likes_acoustic": True}
    high_acoustic = {"genre": "x", "mood": "x", "energy": 0.4, "valence": 0.5, "acousticness": 0.9}
    low_acoustic  = {"genre": "x", "mood": "x", "energy": 0.4, "valence": 0.5, "acousticness": 0.3}
    score_high, _ = score_song(prefs, high_acoustic)
    score_low, _  = score_song(prefs, low_acoustic)
    assert score_high > score_low
