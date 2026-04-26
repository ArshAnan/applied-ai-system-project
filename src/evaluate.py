"""
Batch evaluator for VibeFinder 2.0.

Runs a fixed set of test cases through the full pipeline — extraction,
scoring, and critique — and prints a reliability summary.

Usage:
    python3 -m src.evaluate
"""

from src.recommender import load_songs, recommend_songs
from src.profile_extractor import extract_profile
from src.critic import critique_recommendations
from src.logger import log_session

# ---------------------------------------------------------------------------
# Fixed test cases with expected extraction outputs
# ---------------------------------------------------------------------------

TEST_CASES = [
    {
        "input": "chill lofi music for studying",
        "expected_genre": "lofi",
        "expected_mood": "chill",
    },
    {
        "input": "upbeat pop songs for a party",
        "expected_genre": "pop",
        "expected_mood": "happy",
    },
    {
        "input": "intense rock music to get pumped",
        "expected_genre": "rock",
        "expected_mood": "intense",
    },
    {
        "input": "sad acoustic songs for a rainy day",
        "expected_genre": "folk",
        "expected_mood": "sad",
    },
    {
        "input": "electronic music for working out",
        "expected_genre": "electronic",
        "expected_mood": "energetic",
    },
    {
        "input": "relaxing classical music",
        "expected_genre": "classical",
        "expected_mood": "chill",
    },
]


def run_evaluation() -> None:
    songs = load_songs("data/songs.csv")

    extraction_hits = 0
    verdict_counts  = {"good": 0, "mixed": 0, "poor": 0, "unknown": 0}
    confidence_scores = []
    failures = []

    bar = "=" * 70
    print(f"\n{bar}")
    print("  VibeFinder 2.0 — Batch Evaluation")
    print(f"  Running {len(TEST_CASES)} test cases")
    print(bar)

    for i, case in enumerate(TEST_CASES, start=1):
        user_input = case["input"]
        print(f"\n[{i}/{len(TEST_CASES)}] \"{user_input}\"")

        # Step 1 — extraction
        try:
            prefs = extract_profile(user_input)
        except (ValueError, KeyError) as exc:
            print(f"  ✗ Extraction failed: {exc}")
            failures.append({"input": user_input, "stage": "extraction", "error": str(exc)})
            verdict_counts["unknown"] += 1
            continue

        genre_ok = prefs.get("genre") == case["expected_genre"]
        mood_ok  = prefs.get("mood")  == case["expected_mood"]

        if genre_ok and mood_ok:
            extraction_hits += 1
            print(f"  ✓ Extracted: genre={prefs['genre']}, mood={prefs['mood']}")
        else:
            print(
                f"  ~ Extracted: genre={prefs.get('genre')} "
                f"(expected {case['expected_genre']}), "
                f"mood={prefs.get('mood')} (expected {case['expected_mood']})"
            )

        # Step 2 — scoring
        recommendations = recommend_songs(prefs, songs, k=5)

        # Step 3 — critique
        try:
            critique = critique_recommendations(user_input, prefs, recommendations)
        except (ValueError, KeyError) as exc:
            print(f"  ✗ Critique failed: {exc}")
            failures.append({"input": user_input, "stage": "critique", "error": str(exc)})
            verdict_counts["unknown"] += 1
            log_session(user_input, prefs, recommendations,
                        {"verdict": "unknown", "confidence": 0.0, "flags": [], "explanation": str(exc)})
            continue

        verdict    = critique.get("verdict", "unknown")
        confidence = critique.get("confidence", 0.0)
        flags      = critique.get("flags", [])

        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
        confidence_scores.append(confidence)

        log_session(user_input, prefs, recommendations, critique)

        icon = {"good": "✓", "mixed": "~", "poor": "✗"}.get(verdict, "?")
        print(f"  {icon} Verdict: {verdict.upper()}  |  Confidence: {confidence:.2f}  |  Flags: {len(flags)}")

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    total     = len(TEST_CASES)
    attempted = total - verdict_counts.get("unknown", 0)
    avg_conf  = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0

    print(f"\n{bar}")
    print("  SUMMARY")
    print(bar)
    print(f"  Extraction accuracy : {extraction_hits}/{total} genre+mood pairs matched expected output")
    print(f"  Verdicts            : {verdict_counts['good']} good  |  {verdict_counts['mixed']} mixed  |  {verdict_counts['poor']} poor  |  {verdict_counts.get('unknown', 0)} failed")
    print(f"  Avg confidence      : {avg_conf:.2f} / 1.00  (across {len(confidence_scores)} completed critiques)")

    if failures:
        print(f"\n  Failures ({len(failures)}):")
        for f in failures:
            print(f"    • [{f['stage']}] \"{f['input']}\" → {f['error']}")
    print()


if __name__ == "__main__":
    run_evaluation()
