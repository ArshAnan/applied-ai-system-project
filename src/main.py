"""
Command-line runner for the Music Recommender Simulation.

Modes:
    python -m src.main                  # batch mode: all 5 profiles
    python -m src.main --experiment     # weight-shift experiment
    python -m src.main --chat           # conversational mode (uses Gemini API)
"""

from src.recommender import load_songs, recommend_songs

# ---------------------------------------------------------------------------
# User profiles used for batch evaluation
# ---------------------------------------------------------------------------

PROFILES = [
    {
        "label": "Chill Lofi Listener",
        "prefs": {
            "genre": "lofi",
            "mood": "chill",
            "energy": 0.40,
            "valence": 0.60,
            "likes_acoustic": True,
        },
    },
    {
        "label": "High-Energy Pop Fan",
        "prefs": {
            "genre": "pop",
            "mood": "happy",
            "energy": 0.85,
            "valence": 0.82,
            "likes_acoustic": False,
        },
    },
    {
        "label": "Intense Rock Head",
        "prefs": {
            "genre": "rock",
            "mood": "intense",
            "energy": 0.90,
            "valence": 0.45,
            "likes_acoustic": False,
        },
    },
    {
        "label": "ADVERSARIAL: High Energy + Sad (conflicting vibe)",
        "prefs": {
            "genre": "folk",
            "mood": "sad",
            "energy": 0.92,
            "valence": 0.30,
            "likes_acoustic": True,
        },
    },
    {
        "label": "ADVERSARIAL: Genre Not In Catalog (metal)",
        "prefs": {
            "genre": "metal",
            "mood": "intense",
            "energy": 0.95,
            "valence": 0.25,
            "likes_acoustic": False,
        },
    },
]


def print_recommendations(label: str, recommendations: list, k: int = 5) -> None:
    """Print a formatted recommendation table for one user profile."""
    bar = "=" * 70
    print(f"\n{bar}")
    print(f"  Profile: {label}")
    print(bar)
    print(f"  {'Rank':<5} {'Title':<25} {'Artist':<22} {'Score':>5}")
    print(f"  {'-' * 62}")
    for rank, (song, score, explanation) in enumerate(recommendations, start=1):
        print(f"  {rank:<5} {song['title']:<25} {song['artist']:<22} {score:>5.2f}")
        for reason in explanation.split(" | "):
            print(f"         → {reason}")
        print()


# ---------------------------------------------------------------------------
# Chat mode
# ---------------------------------------------------------------------------

def chat() -> None:
    """
    Interactive conversational loop.

    Each iteration:
      1. Reads a natural language request from the user.
      2. Calls Gemini to extract a structured user_prefs dict.
      3. Runs the existing scoring algorithm.
      4. Calls Gemini to critique the top-5 list.
      5. Displays results with the AI's explanation.
      6. Logs the full session to logs/sessions.jsonl.
    """
    from src.profile_extractor import extract_profile
    from src.critic import critique_recommendations
    from src.logger import log_session

    songs = load_songs("data/songs.csv")
    print("\n🎵  VibeFinder — Chat Mode  (type 'quit' to exit)\n")

    while True:
        user_input = input("What are you in the mood for? › ").strip()
        if user_input.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break
        if not user_input:
            continue

        # Step 1 — extract structured preferences from natural language
        print("\n⏳  Parsing your request...")
        try:
            user_prefs = extract_profile(user_input)
        except (ValueError, KeyError) as exc:
            print(f"⚠️  Could not parse your request: {exc}")
            continue

        print(f"   Extracted → genre: {user_prefs.get('genre')}, "
              f"mood: {user_prefs.get('mood')}, "
              f"energy: {user_prefs.get('energy'):.2f}")

        # Step 2 — run the scoring algorithm
        recommendations = recommend_songs(user_prefs, songs, k=5)

        # Step 3 — AI critique
        print("⏳  Reviewing recommendations...")
        try:
            critique = critique_recommendations(user_input, user_prefs, recommendations)
        except (ValueError, KeyError) as exc:
            print(f"⚠️  Critic failed: {exc}")
            critique = {"verdict": "unknown", "flags": [], "explanation": ""}

        # Step 4 — log the session
        log_session(user_input, user_prefs, recommendations, critique)

        # Step 5 — display results
        bar = "=" * 70
        print(f"\n{bar}")
        print(f"  Top picks for: \"{user_input}\"")
        print(bar)
        print(f"  {'Rank':<5} {'Title':<25} {'Artist':<22} {'Score':>5}")
        print(f"  {'-' * 62}")
        for rank, (song, score, _) in enumerate(recommendations, start=1):
            print(f"  {rank:<5} {song['title']:<25} {song['artist']:<22} {score:>5.2f}")
        print()

        verdict_icon = {"good": "✅", "mixed": "⚠️", "poor": "❌"}.get(
            critique.get("verdict", ""), "🤔"
        )
        print(f"  {verdict_icon}  AI verdict: {critique.get('verdict', 'unknown').upper()}")
        print(f"  {critique.get('explanation', '')}")

        if critique.get("flags"):
            print("\n  Flags:")
            for flag in critique["flags"]:
                print(f"    • {flag}")

        print()


# ---------------------------------------------------------------------------
# Batch / experiment modes (unchanged)
# ---------------------------------------------------------------------------

def main(experiment: bool = False) -> None:
    songs = load_songs("data/songs.csv")
    print(f"\nLoaded {len(songs)} songs.")

    if experiment:
        print("\n*** EXPERIMENT: genre weight ×0.5 | energy weight ×2 ***")

        def experimental_score(user_prefs, song):
            """Modified scorer: genre=1.0, energy=3.0, rest unchanged."""
            score = 0.0
            reasons = []
            if song["genre"] == user_prefs.get("genre"):
                score += 1.0
                reasons.append(f"genre match ({song['genre']}) +1.0")
            if song["mood"] == user_prefs.get("mood"):
                score += 1.0
                reasons.append(f"mood match ({song['mood']}) +1.0")
            energy_pts = 3.0 * (1 - abs(song["energy"] - user_prefs["energy"]))
            score += energy_pts
            reasons.append(f"energy fit +{energy_pts:.2f}")
            if "valence" in user_prefs:
                valence_pts = 0.5 * (1 - abs(song["valence"] - user_prefs["valence"]))
                score += valence_pts
                reasons.append(f"valence fit +{valence_pts:.2f}")
            if user_prefs.get("likes_acoustic") and song["acousticness"] > 0.6:
                score += 0.5
                reasons.append("acoustic match +0.5")
            return score, reasons

        for profile in PROFILES[:3]:
            scored = []
            for song in songs:
                score, reasons = experimental_score(profile["prefs"], song)
                scored.append((song, score, " | ".join(reasons)))
            ranked = sorted(scored, key=lambda x: x[1], reverse=True)[:5]
            print_recommendations(f"[EXP] {profile['label']}", ranked)
        return

    for profile in PROFILES:
        results = recommend_songs(profile["prefs"], songs, k=5)
        print_recommendations(profile["label"], results)


if __name__ == "__main__":
    import sys
    if "--chat" in sys.argv:
        chat()
    elif "--experiment" in sys.argv:
        main(experiment=True)
    else:
        main()
