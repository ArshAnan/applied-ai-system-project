"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

from recommender import load_songs, recommend_songs


def main() -> None:
    songs = load_songs("data/songs.csv") 

    # Taste profile: a late-night lofi listener who values chill, acoustic, low-energy tracks.
    # favorite_genre / favorite_mood are categorical matches (exact string comparison).
    # target_energy and target_valence are on a 0.0–1.0 scale; scorer rewards closeness.
    # likes_acoustic adds a bonus when acousticness > 0.6.
    user_prefs = {
        "genre": "lofi",
        "mood": "chill",
        "energy": 0.40,
        "valence": 0.60,
        "likes_acoustic": True,
    }

    recommendations = recommend_songs(user_prefs, songs, k=5)

    print("\nTop recommendations:\n")
    for rec in recommendations:
        # You decide the structure of each returned item.
        # A common pattern is: (song, score, explanation)
        song, score, explanation = rec
        print(f"{song['title']} - Score: {score:.2f}")
        print(f"Because: {explanation}")
        print()


if __name__ == "__main__":
    main()
