"""
Session Logger — appends one JSON-Lines entry per chat session to
logs/sessions.jsonl so every run is auditable.
"""

import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Tuple

LOG_PATH = "logs/sessions.jsonl"


def log_session(
    user_input: str,
    user_prefs: Dict,
    recommendations: List[Tuple[Dict, float, str]],
    critique: Dict,
) -> None:
    """Append a structured record for one session to LOG_PATH."""
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_input": user_input,
        "extracted_prefs": user_prefs,
        "top_5": [
            {
                "title": song["title"],
                "artist": song["artist"],
                "genre": song["genre"],
                "score": round(score, 3),
            }
            for song, score, _ in recommendations
        ],
        "critique": critique,
    }

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
