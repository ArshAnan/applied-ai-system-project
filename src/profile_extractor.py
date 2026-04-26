"""
Profile Extractor — converts a natural language music request into a
structured user_prefs dict using Gemini 2.5 Flash Lite.
"""

import json
import os

from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

_VALID_GENRES = ["lofi", "pop", "rock", "electronic", "folk", "classical", "jazz", "latin", "rnb"]
_VALID_MOODS  = ["chill", "happy", "intense", "sad", "romantic", "energetic"]

_SYSTEM_PROMPT = """You are a music preference parser. Extract structured listening preferences from the user's natural language request.

Return ONLY a valid JSON object with exactly these keys:
- "genre": one of {genres}
- "mood": one of {moods}
- "energy": float 0.0-1.0  (0 = very calm, 1 = very intense)
- "valence": float 0.0-1.0  (0 = negative/melancholic, 1 = positive/uplifting)
- "likes_acoustic": boolean

Pick the closest match for genre and mood even if the user does not use those exact words.
Do not include any explanation, markdown, or extra keys - only the JSON object.
""".format(genres=_VALID_GENRES, moods=_VALID_MOODS)


def extract_profile(user_input: str) -> dict:
    """
    Parse a natural language music request into a user_prefs dict.

    Returns a dict with keys: genre, mood, energy, valence, likes_acoustic.
    Raises ValueError if the model response cannot be parsed as valid JSON.
    """
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=user_input,
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM_PROMPT,
        ),
    )

    raw = response.text.strip()

    # Strip markdown code fences if the model wraps the JSON
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

    try:
        prefs = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Gemini returned non-JSON output: {raw!r}") from exc

    return prefs
