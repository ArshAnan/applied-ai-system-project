"""
AI Critic — reviews the top-5 recommendations produced by the scoring
algorithm and flags any mismatches with the user's original request.
"""

import json
import os
from typing import Dict, List, Tuple

from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

_SYSTEM_PROMPT = """You are a music recommendation reviewer. You will be given:
1. The user's original request in their own words.
2. The structured preferences extracted from that request.
3. The top-5 songs chosen by a scoring algorithm.

Your job is to judge whether the recommendations genuinely match the user's intent.

Return ONLY a valid JSON object with exactly these keys:
- "verdict": one of "good", "mixed", or "poor"
- "flags": a list of plain-English strings describing specific mismatches (empty list if none)
- "explanation": 1-2 sentences addressed to the user explaining why these picks fit (or don't fit) their request

Do not include any markdown, code fences, or extra keys - only the JSON object.
"""


def critique_recommendations(
    user_input: str,
    user_prefs: Dict,
    recommendations: List[Tuple[Dict, float, str]],
) -> Dict:
    """
    Ask Gemini to review the top-5 song list against the user's original request.

    Returns a dict with keys: verdict, flags, explanation.
    Raises ValueError if the model response cannot be parsed as valid JSON.
    """
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    songs_summary = "\n".join(
        f"{i + 1}. \"{song['title']}\" by {song['artist']}"
        f" - genre: {song['genre']}, mood: {song['mood']}"
        f", energy: {song['energy']:.2f}, score: {score:.2f}"
        for i, (song, score, _) in enumerate(recommendations)
    )

    user_message = (
        f"User's request: \"{user_input}\"\n\n"
        f"Extracted preferences:\n{json.dumps(user_prefs, indent=2)}\n\n"
        f"Algorithm's top-5 picks:\n{songs_summary}"
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM_PROMPT,
        ),
    )

    raw = response.text.strip()

    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

    try:
        result = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Gemini returned non-JSON output: {raw!r}") from exc

    return result
