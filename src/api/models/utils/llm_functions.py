"""
This script contains functions used for the LLM in the API endpoints for generating the
transcript and the medical resume
"""

import os
from typing import Any

from src.api.models.utils.llm_info import *

import httpx

# set up the API configuration
GEMINI_API_KEY = os.getenv('ENV_API_KEY') or os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent"

def _extract_gemini_text(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidates") or []
    if not candidates:
        return ""

    content = candidates[0].get("content") or {}
    parts = content.get("parts") or []
    texts: list[str] = []
    for part in parts:
        text = part.get("text")
        if isinstance(text, str) and text.strip():
            texts.append(text)
    return "\n".join(texts).strip()


async def _generate_medical_report(transcription: str) -> str | None:
    if not GEMINI_API_KEY:
        print("[Gemini] :: API key missing. Set ENV_API_KEY (or GEMINI_API_KEY/GOOGLE_API_KEY).", flush=True)
        return None

    if not transcription.strip():
        return None

    request_payload: dict[str, Any] = {
        "contents": [
            {
                "parts": [
                    {
                        "text": f"{MEDICAL_PROMPT_TEMPLATE}{transcription}",
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "topP": 0.9,
            "topK": 20,
        },
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            GEMINI_API_URL,
            headers={
                "Content-Type": "application/json",
                "X-goog-api-key": GEMINI_API_KEY,
            },
            json=request_payload,
        )

    if response.status_code != 200:
        print(f"[Gemini] :: Request failed ({response.status_code}): {response.text}", flush=True)
        return None

    payload = response.json()
    model_text = _extract_gemini_text(payload)
    if not model_text:
        print("[Gemini] :: Empty model response", flush=True)
        return None

    return _ensure_complete_report(model_text)


def _ensure_complete_report(report_text: str) -> str:
    cleaned = report_text.strip()
    normalized = cleaned.upper()

    missing_sections: list[str] = []
    for section in REPORT_SECTIONS:
        marker = f"{section}:"
        if marker not in normalized:
            missing_sections.append(section)

    if not missing_sections:
        return cleaned

    # Ensure every section exists to avoid incomplete rendering downstream.
    completed = cleaned
    if completed:
        completed += "\n\n"
    completed += "\n\n".join([f"{section}:\nNo referido." for section in missing_sections])
    return completed
