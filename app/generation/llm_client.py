"""LLM generation interface for grounded SEC filing questions."""

from __future__ import annotations

from collections.abc import Iterator

from groq import Groq

from app.config import settings


def generate_answer(prompt: str) -> str:
    """Send a prompt to the configured LLM provider and return the model's text."""
    api_key = settings.GROQ_API_KEY
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing or empty. Add it to your .env file before running generation.")

    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:  # pragma: no cover - simple runtime guard
        raise RuntimeError(f"Groq API call failed: {exc}") from exc


def generate_answer_stream(prompt: str) -> Iterator[str]:
    """Yield answer text fragments as they arrive from Groq's streaming API."""
    api_key = settings.GROQ_API_KEY
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing or empty. Add it to your .env file before running generation.")

    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.2,
            stream=True,
        )
        for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                yield content
    except Exception as exc:  # pragma: no cover - simple runtime guard
        raise RuntimeError(f"Groq streaming API call failed: {exc}") from exc
