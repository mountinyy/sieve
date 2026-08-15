from __future__ import annotations


KNOWN_API_MODELS = {
    "gpt-4o",
    "gpt-4.1",
    "gpt-5",
    "gpt-5.2",
    "gpt-5.4",
    "gemini-3.5-flash",
    "claude-opus-4-7",
}
KNOWN_AZURE_API_MODELS = set()


def is_api_model(model_name: str) -> bool:
    return str(model_name).strip().lower() in KNOWN_API_MODELS
