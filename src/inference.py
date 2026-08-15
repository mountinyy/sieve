from __future__ import annotations

import os

from google import genai
from openai import OpenAI


MODEL_DICT = {
    "gemma3_12b": "google/gemma-3-12b-it",
    "gemma3_27b": "google/gemma-3-27b-it",
    "qwen2.5_14b": "Qwen/Qwen2.5-14B-Instruct",
    "qwen2.5_3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen3_14b": "Qwen/Qwen3-14B",
    "llama3.3_70b": "meta-llama/Llama-3.3-70B-Instruct",
}

API_MODELS = {
    "gpt-4o": "gpt-4o",
    "gpt-4.1": "gpt-4.1",
    "gpt-5": "gpt-5",
    "gpt-5.2": "gpt-5.2",
    "gpt-5.4": "gpt-5.4",
    "gemini-3.5-flash": "gemini-3.5-flash",
}


class EquilibriumManager:
    """Small public API wrapper used by the SIEVE release scripts."""

    def __init__(self, model_name: str, temperature: float = 0.0) -> None:
        self.model_name = model_name
        self.api_model_name = API_MODELS.get(model_name, model_name)
        self.temperature = temperature
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def prompt_llm(
        self,
        prompt_kwargs: dict | None,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1024,
        enable_thinking: bool = False,
        **_: object,
    ) -> str:
        del prompt_kwargs, enable_thinking
        if self.model_name.startswith("gemini"):
            api_key = os.environ.get("GEMINI_API_KEY", "")
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=self.api_model_name,
                contents=user_prompt,
                config={"max_output_tokens": int(max_tokens), "temperature": self.temperature},
            )
            return getattr(response, "text", "") or ""

        api_key = os.environ.get("OPENAI_API_KEY", "")
        client = OpenAI(api_key=api_key)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        response = client.chat.completions.create(
            model=self.api_model_name,
            messages=messages,
            max_tokens=int(max_tokens),
            temperature=self.temperature,
        )
        return response.choices[0].message.content or ""
