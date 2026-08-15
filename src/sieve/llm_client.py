from __future__ import annotations

"""
PRISM: LLM client abstraction.
Implement a subclass for your API (OpenAI, Anthropic, vLLM, etc.)
"""

import json
import os
import re
import tempfile
import threading
from abc import ABC, abstractmethod
from pathlib import Path

from openai import BadRequestError
from src.light_check import KNOWN_API_MODELS, KNOWN_AZURE_API_MODELS, is_api_model

thinking_models = ["qwen3"]
DEFAULT_REFUSAL = "I'm sorry, I cannot help with that request."


def _resolve_local_model_source(model_name: str) -> str:
    """Resolve a local-model identifier to a concrete filesystem path when possible.

    Resolution order:
    1. Existing local path -> use directly
    2. Hugging Face cache lookup only
    3. Hugging Face download into the default cache
    4. Fallback to the original identifier
    """
    expanded = Path(str(model_name)).expanduser()
    if expanded.exists():
        return str(expanded)

    try:
        from huggingface_hub import snapshot_download  # noqa: WPS433
        from huggingface_hub.utils import LocalEntryNotFoundError  # noqa: WPS433
    except Exception:
        return str(model_name)

    try:
        resolved = snapshot_download(
            repo_id=str(model_name),
            local_files_only=True,
            resume_download=True,
        )
        print(f"[INFO] Using cached Hugging Face model: {model_name}")
        return str(resolved)
    except LocalEntryNotFoundError:
        pass
    except Exception as exc:
        print(f"[WARN] Could not resolve cached Hugging Face model '{model_name}': {exc}")

    try:
        print(f"[INFO] Downloading Hugging Face model: {model_name}")
        resolved = snapshot_download(
            repo_id=str(model_name),
            local_files_only=False,
            resume_download=True,
        )
        return str(resolved)
    except Exception as exc:
        print(f"[WARN] Falling back to model identifier '{model_name}': {exc}")
        return str(model_name)


def _supports_enable_thinking(model_name: str) -> bool:
    normalized = str(model_name).strip().lower()
    return any(thinking_model in normalized for thinking_model in thinking_models)


def _build_thinking_kwargs(model_name: str, enable_thinking: bool | None) -> dict:
    if enable_thinking is None or not _supports_enable_thinking(model_name):
        return {}
    return {"chat_template_kwargs": {"enable_thinking": enable_thinking}}


def _coerce_text_or_refusal(
    text: str | None,
    *,
    reason: str | None = None,
    model_name: str | None = None,
) -> str:
    normalized = (text or "").strip()
    if normalized:
        return normalized
    if reason:
        model_part = f" for {model_name}" if model_name else ""
        print(
            f"[WARN] Returning DEFAULT_REFUSAL{model_part}: {reason}",
            flush=True,
        )
    return DEFAULT_REFUSAL


def is_default_refusal_text(text: str | None) -> bool:
    return (text or "").strip() == DEFAULT_REFUSAL


def render_chat_messages_as_text(messages: list[dict]) -> str:
    return "\n".join(
        f"{str(message.get('role', 'user')).title()}: {message.get('content', '')}"
        for message in messages
    )


def _response_field_to_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(
            text for item in value if (text := _response_field_to_text(item))
        ).strip()
    if isinstance(value, dict):
        preferred = []
        for key in ("text", "content", "reasoning_content", "reasoning"):
            if key in value:
                text = _response_field_to_text(value.get(key))
                if text:
                    preferred.append(text)
        if preferred:
            return "\n".join(preferred).strip()
        try:
            return json.dumps(value, ensure_ascii=False, default=str).strip()
        except Exception:
            return str(value).strip()
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _response_field_to_text(model_dump())
        except Exception:
            pass
    return str(value).strip()


def _extract_message_reasoning(message: object) -> str:
    for field_name in ("reasoning_content", "reasoning"):
        text = _response_field_to_text(getattr(message, field_name, None))
        if text:
            return text

    model_extra = getattr(message, "model_extra", None)
    if isinstance(model_extra, dict):
        for field_name in ("reasoning_content", "reasoning"):
            text = _response_field_to_text(model_extra.get(field_name))
            if text:
                return text

    model_dump = getattr(message, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump()
        except Exception:
            dumped = None
        if isinstance(dumped, dict):
            for field_name in ("reasoning_content", "reasoning"):
                text = _response_field_to_text(dumped.get(field_name))
                if text:
                    return text
    return ""


class LLMClient(ABC):
    """Abstract LLM client. Subclass for your API."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float | None = None,
    ) -> str:
        """Generate a response from the LLM."""
        raise NotImplementedError

    def generate_with_metadata(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float | None = None,
    ) -> dict:
        """
        Generate a response plus lightweight metadata.

        Subclasses can override this to return provider-native token counts.
        The default implementation preserves backward compatibility by calling
        `generate()` and estimating output length from whitespace tokens.
        """
        text = self.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return {
            "text": text,
            "output_tokens": len(text.split()),
            "temperature": temperature,
        }

    def generate_chat(
        self,
        messages: list[dict],
        max_tokens: int = 1024,
        temperature: float | None = None,
    ) -> str:
        return self.generate(
            render_chat_messages_as_text(messages),
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def generate_chat_with_metadata(
        self,
        messages: list[dict],
        max_tokens: int = 1024,
        temperature: float | None = None,
    ) -> dict:
        text = self.generate_chat(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return {
            "text": text,
            "output_tokens": len(text.split()),
            "temperature": temperature,
        }

    def generate_batch(
        self,
        prompts: list[str],
        *,
        max_tokens: int = 1024,
        temperature: float | None = None,
        save_dir: str | Path | None = None,
        batch_name: str = "batch",
    ) -> list[str]:
        outputs = []
        for prompt in prompts:
            try:
                outputs.append(
                    self.generate(
                        prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                )
            except TypeError as exc:
                if "temperature" not in str(exc):
                    raise
                outputs.append(self.generate(prompt, max_tokens=max_tokens))
        return outputs

    def parse_json(self, original_text: str) -> dict:
        """Extract JSON from LLM response, handling markdown fences."""
        text = original_text.strip()

        fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        candidates = []
        if fenced_match:
            candidates.append(fenced_match.group(1).strip())

        direct_match = re.search(r"\{.*\}", text, re.DOTALL)
        if direct_match:
            candidates.append(direct_match.group(0).strip())

        if text.startswith("{") and text.endswith("}"):
            candidates.append(text)

        seen = set()
        ordered_candidates = []
        for candidate in candidates:
            if candidate not in seen:
                ordered_candidates.append(candidate)
                seen.add(candidate)

        for candidate in ordered_candidates:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                repaired = self._repair_json_candidate(candidate)
                if repaired != candidate:
                    try:
                        return json.loads(repaired)
                    except json.JSONDecodeError:
                        continue

        raise ValueError(
            "Failed to parse JSON from LLM response. "
            f"Raw response preview: {original_text[:500]}"
        )

    def _repair_json_candidate(self, candidate: str) -> str:
        """Apply minimal structural repairs for common truncated JSON outputs."""
        text = candidate.strip()
        if not text:
            return text

        open_square = text.count("[")
        close_square = text.count("]")
        open_curly = text.count("{")
        close_curly = text.count("}")

        if open_square > close_square:
            text = text + ("]" * (open_square - close_square))
        if open_curly > close_curly:
            text = text + ("}" * (open_curly - close_curly))
        return text


# ==============================================================================
# Example implementations (uncomment and fill in your API key)
# ==============================================================================

# class OpenAIClient(LLMClient):
#     def __init__(self, model="gpt-4o", api_key=None):
#         from openai import OpenAI
#         self.client = OpenAI(api_key=api_key)
#         self.model = model
#
#     def generate(self, prompt, max_tokens=1024):
#         response = self.client.chat.completions.create(
#             model=self.model,
#             messages=[{"role": "user", "content": prompt}],
#             max_tokens=max_tokens,
#             temperature=0.0,
#         )
#         return response.choices[0].message.content


# class AnthropicClient(LLMClient):
#     def __init__(self, model="claude-sonnet-4-20250514", api_key=None):
#         from anthropic import Anthropic
#         self.client = Anthropic(api_key=api_key)
#         self.model = model
#
#     def generate(self, prompt, max_tokens=1024):
#         response = self.client.messages.create(
#             model=self.model,
#             max_tokens=max_tokens,
#             messages=[{"role": "user", "content": prompt}],
#         )
#         return response.content[0].text


# class VLLMClient(LLMClient):
#     def __init__(self, model_name, base_url="http://localhost:8000/v1"):
#         from openai import OpenAI
#         self.client = OpenAI(base_url=base_url, api_key="dummy")
#         self.model = model_name
#
#     def generate(self, prompt, max_tokens=1024):
#         response = self.client.chat.completions.create(
#             model=self.model,
#             messages=[{"role": "user", "content": prompt}],
#             max_tokens=max_tokens,
#             temperature=0.0,
#         )
#         return response.choices[0].message.content


def _normalize_azure_model_name(model_name: str) -> str:
    normalized = str(model_name).strip().lower()
    azure_aliases = {
        "gpt-4o": "gpt-4o",
        "gpt-4.1": "gpt-4.1",
        "gpt-5": "gpt-5.2",
        "gpt-5.2": "gpt-5.2",
        "o3": "o3",
        "o3-mini": "o3-mini",
        "deepseek-v3.2": "DeepSeek-V3.2",
        "kimi-k2.5": "Kimi-K2.5",
        "llama-4-maverick-17b-128e-instruct-fp8": "Llama-4-Maverick-17B-128E-Instruct-FP8",
        "mistral-large-3": "Mistral-Large-3",
        "azure_gpt4": "gpt-4o",
        "azure_gpt-4o": "gpt-4o",
        "azure_gpt-4.1": "gpt-4.1",
        "azure_gpt5": "gpt-5.2",
        "azure_gpt-5": "gpt-5.2",
        "azure_gpt-5.2": "gpt-5.2",
    }
    return azure_aliases.get(normalized, model_name)


def _is_azure_api_model_name(model_name: str) -> bool:
    return is_api_model(model_name)


def _supports_temperature_parameter(model_name: str) -> bool:
    return not str(model_name).strip().lower().startswith("gpt-5")


class AzureInferenceClient(LLMClient):
    """Inference-only Azure client used by SIEVE eval paths."""

    def __init__(
        self,
        model_name: str,
        *,
        max_gen_len: int | None = None,
        temperature: float = 0.0,
        reasoning_effort: str | None = None,
    ):
        from src.azure import get_client

        self.model_name = model_name
        self.api_model_name = _normalize_azure_model_name(model_name)
        self.client = get_client(self.api_model_name)
        self.max_gen_len = max_gen_len
        self.default_temperature = float(temperature)
        self.reasoning_effort = reasoning_effort
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float | None = None,
    ) -> str:
        return self.generate_chat(
            [{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def generate_chat(
        self,
        messages: list[dict],
        max_tokens: int = 1024,
        temperature: float | None = None,
    ) -> str:
        if self.reasoning_effort:
            kwargs = {"reasoning": {"effort": self.reasoning_effort}}
        else:
            kwargs = {"effort": "none"} if self.api_model_name == "gpt-5.2" else {}
        resolved_max_tokens = min(
            int(max_tokens),
            int(self.max_gen_len) if self.max_gen_len is not None else int(max_tokens),
        )
        request_kwargs = {
            "model": self.api_model_name,
            "input": messages,
            "max_output_tokens": resolved_max_tokens,
            **kwargs,
        }
        if _supports_temperature_parameter(self.api_model_name):
            request_kwargs["temperature"] = (
                self.default_temperature if temperature is None else float(temperature)
            )
        try:
            completion = self.client.responses.create(**request_kwargs)
        except Exception as exc:
            print(f"AzureInferenceClient error for {self.api_model_name}: {exc}")
            return DEFAULT_REFUSAL

        usage = getattr(completion, "usage", None)
        if usage is not None:
            input_tokens = getattr(usage, "input_tokens", None)
            output_tokens = getattr(usage, "output_tokens", None)
            if input_tokens is None:
                input_tokens = getattr(usage, "prompt_tokens", 0)
            if output_tokens is None:
                output_tokens = getattr(usage, "completion_tokens", 0)
            self.prompt_tokens += input_tokens or 0
            self.completion_tokens += output_tokens or 0

        output_text = getattr(completion, "output_text", None)
        if output_text:
            return _coerce_text_or_refusal(output_text)

        texts = []
        for output_item in getattr(completion, "output", []):
            if getattr(output_item, "type", None) != "message":
                continue
            for content_item in getattr(output_item, "content", []):
                if getattr(content_item, "type", None) == "output_text":
                    texts.append(getattr(content_item, "text", ""))
        return _coerce_text_or_refusal("\n".join(texts))


class DirectAPIBatchInferenceClient(LLMClient):
    """Direct OpenAI/Gemini/Anthropic client with provider-native batch support."""

    def __init__(
        self,
        model_name: str,
        *,
        openai_api_key: str = "",
        gemini_api_key: str = "",
        anthropic_api_key: str = "",
        max_model_len: int | None = 8192,
        max_gen_len: int | None = 4096,
        temperature: float = 0.0,
        reasoning_effort: str | None = None,
        poll_interval: int = 30,
        max_polls: int = 720,
    ):
        self.model_name = str(model_name)
        self.max_model_len = max_model_len
        self.max_gen_len = max_gen_len
        self.default_temperature = float(temperature)
        self.reasoning_effort = reasoning_effort
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY", "")
        self.gemini_api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
        self.anthropic_api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.poll_interval = int(poll_interval)
        self.max_polls = int(max_polls)
        self.api_model_name = self.model_name
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.batch_estimated_cost_usd = 0.0
        self.batch_full_price_cost_usd = 0.0

    def _provider(self) -> str:
        normalized = self.model_name.lower()
        if "gemini" in normalized:
            return "gemini"
        if "claude" in normalized or "anthropic" in normalized:
            return "anthropic"
        return "openai"

    def _resolved_max_tokens(self, max_tokens: int) -> int:
        if self.max_gen_len is None:
            return int(max_tokens)
        return min(int(max_tokens), int(self.max_gen_len))

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float | None = None,
    ) -> str:
        provider = self._provider()
        resolved_max_tokens = self._resolved_max_tokens(max_tokens)
        try:
            if provider == "openai":
                from openai import OpenAI

                client = OpenAI(api_key=self.openai_api_key)
                request_kwargs = {
                    "model": self.model_name,
                    "input": [{"role": "user", "content": prompt}],
                    "max_output_tokens": resolved_max_tokens,
                }
                if self.reasoning_effort:
                    request_kwargs["reasoning"] = {"effort": self.reasoning_effort}
                if _supports_temperature_parameter(self.model_name):
                    request_kwargs["temperature"] = (
                        self.default_temperature if temperature is None else float(temperature)
                    )
                response = client.responses.create(**request_kwargs)
                usage = getattr(response, "usage", None)
                if usage is not None:
                    self.prompt_tokens += getattr(usage, "input_tokens", 0) or 0
                    self.completion_tokens += getattr(usage, "output_tokens", 0) or 0
                output_text = getattr(response, "output_text", None)
                if output_text:
                    return _coerce_text_or_refusal(output_text)
                texts = []
                for output_item in getattr(response, "output", []) or []:
                    if getattr(output_item, "type", None) != "message":
                        continue
                    for content_item in getattr(output_item, "content", []) or []:
                        if getattr(content_item, "type", None) == "output_text":
                            texts.append(getattr(content_item, "text", ""))
                return _coerce_text_or_refusal("\n".join(texts))

            if provider == "gemini":
                from google import genai
                from google.genai import types

                client = genai.Client(api_key=self.gemini_api_key)
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        max_output_tokens=resolved_max_tokens,
                    ),
                )
                usage = getattr(response, "usage_metadata", None)
                if usage is not None:
                    self.prompt_tokens += getattr(usage, "prompt_token_count", 0) or 0
                    self.completion_tokens += getattr(usage, "candidates_token_count", 0) or 0
                return _coerce_text_or_refusal(getattr(response, "text", ""))

            import anthropic

            client = anthropic.Anthropic(api_key=self.anthropic_api_key)
            response = client.messages.create(
                model=self.model_name,
                max_tokens=resolved_max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            usage = getattr(response, "usage", None)
            if usage is not None:
                self.prompt_tokens += getattr(usage, "input_tokens", 0) or 0
                self.completion_tokens += getattr(usage, "output_tokens", 0) or 0
            texts = [
                getattr(block, "text", "")
                for block in getattr(response, "content", []) or []
                if getattr(block, "type", "") == "text"
            ]
            return _coerce_text_or_refusal("\n".join(texts))
        except Exception as exc:
            print(f"DirectAPIBatchInferenceClient error for {self.model_name}: {exc}")
            return DEFAULT_REFUSAL

    def generate_batch(
        self,
        prompts: list[str],
        *,
        max_tokens: int = 1024,
        temperature: float | None = None,
        save_dir: str | Path | None = None,
        batch_name: str = "batch",
    ) -> list[str]:
        if not prompts:
            return []

        provider = self._provider()
        resolved_max_tokens = self._resolved_max_tokens(max_tokens)
        work_dir = Path(save_dir) if save_dir is not None else Path(tempfile.mkdtemp(prefix="sieve-api-batch-"))
        work_dir.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", batch_name).strip("-") or "batch"
        print(
            f"[INFO] Submitting {provider} batch for {self.model_name}: "
            f"requests={len(prompts)}, max_output_tokens={resolved_max_tokens}, save_dir={work_dir}",
            flush=True,
        )

        try:
            from src.runs.schema_label import (
                anthropic_text_or_tool_input,
                anthropic_usage,
                download_anthropic_results,
                download_batch_output,
                download_gemini_responses,
                estimate_batch_cost,
                extract_response_text,
                extract_usage,
                gemini_custom_id,
                gemini_response_text,
                gemini_usage,
                read_jsonl,
                sdk_to_dict,
                submit_anthropic_batch,
                submit_batch,
                submit_gemini_batch,
                wait_for_anthropic_batch,
                wait_for_batch,
                wait_for_gemini_batch,
                write_json,
                write_jsonl,
            )
        except Exception as exc:
            raise RuntimeError("src.runs.schema_label batch helpers are required for API batch inference.") from exc

        usage_totals = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_input_tokens": 0,
            "reasoning_tokens": 0,
        }

        def add_usage(usage: dict) -> None:
            for key in usage_totals:
                usage_totals[key] += int(usage.get(key, 0) or 0)

        outputs_by_id: dict[str, str] = {}
        if provider == "openai":
            from openai import OpenAI

            client = OpenAI(api_key=self.openai_api_key)
            requests = []
            for idx, prompt in enumerate(prompts):
                body = {
                    "model": self.model_name,
                    "input": [{"role": "user", "content": prompt}],
                    "max_output_tokens": resolved_max_tokens,
                }
                if self.reasoning_effort:
                    body["reasoning"] = {"effort": self.reasoning_effort}
                if _supports_temperature_parameter(self.model_name):
                    body["temperature"] = (
                        self.default_temperature if temperature is None else float(temperature)
                    )
                requests.append(
                    {
                        "custom_id": f"sample-{idx}",
                        "method": "POST",
                        "url": "/v1/responses",
                        "body": body,
                    }
                )
            batch_file = work_dir / f"{safe_name}-openai.jsonl"
            output_file = work_dir / f"{safe_name}-openai-output.jsonl"
            write_jsonl(batch_file, requests)
            batch_id = submit_batch(client, batch_file, "24h")
            (work_dir / f"{safe_name}-openai-batch_id.txt").write_text(str(batch_id), encoding="utf-8")
            batch = wait_for_batch(
                client,
                str(batch_id),
                poll_interval=self.poll_interval,
                max_polls=self.max_polls,
            )
            if getattr(batch, "status", None) != "completed":
                raise RuntimeError(f"OpenAI batch ended with status={getattr(batch, 'status', None)}")
            download_batch_output(client, batch, output_file)
            for row in read_jsonl(output_file):
                custom_id = str(row.get("custom_id", ""))
                body = row.get("response", {}).get("body", {})
                outputs_by_id[custom_id] = extract_response_text(body)
                add_usage(extract_usage(body))

        elif provider == "gemini":
            requests_payload = []
            for idx, prompt in enumerate(prompts):
                custom_id = f"sample-{idx}"
                requests_payload.append(
                    {
                        "request": {
                            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                            "config": {"max_output_tokens": resolved_max_tokens},
                        },
                        "metadata": {"key": custom_id, "custom_id": custom_id},
                    }
                )
            write_json(work_dir / f"{safe_name}-gemini-requests.json", requests_payload)
            batch_name_id = submit_gemini_batch(
                model=self.model_name,
                api_key=self.gemini_api_key,
                requests_payload=requests_payload,
                display_name=safe_name,
            )
            (work_dir / f"{safe_name}-gemini-batch_name.txt").write_text(str(batch_name_id), encoding="utf-8")
            batch = wait_for_gemini_batch(
                batch_name=str(batch_name_id),
                api_key=self.gemini_api_key,
                poll_interval=self.poll_interval,
                max_polls=self.max_polls,
            )
            write_json(work_dir / f"{safe_name}-gemini-status.json", sdk_to_dict(batch))
            state = str(
                getattr(batch, "state", "")
                or sdk_to_dict(batch).get("state", "")
                or sdk_to_dict(batch).get("response", {}).get("state", "")
            )
            if state and "SUCCEEDED" not in state:
                raise RuntimeError(f"Gemini batch ended with state={state}")
            rows = download_gemini_responses(
                batch=batch,
                api_key=self.gemini_api_key,
                requests_payload=requests_payload,
            )
            write_json(work_dir / f"{safe_name}-gemini-output.json", rows)
            for idx, row in enumerate(rows):
                custom_id = gemini_custom_id(row, idx)
                outputs_by_id[custom_id] = gemini_response_text(row)
                add_usage(gemini_usage(row))

        else:
            model_requests = [
                {
                    "custom_id": f"sample-{idx}",
                    "params": {
                        "model": self.model_name,
                        "max_tokens": resolved_max_tokens,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                }
                for idx, prompt in enumerate(prompts)
            ]
            write_jsonl(work_dir / f"{safe_name}-anthropic-requests.jsonl", model_requests)
            batch_id = submit_anthropic_batch(
                model_requests=model_requests,
                api_key=self.anthropic_api_key,
            )
            (work_dir / f"{safe_name}-anthropic-batch_id.txt").write_text(str(batch_id), encoding="utf-8")
            batch = wait_for_anthropic_batch(
                batch_id=str(batch_id),
                api_key=self.anthropic_api_key,
                poll_interval=self.poll_interval,
                max_polls=self.max_polls,
            )
            write_json(work_dir / f"{safe_name}-anthropic-status.json", sdk_to_dict(batch))
            rows = download_anthropic_results(batch_id=str(batch_id), api_key=self.anthropic_api_key)
            write_jsonl(work_dir / f"{safe_name}-anthropic-output.jsonl", rows)
            for row in rows:
                custom_id = str(row.get("custom_id", ""))
                text, _ = anthropic_text_or_tool_input(row)
                outputs_by_id[custom_id] = text
                add_usage(anthropic_usage(row))

        self.prompt_tokens += usage_totals["input_tokens"]
        self.completion_tokens += usage_totals["output_tokens"]
        cost = estimate_batch_cost(self.model_name, usage_totals)
        self.batch_estimated_cost_usd += float(cost.get("estimated_cost_usd", 0.0) or 0.0)
        self.batch_full_price_cost_usd += float(
            cost.get("estimated_full_price_cost_usd", 0.0) or 0.0
        )
        write_json(work_dir / f"{safe_name}-cost.json", cost)
        print(
            f"[COST] {self.model_name} batch: ${float(cost.get('estimated_cost_usd', 0.0)):.6f} "
            f"(input={usage_totals['input_tokens']}, output={usage_totals['output_tokens']})",
            flush=True,
        )
        print(
            f"[COST] {self.model_name} cumulative batch: "
            f"${self.batch_estimated_cost_usd:.6f} "
            f"(full price ${self.batch_full_price_cost_usd:.6f})",
            flush=True,
        )
        return [
            _coerce_text_or_refusal(outputs_by_id.get(f"sample-{idx}", ""))
            for idx in range(len(prompts))
        ]


class VLLMServerInferenceClient(LLMClient):
    """Inference client backed by an external vLLM OpenAI-compatible server."""

    def __init__(
        self,
        model_name: str,
        port: int = 8000,
        *,
        max_model_len: int | None = None,
        temperature: float = 0.0,
        max_gen_len: int | None = None,
        enable_thinking: bool | None = None,
    ):
        from openai import OpenAI
        from src.inference import MODEL_DICT
        from transformers import AutoTokenizer

        self.model_name = model_name
        self.vllm_port = port
        self.max_model_len = max_model_len
        self.default_temperature = float(temperature)
        self.max_gen_len = max_gen_len
        self.enable_thinking = enable_thinking
        self._request_state = threading.local()
        self.last_error = ""
        self.last_reasoning_content = ""
        self.last_reasoning_unavailable_reason = ""
        self.client = OpenAI(
            base_url=f"http://127.0.0.1:{self.vllm_port}/v1",
            api_key="EMPTY",
        )
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.tokenizer = None
        try:
            tokenizer_name = MODEL_DICT.get(model_name, model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                trust_remote_code=True,
                local_files_only=True,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as exc:
            print(
                f"[WARN] Could not load cached tokenizer for {self.model_name}; "
                "vLLM requests will continue, but client-side prompt token counting "
                f"and max-token clamping are disabled. Error: {exc}"
            )

    @property
    def last_error(self) -> str:
        return str(getattr(self._request_state, "last_error", "") or "")

    @last_error.setter
    def last_error(self, value: object) -> None:
        self._request_state.last_error = str(value or "")

    @property
    def last_reasoning_content(self) -> str:
        return str(getattr(self._request_state, "last_reasoning_content", "") or "")

    @last_reasoning_content.setter
    def last_reasoning_content(self, value: object) -> None:
        self._request_state.last_reasoning_content = str(value or "")

    @property
    def last_reasoning_unavailable_reason(self) -> str:
        return str(
            getattr(self._request_state, "last_reasoning_unavailable_reason", "") or ""
        )

    @last_reasoning_unavailable_reason.setter
    def last_reasoning_unavailable_reason(self, value: object) -> None:
        self._request_state.last_reasoning_unavailable_reason = str(value or "")

    def _render_messages_for_counting(self, messages: list[dict]) -> str:
        if self.tokenizer is None:
            return render_chat_messages_as_text(messages)
        try:
            apply_kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if _supports_enable_thinking(self.model_name) and self.enable_thinking is not None:
                apply_kwargs["enable_thinking"] = self.enable_thinking
            return self.tokenizer.apply_chat_template(messages, **apply_kwargs)
        except Exception:
            return render_chat_messages_as_text(messages)

    def _resolve_max_tokens_for_messages(self, messages: list[dict], max_tokens: int) -> int:
        resolved_max_tokens = min(
            int(max_tokens),
            int(self.max_gen_len) if self.max_gen_len is not None else int(max_tokens),
        )
        if self.max_model_len is None or self.tokenizer is None:
            return max(1, resolved_max_tokens)

        rendered_prompt = self._render_messages_for_counting(messages)
        try:
            prompt_tokens = len(
                self.tokenizer.encode(rendered_prompt, add_special_tokens=False)
            )
        except TypeError:
            prompt_tokens = len(self.tokenizer.encode(rendered_prompt))
        except Exception:
            return max(1, resolved_max_tokens)

        # Leave a small buffer for template/accounting differences.
        available_generation = max(1, int(self.max_model_len) - int(prompt_tokens) - 16)
        if available_generation < resolved_max_tokens:
            print(
                "[INFO] Clamping vLLM server max_tokens for "
                f"{self.model_name}: requested={resolved_max_tokens}, "
                f"prompt_tokens={prompt_tokens}, available={available_generation}, "
                f"max_model_len={self.max_model_len}"
            )
        return min(resolved_max_tokens, available_generation)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float | None = None,
    ) -> str:
        return self.generate_chat(
            [{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def generate_chat(
        self,
        messages: list[dict],
        max_tokens: int = 1024,
        temperature: float | None = None,
    ) -> str:
        self.last_error = ""
        self.last_reasoning_content = ""
        self.last_reasoning_unavailable_reason = ""
        request_kwargs = _build_thinking_kwargs(self.model_name, self.enable_thinking)
        resolved_max_tokens = self._resolve_max_tokens_for_messages(messages, max_tokens)
        resolved_temperature = (
            self.default_temperature if temperature is None else float(temperature)
        )
        try:
            create_kwargs = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": resolved_max_tokens,
                "temperature": resolved_temperature,
            }
            if request_kwargs:
                create_kwargs["extra_body"] = request_kwargs
            response = self.client.chat.completions.create(**create_kwargs)
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.last_reasoning_unavailable_reason = (
                "The vLLM server request failed before a parseable response message "
                "was returned."
            )
            print(
                "VLLMServerInferenceClient returning DEFAULT_REFUSAL after error "
                f"for {self.model_name}: {self.last_error}",
                flush=True,
            )
            return DEFAULT_REFUSAL
        self.last_error = ""
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            self.completion_tokens += getattr(usage, "completion_tokens", 0) or 0
        choices = getattr(response, "choices", None) or []
        if not choices:
            self.last_reasoning_unavailable_reason = "The vLLM response had no choices."
            return _coerce_text_or_refusal(
                "",
                reason="vLLM response had no choices.",
                model_name=self.model_name,
            )
        message = getattr(choices[0], "message", None)
        if message is None:
            self.last_reasoning_unavailable_reason = (
                "The vLLM response choice had no message."
            )
            return _coerce_text_or_refusal(
                "",
                reason="vLLM response choice had no message.",
                model_name=self.model_name,
            )
        self.last_reasoning_content = _extract_message_reasoning(message)
        if not self.last_reasoning_content:
            self.last_reasoning_unavailable_reason = (
                "The vLLM response message did not expose reasoning_content or reasoning."
            )
        content = getattr(message, "content", None)
        if isinstance(content, list):
            texts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        texts.append(item.get("text", ""))
                elif getattr(item, "type", None) == "text":
                    texts.append(getattr(item, "text", ""))
            content = "".join(texts)
        finish_reason = getattr(choices[0], "finish_reason", None)
        return _coerce_text_or_refusal(
            content,
            reason=(
                "vLLM returned an empty message content "
                f"(finish_reason={finish_reason!r}, "
                f"requested_max_tokens={resolved_max_tokens}, "
                f"temperature={resolved_temperature})."
            ),
            model_name=self.model_name,
        )


class LocalVLLMInferenceClient(LLMClient):
    """Inference client that loads a local vLLM engine in-process."""

    @staticmethod
    def _select_gpu_memory_utilization(use_sieve_memory_profile: bool) -> float:
        if not use_sieve_memory_profile:
            return 0.9
        try:
            import torch

            if not torch.cuda.is_available():
                return 0.9
            total_memory_bytes = torch.cuda.get_device_properties(0).total_memory
            total_memory_gb = total_memory_bytes / (1024**3)
            ratio = 0.8 if total_memory_gb >= 100 else 0.6
            free_bytes, total_bytes = torch.cuda.mem_get_info(device=0)
            print("[INFO] Detected GPU with total memory: {:.2f} GB".format(total_memory_gb))
            print("[INFO] VLLM GPU memory: {:.2f} GB".format(total_memory_gb * ratio))
            print("[INFO] Free GPU memory: {:.2f} GB".format(free_bytes / (1024**3)))
            return ratio
        except Exception:
            return 0.9

    def __init__(
        self,
        model_name: str,
        *,
        max_model_len: int | None = None,
        max_gen_len: int | None = None,
        temperature: float = 0.0,
        use_sieve_memory_profile: bool = False,
        enable_thinking: bool | None = None,
    ):
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        from vllm import LLM, SamplingParams
        from src.inference import MODEL_DICT

        self.model_name = model_name
        self.api_model_name = None
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self._sampling_params_cls = SamplingParams
        self._resolved_model_name = _resolve_local_model_source(
            MODEL_DICT.get(model_name, model_name)
        )
        self.max_model_len = max_model_len
        self.max_gen_len = max_gen_len
        self.default_temperature = float(temperature)
        self.enable_thinking = enable_thinking

        llm_kwargs = {
            "model": self._resolved_model_name,
            "trust_remote_code": True,
            "gpu_memory_utilization": self._select_gpu_memory_utilization(
                use_sieve_memory_profile
            ),
        }
        tensor_parallel_size = int(os.environ.get("VLLM_TENSOR_PARALLEL_SIZE", "1") or "1")
        if tensor_parallel_size > 1:
            llm_kwargs["tensor_parallel_size"] = tensor_parallel_size
        if self.max_model_len is not None:
            llm_kwargs["max_model_len"] = int(self.max_model_len)
        if self._resolved_model_name != MODEL_DICT.get(model_name, model_name):
            print(f"[INFO] Resolved local model source: {self._resolved_model_name}")
        self.llm = LLM(
            **llm_kwargs,
        )
        self.tokenizer = self.llm.get_tokenizer()

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float | None = None,
    ) -> str:
        return self.generate_chat(
            [{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def generate_chat(
        self,
        messages: list[dict],
        max_tokens: int = 1024,
        temperature: float | None = None,
    ) -> str:
        try:
            apply_kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if _supports_enable_thinking(self.model_name) and self.enable_thinking is not None:
                apply_kwargs["enable_thinking"] = self.enable_thinking
            rendered_prompt = self.tokenizer.apply_chat_template(messages, **apply_kwargs)
        except Exception:
            rendered_prompt = prompt

        resolved_max_tokens = min(
            int(max_tokens),
            int(self.max_gen_len) if self.max_gen_len is not None else int(max_tokens),
        )
        resolved_temperature = (
            self.default_temperature if temperature is None else float(temperature)
        )
        sampling_params = self._sampling_params_cls(
            temperature=resolved_temperature,
            max_tokens=resolved_max_tokens,
        )
        outputs = self.llm.generate([rendered_prompt], sampling_params, use_tqdm=False)
        if not outputs:
            return DEFAULT_REFUSAL

        first_output = outputs[0]
        self.prompt_tokens += len(getattr(first_output, "prompt_token_ids", []) or [])

        candidates = getattr(first_output, "outputs", [])
        if not candidates:
            return DEFAULT_REFUSAL

        candidate = candidates[0]
        self.completion_tokens += len(getattr(candidate, "token_ids", []) or [])
        return _coerce_text_or_refusal(getattr(candidate, "text", ""))


def build_inference_llm_client(
    model_name: str,
    port: int = 8000,
    *,
    use_vllm_server: bool = True,
    api_use_batch: bool = False,
    gpt4_api_key: str = "",
    gemini_api_key: str = "",
    anthropic_api_key: str = "",
    max_model_len: int | None = None,
    max_gen_len: int | None = None,
    temperature: float = 0.0,
    use_sieve_memory_profile: bool = False,
    enable_thinking: bool | None = None,
    reasoning_effort: str | None = None,
) -> LLMClient:
    normalized = str(model_name).strip().lower()
    if api_use_batch and is_api_model(model_name):
        return DirectAPIBatchInferenceClient(
            model_name,
            openai_api_key=gpt4_api_key,
            gemini_api_key=gemini_api_key,
            anthropic_api_key=anthropic_api_key,
            max_model_len=max_model_len or 8192,
            max_gen_len=max_gen_len or 4096,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )
    if is_api_model(model_name) and ("gemini" in normalized or "claude" in normalized):
        return DirectAPIBatchInferenceClient(
            model_name,
            openai_api_key=gpt4_api_key,
            gemini_api_key=gemini_api_key,
            anthropic_api_key=anthropic_api_key,
            max_model_len=max_model_len or 8192,
            max_gen_len=max_gen_len or 4096,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )
    if _is_azure_api_model_name(model_name):
        return AzureInferenceClient(
            model_name,
            max_gen_len=max_gen_len,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )
    if use_vllm_server:
        return VLLMServerInferenceClient(
            model_name,
            port=port,
            max_model_len=max_model_len,
            temperature=temperature,
            max_gen_len=max_gen_len,
            enable_thinking=enable_thinking,
        )
    return LocalVLLMInferenceClient(
        model_name,
        max_model_len=max_model_len,
        max_gen_len=max_gen_len,
        temperature=temperature,
        use_sieve_memory_profile=use_sieve_memory_profile,
        enable_thinking=enable_thinking,
    )
