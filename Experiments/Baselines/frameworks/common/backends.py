"""Temperature-keyed LLM backend pool + JSON-typed chat helpers.

`LLM_judge/llm_backend.py`'s OpenAIBackend fixes temperature/max_tokens at
construction, so per-call-type decoding is achieved by caching one backend
instance per (temperature, max_tokens) pair — all pointing at the same server.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Tuple

import common  # noqa: F401  (sys.path setup)

# Single-sourced from the existing single-prompt baseline / judge backend.
from baseline_generate import MODEL_REGISTRY, extract_json  # noqa: F401
from llm_backend import LLMBackend, build_backend

logger = logging.getLogger(__name__)


class JSONParseError(RuntimeError):
    """Raised when an LLM response cannot be parsed as JSON after retries."""


class BackendPool:
    """Caches one LLMBackend per (temperature, max_tokens) for a single model."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self._cfg: Dict[str, Any] = dict(cfg)
        self._backends: Dict[Tuple[float, int], LLMBackend] = {}
        self.num_calls: int = 0

    @property
    def cfg(self) -> Dict[str, Any]:
        return dict(self._cfg)

    def backend(self, temperature: float, max_tokens: int) -> LLMBackend:
        key = (round(float(temperature), 3), int(max_tokens))
        if key not in self._backends:
            cfg = dict(self._cfg)
            cfg["temperature"], cfg["max_tokens"] = key
            self._backends[key] = build_backend(cfg)
        return self._backends[key]

    def chat_text(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float,
        max_tokens: int,
    ) -> str:
        self.num_calls += 1
        return self.backend(temperature, max_tokens).chat(system_prompt, user_prompt)

    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        label: str = "call",
    ) -> Dict[str, Any]:
        """chat_text + extract_json, retrying with fresh sampling on parse failure."""
        max_retries = int(self._cfg.get("max_retries", 3))
        retry_delay = float(self._cfg.get("retry_delay", 10.0))
        raw = ""
        for attempt in range(1, max_retries + 1):
            raw = self.chat_text(
                system_prompt, user_prompt,
                temperature=temperature, max_tokens=max_tokens,
            )
            try:
                return extract_json(raw)
            except (ValueError, json.JSONDecodeError) as exc:
                logger.warning(
                    "JSON parse failed for %s (attempt %d/%d): %s",
                    label, attempt, max_retries, exc,
                )
                if attempt < max_retries:
                    time.sleep(retry_delay)
        logger.warning("Raw response for %s (first 1000 chars):\n%s", label, raw[:1000])
        raise JSONParseError(f"Could not parse JSON for {label} after {max_retries} attempts")

    def cleanup(self) -> None:
        for backend in self._backends.values():
            try:
                backend.cleanup()
            except Exception:  # noqa: BLE001 - best-effort cleanup
                pass
        self._backends.clear()
