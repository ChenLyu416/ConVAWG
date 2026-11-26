"""
Lightweight LLM backend for the dialogue judge.
Supports OpenAI-compatible REST APIs (GPT, vLLM server) and native vLLM.
"""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class LLMBackend(ABC):
    @abstractmethod
    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """Send a chat request and return the assistant reply."""

    def cleanup(self) -> None:
        pass


# ---------------------------------------------------------------------------
# OpenAI / vLLM-as-OpenAI backend
# ---------------------------------------------------------------------------

class OpenAIBackend(LLMBackend):
    """
    Works for:
    - Real OpenAI API  (base_url = "https://api.openai.com/v1")
    - vLLM server      (base_url = "http://localhost:8000/v1")
    """

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.1,
        max_tokens: int = 1024,
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Install openai: pip install openai")

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
        self.client = OpenAI(api_key=resolved_key, base_url=base_url)
        self.base_url = base_url
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        logger.info(f"OpenAI backend ready: {model_name} @ {base_url}")

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        _is_deepseek = "deepseek" in self.base_url.lower() or "deepseek" in self.model_name.lower()
        # DeepSeek: prepend /no_think so the model skips chain-of-thought
        if _is_deepseek:
            system_prompt = "/no_think\n" + system_prompt
        params: dict = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
        }
        # GPT-4o and GPT-5.x use max_completion_tokens
        if any(tag in self.model_name.lower() for tag in ("gpt-4o", "gpt-5", "o1", "o3")):
            params["max_completion_tokens"] = self.max_tokens
        else:
            params["max_tokens"] = self.max_tokens
        # Qwen3-series vLLM server: disable thinking mode so output is clean JSON
        if "qwen3" in self.model_name.lower():
            params["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(**params)
                content = resp.choices[0].message.content
                return (content or "").strip()
            except Exception as exc:
                logger.warning(f"LLM call failed (attempt {attempt}/{self.max_retries}): {exc}")
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
                else:
                    raise

    def cleanup(self) -> None:
        if hasattr(self, "client"):
            del self.client


# ---------------------------------------------------------------------------
# Gemini backend (OpenAI-compatible endpoint)
# ---------------------------------------------------------------------------

_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
_GEMINI_DEFAULT_MODEL = "gemini-3.1-flash-lite"


class GeminiBackend(OpenAIBackend):
    """
    Google Gemini via the OpenAI-compatible endpoint.
    Requires GEMINI_API_KEY env var or api_key in config.
    Default model: gemini-3.1-flash-lite (stable).
    """

    def __init__(
        self,
        model_name: str = _GEMINI_DEFAULT_MODEL,
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ) -> None:
        resolved_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Gemini backend requires an API key. "
                "Set GEMINI_API_KEY env var or api_key in the config."
            )
        super().__init__(
            model_name=model_name,
            api_key=resolved_key,
            base_url=_GEMINI_BASE_URL,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )


# ---------------------------------------------------------------------------
# Native vLLM backend (loads model in-process — no server required)
# ---------------------------------------------------------------------------

class VLLMNativeBackend(LLMBackend):
    """
    Runs a model directly via vLLM (no external server).
    Suitable for local Qwen3 / Qwen3.5 MoE models.

    For Qwen3-series models, thinking mode is disabled by default so the judge
    returns clean structured output rather than <think>...</think> blocks.
    Set enable_thinking=True in config to re-enable it.
    """

    # Architectures not yet natively registered in vLLM that should be routed
    # through the generic HuggingFace Transformers fallback backend.
    # Note: Qwen3_5Moe uses a vision-language config schema and linear attention
    # that is incompatible with the native Qwen3MoeForCausalLM vLLM kernel.
    _TRANSFORMERS_FALLBACK_ARCHS: frozenset[str] = frozenset({
        "Qwen3_5MoeForConditionalGeneration",
        "Qwen3_5MoeForCausalLM",
    })

    # For architectures that go through model_impl='transformers', vLLM
    # mis-classifies them as multimodal if the class name contains
    # 'ForConditionalGeneration'.  Map them to a CausalLM alias so vLLM picks
    # TransformersMoEForCausalLM (text-only) instead of the multimodal wrapper.
    _TRANSFORMERS_ARCH_ALIAS: dict[str, str] = {
        "Qwen3_5MoeForConditionalGeneration": "Qwen3_5MoeForCausalLM",
    }

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        max_num_seqs: int = 256,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        max_retries: int = 3,
        retry_delay: float = 5.0,
        enable_thinking: bool = False,
        cuda_visible_devices: Optional[str] = None,
    ) -> None:
        try:
            from vllm import LLM, SamplingParams  # noqa: F401
        except ImportError:
            raise ImportError("Install vllm: pip install vllm")

        from vllm import LLM, SamplingParams
        self._SamplingParams = SamplingParams

        # Pin specific GPUs before vLLM spawns workers (must be set before LLM())
        if cuda_visible_devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
            logger.info(f"CUDA_VISIBLE_DEVICES set to: {cuda_visible_devices}")

        # Detect whether the model needs the Transformers fallback backend.
        model_impl, hf_overrides = self._resolve_model_impl(model_name)

        if model_impl == "transformers":
            self._patch_attention_backend_compat(model_name, hf_overrides)

        logger.info(
            f"Loading vLLM model: {model_name}  "
            f"(tp={tensor_parallel_size}, gpu_mem={gpu_memory_utilization}, "
            f"max_num_seqs={max_num_seqs}, model_impl={model_impl}"
            + (f", hf_overrides={hf_overrides}" if hf_overrides else "") + ")"
        )
        llm_kwargs: dict = dict(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
            trust_remote_code=True,
            dtype="auto",
            model_impl=model_impl,
        )
        if hf_overrides:
            llm_kwargs["hf_overrides"] = hf_overrides
        self.llm = LLM(**llm_kwargs)
        self.tokenizer = self.llm.get_tokenizer()
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.enable_thinking = enable_thinking
        logger.info(f"vLLM native backend ready: {model_name}  thinking={enable_thinking}  model_impl={model_impl}")

    @classmethod
    def _resolve_model_impl(cls, model_name: str) -> tuple[str, dict]:
        """Return (model_impl, hf_overrides) for loading this model in vLLM."""
        try:
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            archs = getattr(cfg, "architectures", []) or []
            if any(a in cls._TRANSFORMERS_FALLBACK_ARCHS for a in archs):
                # Use CausalLM alias to prevent vLLM from routing through the
                # multimodal pipeline (which crashes on text-only inference).
                aliased = [cls._TRANSFORMERS_ARCH_ALIAS.get(a, a) for a in archs]
                hf_overrides = {"architectures": aliased} if aliased != archs else {}
                logger.info(
                    f"Architecture {archs} → transformers fallback"
                    + (f" (alias: {aliased})" if hf_overrides else "")
                )
                return "transformers", hf_overrides
        except Exception as exc:
            logger.debug(f"Could not pre-read model config for {model_name}: {exc}")
        return "auto", {}

    @classmethod
    def _patch_attention_backend_compat(cls, model_name: str, hf_overrides: dict) -> None:
        """Patch _supports_attention_backend=True on classes that need it for vLLM's
        Transformers backend compatibility check."""
        try:
            from transformers import AutoConfig
            import transformers as _transformers
            cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            # Use aliased arch names if overridden
            archs = hf_overrides.get("architectures") or (getattr(cfg, "architectures", []) or [])
            for arch in archs:
                model_cls = getattr(_transformers, arch, None)
                if model_cls is None:
                    continue
                if not getattr(model_cls, "_supports_attention_backend", True):
                    logger.info(f"Patching {arch}._supports_attention_backend = True")
                    model_cls._supports_attention_backend = True
        except Exception as exc:
            logger.debug(f"_patch_attention_backend_compat skipped: {exc}")

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]

        # apply_chat_template kwargs — Qwen3 supports enable_thinking
        template_kwargs: dict = {"tokenize": False, "add_generation_prompt": True}
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                **template_kwargs,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
            # Model tokenizer does not support enable_thinking — fall back
            prompt = self.tokenizer.apply_chat_template(messages, **template_kwargs)

        sampling_params = self._SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        for attempt in range(1, self.max_retries + 1):
            try:
                outputs = self.llm.generate([prompt], sampling_params)
                text = outputs[0].outputs[0].text or ""
                # Strip any residual <think>...</think> block just in case
                if not self.enable_thinking and "<think>" in text:
                    import re
                    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
                return text.strip()
            except Exception as exc:
                logger.warning(f"vLLM call failed (attempt {attempt}/{self.max_retries}): {exc}")
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
                else:
                    raise

    def cleanup(self) -> None:
        if hasattr(self, "llm"):
            del self.llm


# ---------------------------------------------------------------------------
# HuggingFace Transformers backend (no vLLM — direct model.generate())
# ---------------------------------------------------------------------------

class HFTransformersBackend(LLMBackend):
    """
    Runs a model directly via HuggingFace Transformers (no vLLM server required).
    Suitable for architectures that vLLM does not yet natively support, such as
    Qwen3.5 MoE (Qwen3_5MoeForConditionalGeneration).

    Uses device_map="auto" so the model is automatically sharded across all
    GPUs visible in CUDA_VISIBLE_DEVICES.
    """

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.9,
        max_tokens: int = 32000,
        max_retries: int = 3,
        retry_delay: float = 5.0,
        cuda_visible_devices: Optional[str] = None,
        torch_dtype: str = "bfloat16",
    ) -> None:
        try:
            import torch
            import transformers as _transformers
        except ImportError:
            raise ImportError("Install transformers and torch.")

        if cuda_visible_devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
            logger.info(f"CUDA_VISIBLE_DEVICES set to: {cuda_visible_devices}")

        _dtype = getattr(torch, torch_dtype, torch.bfloat16)

        logger.info(f"Loading HF Transformers model: {model_name}  dtype={torch_dtype}")

        self.tokenizer = _transformers.AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

        # Try the specific architecture class first so that models with non-CausalLM
        # class names (e.g. "ForConditionalGeneration") load correctly.
        try:
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            archs = getattr(cfg, "architectures", []) or []
            model_cls = None
            for arch in archs:
                model_cls = getattr(_transformers, arch, None)
                if model_cls is not None:
                    break
        except Exception:
            model_cls = None

        if model_cls is None:
            model_cls = _transformers.AutoModelForCausalLM

        self.model = model_cls.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=_dtype,
            trust_remote_code=True,
        )
        self.model.eval()
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        logger.info(f"HF Transformers backend ready: {model_name}")

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        import torch

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # apply_chat_template for text-only input
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        inputs = self.tokenizer(prompt, return_tensors="pt")
        # Move inputs to the same device as the first model parameter
        first_param = next(self.model.parameters())
        inputs = {k: v.to(first_param.device) for k, v in inputs.items()}

        for attempt in range(1, self.max_retries + 1):
            try:
                with torch.no_grad():
                    out_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_tokens,
                        do_sample=(self.temperature > 0),
                        temperature=self.temperature if self.temperature > 0 else None,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                # Decode only the newly generated tokens
                new_ids = out_ids[0][inputs["input_ids"].shape[1]:]
                return self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            except Exception as exc:
                logger.warning(f"HF generate failed (attempt {attempt}/{self.max_retries}): {exc}")
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
                else:
                    raise

    def cleanup(self) -> None:
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_OPENAI_BASE_URL = "https://api.openai.com/v1"
_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
_DEEPSEEK_DEFAULT_MODEL = "deepseek-v4-flash"


def build_backend(cfg: dict) -> LLMBackend:
    """
    Build an LLM backend from a config dict.

    Expected keys (matching judge_config.yaml):
      backend, model_name, temperature, max_tokens, base_url, api_key,
      max_retries, retry_delay

    Supported backend values:
      "openai"      — OpenAI API (default base_url: https://api.openai.com/v1)
      "vllm"        — vLLM server (point base_url to the server)
      "deepseek"    — DeepSeek API (default base_url: https://api.deepseek.com)
                      API key read from DEEPSEEK_API_KEY env var or api_key config.
      "gemini"      — Google Gemini via OpenAI-compatible endpoint.
                      API key read from GEMINI_API_KEY env var or api_key config.
                      Default model: gemini-3.1-flash-lite
      "vllm_native" — Load model in-process via vLLM (no server needed).
                      Uses tensor_parallel_size and gpu_memory_utilization from config.
    """
    backend_type: str = cfg.get("backend", "openai").lower()

    if backend_type in ("openai", "vllm"):
        return OpenAIBackend(
            model_name=cfg["model_name"],
            api_key=cfg.get("api_key"),
            base_url=cfg.get("base_url", "https://api.openai.com/v1"),
            temperature=cfg.get("temperature", 0.1),
            max_tokens=cfg.get("max_tokens", 1024),
            max_retries=cfg.get("max_retries", 3),
            retry_delay=cfg.get("retry_delay", 5.0),
        )

    if backend_type == "deepseek":
        resolved_key = (
            cfg.get("api_key")
            or os.environ.get("DEEPSEEK_API_KEY")
        )
        if not resolved_key:
            raise ValueError(
                "DeepSeek backend requires an API key. "
                "Set DEEPSEEK_API_KEY env var or api_key in the config."
            )
        # Only use base_url from config if it has been changed from the OpenAI
        # default — otherwise the shared config value would silently send
        # DeepSeek requests to api.openai.com.
        configured_url = cfg.get("base_url", "")
        base_url = (
            configured_url
            if configured_url and configured_url != _OPENAI_BASE_URL
            else _DEEPSEEK_BASE_URL
        )
        return OpenAIBackend(
            model_name=cfg.get("model_name", _DEEPSEEK_DEFAULT_MODEL),
            api_key=resolved_key,
            base_url=base_url,
            temperature=cfg.get("temperature", 0.1),
            max_tokens=cfg.get("max_tokens", 1024),
            max_retries=cfg.get("max_retries", 3),
            retry_delay=cfg.get("retry_delay", 5.0),
        )

    if backend_type == "gemini":
        return GeminiBackend(
            model_name=cfg.get("model_name", _GEMINI_DEFAULT_MODEL),
            api_key=cfg.get("api_key"),
            temperature=cfg.get("temperature", 0.1),
            max_tokens=cfg.get("max_tokens", 1024),
            max_retries=cfg.get("max_retries", 3),
            retry_delay=cfg.get("retry_delay", 5.0),
        )

    if backend_type == "vllm_native":
        return VLLMNativeBackend(
            model_name=cfg["model_name"],
            tensor_parallel_size=cfg.get("tensor_parallel_size", 1),
            gpu_memory_utilization=cfg.get("gpu_memory_utilization", 0.85),
            max_num_seqs=cfg.get("max_num_seqs", 256),
            temperature=cfg.get("temperature", 0.1),
            max_tokens=cfg.get("max_tokens", 1024),
            max_retries=cfg.get("max_retries", 3),
            retry_delay=cfg.get("retry_delay", 5.0),
            enable_thinking=cfg.get("enable_thinking", False),
            cuda_visible_devices=cfg.get("cuda_visible_devices"),
        )

    if backend_type == "hf_transformers":
        return HFTransformersBackend(
            model_name=cfg["model_name"],
            temperature=cfg.get("temperature", 0.9),
            max_tokens=cfg.get("max_tokens", 32000),
            max_retries=cfg.get("max_retries", 3),
            retry_delay=cfg.get("retry_delay", 5.0),
            cuda_visible_devices=cfg.get("cuda_visible_devices"),
            torch_dtype=cfg.get("torch_dtype", "bfloat16"),
        )

    raise ValueError(
        f"Unsupported backend: {backend_type!r}. "
        "Use 'openai', 'vllm', 'deepseek', 'gemini', 'vllm_native', or 'hf_transformers'."
    )
