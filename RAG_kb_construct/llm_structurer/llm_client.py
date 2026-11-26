"""Unified LLM client supporting multiple backends."""

import os
import json
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod
from tenacity import retry, stop_after_attempt, wait_exponential

from llm_structurer.config import Config

# Try to import new backends module
try:
    from llm_structurer.llm_backends import create_backend as create_new_backend
    HAS_NEW_BACKENDS = True
except ImportError:
    HAS_NEW_BACKENDS = False

logger = logging.getLogger(__name__)


class LLMJsonParseError(ValueError):
    """Raised when the LLM returns a response that cannot be parsed as JSON."""

    def __init__(self, message: str, response: str):
        super().__init__(message)
        self.response = response


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
        pass

    @abstractmethod
    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate text for multiple prompts."""
        pass


class OpenAIBackend(LLMBackend):
    """OpenAI API backend."""

    def __init__(self, model_name: str, api_key: str, **kwargs):
        """Initialize OpenAI backend."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")
        
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.default_kwargs = kwargs
        self.logger = logging.getLogger(__name__)

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using OpenAI API."""
        merged_kwargs = {**self.default_kwargs, **kwargs}
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=merged_kwargs.get('temperature', 0.0),
                max_tokens=merged_kwargs.get('max_tokens', 4096),
                response_format={"type": "json_object"} if merged_kwargs.get('json_mode', True) else None
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            raise

    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate text for multiple prompts."""
        results = []
        for prompt in prompts:
            result = self.generate(prompt, **kwargs)
            results.append(result)
        return results


class HuggingFaceBackend(LLMBackend):
    """HuggingFace Transformers backend."""

    def __init__(self, model_name: str, device: str = "cuda", load_in_8bit: bool = False,
                 cuda_visible_devices: Optional[str] = None, trust_remote_code: bool = True,
                 torch_dtype: str = "bfloat16", **kwargs):
        """Initialize HuggingFace backend."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
            import torch
        except ImportError:
            raise ImportError("transformers and torch required. Install with: pip install transformers torch bitsandbytes accelerate")

        # Pin specific GPUs before model loading (mirrors judge's HFTransformersBackend)
        if cuda_visible_devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.default_kwargs = kwargs
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Loading HuggingFace model: {model_name}")
        self.logger.info(f"Device: {self.device}, 8-bit quantization: {load_in_8bit}, trust_remote_code: {trust_remote_code}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        
        # Resolve dtype (bfloat16 recommended for Qwen3.5 MoE)
        _dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        _resolved_dtype = _dtype_map.get(torch_dtype, torch.bfloat16)

        # Configure model loading with optional 8-bit quantization
        model_kwargs = {"trust_remote_code": trust_remote_code}
        
        # When using device_map="auto", accelerate handles device placement
        # and we cannot specify device in pipeline
        use_device_map = False
        
        if load_in_8bit and self.device == "cuda":
            # Use 8-bit quantization for memory efficiency with large models
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False
            )
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto"
            model_kwargs["torch_dtype"] = _resolved_dtype
            use_device_map = True
            self.logger.info("Using 8-bit quantization with bitsandbytes and device_map='auto'")
        elif self.device == "cuda":
            model_kwargs["device_map"] = "auto"
            model_kwargs["torch_dtype"] = _resolved_dtype
            use_device_map = True
        else:
            model_kwargs["torch_dtype"] = torch.float32
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        # When using device_map="auto" with accelerate, don't specify device in pipeline
        pipeline_kwargs = {
            "model": self.model,
            "tokenizer": self.tokenizer,
        }
        
        # Only add device if NOT using device_map
        if not use_device_map:
            pipeline_kwargs["device"] = 0 if self.device == "cuda" else -1
        
        self.pipeline = pipeline(
            "text-generation",
            **pipeline_kwargs
        )

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using HuggingFace pipeline."""
        merged_kwargs = {**self.default_kwargs, **kwargs}
        
        # Add JSON formatting instruction to prompt if json_mode is requested
        if merged_kwargs.get('json_mode', True):
            prompt = f"{prompt}\n\nYou must respond with valid JSON only. Do not include any text before or after the JSON."

        outputs = self.pipeline(
            prompt,
            max_new_tokens=merged_kwargs.get('max_tokens', 1024),
            truncation=True,
            temperature=merged_kwargs.get('temperature', 0.0),
            do_sample=merged_kwargs.get('temperature', 0.0) > 0,
            return_full_text=False
        )
        
        generated_text = outputs[0]['generated_text']
        
        # Extract JSON from response if json_mode was requested
        if merged_kwargs.get('json_mode', True):
            generated_text = self._extract_json(generated_text)
        
        return generated_text
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from text that may contain markdown or extra content."""
        import re
        
        # Remove markdown code blocks
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*$', '', text)
        
        # Try to find JSON object or array
        json_match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
        if json_match:
            return json_match.group(1)
        
        return text.strip()

    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate text for multiple prompts."""
        merged_kwargs = {**self.default_kwargs, **kwargs}
        
        outputs = self.pipeline(
            prompts,
            max_new_tokens=merged_kwargs.get('max_tokens', 1024),
            truncation=True,
            temperature=merged_kwargs.get('temperature', 0.0),
            do_sample=merged_kwargs.get('temperature', 0.0) > 0,
            return_full_text=False,
            batch_size=merged_kwargs.get('batch_size', 4)
        )
        
        return [output[0]['generated_text'] for output in outputs]


class VLLMBackend(LLMBackend):
    """vLLM backend for local deployment."""

    def __init__(self, model_name: str, api_base: str, **kwargs):
        """Initialize vLLM backend."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required for vLLM. Install with: pip install openai")
        
        self.client = OpenAI(
            api_key="EMPTY",  # vLLM doesn't require API key
            base_url=api_base
        )
        self.model_name = model_name
        self.default_kwargs = kwargs
        self.logger = logging.getLogger(__name__)

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using vLLM endpoint."""
        merged_kwargs = {**self.default_kwargs, **kwargs}
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=merged_kwargs.get('temperature', 0.0),
                max_tokens=merged_kwargs.get('max_tokens', 4096)
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"vLLM API error: {e}")
            raise

    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate text for multiple prompts."""
        results = []
        for prompt in prompts:
            result = self.generate(prompt, **kwargs)
            results.append(result)
        return results


class LLMClient:
    """Unified LLM client with retry logic and batching support."""

    def __init__(self, config: Config):
        """
        Initialize LLM client.

        Args:
            config: Configuration object
        """
        self.config = config
        self.backend = self._initialize_backend()
        self.logger = logging.getLogger(__name__)
        self.json_retry_attempts = getattr(config, "json_retry_attempts", 2)

    def _initialize_backend(self) -> LLMBackend:
        """Initialize the appropriate backend based on configuration."""
        backend_type = self.config.llm_backend
        model_name = self.config.model_name
        
        backend_kwargs = {
            'temperature': self.config.temperature,
            'max_tokens': self.config.max_tokens
        }
        
        # Check if using new backends (GGUF)
        if HAS_NEW_BACKENDS and backend_type in ["gguf", "llama-cpp"]:
            # Use GGUF backend system
            gguf_file = getattr(self.config, 'gguf_file', None)
            n_gpu_layers = getattr(self.config, 'n_gpu_layers', -1)
            n_ctx = getattr(self.config, 'n_ctx', 8192)
            return create_new_backend(
                "gguf",
                model_name,
                gguf_file=gguf_file,
                n_gpu_layers=n_gpu_layers,
                n_ctx=n_ctx
            )

        if backend_type == "openai":
            api_key = self.config.get_openai_api_key()
            return OpenAIBackend(model_name, api_key, **backend_kwargs)
        
        elif backend_type == "huggingface":
            device = getattr(self.config, 'device', 'cuda')
            load_in_8bit = getattr(self.config, 'load_in_8bit', False)
            cuda_visible_devices = self.config.get('llm.cuda_visible_devices')
            trust_remote_code = self.config.get('llm.trust_remote_code', True)
            torch_dtype = self.config.get('llm.torch_dtype', 'bfloat16')
            return HuggingFaceBackend(
                model_name, device=device, load_in_8bit=load_in_8bit,
                cuda_visible_devices=cuda_visible_devices,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch_dtype,
                **backend_kwargs
            )
        
        elif backend_type == "vllm":
            api_base = self.config.api_base
            if not api_base:
                raise ValueError("api_base required for vLLM backend")
            return VLLMBackend(model_name, api_base, **backend_kwargs)
        
        else:
            raise ValueError(f"Unsupported backend: {backend_type}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def generate(self, prompt: str, json_mode: bool = True, **kwargs) -> str:
        """
        Generate text from prompt with retry logic.

        Args:
            prompt: Input prompt
            json_mode: Whether to enforce JSON output
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        self.logger.debug(f"Generating with prompt length: {len(prompt)}")
        return self.backend.generate(prompt, json_mode=json_mode, **kwargs)

    def generate_json(self, prompt: str, return_raw: bool = False, **kwargs) -> Union[Dict[str, Any], tuple[Dict[str, Any], str]]:
        """
        Generate and parse JSON response with automatic repair attempts.

        Args:
            prompt: Input prompt
            return_raw: Whether to also return the original LLM text response
            **kwargs: Additional generation parameters

        Returns:
            Parsed JSON dictionary
        """
        original_prompt = prompt
        current_prompt = prompt
        attempts = max(1, self.json_retry_attempts)
        last_error: Optional[Exception] = None
        last_response: Optional[str] = None

        for attempt in range(1, attempts + 1):
            response = self.generate(current_prompt, json_mode=True, **kwargs)
            last_response = response
            
            # Always save raw response before parsing
            self._save_raw_response(response, attempt)

            cleaned_response = self._clean_json_response(response)

            try:
                parsed = json.loads(cleaned_response)
                if return_raw:
                    return parsed, response
                return parsed
            except json.JSONDecodeError as exc:
                last_error = exc
                self.logger.error(
                    "Failed to parse JSON response (attempt %s/%s): %s",
                    attempt,
                    attempts,
                    exc,
                )
                self.logger.debug("Response: %s", response)

                if attempt == attempts:
                    break

                # Prepare a stricter prompt for the next attempt
                current_prompt = self._build_json_retry_prompt(original_prompt, cleaned_response)

        snippet = (last_response or "")[:200]
        raise LLMJsonParseError(
            f"Could not parse JSON from LLM response after {attempts} attempts: {snippet}",
            last_response or "",
        )

    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Generate text for multiple prompts.

        Args:
            prompts: List of prompts
            **kwargs: Additional generation parameters

        Returns:
            List of generated texts
        """
        batch_size = self.config.batch_size
        results = []
        
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            self.logger.info(f"Processing batch {i // batch_size + 1} of {len(prompts) // batch_size + 1}")
            batch_results = self.backend.generate_batch(batch, **kwargs)
            results.extend(batch_results)
        
        return results

    def generate_batch_json(self, prompts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """
        Generate and parse JSON responses for multiple prompts.

        Args:
            prompts: List of prompts
            **kwargs: Additional generation parameters

        Returns:
            List of parsed JSON dictionaries
        """
        responses = self.generate_batch(prompts, json_mode=True, **kwargs)
        
        results = []
        for response in responses:
            try:
                cleaned = self._clean_json_response(response)
                results.append(json.loads(cleaned))
            except json.JSONDecodeError:
                self.logger.error(f"Failed to parse JSON in batch: {response[:100]}")
                results.append({})
        
        return results

    def _clean_json_response(self, response: str) -> str:
        """Strip markdown/code fences and isolate potential JSON payload."""
        response_cleaned = response.strip()
        if response_cleaned.startswith("```json"):
            response_cleaned = response_cleaned[7:]
        if response_cleaned.endswith("```"):
            response_cleaned = response_cleaned[:-3]

        response_cleaned = response_cleaned.strip()

        # If the response contains JSON among other text, extract the first JSON structure
        if response_cleaned:
            import re

            match = re.search(r'(\{.*\}|\[.*\])', response_cleaned, re.DOTALL)
            if match:
                response_cleaned = match.group(1)

        return response_cleaned

    def _build_json_retry_prompt(self, original_prompt: str, invalid_response: str) -> str:
        """Create a stricter prompt that reminds the model to output valid JSON."""
        invalid_response = invalid_response.strip()
        if len(invalid_response) > 1500:
            invalid_response = invalid_response[:1500] + "\n..."

        reminder = (
            "\n\nIMPORTANT: Your previous answer was not valid JSON. "
            "Regenerate the answer strictly as valid JSON only. "
            "Do not include explanations, bullet points, or markdown fences. "
            "Respond with JSON matching the schema exactly."
        )

        return (
            f"{original_prompt}{reminder}\n\nPrevious invalid answer (for reference only):\n"
            f"{invalid_response}"
        )
    
    def _save_raw_response(self, response: str, attempt: int = 1) -> None:
        """Save raw LLM response to file before parsing."""
        try:
            # Create raw_responses directory in outputs
            raw_dir = Path("outputs") / "raw_responses"
            raw_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"raw_response_{timestamp}_attempt{attempt}.txt"
            filepath = raw_dir / filename
            
            # Write response to file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"=== RAW LLM RESPONSE (Attempt {attempt}) ===\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Response Length: {len(response)} characters\n")
                f.write("=" * 80 + "\n\n")
                f.write(response)
            
            self.logger.debug(f"Saved raw response to {filepath}")
        except Exception as e:
            self.logger.warning(f"Failed to save raw response: {e}")
