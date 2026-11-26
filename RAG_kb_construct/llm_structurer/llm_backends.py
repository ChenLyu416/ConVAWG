"""
LLM Backend implementations for different model types.
Supports: HuggingFace Transformers, GGUF (llama.cpp), OpenAI API
"""

from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class LLMBackend(ABC):
    """Abstract base class for LLM backends"""
    
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate text from prompt"""
        pass
    
    @abstractmethod
    def cleanup(self):
        """Clean up resources"""
        pass


class OpenAIBackend(LLMBackend):
    """OpenAI API backend for GPT models"""
    
    def __init__(self, model_name: str, api_key: str, base_url: Optional[str] = None):
        """
        Initialize OpenAI backend.
        
        Args:
            model_name: Model name (e.g., 'gpt-5.2')
            api_key: OpenAI API key
            base_url: Optional custom API base URL
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI backend requires the 'openai' package. "
                "Install it with: pip install openai"
            )
        
        logger.info(f"Initializing OpenAI backend: {model_name}")
        
        self.model_name = model_name
        self.base_url = base_url or ""
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        
        logger.info(f"OpenAI backend initialized for {model_name}")
    
    def generate(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.7) -> str:
        """
        Generate text using OpenAI API.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        
        Returns:
            Generated text
        """
        try:
            # Use max_completion_tokens for newer models (GPT-4o, GPT-5+)
            # Fall back to max_tokens for older models
            _is_deepseek = 'deepseek' in self.base_url.lower() or 'deepseek' in self.model_name.lower()
            messages = [{"role": "user", "content": prompt}]
            if _is_deepseek:
                # /no_think token disables chain-of-thought for DeepSeek reasoning models
                messages = [
                    {"role": "system", "content": "/no_think"},
                    {"role": "user", "content": prompt},
                ]
            completion_params = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "timeout": 60.0  # Add 60 second timeout
            }

            # GPT-4o and GPT-5+ use max_completion_tokens
            if 'gpt-4o' in self.model_name.lower() or 'gpt-5' in self.model_name.lower():
                completion_params["max_completion_tokens"] = max_tokens
            else:
                completion_params["max_tokens"] = max_tokens
            
            logger.info(f"Calling OpenAI API with model {self.model_name}")
            response = self.client.chat.completions.create(**completion_params)
            logger.info(f"Got response from OpenAI API")
            
            content = response.choices[0].message.content
            if content:
                return content.strip()
            else:
                logger.warning(f"OpenAI returned empty content. Finish reason: {response.choices[0].finish_reason}")
                return ""
            
        except Exception as e:
            logger.error(f"OpenAI API error: {type(e).__name__}: {e}")
            raise
    
    def cleanup(self):
        """Clean up OpenAI client resources"""
        if hasattr(self, 'client'):
            del self.client


class HuggingFaceBackend(LLMBackend):
    """HuggingFace Transformers backend (full precision + 8bit quantization)"""
    
    def __init__(self, model_name: str, load_in_8bit: bool = True, device_map: str = "auto",
                 trust_remote_code: bool = True):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        
        logger.info(f"Loading HuggingFace model: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True, trust_remote_code=trust_remote_code)
        
        model_kwargs = {
            "device_map": device_map,
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": trust_remote_code,
        }
        
        if load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            model_kwargs["quantization_config"] = quantization_config
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        # Get the device of the first model parameter for multi-GPU setups
        self.device = next(self.model.parameters()).device
        logger.info(f"Model loaded on {self.device}")
    
    def generate(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.1) -> str:
        import torch
        
        # When device_map="auto", let the model handle device placement
        inputs = self.tokenizer(prompt, return_tensors="pt")
        # Move inputs to the device of the first model parameter
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the prompt from response
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        
        return response
    
    def cleanup(self):
        import torch
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        torch.cuda.empty_cache()


class GGUFBackend(LLMBackend):
    """GGUF backend using llama-cpp-python"""
    
    def __init__(self, model_name: str, gguf_file: Optional[str] = None, n_gpu_layers: int = -1, n_ctx: int = 8192):
        from llama_cpp import Llama
        from huggingface_hub import hf_hub_download, list_repo_files
        import os
        
        logger.info(f"Loading GGUF model: {model_name}")
        
        # Download GGUF file from HuggingFace
        if gguf_file is None:
            # Try to auto-detect GGUF file
            gguf_file = self._find_gguf_file(model_name)
        
        # Download main file
        model_path = hf_hub_download(
            repo_id=model_name,
            filename=gguf_file
        )
        
        # Check if this is a split file (e.g., -00001-of-00004)
        if '-00001-of-' in gguf_file:
            # Download all parts in the split
            base_name = gguf_file.split('-00001-of-')[0]
            total_parts = int(gguf_file.split('-00001-of-')[1].split('.')[0])
            
            logger.info(f"Detected split GGUF with {total_parts} parts, downloading all...")
            model_dir = os.path.dirname(model_path)
            
            for i in range(2, total_parts + 1):
                part_name = f"{base_name}-{i:05d}-of-{total_parts:05d}.gguf"
                part_path = hf_hub_download(
                    repo_id=model_name,
                    filename=part_name
                )
                logger.info(f"Downloaded part {i}/{total_parts}")
        
        logger.info(f"Loading GGUF from: {model_path}")
        
        self.model = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,  # -1 for all layers on GPU
            n_ctx=n_ctx,
            verbose=False
        )
        
        logger.info(f"GGUF model loaded (GPU layers: {n_gpu_layers})")
    
    def _find_gguf_file(self, model_name: str) -> str:
        """Auto-detect GGUF file from repo"""
        from huggingface_hub import list_repo_files
        
        files = list(list_repo_files(model_name))
        gguf_files = [f for f in files if f.endswith('.gguf')]
        
        if not gguf_files:
            raise ValueError(f"No GGUF files found in {model_name}")
        
        # Prefer Q6_K or Q4_K_M
        for pattern in ['Q6_K_L', 'Q6_K', 'Q4_K_M', 'q4_0']:
            for f in gguf_files:
                if pattern in f:
                    logger.info(f"Auto-selected GGUF file: {f}")
                    return f
        
        # Fallback to first GGUF file
        logger.warning(f"Using first GGUF file: {gguf_files[0]}")
        return gguf_files[0]
    
    def generate(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.1, json_mode: bool = False, **kwargs) -> str:
        # Add JSON instruction to prompt if json_mode is enabled
        if json_mode:
            prompt = f"{prompt}\n\nYou must respond with valid JSON only. Do not include any text before or after the JSON."
        
        output = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["</s>", "<|end|>", "<|eot_id|>"],
            echo=False
        )
        
        return output['choices'][0]['text'].strip()
    
    def cleanup(self):
        if hasattr(self, 'model'):
            del self.model


class VLLMBackend(LLMBackend):
    """Client for a vLLM server exposing the OpenAI-compatible API.

    The model is served out-of-process (start it once, share it across all
    pipeline stages), e.g.:

        python -m vllm.entrypoints.openai.api_server \\
            --model <model_name> --port 8000 --tensor-parallel-size 4

    Legacy in-process arguments (tensor_parallel_size, gpu_memory_utilization,
    max_model_len, dtype, ...) are accepted for backward compatibility but are
    now server-side concerns and are ignored by the client.
    """

    DEFAULT_BASE_URL = "http://localhost:8000/v1"

    def __init__(
        self,
        model_name: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        trust_remote_code: bool = True,
        dtype: str = "auto",
        hf_token: Optional[str] = None,
        enforce_eager: bool = False,
        disable_custom_all_reduce: bool = False,
    ):
        """
        Initialize vLLM server client.

        Args:
            model_name: Model identifier as served by the vLLM server
            base_url: vLLM server URL (default: $VLLM_BASE_URL or
                      http://localhost:8000/v1)
            api_key: API key (vLLM servers accept any value; default "EMPTY")
            tensor_parallel_size, gpu_memory_utilization, max_model_len,
            trust_remote_code, dtype, hf_token, enforce_eager,
            disable_custom_all_reduce: legacy in-process options — ignored,
                configure them on the server instead
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "vLLM server backend requires the 'openai' package. "
                "Install it with: pip install openai"
            )

        import os
        self.model_name = model_name
        self.base_url = (
            base_url
            or os.environ.get("VLLM_BASE_URL")
            or self.DEFAULT_BASE_URL
        )

        if tensor_parallel_size != 1 or max_model_len is not None:
            logger.info(
                "VLLMBackend now targets a vLLM server; GPU options "
                "(tensor_parallel_size, gpu_memory_utilization, max_model_len, "
                "dtype) are ignored — set them when launching the server."
            )

        logger.info(f"Initializing vLLM server client: {model_name} @ {self.base_url}")

        self.client = OpenAI(
            api_key=api_key or os.environ.get("VLLM_API_KEY", "EMPTY"),
            base_url=self.base_url,
        )

        # Fail fast with a clear message if the server is not up.
        try:
            self.client.models.list()
        except Exception as e:
            raise ConnectionError(
                f"Cannot reach vLLM server at {self.base_url} ({e}). "
                f"Start it first, e.g.: python -m vllm.entrypoints.openai.api_server "
                f"--model {model_name} --port 8000"
            ) from e

        logger.info(f"vLLM server client initialized for {model_name}")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.95,
        **kwargs
    ) -> str:
        """
        Generate text via the vLLM server's chat completions endpoint.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter

        Returns:
            Generated text
        """
        try:
            # For Qwen3 reasoning models send a /no_think system message
            # to disable chain-of-thought (same behaviour as the old
            # in-process chat() path).
            messages = [{"role": "user", "content": prompt}]
            if 'qwen' in self.model_name.lower():
                messages = [
                    {"role": "system", "content": "/no_think"},
                    {"role": "user", "content": prompt},
                ]

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )

            content = response.choices[0].message.content
            return content.strip() if content else ""

        except Exception as e:
            logger.error(f"vLLM server generation error: {e}")
            raise

    def cleanup(self):
        """Clean up client resources (server keeps running)"""
        if hasattr(self, 'client'):
            del self.client


def create_backend(backend_type: str, model_name: str, **kwargs) -> LLMBackend:
    """Factory function to create appropriate backend"""
    
    backend_type = backend_type.lower()
    
    if backend_type == "huggingface" or backend_type == "transformers":
        return HuggingFaceBackend(model_name, **kwargs)
    elif backend_type == "gguf" or backend_type == "llama-cpp":
        return GGUFBackend(model_name, **kwargs)
    elif backend_type == "openai":
        return OpenAIBackend(model_name, **kwargs)
    elif backend_type == "vllm":
        return VLLMBackend(model_name, **kwargs)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}. Supported: huggingface, gguf, openai, vllm")
