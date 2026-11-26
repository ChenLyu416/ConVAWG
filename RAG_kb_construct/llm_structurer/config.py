"""Configuration management for LLM Structurer."""

import os
import yaml
from typing import Any, Dict, Optional
from pathlib import Path


class Config:
    """Configuration manager for the LLM structurer system."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.

        Args:
            config_path: Path to YAML config file. If None, uses default.
        """
        if config_path is None:
            config_path = "configs/qwen_14b_gguf.yaml"
        
        self.config_path = config_path
        self.config = self._load_config()
        self._validate_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        path = Path(self.config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def _validate_config(self):
        """Validate required configuration fields."""
        required_sections = ['llm', 'processing', 'paths']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required config section: {section}")
        
        required_llm_fields = ['backend', 'model_name']
        for field in required_llm_fields:
            if field not in self.config['llm']:
                raise ValueError(f"Missing required LLM config field: {field}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key: Configuration key (e.g., 'llm.model_name')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value

    @property
    def llm_backend(self) -> str:
        """Get LLM backend type."""
        return self.config['llm']['backend']

    @property
    def model_name(self) -> str:
        """Get model name."""
        return self.config['llm']['model_name']

    @property
    def api_base(self) -> Optional[str]:
        """Get API base URL for vLLM."""
        return self.config['llm'].get('api_base')

    @property
    def gguf_file(self) -> Optional[str]:
        """Get GGUF file name for GGUF backend."""
        return self.config['llm'].get('gguf_file')

    @property
    def n_gpu_layers(self) -> int:
        """Get number of GPU layers for GGUF backend."""
        return self.config['llm'].get('n_gpu_layers', -1)

    @property
    def n_ctx(self) -> int:
        """Get context window size for GGUF backend."""
        return self.config['llm'].get('n_ctx', 8192)

    @property
    def max_tokens(self) -> int:
        """Get max tokens for generation."""
        return self.config['llm'].get('max_tokens', 4096)

    @property
    def temperature(self) -> float:
        """Get temperature for generation."""
        return self.config['llm'].get('temperature', 0.0)

    @property
    def batch_size(self) -> int:
        """Get batch size for processing."""
        return self.config['llm'].get('batch_size', 5)

    @property
    def max_retries(self) -> int:
        """Get maximum number of retries."""
        return self.config['llm'].get('max_retries', 3)

    @property
    def retry_delay(self) -> int:
        """Get retry delay in seconds."""
        return self.config['llm'].get('retry_delay', 2)

    @property
    def json_retry_attempts(self) -> int:
        """Number of attempts to repair invalid JSON responses."""
        return self.config['llm'].get('json_retry_attempts', 2)

    @property
    def chunk_size(self) -> int:
        """Get chunk size for long documents."""
        return self.config['processing'].get('chunk_size', 2000)

    @property
    def enable_storage(self) -> bool:
        """Check if database storage is enabled."""
        return self.config['processing'].get('enable_storage', True)

    @property
    def output_formats(self) -> list:
        """Get output formats."""
        return self.config['processing'].get('output_formats', ['json', 'jsonl'])

    @property
    def prompts_dir(self) -> str:
        """Get prompts directory path."""
        return self.config['paths'].get('prompts_dir', 'llm_structurer/prompts')

    @property
    def output_dir(self) -> str:
        """Get output directory path."""
        return self.config['paths'].get('output_dir', 'outputs')

    @property
    def db_path(self) -> str:
        """Get database path."""
        return self.config['paths'].get('db_path', 'outputs/cases.db')

    def get_openai_api_key(self) -> str:
        """Get OpenAI API key from environment."""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key and self.llm_backend == 'openai':
            raise ValueError("OPENAI_API_KEY environment variable not set")
        return api_key

    def get_hf_token(self) -> Optional[str]:
        """Get HuggingFace token from environment."""
        return os.getenv('HF_TOKEN')
