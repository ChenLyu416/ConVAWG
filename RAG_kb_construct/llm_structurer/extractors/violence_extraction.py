"""Violence type classification module using LLM."""

import logging
from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Tuple
from llm_structurer.llm_client import LLMClient, LLMJsonParseError
from llm_structurer.schema import ViolenceType


logger = logging.getLogger(__name__)


class ViolenceExtractor:
    """Classify violence types using LLM."""

    def __init__(self, llm_client: LLMClient, prompts_dir: str = "llm_structurer/prompts"):
        """
        Initialize violence type extractor.

        Args:
            llm_client: LLM client instance
            prompts_dir: Directory containing prompt templates
        """
        self.llm_client = llm_client
        self.prompt_template = self._load_prompt(prompts_dir)
        self.logger = logging.getLogger(__name__)

    def _load_prompt(self, prompts_dir: str) -> str:
        """Load the violence classification prompt template."""
        prompt_path = Path(prompts_dir) / "violence_types.txt"
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()

    def extract(self, case_text: str, return_raw: bool = False) -> Union[List[ViolenceType], Tuple[List[ViolenceType], str]]:
        """
        Classify violence types present in case.

        Args:
            case_text: Raw case narrative text

        Returns:
            List of ViolenceType enums
        """
        self.logger.info("Classifying violence types")
        
        prompt = self.prompt_template.format(case_text=case_text)
        
        raw_response: Optional[str] = None

        try:
            if return_raw:
                response, raw_response = self.llm_client.generate_json(
                    prompt, return_raw=True
                )
            else:
                response = self.llm_client.generate_json(prompt)
            violence_types_raw = response.get('violence_types', [])
            
            violence_types = []
            for vtype in violence_types_raw:
                try:
                    violence_types.append(ViolenceType(vtype))
                except ValueError:
                    self.logger.warning(f"Invalid violence type: {vtype}")
            
            self.logger.info(f"Classified {len(violence_types)} violence types")
            if return_raw:
                return violence_types, raw_response or ""
            return violence_types

        except LLMJsonParseError as exc:
            raw_response = exc.response
            self.logger.error(f"Violence classification failed: {exc}")
        except Exception as e:
            self.logger.error(f"Violence classification failed: {e}")

        fallback = [ViolenceType.OTHER]
        if return_raw:
            return fallback, raw_response or ""
        return fallback

    def extract_chunked(
        self,
        case_text: str,
        chunk_size: int,
        return_raw: bool = False
    ) -> Union[List[ViolenceType], Tuple[List[ViolenceType], List[str]]]:
        """
        Extract violence types from case text by chunking.
        
        Args:
            case_text: Raw case narrative text
            chunk_size: Maximum characters per chunk
            return_raw: Whether to return raw LLM responses
            
        Returns:
            List of ViolenceType objects (or tuple with raw responses)
        """
        self.logger.info(f"Extracting violence types using chunking (chunk_size={chunk_size})")
        
        # Split text into chunks
        chunks = []
        for i in range(0, len(case_text), chunk_size):
            chunks.append(case_text[i:i + chunk_size])
        
        self.logger.info(f"Processing {len(chunks)} chunks")
        
        all_violence_types: set = set()  # Use set to avoid duplicates
        all_raw_responses: List[str] = []
        
        for idx, chunk in enumerate(chunks):
            self.logger.info(f"Processing chunk {idx + 1}/{len(chunks)}")
            
            if return_raw:
                chunk_types, raw_response = self.extract(chunk, return_raw=True)
                all_raw_responses.append(raw_response)
            else:
                chunk_types = self.extract(chunk, return_raw=False)
            
            all_violence_types.update(chunk_types)
        
        violence_list = list(all_violence_types)
        self.logger.info(f"Extracted {len(violence_list)} unique violence types from all chunks")
        
        if return_raw:
            return violence_list, all_raw_responses
        return violence_list
