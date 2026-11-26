"""Summary generation module using LLM."""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union
from llm_structurer.llm_client import LLMClient, LLMJsonParseError


logger = logging.getLogger(__name__)


class SummaryExtractor:
    """Generate case summaries using LLM."""

    def __init__(self, llm_client: LLMClient, prompts_dir: str = "llm_structurer/prompts"):
        """
        Initialize summary extractor.

        Args:
            llm_client: LLM client instance
            prompts_dir: Directory containing prompt templates
        """
        self.llm_client = llm_client
        self.prompt_template = self._load_prompt(prompts_dir)
        self.logger = logging.getLogger(__name__)

    def _load_prompt(self, prompts_dir: str) -> str:
        """Load the summary generation prompt template."""
        prompt_path = Path(prompts_dir) / "summary.txt"
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()

    def extract(
        self,
        case_text: str,
        return_raw: bool = False,
    ) -> Union[Optional[str], Tuple[Optional[str], str]]:
        """
        Generate a summary of the case.

        Args:
            case_text: Raw case narrative text

        Returns:
            Summary string or None
        """
        self.logger.info("Generating case summary")
        
        prompt = self.prompt_template.format(case_text=case_text)
        
        raw_response: Optional[str] = None

        try:
            if return_raw:
                response, raw_response = self.llm_client.generate_json(
                    prompt, return_raw=True
                )
            else:
                response = self.llm_client.generate_json(prompt)
            summary = response.get('summary', '')
            
            if summary:
                self.logger.info(f"Generated summary ({len(summary)} chars)")
                if return_raw:
                    return summary, raw_response or ""
                return summary
            else:
                self.logger.warning("Empty summary returned")
                if return_raw:
                    return None, raw_response or ""
                return None
            
        except LLMJsonParseError as exc:
            raw_response = exc.response
            self.logger.error(f"Summary generation failed: {exc}")
        except Exception as e:
            self.logger.error(f"Summary generation failed: {e}")

        if return_raw:
            return None, raw_response or ""
        return None
