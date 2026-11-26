"""Actor extraction module using LLM."""

import logging
from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Tuple
from llm_structurer.llm_client import LLMClient, LLMJsonParseError
from llm_structurer.schema import Actor, ActorRole


logger = logging.getLogger(__name__)


class ActorExtractor:
    """Extract actors from case text using LLM."""

    def __init__(self, llm_client: LLMClient, prompts_dir: str = "llm_structurer/prompts"):
        """
        Initialize actor extractor.

        Args:
            llm_client: LLM client instance
            prompts_dir: Directory containing prompt templates
        """
        self.llm_client = llm_client
        self.prompt_template = self._load_prompt(prompts_dir)
        self.logger = logging.getLogger(__name__)

    def _load_prompt(self, prompts_dir: str) -> str:
        """Load the actor extraction prompt template."""
        prompt_path = Path(prompts_dir) / "actors.txt"
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()

    def extract(self, case_text: str, return_raw: bool = False) -> Union[List[Actor], Tuple[List[Actor], str]]:
        """
        Extract actors from case text.

        Args:
            case_text: Raw case narrative text

        Returns:
            List of Actor objects
        """
        self.logger.info("Extracting actors from case text")
        
        prompt = self.prompt_template.format(case_text=case_text)
        
        raw_response: Optional[str] = None

        try:
            if return_raw:
                response, raw_response = self.llm_client.generate_json(
                    prompt, return_raw=True
                )
            else:
                response = self.llm_client.generate_json(prompt)
            actors_data = response.get('actors', [])
            
            actors = []
            for actor_dict in actors_data:
                try:
                    actor = Actor(**actor_dict)
                    actors.append(actor)
                except Exception as e:
                    self.logger.error(f"Failed to parse actor: {e}")
                    self.logger.debug(f"Actor data: {actor_dict}")
            
            self.logger.info(f"Extracted {len(actors)} actors")
            if return_raw:
                return actors, raw_response or ""
            return actors

        except LLMJsonParseError as exc:
            raw_response = exc.response
            self.logger.error(f"Actor extraction failed: {exc}")
        except Exception as e:
            self.logger.error(f"Actor extraction failed: {e}")

        fallback = self._fallback_extraction(case_text)
        if return_raw:
            return fallback, raw_response or ""
        return fallback

    def _fallback_extraction(self, case_text: str) -> List[Actor]:
        """
        Fallback heuristic for actor extraction if LLM fails.

        Args:
            case_text: Raw case text

        Returns:
            List of Actor objects (minimal)
        """
        self.logger.warning("Using fallback actor extraction")
        
        actors = []
        
        if "victim" in case_text.lower():
            actors.append(Actor(
                actor_id="actor_victim",
                name="Victim",
                role=ActorRole.VICTIM,
                mentions=[]
            ))
        
        if "defendant" in case_text.lower() or "perpetrator" in case_text.lower():
            actors.append(Actor(
                actor_id="actor_perpetrator",
                name="Defendant",
                role=ActorRole.PERPETRATOR,
                mentions=[]
            ))
        
        return actors

    def extract_chunked(
        self, 
        case_text: str, 
        chunk_size: int, 
        return_raw: bool = False
    ) -> Union[List[Actor], Tuple[List[Actor], List[str]]]:
        """
        Extract actors from case text by chunking.
        
        Args:
            case_text: Raw case narrative text
            chunk_size: Maximum characters per chunk
            return_raw: Whether to return raw LLM responses
            
        Returns:
            List of Actor objects (or tuple with raw responses)
        """
        self.logger.info(f"Extracting actors from case text using chunking (chunk_size={chunk_size})")
        
        # Split text into chunks
        chunks = []
        for i in range(0, len(case_text), chunk_size):
            chunks.append(case_text[i:i + chunk_size])
        
        self.logger.info(f"Processing {len(chunks)} chunks")
        
        all_actors: Dict[str, Actor] = {}  # Use dict to deduplicate by actor_id
        all_raw_responses: List[str] = []
        
        for idx, chunk in enumerate(chunks):
            self.logger.info(f"Processing chunk {idx + 1}/{len(chunks)}")
            
            if return_raw:
                chunk_actors, raw_response = self.extract(chunk, return_raw=True)
                all_raw_responses.append(raw_response)
            else:
                chunk_actors = self.extract(chunk, return_raw=False)
            
            # Merge actors by actor_id, combining mentions
            for actor in chunk_actors:
                if actor.actor_id in all_actors:
                    # Merge mentions
                    existing = all_actors[actor.actor_id]
                    existing.mentions.extend(actor.mentions)
                    # Update non-null fields that exist in the schema
                    if actor.age and not existing.age:
                        existing.age = actor.age
                    if actor.gender and not existing.gender:
                        existing.gender = actor.gender
                    if actor.occupation and not existing.occupation:
                        existing.occupation = actor.occupation
                    if actor.ethnicity and not existing.ethnicity:
                        existing.ethnicity = actor.ethnicity
                    if actor.marital_status and not existing.marital_status:
                        existing.marital_status = actor.marital_status
                    if actor.name and not existing.name:
                        existing.name = actor.name
                    if actor.demographics:
                        existing.demographics.update(actor.demographics)
                else:
                    all_actors[actor.actor_id] = actor
        
        actors_list = list(all_actors.values())
        self.logger.info(f"Extracted {len(actors_list)} unique actors from all chunks")
        
        if return_raw:
            return actors_list, all_raw_responses
        return actors_list
