"""Relationship extraction module using LLM."""

import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Tuple
from llm_structurer.llm_client import LLMClient, LLMJsonParseError
from llm_structurer.schema import Relationship, Actor


logger = logging.getLogger(__name__)


class RelationshipExtractor:
    """Extract relationships between actors using LLM."""

    def __init__(self, llm_client: LLMClient, prompts_dir: str = "llm_structurer/prompts"):
        """
        Initialize relationship extractor.

        Args:
            llm_client: LLM client instance
            prompts_dir: Directory containing prompt templates
        """
        self.llm_client = llm_client
        self.prompt_template = self._load_prompt(prompts_dir)
        self.logger = logging.getLogger(__name__)

    def _load_prompt(self, prompts_dir: str) -> str:
        """Load the relationship extraction prompt template."""
        prompt_path = Path(prompts_dir) / "relationships.txt"
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()

    def extract(self, case_text: str, actors: List[Actor], return_raw: bool = False) -> Union[List[Relationship], Tuple[List[Relationship], str]]:
        """
        Extract relationships between actors.

        Args:
            case_text: Raw case narrative text
            actors: List of extracted actors

        Returns:
            List of Relationship objects
        """
        self.logger.info("Extracting relationships between actors")
        
        if len(actors) < 2:
            self.logger.warning("Fewer than 2 actors, skipping relationship extraction")
            return []
        
        actors_json = json.dumps([{
            'actor_id': a.actor_id,
            'name': a.name,
            'role': a.role
        } for a in actors], indent=2)
        
        prompt = self.prompt_template.format(
            case_text=case_text,
            actors_json=actors_json
        )
        
        raw_response: Optional[str] = None

        try:
            if return_raw:
                response, raw_response = self.llm_client.generate_json(
                    prompt, return_raw=True
                )
            else:
                response = self.llm_client.generate_json(prompt)
            relationships_data = response.get('relationships', [])
            
            relationships = []
            for rel_dict in relationships_data:
                try:
                    relationship = Relationship(**rel_dict)
                    relationships.append(relationship)
                except Exception as e:
                    self.logger.error(f"Failed to parse relationship: {e}")
                    self.logger.debug(f"Relationship data: {rel_dict}")
            
            self.logger.info(f"Extracted {len(relationships)} relationships")
            if return_raw:
                return relationships, raw_response or ""
            return relationships

        except LLMJsonParseError as exc:
            raw_response = exc.response
            self.logger.error(f"Relationship extraction failed: {exc}")
        except Exception as e:
            self.logger.error(f"Relationship extraction failed: {e}")

        if return_raw:
            return [], raw_response or ""
        return []

    def extract_chunked(
        self,
        case_text: str,
        actors: List[Actor],
        chunk_size: int,
        return_raw: bool = False
    ) -> Union[List[Relationship], Tuple[List[Relationship], List[str]]]:
        """
        Extract relationships from case text by chunking.
        
        Args:
            case_text: Raw case narrative text
            actors: List of actors to find relationships between
            chunk_size: Maximum characters per chunk
            return_raw: Whether to return raw LLM responses
            
        Returns:
            List of Relationship objects (or tuple with raw responses)
        """
        self.logger.info(f"Extracting relationships using chunking (chunk_size={chunk_size})")
        
        # Split text into chunks
        chunks = []
        for i in range(0, len(case_text), chunk_size):
            chunks.append(case_text[i:i + chunk_size])
        
        self.logger.info(f"Processing {len(chunks)} chunks")
        
        all_relationships: List[Relationship] = []
        all_raw_responses: List[str] = []
        seen_relationships: set = set()  # Track unique relationships
        
        for idx, chunk in enumerate(chunks):
            self.logger.info(f"Processing chunk {idx + 1}/{len(chunks)}")
            
            if return_raw:
                chunk_rels, raw_response = self.extract(chunk, actors, return_raw=True)
                all_raw_responses.append(raw_response)
            else:
                chunk_rels = self.extract(chunk, actors, return_raw=False)
            
            # Deduplicate relationships
            for rel in chunk_rels:
                rel_key = (rel.actor_1_id, rel.actor_2_id, rel.relationship_type)
                if rel_key not in seen_relationships:
                    seen_relationships.add(rel_key)
                    all_relationships.append(rel)
        
        self.logger.info(f"Extracted {len(all_relationships)} unique relationships from all chunks")
        
        if return_raw:
            return all_relationships, all_raw_responses
        return all_relationships
