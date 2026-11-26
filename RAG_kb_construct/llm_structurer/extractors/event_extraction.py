"""Event extraction module using LLM."""

import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Union, Optional
from llm_structurer.llm_client import LLMClient
from llm_structurer.schema import Event, Actor


logger = logging.getLogger(__name__)


class EventExtractor:
    """Extract events from case timeline using LLM."""

    def __init__(self, llm_client: LLMClient, prompts_dir: str = "llm_structurer/prompts"):
        """
        Initialize event extractor.

        Args:
            llm_client: LLM client instance
            prompts_dir: Directory containing prompt templates
        """
        self.llm_client = llm_client
        self.prompt_template = self._load_prompt(prompts_dir)
        self.logger = logging.getLogger(__name__)

    def _load_prompt(self, prompts_dir: str) -> str:
        """Load the event extraction prompt template."""
        prompt_path = Path(prompts_dir) / "events.txt"
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()

    def extract(
        self,
        case_text: str,
        actors: List[Actor],
        return_raw: bool = False,
    ) -> Union[List[Event], Tuple[List[Event], str]]:
        """
        Extract events from case timeline.

        Args:
            case_text: Raw case narrative text
            actors: List of extracted actors

        Returns:
            List of Event objects
        """
        self.logger.info("Extracting events from case timeline")
        
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
            events_data = response.get('events', [])
            
            events = []
            for event_dict in events_data:
                try:
                    event = Event(**event_dict)
                    events.append(event)
                except Exception as e:
                    self.logger.error(f"Failed to parse event: {e}")
                    self.logger.debug(f"Event data: {event_dict}")
            
            events.sort(key=lambda e: e.temporal_order or 999)
            
            self.logger.info(f"Extracted {len(events)} events")
            if return_raw:
                return events, raw_response or ""
            return events
            
        except Exception as e:
            self.logger.error(f"Event extraction failed: {e}")
            if return_raw:
                return [], raw_response or ""
            return []

    def extract_chunked(
        self,
        case_text: str,
        actors: List[Actor],
        chunk_size: int = 2000,
        return_raw: bool = False,
    ) -> Union[List[Event], Tuple[List[Event], List[str]]]:
        """
        Extract events from long case text by chunking.

        Args:
            case_text: Raw case narrative text
            actors: List of extracted actors
            chunk_size: Maximum characters per chunk

        Returns:
            List of Event objects
        """
        if len(case_text) <= chunk_size:
            return self.extract(case_text, actors, return_raw=return_raw)
        
        self.logger.info(f"Chunking case text ({len(case_text)} chars) into {chunk_size} char chunks")
        
        chunks = []
        for i in range(0, len(case_text), chunk_size):
            chunks.append(case_text[i:i + chunk_size])
        
        all_events: List[Event] = []
        raw_chunks: Optional[List[str]] = [] if return_raw else None
        for i, chunk in enumerate(chunks):
            self.logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            if return_raw:
                chunk_events, chunk_raw = self.extract(
                    chunk, actors, return_raw=True
                )
                raw_chunks.append(chunk_raw)
            else:
                chunk_events = self.extract(chunk, actors)
            all_events.extend(chunk_events)
        
        for i, event in enumerate(all_events):
            event.temporal_order = i + 1
        
        if return_raw:
            return all_events, raw_chunks
        return all_events
