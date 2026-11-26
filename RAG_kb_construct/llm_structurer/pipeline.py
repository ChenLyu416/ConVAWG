"""Main extraction pipeline orchestrator."""

import logging
from typing import Dict, Any, Optional, Tuple, Union
from datetime import datetime

from llm_structurer.config import Config
from llm_structurer.llm_client import LLMClient
from llm_structurer.schema import Case
from llm_structurer.extractors.actor_extraction import ActorExtractor
from llm_structurer.extractors.relationship_extraction import RelationshipExtractor
from llm_structurer.extractors.violence_extraction import ViolenceExtractor
from llm_structurer.extractors.event_extraction import EventExtractor
from llm_structurer.extractors.summary_extraction import SummaryExtractor


logger = logging.getLogger(__name__)


class ExtractionPipeline:
    """Orchestrates the complete LLM-based extraction pipeline."""

    def __init__(self, config: Config):
        """
        Initialize the extraction pipeline.

        Args:
            config: Configuration object
        """
        self.config = config
        self.llm_client = LLMClient(config)
        
        prompts_dir = config.prompts_dir
        
        self.actor_extractor = ActorExtractor(self.llm_client, prompts_dir)
        self.relationship_extractor = RelationshipExtractor(self.llm_client, prompts_dir)
        self.violence_extractor = ViolenceExtractor(self.llm_client, prompts_dir)
        self.event_extractor = EventExtractor(self.llm_client, prompts_dir)
        self.summary_extractor = SummaryExtractor(self.llm_client, prompts_dir)
        
        self.logger = logging.getLogger(__name__)

    def process_case(
        self,
        case_data: Dict[str, Any],
        capture_raw: bool = False,
    ) -> Union[Case, Tuple[Case, Dict[str, Any]]]:
        """
        Process a single case through the complete pipeline.

        Args:
            case_data: Dictionary with 'case_id' and 'raw_text'

        Returns:
            Structured Case object
        """
        case_id = case_data.get('case_id', 'unknown')
        raw_text = case_data.get('raw_text', '')
        source = case_data.get('source')
        
        self.logger.info(f"Processing case: {case_id}")
        
        raw_outputs: Dict[str, Any] = {}

        # Use chunked extraction if text is too long
        use_chunking = len(raw_text) > self.config.chunk_size
        
        if use_chunking:
            self.logger.info(f"Text length ({len(raw_text)} chars) exceeds chunk_size ({self.config.chunk_size}), using chunked extraction")
            actors_result = self.actor_extractor.extract_chunked(
                raw_text, self.config.chunk_size, return_raw=capture_raw
            )
        else:
            actors_result = self.actor_extractor.extract(
                raw_text, return_raw=capture_raw
            )
        if capture_raw:
            actors, raw_actors = actors_result  # type: ignore[assignment]
            raw_outputs['actors'] = raw_actors
        else:
            actors = actors_result  # type: ignore[assignment]
        self.logger.info(f"  - Extracted {len(actors)} actors")
        
        if use_chunking:
            relationships_result = self.relationship_extractor.extract_chunked(
                raw_text, actors, self.config.chunk_size, return_raw=capture_raw
            )
        else:
            relationships_result = self.relationship_extractor.extract(
                raw_text, actors, return_raw=capture_raw
            )
        if capture_raw:
            relationships, raw_relationships = relationships_result  # type: ignore[assignment]
            raw_outputs['relationships'] = raw_relationships
        else:
            relationships = relationships_result  # type: ignore[assignment]
        self.logger.info(f"  - Extracted {len(relationships)} relationships")
        
        if use_chunking:
            violence_result = self.violence_extractor.extract_chunked(
                raw_text, self.config.chunk_size, return_raw=capture_raw
            )
        else:
            violence_result = self.violence_extractor.extract(
                raw_text, return_raw=capture_raw
            )
        if capture_raw:
            violence_types, raw_violence = violence_result  # type: ignore[assignment]
            raw_outputs['violence_types'] = raw_violence
        else:
            violence_types = violence_result  # type: ignore[assignment]
        self.logger.info(f"  - Classified {len(violence_types)} violence types")
        
        if use_chunking:
            events_result = self.event_extractor.extract_chunked(
                raw_text,
                actors,
                self.config.chunk_size,
                return_raw=capture_raw,
            )
        else:
            events_result = self.event_extractor.extract(
                raw_text, actors, return_raw=capture_raw
            )

        if capture_raw:
            events, raw_events = events_result  # type: ignore[assignment]
            raw_outputs['events'] = raw_events
        else:
            events = events_result  # type: ignore[assignment]
        self.logger.info(f"  - Extracted {len(events)} events")
        
        summary_result = self.summary_extractor.extract(
            raw_text, return_raw=capture_raw
        )
        if capture_raw:
            summary, raw_summary = summary_result  # type: ignore[assignment]
            raw_outputs['summary'] = raw_summary
        else:
            summary = summary_result  # type: ignore[assignment]
        self.logger.info(f"  - Generated summary")
        
        case = Case(
            case_id=case_id,
            raw_text=raw_text,
            source=source,
            date_processed=datetime.utcnow(),
            actors=actors,
            relationships=relationships,
            events=events,
            violence_types=violence_types,
            summary=summary,
            metadata=case_data.get('metadata', {})
        )
        self.logger.info(f"Completed processing case: {case_id}")

        if capture_raw:
            case.metadata = case.metadata or {}
            case.metadata['raw_llm_outputs'] = raw_outputs
            return case, raw_outputs  # type: ignore[return-value]

        return case

    def validate_case(self, case: Case) -> bool:
        """
        Validate extracted case data.

        Args:
            case: Case object to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            if not case.case_id:
                self.logger.error("Case missing case_id")
                return False
            
            if not case.raw_text:
                self.logger.error("Case missing raw_text")
                return False
            
            if len(case.actors) == 0:
                self.logger.warning("Case has no actors extracted")
            
            self.logger.info(f"Case {case.case_id} validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            return False
