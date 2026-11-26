"""
Retrieval functions for finding similar KB cases based on persona descriptions.
Works directly with case-level documents from KB_homicide.
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from schemas.vawg_scenario import RetrievedCase


logger = logging.getLogger(__name__)


class KBCaseRetriever:
    """Retrieves similar KB cases for a given persona using FAISS index."""
    
    def __init__(
        self,
        persona_index_dir: Path,
        kb_homicide_dir: Path,
        embeddings_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize retriever with case index and KB directory.
        
        Args:
            persona_index_dir: Path to persona_index/ directory with FAISS index
            kb_homicide_dir: Path to KB_homicide/ directory with full case JSON files
            embeddings_model: Sentence transformer model for embeddings
        """
        self.kb_homicide_dir = Path(kb_homicide_dir)
        self.persona_index_dir = Path(persona_index_dir)
        
        # Load case documents from JSONL - check both locations
        case_docs_file = persona_index_dir / "persona_docs.jsonl"
        if not case_docs_file.exists():
            # Try parent outputs directory
            case_docs_file = persona_index_dir.parent / "persona_docs_full.jsonl"
        if not case_docs_file.exists():
            raise FileNotFoundError(f"Case JSONL not found in {persona_index_dir} or parent directory")
        
        self.case_docs = self._load_case_docs(case_docs_file)
        
        # Load index - check both filenames
        index_file = persona_index_dir / "index.faiss"
        if not index_file.exists():
            index_file = persona_index_dir / "faiss_index.bin"
        if not index_file.exists():
            raise FileNotFoundError(f"Index file not found: {index_file}")
        
        # Create embedding backend
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(embeddings_model)
        embedding_dim = model.get_sentence_embedding_dimension()
        
        # Initialize retriever with loaded components
        import faiss
        self.index = faiss.read_index(str(index_file))
        self.embedding_model = model
        
        logger.info(f"Initialized KBCaseRetriever with index: {persona_index_dir}")
        logger.info(f"KB directory: {kb_homicide_dir}")
        logger.info(f"Loaded {len(self.case_docs)} case documents")
    
    def _load_case_docs(self, case_docs_file: Path) -> List[Dict[str, Any]]:
        """Load case documents from JSONL file."""
        cases = []
        with open(case_docs_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    cases.append(data)
        return cases
    
    def retrieve_similar_cases(
        self,
        persona_text: str,
        k: int = 5,
        role_filter: str = "victim"
    ) -> List[RetrievedCase]:
        """
        Retrieve k most similar cases from KB based on persona text.
        
        Args:
            persona_text: Description of the persona to match
            k: Number of similar cases to retrieve
            role_filter: Filter by role (typically "victim" for matching victim profiles)
        
        Returns:
            List of RetrievedCase objects with full case data
        """
        logger.info(f"Retrieving {k} similar cases for persona: {persona_text[:100]}...")
        
        # Encode query
        import numpy as np
        import faiss
        query_embedding = self.embedding_model.encode([persona_text])[0]
        query_embedding = np.array([query_embedding], dtype=np.float32)
        
        # Normalize for cosine similarity (if using IndexFlatIP)
        faiss.normalize_L2(query_embedding)
        
        # Search index
        distances, indices = self.index.search(query_embedding, min(k * 2, len(self.case_docs)))
        
        # Collect results
        retrieved_cases = []
        case_ids_seen = set()
        
        for idx, distance in zip(indices[0], distances[0]):
            if len(retrieved_cases) >= k:
                break
            
            if idx >= len(self.case_docs):
                continue
            
            case_doc = self.case_docs[idx]
            case_id = case_doc.get("case_id", "unknown")
            
            # Skip duplicate cases
            if case_id in case_ids_seen:
                continue
            case_ids_seen.add(case_id)
            
            # Check if case has victim actor (role filter)
            victim_actor = self._extract_actor_by_role(case_doc, "victim")
            if role_filter == "victim" and victim_actor is None:
                logger.debug(f"Skipping case {case_id} - no victim actor found")
                continue
            
            # Calculate similarity score (distance is cosine similarity for IndexFlatIP with normalized vectors)
            similarity_score = float(distance)
            
            # Extract perpetrator
            perpetrator_actor = self._extract_actor_by_role(case_doc, "perpetrator")
            
            # Extract timeline duration
            time_duration = self._extract_timeline_duration(case_doc)
            
            # Build RetrievedCase
            retrieved_case = RetrievedCase(
                case_id=case_id,
                victim_summary=self._create_actor_summary(victim_actor, case_doc),
                perpetrator_summary=self._create_actor_summary(perpetrator_actor, case_doc),
                violence_types=case_doc.get("violence_types", []),
                context=self._extract_case_context(case_doc),
                similarity_score=similarity_score,
                metadata={
                    "full_case": case_doc,  # Include full case data for downstream use
                    "actors": case_doc.get("actors", []),
                    "relationships": case_doc.get("relationships", []),
                    "events": case_doc.get("events", []),
                    "time_duration": time_duration  # Add timeline duration to metadata
                }
            )
            
            retrieved_cases.append(retrieved_case)
        
        logger.info(f"Retrieved {len(retrieved_cases)} cases")
        return retrieved_cases
    
    def _extract_actor_by_role(
        self, 
        case_doc: Dict[str, Any], 
        role: str
    ) -> Optional[Dict[str, Any]]:
        """Extract actor data by role from case document."""
        actors = case_doc.get("actors", [])
        for actor in actors:
            if actor.get("role") == role:
                return actor
        return None
    
    def _extract_case_context(self, case_doc: Dict[str, Any]) -> str:
        """Extract case context from case document."""
        # Try summary field first
        if case_doc.get("summary"):
            return case_doc["summary"]
        
        # Try extracting from raw_text
        raw_text = case_doc.get("raw_text", "")
        if raw_text and "**2. The Facts**" in raw_text:
            facts_section = raw_text.split("**2. The Facts**")[1]
            paragraphs = [p.strip() for p in facts_section.split("\n\n") if p.strip()]
            if paragraphs:
                # Increased from 500 to 1500 for richer context
                return paragraphs[0][:1500]
        
        # Fallback: first 1500 chars of raw_text (increased from 500)
        if raw_text:
            return raw_text[:1500].strip()
        
        # Last resort: use case_id and basic info
        case_id = case_doc.get("case_id", "Unknown")
        actors_count = len(case_doc.get("actors", []))
        return f"Case {case_id} with {actors_count} actors. Limited context available."
    
    def _create_actor_summary(
        self, 
        actor_data: Optional[Dict[str, Any]], 
        case_doc: Dict[str, Any]
    ) -> str:
        """Create a human-readable summary from actor data."""
        if actor_data is None:
            return "[Actor data not available]"
        
        parts = []
        
        # Name and role
        name = actor_data.get("name", "Unknown")
        role = actor_data.get("role", "unknown")
        parts.append(f"{name} ({role})")
        
        # Demographics
        age = actor_data.get("age")
        gender = actor_data.get("gender")
        occupation = actor_data.get("occupation")
        ethnicity = actor_data.get("ethnicity")
        
        if age:
            parts.append(f"age {age}")
        if gender:
            parts.append(gender)
        if occupation:
            parts.append(f"occupation: {occupation}")
        if ethnicity:
            parts.append(f"ethnicity: {ethnicity}")
        
        # Relationship context (from relationships array)
        relationships = case_doc.get("relationships", [])
        actor_id = actor_data.get("actor_id")
        for rel in relationships:
            if rel.get("actor_1_id") == actor_id or rel.get("actor_2_id") == actor_id:
                rel_type = rel.get("relationship_type", "unknown")
                parts.append(f"relationship: {rel_type}")
                break
        
        # Include mention count
        mentions = actor_data.get("mentions", [])
        if mentions:
            parts.append(f"({len(mentions)} mentions)")
        
        return ", ".join(parts)
    
    def _extract_timeline_duration(self, case_doc: Dict[str, Any]) -> Optional[str]:
        """
        Extract relationship/abuse duration from case events and timeline.
        
        Looks for temporal information in:
        - Events with date_time fields
        - Relationship temporal_info fields
        - Summary text mentioning duration
        
        Restricts durations to between 3 months and 3 years.
        
        Returns:
            Duration string like "6 months", "2 years", or None if not found or outside valid range
        """
        import re
        from datetime import datetime
        
        duration_info = None
        
        def validate_duration(duration_str: str) -> bool:
            """Check if duration is between 3 months and 3 years."""
            if not duration_str:
                return False
            
            duration_lower = duration_str.lower()
            
            # Extract number and unit
            match = re.match(r'(\d+)\s*(year|month|week|day)', duration_lower)
            if not match:
                return False
            
            num = int(match.group(1))
            unit = match.group(2)
            
            # Convert to months for comparison
            if unit.startswith('year'):
                months = num * 12
            elif unit.startswith('month'):
                months = num
            elif unit.startswith('week'):
                months = num / 4.33  # approximate weeks to months
            elif unit.startswith('day'):
                months = num / 30.44  # approximate days to months
            else:
                return False
            
            # Check if between 3 months and 3 years (36 months)
            return 3 <= months <= 36
        
        # Method 1: Check relationships for temporal_info
        relationships = case_doc.get("relationships", [])
        for rel in relationships:
            if rel.get("relationship_type") in ["intimate_partner", "ex_partner"]:
                temporal = rel.get("temporal_info")
                if temporal and temporal != "current" and validate_duration(temporal):
                    duration_info = temporal
                    break
        
        # Method 2: Analyze events with date_time
        events = case_doc.get("events", [])
        dated_events = [e for e in events if e.get("date_time")]
        
        if len(dated_events) >= 2:
            # Try to parse dates and calculate duration
            try:
                dates = []
                for event in dated_events:
                    date_str = event.get("date_time", "")
                    # Try various date formats
                    for fmt in ["%d/%m/%Y", "%Y-%m-%d", "%B %Y", "%Y"]:
                        try:
                            dt = datetime.strptime(date_str, fmt)
                            dates.append(dt)
                            break
                        except (ValueError, TypeError):
                            continue
                
                if len(dates) >= 2:
                    dates.sort()
                    time_diff = dates[-1] - dates[0]
                    total_months = time_diff.days / 30.44
                    
                    # Clamp to valid range (3 months to 3 years)
                    if total_months < 3:
                        duration_info = None  # Too short, ignore
                    elif total_months <= 36:
                        # Within valid range
                        years = int(total_months // 12)
                        months = int(total_months % 12)
                        
                        if years > 0:
                            if months > 6:
                                duration_info = f"{years + 1} years"
                            else:
                                duration_info = f"{years} {'year' if years == 1 else 'years'}"
                        else:
                            duration_info = f"{int(total_months)} months"
                    else:
                        # Too long, clamp to 3 years
                        duration_info = "3 years"
            except Exception as e:
                logger.debug(f"Could not parse event dates: {e}")
        
        # Method 3: Search summary and raw text for duration patterns
        if not duration_info:
            summary = case_doc.get("summary") or ""
            raw_text = case_doc.get("raw_text") or ""
            text_to_search = summary + " " + raw_text[:2000]
            
            # Common duration patterns
            duration_patterns = [
                r"(\d+)\s*years?\s*(together|relationship|married|dating)",
                r"(\d+)\s*months?\s*(together|relationship|married|dating)",
                r"(together|relationship|married|dating)\s*for\s*(\d+)\s*(years?|months?)",
                r"relationship\s*lasted\s*(?:approximately\s*)?(\d+)\s*(years?|months?)",
                r"been\s*(?:together|married|in\s*a\s*relationship)\s*(?:for\s*)?(\d+)\s*(years?|months?)"
            ]
            
            for pattern in duration_patterns:
                match = re.search(pattern, text_to_search, re.IGNORECASE)
                if match:
                    if len(match.groups()) >= 2:
                        # Pattern like "2 years together" or "together for 2 years"
                        num = match.group(1) if match.group(1).isdigit() else match.group(2)
                        unit = match.group(2) if match.group(1).isdigit() else match.group(3)
                        candidate_duration = f"{num} {unit}"
                        if validate_duration(candidate_duration):
                            duration_info = candidate_duration
                    break
        
        return duration_info
