"""
Event Graph Construction from VAWG Scenarios

This script:
1. Extracts atomic events from scenario descriptions using LLM
2. Represents events as predicate-argument structures
3. Constructs event graphs with temporal/causal relations
4. Validates persona consistency constraints
"""
import json
import logging
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple, Set
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import sys
import os
import argparse

# ── Fallback name pools for third-party name resolution ───────────────────────
# Used when an actor name is missing, empty, or the placeholder "unknown".
_TP_NAMES_FEMALE = [
    "Anna", "Emma", "Lucy", "Hannah", "Rachel", "Laura", "Sarah", "Rebecca",
    "Priya", "Aisha", "Fatima", "Zara", "Amina", "Maya", "Sofia", "Elena",
    "Mei", "Yasmin", "Amara", "Natasha", "Camila", "Nia", "Riya", "Layla",
]
_TP_NAMES_MALE = [
    "David", "James", "Michael", "Robert", "Daniel", "Matthew", "Andrew",
    "Omar", "Ali", "Rahul", "Kofi", "Jamal", "Marcus", "Dimitri", "Carlos",
    "Wei", "Hassan", "Santiago", "Ivan", "Arjun", "Kwame", "Tariq", "Elijah",
]


def _is_placeholder_name(name: str) -> bool:
    """Return True if *name* is a known placeholder that must be replaced."""
    if not name:
        return True
    lower = name.strip().lower()
    return (
        lower in ("unknown", "n/a", "none", "", "unnamed")
        or lower.startswith("third_party_")
    )


def _generate_third_party_name(
    actor: Dict[str, Any],
    used_names: Set[str],
) -> str:
    """Return a concrete name for a third-party actor.

    Uses the actor’s existing name when it is meaningful; otherwise picks a
    fresh name from the gender-appropriate fallback pool, avoiding names
    already in *used_names*.
    """
    existing = (actor.get("name") or "").strip()
    if existing and not _is_placeholder_name(existing):
        return existing

    gender = (actor.get("gender") or "").lower()
    pool = _TP_NAMES_FEMALE if "female" in gender else _TP_NAMES_MALE

    available = [n for n in pool if n not in used_names]
    if not available:
        # If pool is exhausted fall back to the other gender pool
        alt_pool = _TP_NAMES_MALE if "female" in gender else _TP_NAMES_FEMALE
        available = [n for n in alt_pool if n not in used_names]
    if not available:
        available = pool  # allow reuse as last resort

    chosen = random.choice(available)
    used_names.add(chosen)
    return chosen

# Add RAG_kb_construct to path for LLM backend
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "RAG_kb_construct"))

from llm_structurer.llm_backends import HuggingFaceBackend, OpenAIBackend
try:
    from llm_structurer.llm_backends import VLLMBackend
except ImportError:
    VLLMBackend = None
from llm_structurer.config import Config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _extract_first_json(text: str) -> Any:
    """Extract the first balanced {...} JSON object from text.

    Handles models that emit extra prose or multiple JSON blocks after the
    main object (e.g. LLaMA-3.1-8B-Instruct), which cause json.loads to
    raise 'Extra data'.
    """
    start = text.find('{')
    if start == -1:
        raise json.JSONDecodeError("No JSON object found", text, 0)
    depth = 0
    in_string = False
    escape_next = False
    for i, ch in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise json.JSONDecodeError("Unbalanced braces", text, len(text))


class RelationType(Enum):
    """Types of relations between events."""
    TEMPORAL_BEFORE = "TEMPORAL_BEFORE"  # Event A happens before Event B
    CAUSES = "CAUSES"                     # Event A causes Event B
    ENABLES = "ENABLES"                   # Event A enables Event B
    ESCALATES_TO = "ESCALATES_TO"        # Event A escalates to Event B
    EVIDENCE_OF = "EVIDENCE_OF"          # Event A is evidence of Event B
    CONTRASTS_WITH = "CONTRASTS_WITH"    # Event A contrasts with Event B


@dataclass
class EventArgument:
    """Argument in event structure."""
    name: str  # e.g., "Bronwyn", "police", "unknown"
    role: str  # For agent: "victim", "perpetrator", "third_party", "unknown"
    type: str  # For patient: "person", "org", "object", "unknown"


@dataclass
class Event:
    """Event with enhanced schema (composite events allowed, 5-8 events per summary)."""
    event_id: str
    sentence_id: int
    text_span: str
    description: str  # Coherent description with actor, time, location, and actions
    agent: EventArgument
    action: str
    patient: EventArgument
    time: str  # Temporal information extracted from text
    timestamp: str  # LLM-generated timestamp in YYYY-MM-DD format
    location: str  # Location information extracted from text
    attributes: Dict[str, Any]  # polarity, modality, time_hint, crime_type_hints
    evidence: Dict[str, str]  # summary_sentence, notes


@dataclass
class EventRelation:
    """Relation between two events."""
    src_event_id: str
    dst_event_id: str
    relation_type: RelationType
    confidence: float  # 0.0 to 1.0
    evidence: str  # Justification for the relation


@dataclass
class EventGraph:
    """Complete event graph for a scenario."""
    scenario_id: str
    events: List[Event]
    relations: List[EventRelation]
    victim_name: str
    perpetrator_name: str
    third_party_names: List[str]
    metadata: Dict[str, Any]


class EventGraphBuilder:
    """Builds event graphs from VAWG scenarios using LLM."""
    
    def __init__(self, model_config_path: Path = None, backend_type: str = "huggingface",
                 model_name: str = None, api_key: str = None,
                 base_url: str = None, tensor_parallel_size: int = 4,
                 max_model_len: int = 8192):
        """
        Initialize event graph builder.

        Args:
            model_config_path: Path to LLM config (e.g., qwen35_35b_hf.yaml) - for HuggingFace backend
            backend_type: Type of backend - "huggingface", "vllm", or "openai"
            model_name: Model name for vLLM or OpenAI backends
            api_key: API key for OpenAI backend
            base_url: Optional base URL override for OpenAI-compatible endpoints (DeepSeek, Gemini, etc.)
            tensor_parallel_size: Number of GPUs for vLLM tensor parallelism (default 4)
        """
        self.backend_type = backend_type

        # Initialize backend based on type
        if backend_type == "huggingface":
            self.config = Config(str(model_config_path))
            cuda_visible = self.config.get('llm.cuda_visible_devices', None)
            if cuda_visible:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible)
            logger.info(f"Loading HuggingFace model: {self.config.model_name}")
            self.backend = HuggingFaceBackend(
                model_name=self.config.model_name,
                load_in_8bit=self.config.get('llm.load_in_8bit', False),
                device_map=self.config.get('llm.device_map', 'auto'),
                trust_remote_code=self.config.get('llm.trust_remote_code', True),
            )

        elif backend_type == "vllm":
            if VLLMBackend is None:
                raise ImportError("vLLM backend is not available. Install vllm package or use a different backend.")
            logger.info(f"Connecting to vLLM server for model: {model_name}")
            self.backend = VLLMBackend(
                model_name=model_name,
                base_url=base_url,
                api_key=api_key,
            )

        elif backend_type == "openai":
            logger.info(f"Using OpenAI-compatible API: {model_name} (base_url={base_url or 'default'})")
            self.backend = OpenAIBackend(
                model_name=model_name,
                api_key=api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("GPT52_API_KEY"),
                base_url=base_url
            )

        else:
            raise ValueError(f"Unknown backend type: {backend_type}")
        
        self.stats = {
            "total_processed": 0,
            "total_events_extracted": 0,
            "total_relations_extracted": 0,
            "failed": 0
        }
    
    def load_scenario(self, filepath: Path) -> Dict[str, Any]:
        """Load scenario from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def extract_events_and_relations(self, scenario: Dict[str, Any]) -> Tuple[List[Event], List[EventRelation], Dict[str, Any]]:
        """
        Extract events and relations in one unified call using new prompt structure.
        Expects 5-8 composite events per summary (not strictly atomic).
        
        Args:
            scenario: Full scenario dict
        
        Returns:
            Tuple of (events, relations, quality_checks)
        """
        # Prepare JSON record for prompt
        time_duration = scenario.get('time_duration', 'unknown')
        record_json = json.dumps({
            "persona_id": scenario.get('persona_id', 'unknown'),
            "scenario_description": scenario['scenario_description'],
            "victim_profile": scenario['victim_profile'],
            "perpetrator_profile": scenario['perpetrator_profile'],
            "third_party_actors": scenario.get('third_party_actors', []),
            "crime_types": scenario.get('crime_types', []),
            "time_duration": time_duration,
            "persona_text": scenario.get('persona_text', '')
        }, indent=2)
        
        prompt = f"""You are an event extraction expert for VAWG (Violence Against Women and Girls) scenarios. Extract a structured event graph from the given JSON record.

REQUIREMENTS
- Output VALID JSON ONLY (no extra text).
- Extract EXACTLY 5–8 composite events covering the full scenario timeline.
- Do NOT invent facts. If uncertain, use "unknown" and lower confidence.
- Resolve pronouns only when unambiguous; otherwise agent.name="unknown".
- Event IDs: "E1", "E2", … in sentence order.
- crime_type_hints must be a subset of provided crime_types (or [] if none).

CRITICAL: REALISTIC PROGRESSION OF ABUSE
- The FIRST 40% of events (typically E1, E2, E3) should represent NORMAL or EARLY relationship behaviors
- These early events should have EMPTY crime_type_hints: [] (no crime types)
- Examples of early normal events: meeting, dating, moving in together, spending time together
- Only assign crime_type_hints to events that explicitly describe harmful behaviors
- Escalation should be GRADUAL: normal → subtle control → harassment → violence
- DO NOT label every event with crime types - maintain realism with normal relationship stages

RELATION TYPES (choose from):
TEMPORAL_BEFORE, CAUSES, ENABLES, ESCALATES_TO, EVIDENCE_OF, CONTRASTS_WITH

TASK
1) Extract ordered events from scenario_description.
2) Extract directed relations between events (temporal, causal, escalation, evidence).


OUTPUT SCHEMA (JSON only)
{{
  "persona_id": "<string>",
  "source_fields_used": ["scenario_description","victim_profile","perpetrator_profile","third_party_actors","crime_types","persona_text"],
  "events": [
    {{
      "event_id": "E1",
      "sentence_id": 0,
      "text_span": "<verbatim snippet from the sentence supporting this event>",
      "description": "<coherent 1-2 sentence description integrating actor, time, location, and actions>",
      "agent": {{"name": "<string>", "role": "victim|perpetrator|third_party|unknown"}},
      "action": "<normalized verb lemma or short action phrase>",
      "patient": {{"name": "<string or unknown>", "type": "person|org|object|unknown"}},
      "time": "<temporal expression from text or 'unknown'>",
      "timestamp": "<YYYY-MM-DD format - generate realistic date within time_duration>",
      "location": "<location from text or 'unknown'>",
      "attributes": {{
        "polarity": "negative|neutral|positive",
        "modality": "asserted|reported|hypothetical",
        "time_hint": "early|mid|late|unknown",
        "crime_type_hints": ["<subset of crime_types>"]
      }},
      "evidence": {{
        "summary_sentence": "<full sentence text>",
        "notes": "<very short note on how you mapped agent/action/patient>"
      }}
    }}
  ],
  "relations": [
    {{
      "src_event_id": "E1",
      "dst_event_id": "E2",
      "type": "TEMPORAL_BEFORE|CAUSES|ENABLES|ESCALATES_TO|EVIDENCE_OF|CONTRASTS_WITH",
      "confidence": 0.0,
      "evidence": "<brief justification; cite cue words like 'this evolved into', 'when', 'then'>"
    }}
  ],
  "quality_checks": {{
    "num_events": 0,
    "num_relations": 0,
    "has_escalation": true,
    "unresolved_pronouns": ["<list>"]
  }}
}}

TEMPORAL & CONTENT GUIDELINES
- Assign realistic YYYY-MM-DD timestamps, in chronological order, distributed across
  time_duration = {scenario.get('time_duration', 'unknown')}, working backwards from 2026-03-01.
- Temporal pacing MUST be asymmetric:
  • Early-stage events should be separated by long gaps (can be more than 3 months).
  • Once escalation or violence appears, subsequent events MUST occur with much shorter gaps (weeks or very few months).


CRIME TYPE HINTS (use provided crime_types only)
- digital: phone checks, passwords, location tracking, online messages/posts
- economic: restricting funds, financial control
- coercive_control: monitoring, routine control, isolation, threats
- harassment/stalking: repeated or persistent contact, tracking
- domestic_violence: physical abuse, threats of harm
- sexual_abuse: unwanted sexual contact, coercion
- psychological/emotional: volatility, dismissal, manipulation, humiliation

RELATION HEURISTICS
- TEMPORAL_BEFORE: default by sentence order
- ESCALATES_TO: “escalated”, “increasingly”, “constant”
- CAUSES / ENABLES: “when”, “after”, “because”, “led to”
- EVIDENCE_OF: concrete acts supporting broader patterns

NOW PROCESS THIS JSON RECORD:
{record_json}

JSON OUTPUT:"""
        
        def _generate(prompt_text: str, max_tok: int) -> str:
            if hasattr(self.backend, 'default_temperature'):
                return self.backend.generate(
                    prompt=prompt_text,
                    max_tokens=max_tok,
                    temperature=getattr(self.backend, 'default_temperature', 0.3)
                ).strip()
            else:
                return self.backend.generate(
                    prompt=prompt_text,
                    max_tokens=max_tok,
                    temperature=0.3
                ).strip()

        try:
            response = _generate(prompt, 16384)
            response = response.replace('```json', '').replace('```', '').strip()
            # Strip <think>...</think> reasoning chains (DeepSeek/Qwen reasoning models)
            import re as _re
            response = _re.sub(r'<think>.*?</think>', '', response, flags=_re.DOTALL).strip()

            # Parse JSON — retry once with a larger budget if truncated
            try:
                result = _extract_first_json(response)
            except json.JSONDecodeError as first_err:
                logger.warning(f"  JSON parse failed ({first_err}), retrying with increased token budget…")
                retry_prompt = prompt + "\n\nIMPORTANT: The previous response was truncated. Output ONLY the complete JSON object with no extra text."
                response = _generate(retry_prompt, 32768)
                response = response.replace('```json', '').replace('```', '').strip()
                response = _re.sub(r'<think>.*?</think>', '', response, flags=_re.DOTALL).strip()
                result = _extract_first_json(response)
            
            # Convert to Event objects
            events = []
            for event_data in result.get('events', []):
                agent = EventArgument(
                    name=event_data['agent']['name'],
                    role=event_data['agent']['role'],
                    type=''
                )
                patient = EventArgument(
                    name=event_data['patient']['name'],
                    role='',
                    type=event_data['patient']['type']
                )
                
                event = Event(
                    event_id=event_data['event_id'],
                    sentence_id=event_data.get('sentence_id', 0),
                    text_span=event_data.get('text_span', ''),
                    description=event_data.get('description', ''),
                    agent=agent,
                    action=event_data['action'],
                    patient=patient,
                    time=event_data.get('time', 'unknown'),
                    timestamp=event_data.get('timestamp', 'unknown'),
                    location=event_data.get('location', 'unknown'),
                    attributes=event_data.get('attributes', {}),
                    evidence=event_data.get('evidence', {})
                )
                events.append(event)
            
            # Convert to EventRelation objects
            relations = []
            for rel_data in result.get('relations', []):
                relation = EventRelation(
                    src_event_id=rel_data['src_event_id'],
                    dst_event_id=rel_data['dst_event_id'],
                    relation_type=RelationType(rel_data['type']),
                    confidence=float(rel_data.get('confidence', 0.5)),
                    evidence=rel_data.get('evidence', '')
                )
                relations.append(relation)
            
            quality_checks = result.get('quality_checks', {})
            
            logger.info(f"  Extracted {len(events)} events and {len(relations)} relations")
            self.stats['total_events_extracted'] += len(events)
            self.stats['total_relations_extracted'] += len(relations)
            
            return events, relations, quality_checks
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON after retry: {e}")
            logger.error(f"Response: {response[:500]}")
            return [], [], {}
        except Exception as e:
            logger.error(f"Error extracting events and relations: {e}")
            return [], [], {}
    

    
    def validate_persona_consistency(
        self,
        events: List[Event],
        quality_checks: Dict[str, Any],
        scenario: Dict[str, Any]
    ) -> Tuple[List[Event], List[str]]:
        """
        Validate and refine events for persona consistency.
        
        Args:
            events: List of events (composite allowed)
            quality_checks: Quality check metadata from extraction
            scenario: Full scenario dict with persona profiles
        
        Returns:
            Tuple of (refined_events, issues_found)
        """
        issues = []
        victim = scenario['victim_profile']
        perp = scenario['perpetrator_profile']
        victim_name = victim.get('name', 'victim')
        perp_name = perp.get('name', 'perpetrator')
        
        # Check if perpetrator is acting as agent in harmful events
        for event in events:
            crime_hints = event.attributes.get('crime_type_hints', [])
            if crime_hints and event.agent.name == victim_name:
                # Victim as agent in crime event - flag unless it's reporting/seeking help
                if event.action not in ['report', 'seek_help', 'contact_police', 'tell']:
                    issues.append(
                        f"Event {event.event_id}: Victim as agent in crime event - "
                        f"verify action '{event.action}' is appropriate"
                    )
        
        # Check for unresolved pronouns from quality checks
        unresolved = quality_checks.get('unresolved_pronouns', [])
        if unresolved:
            issues.append(f"Unresolved pronouns detected: {', '.join(unresolved)}")
        
        logger.info(f"  Found {len(issues)} potential persona inconsistencies")
        
        return events, issues
    
    def build_event_graph(self, scenario: Dict[str, Any]) -> EventGraph:
        """
        Build complete event graph for a scenario.
        
        Args:
            scenario: Full scenario dict
        
        Returns:
            EventGraph object
        """
        scenario_id = scenario.get('persona_id', 'unknown')
        logger.info(f"Building event graph for scenario {scenario_id}")
        
        # Extract events and relations in one unified call
        events, relations, quality_checks = self.extract_events_and_relations(scenario)
        
        if not events:
            logger.warning(f"  No events extracted for scenario {scenario_id}")
            return None
        
        # Validate persona consistency
        events, issues = self.validate_persona_consistency(events, quality_checks, scenario)
        
        # ── Resolve third-party names (replace any "unknown" placeholders) ──
        _used = {
            scenario['victim_profile'].get('name', 'victim'),
            scenario['perpetrator_profile'].get('name', 'perpetrator'),
        }
        resolved_tp_names = [
            _generate_third_party_name(actor, _used)
            for actor in scenario.get('third_party_actors', [])
        ]

        # ── Fix event-level "unknown" agent / patient names for third-party roles ──
        _tp_iter = iter(resolved_tp_names)  # assign in encounter order
        _event_tp_map: Dict[str, str] = {}  # cache name-per-event so it's stable
        for event in events:
            for slot_name, slot_role in [
                (event.agent.name, event.agent.role),
                (event.patient.name, event.patient.type),
            ]:
                if _is_placeholder_name(slot_name) and slot_role in ("third_party", "person", "unknown"):
                    pass  # handled below per object reference
            # Fix agent
            if _is_placeholder_name(event.agent.name) and event.agent.role == "third_party":
                if event.event_id not in _event_tp_map:
                    _event_tp_map[event.event_id] = (
                        resolved_tp_names[0] if resolved_tp_names
                        else _generate_third_party_name({}, _used)
                    )
                event.agent.name = _event_tp_map[event.event_id]
                logger.debug(
                    f"  Resolved unknown agent name in {event.event_id} "
                    f"→ '{event.agent.name}'"
                )
            # Fix patient
            if _is_placeholder_name(event.patient.name) and event.patient.type in ("person", "unknown"):
                candidate = (
                    resolved_tp_names[0] if resolved_tp_names
                    else _generate_third_party_name({}, _used)
                )
                event.patient.name = candidate
                logger.debug(
                    f"  Resolved unknown patient name in {event.event_id} "
                    f"→ '{event.patient.name}'"
                )

        # Build event graph
        graph = EventGraph(
            scenario_id=scenario_id,
            events=events,
            relations=relations,
            victim_name=scenario['victim_profile'].get('name', 'victim'),
            perpetrator_name=scenario['perpetrator_profile'].get('name', 'perpetrator'),
            third_party_names=resolved_tp_names,
            metadata={
                'crime_types': scenario.get('crime_types', []),
                'time_duration': scenario.get('time_duration', 'unknown'),
                'quality_checks': quality_checks,
                'persona_consistency_issues': issues,
                'timestamp': datetime.now().isoformat()
            }
        )
        
        self.stats['total_processed'] += 1
        return graph
    
    def save_event_graph(self, graph: EventGraph, output_path: Path) -> None:
        """
        Save event graph to JSON file.
        
        Args:
            graph: EventGraph object
            output_path: Path to save JSON file
        """
        # Convert to serializable dict
        graph_dict = {
            'scenario_id': graph.scenario_id,
            'victim_name': graph.victim_name,
            'perpetrator_name': graph.perpetrator_name,
            'third_party_names': graph.third_party_names,
            'events': [
                {
                    'event_id': e.event_id,
                    'sentence_id': e.sentence_id,
                    'text_span': e.text_span,
                    'description': e.description,
                    'agent': {
                        'name': e.agent.name,
                        'role': e.agent.role
                    },
                    'action': e.action,
                    'patient': {
                        'name': e.patient.name,
                        'type': e.patient.type
                    },
                    'time': e.time,
                    'timestamp': e.timestamp,
                    'location': e.location,
                    'attributes': e.attributes,
                    'evidence': e.evidence
                }
                for e in graph.events
            ],
            'relations': [
                {
                    'src_event_id': r.src_event_id,
                    'dst_event_id': r.dst_event_id,
                    'relation_type': r.relation_type.value,
                    'confidence': r.confidence,
                    'evidence': r.evidence
                }
                for r in graph.relations
            ],
            'metadata': graph.metadata
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(graph_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"  Saved event graph to {output_path.name}")
    
    def process_scenario_file(
        self,
        input_path: Path,
        output_dir: Path
    ) -> bool:
        """
        Process a single scenario file and generate event graph.
        
        Args:
            input_path: Path to input scenario JSON
            output_dir: Directory to save output event graph
        
        Returns:
            True if successful, False otherwise
        """
        try:
            scenario = self.load_scenario(input_path)
            graph = self.build_event_graph(scenario)
            
            if graph:
                output_path = output_dir / input_path.name
                self.save_event_graph(graph, output_path)
                return True
            else:
                self.stats['failed'] += 1
                return False
                
        except Exception as e:
            logger.error(f"Error processing {input_path.name}: {e}")
            self.stats['failed'] += 1
            return False
    
    def process_directory(
        self,
        input_dir: Path,
        output_dir: Path
    ) -> Dict[str, Any]:
        """
        Process all scenario files in a directory.
        
        Args:
            input_dir: Directory containing scenario JSON files
            output_dir: Directory to save event graphs
        
        Returns:
            Statistics dictionary
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        _REPORT_NAMES = {"postprocessing_report.json", "event_graph_report.json", "expansion_report.json", "generation_report.json"}
        scenario_files = sorted(f for f in input_dir.glob("*.json") if f.name not in _REPORT_NAMES)
        logger.info(f"Found {len(scenario_files)} scenarios to process")
        logger.info(f"Output directory: {output_dir}")
        logger.info("=" * 80)
        
        for i, filepath in enumerate(scenario_files, 1):
            logger.info(f"\n[{i}/{len(scenario_files)}] Processing {filepath.name}")
            self.process_scenario_file(filepath, output_dir)
            
            # Log progress every 5 files
            if i % 5 == 0:
                logger.info(f"\nProgress: {i}/{len(scenario_files)} completed")
                logger.info(f"Total events extracted: {self.stats['total_events_extracted']}")
                logger.info(f"Total relations extracted: {self.stats['total_relations_extracted']}")
                logger.info(f"Failed: {self.stats['failed']}")
        
        logger.info("\n" + "=" * 80)
        logger.info("EVENT GRAPH CONSTRUCTION COMPLETE")
        logger.info(f"Total scenarios processed: {self.stats['total_processed']}")
        logger.info(f"Total events extracted: {self.stats['total_events_extracted']}")
        logger.info(f"Total relations extracted: {self.stats['total_relations_extracted']}")
        logger.info(f"Failed: {self.stats['failed']}")
        
        return self.stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build event graphs from VAWG scenarios using LLM"
    )
    parser.add_argument(
        "--model",
        choices=["gpt52", "deepseek_v4_flash", "gemini_flash_lite", "qwen35_35b", "llama3_8b"],
        default="gpt52",
        help="Model to use for event extraction"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help="Override tensor parallel size for vLLM models (default: model-specific)"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).parent / "example_summary",
        help="Input directory containing scenario JSON files"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output directory for event graphs (default: timestamped dir in event_graphs/)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to model config file (for HuggingFace backend)"
    )
    parser.add_argument(
        "--api-key",
        help="API key for OpenAI backend (or set OPENAI_API_KEY or GPT52_API_KEY env var)"
    )
    parser.add_argument(
        "--base-url",
        help="vLLM server URL for vllm-backed models "
             "(default: $VLLM_BASE_URL or http://localhost:8000/v1)"
    )

    args = parser.parse_args()
    
    # Paths
    base_dir = Path(__file__).parent.parent.parent
    input_dir = args.input
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output:
        output_dir = args.output
    else:
        output_dir = base_dir / "event_generation" / "event_graph_from_summary" / "event_graphs" / f"{args.model}_{timestamp}"
    
    # Verify input directory
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)
    
    logger.info(f"Model: {args.model}")
    logger.info(f"Input: {input_dir}")
    logger.info(f"Output: {output_dir}")
    
    # Initialize builder based on model choice
    if args.model == "llama3_8b":
        model_name = os.environ.get("LLAMA3_8B_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
        builder = EventGraphBuilder(
            backend_type="vllm",
            model_name=model_name,
            base_url=args.base_url,
        )

    elif args.model == "qwen35_35b":
        model_config = args.config or base_dir / "RAG_kb_construct" / "configs" / "qwen35_35b_hf.yaml"
        if not model_config.exists():
            logger.error(f"Model config not found: {model_config}")
            sys.exit(1)
        builder = EventGraphBuilder(
            model_config_path=model_config,
            backend_type="huggingface",
        )

    elif args.model == "gpt52":
        model_name = os.environ.get("GPT52_MODEL", "gpt-5.2")
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("GPT52_API_KEY")
        if not api_key:
            logger.error("GPT-5.2 requires API key. Set --api-key or OPENAI_API_KEY or GPT52_API_KEY env var")
            sys.exit(1)
        builder = EventGraphBuilder(
            backend_type="openai",
            model_name=model_name,
            api_key=api_key,
        )

    elif args.model == "deepseek_v4_flash":
        model_name = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")
        api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.error("DeepSeek requires API key. Set --api-key or DEEPSEEK_API_KEY or OPENAI_API_KEY env var")
            sys.exit(1)
        builder = EventGraphBuilder(
            backend_type="openai",
            model_name=model_name,
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )

    elif args.model == "gemini_flash_lite":
        model_name = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
        api_key = args.api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.error("Gemini requires API key. Set --api-key or GEMINI_API_KEY or OPENAI_API_KEY env var")
            sys.exit(1)
        builder = EventGraphBuilder(
            backend_type="openai",
            model_name=model_name,
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    
    # Process all scenarios
    stats = builder.process_directory(input_dir, output_dir)
    
    # Save processing report
    report = {
        "timestamp": datetime.now().isoformat(),
        "input_directory": str(input_dir),
        "output_directory": str(output_dir),
        "statistics": stats
    }
    
    report_file = output_dir / "event_graph_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"\nReport saved to {report_file}")


if __name__ == "__main__":
    main()
