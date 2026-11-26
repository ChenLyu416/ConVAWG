"""
Hierarchical Event Graph Expansion

This script expands coarse-grained event graphs into detailed, real-life centric graphs by:
1. Decomposing each event into 1-5 sub-events based on escalation level:
   - Escalation 0 (normal): 1-3 sub-events
   - Escalation 1 (subtle control): 2-4 sub-events
   - Escalation 2-3 (manipulation/intimidation): 2-4 sub-events
   - Escalation 4 (severe violence): 3-5 sub-events
2. Assigning specific locations (online platforms or physical locations)
3. Maintaining persona and crime type consistency
4. Creating causal/temporal/enabling relations between sub-events

Note: Timestamps are now assigned in generate_chat_scripts.py with escalation-aware timeline rules.
"""
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta
import random
import sys
import argparse

# Add RAG_kb_construct to path for LLM backend
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "RAG_kb_construct"))

from llm_structurer.llm_backends import HuggingFaceBackend
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


def _extract_first_json(text: str):
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


def parse_duration_to_days(duration_str: str) -> int:
    """
    Parse duration string into number of days.
    
    Args:
        duration_str: Duration string like "18 months", "2 years", "6 weeks", "3-6 months"
    
    Returns:
        Number of days (approximated for months/years)
    """
    if not duration_str or duration_str.lower() in ['unknown', 'null', 'not specified']:
        return None
    
    duration_str = duration_str.lower().strip()
    
    # Handle ranges like "3-6 months" by taking the midpoint
    if '-' in duration_str:
        parts = duration_str.split('-')
        try:
            low = int(''.join(filter(str.isdigit, parts[0])))
            high = int(''.join(filter(str.isdigit, parts[1])))
            mid_value = (low + high) // 2
            # Extract unit from the second part
            unit = ''.join(filter(str.isalpha, parts[1])).strip()
        except (ValueError, IndexError):
            return None
    else:
        # Extract number and unit
        import re
        match = re.search(r'(\d+(?:\.\d+)?)\s*(\w+)', duration_str)
        if not match:
            return None
        mid_value = float(match.group(1))
        unit = match.group(2)
    
    # Convert to days based on unit
    if 'year' in unit:
        return int(mid_value * 365)
    elif 'month' in unit:
        return int(mid_value * 30)
    elif 'week' in unit:
        return int(mid_value * 7)
    elif 'day' in unit:
        return int(mid_value)
    else:
        return None


class EventGraphExpander:
    """Expands coarse-grained event graphs into detailed hierarchical structures."""
    
    def __init__(self, model_config_path: Path):
        """
        Initialize event graph expander.
        
        Args:
            model_config_path: Path to LLM config (e.g., gpt52_openai.yaml)
        """
        self.config = Config(str(model_config_path))
        
        # Initialize LLM backend based on config
        backend_type = self.config.llm_backend.lower()
        logger.info(f"Loading model: {self.config.model_name} (backend: {backend_type})")
        
        if backend_type == "huggingface":
            cuda_visible = self.config.get('llm.cuda_visible_devices', None)
            if cuda_visible:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible)
            self.backend = HuggingFaceBackend(
                model_name=self.config.model_name,
                load_in_8bit=self.config.get('llm.load_in_8bit', False),
                device_map=self.config.get('llm.device_map', 'auto'),
                trust_remote_code=self.config.get('llm.trust_remote_code', True),
            )
        elif backend_type == "openai":
            from llm_structurer.llm_backends import OpenAIBackend
            api_key = self.config.get('llm.api_key')
            if not api_key:
                # Use model-specific env vars before falling back to OPENAI_API_KEY
                base_url = self.config.get('llm.base_url') or ''
                model_name = self.config.model_name or ''
                if 'deepseek' in base_url.lower() or 'deepseek' in model_name.lower():
                    api_key = os.environ.get('DEEPSEEK_API_KEY') or os.environ.get('OPENAI_API_KEY')
                elif 'generativelanguage' in base_url.lower() or 'gemini' in model_name.lower():
                    api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('OPENAI_API_KEY')
                else:
                    api_key = self.config.get_openai_api_key()
            self.backend = OpenAIBackend(
                model_name=self.config.model_name,
                api_key=api_key,
                base_url=self.config.get('llm.base_url')
            )
        elif backend_type == "vllm":
            if VLLMBackend is None:
                raise ImportError("vLLM backend is not available. Install vllm package or use a different backend.")
            self.backend = VLLMBackend(
                model_name=self.config.model_name,
                base_url=self.config.get('llm.base_url', None),
                api_key=self.config.get('llm.api_key', None),
            )
        else:
            raise ValueError(f"Unsupported backend: {backend_type}. Supported: huggingface, openai, vllm")
        
        self.stats = {
            "total_graphs_processed": 0,
            "total_events_expanded": 0,
            "total_subevents_generated": 0,
            "failed": 0
        }
    
    def load_event_graph(self, filepath: Path) -> Dict[str, Any]:
        """Load event graph from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _infer_subevent_count(self,
                            event: Dict[str, Any],
                            event_index: int,
                            total_events: int,
                            previous_expansions: List[Dict[str, Any]] = None) -> int:
        """
        Infer the appropriate number of sub-events based on estimated escalation level:
        - Escalation 0: 1-2 sub-events (normal interactions)
        - Escalation 1: 1-3 sub-events (subtle control)
        - Escalation 2: 1-3 sub-events (clear manipulation)
        - Escalation 3: 2-3 sub-events (aggressive intimidation)
        - Escalation 4: 2-4 sub-events (severe threats/violence - needs detail)
        
        Returns:
            Target number of sub-events (1-4 depending on escalation)
        """
        # Calculate position in timeline (0.0 = beginning, 1.0 = end)
        timeline_position = event_index / max(1, total_events - 1) if total_events > 1 else 0.0
        
        # Extract violence/crime indicators
        crime_hints = event.get('attributes', {}).get('crime_type_hints', [])
        description = event.get('description', '').lower()
        action = event.get('action', '').lower()
        
        # Estimate escalation level based on content
        # (similar logic to _calculate_escalation_level in generate_chat_scripts.py)
        violence_keywords = ['assault', 'violence', 'physical', 'threat', 'harm', 'attack', 'hit', 'hurt']
        severe_threat_keywords = ['kill', 'murder', 'weapon', 'death', 'injure', 'destroy']
        
        has_violence = any(kw in description or kw in action for kw in violence_keywords)
        has_severe_threat = any(kw in description or kw in action for kw in severe_threat_keywords)
        has_stalking = 'stalking' in crime_hints or 'stalk' in description or 'follow' in description
        has_harassment = 'harassment' in crime_hints or 'harass' in description
        has_coercive = 'coercive_control' in crime_hints
        
        # Determine estimated escalation level
        if has_severe_threat or ('domestic_violence' in crime_hints and has_violence):
            estimated_escalation = 4
        elif has_violence or has_stalking or (has_coercive and timeline_position > 0.5):
            estimated_escalation = 3
        elif has_harassment or has_coercive or timeline_position > 0.33:
            estimated_escalation = 2
        elif timeline_position > 0.1:
            estimated_escalation = 1
        else:
            estimated_escalation = 0
        
        # Determine sub-event count based on estimated escalation level
        # Use random.randint to add variety within the specified ranges
        import random
        
        if estimated_escalation == 0:
            # Normal interactions: 1-2 sub-events
            target_count = random.randint(1, 2)
        elif estimated_escalation == 1:
            # Subtle control: 1-3 sub-events
            target_count = random.randint(1, 3)
        elif estimated_escalation == 2:
            # Clear manipulation: 1-3 sub-events
            target_count = random.randint(1, 3)
        elif estimated_escalation == 3:
            # Aggressive intimidation: 2-3 sub-events
            target_count = random.randint(2, 3)
        else:  # estimated_escalation == 4
            # Severe threats/violence: 2-4 sub-events (needs more detail)
            target_count = random.randint(2, 4)
        
        logger.info(f"    Estimated escalation {estimated_escalation} for {event.get('event_id')} -> {target_count} sub-events")
        return target_count
    
    def expand_event_to_subevents(
        self,
        event: Dict[str, Any],
        event_base_timestamp: str,
        graph_metadata: Dict[str, Any],
        event_index: int,
        total_events: int,
        previous_expansions: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Expand a single event into a hierarchical subgraph of sub-events.
        
        Args:
            event: The event to expand
            event_base_timestamp: ISO timestamp for this event's timeframe
            graph_metadata: Graph metadata including crime_types, personas
            event_index: Index of this event in the sequence
            total_events: Total number of events in the graph
            previous_expansions: List of previously expanded events for context
        
        Returns:
            Expanded event with subevents and sub-relations (1-5 sub-events depending on escalation)
        """
        
        # Infer target number of sub-events using LLM
        target_subevent_count = self._infer_subevent_count(
            event=event,
            event_index=event_index,
            total_events=total_events,
            previous_expansions=previous_expansions
        )
        # Prepare context for LLM
        event_id = event['event_id']
        description = event.get('description', '')
        text_span = event.get('text_span', '')
        action = event.get('action', '')
        agent = event.get('agent', {})
        patient = event.get('patient', {})
        time_info = event.get('time', 'unknown')
        location_info = event.get('location', 'unknown')
        crime_hints = event.get('attributes', {}).get('crime_type_hints', [])
        
        # Context from graph
        crime_types = graph_metadata.get('crime_types', [])
        time_duration = graph_metadata.get('time_duration', 'unknown')
        victim_name = graph_metadata.get('victim_name', 'victim')
        perp_name = graph_metadata.get('perpetrator_name', 'perpetrator')
        third_party_names = graph_metadata.get('third_party_names', [])
        
        # Build context from previous expansions
        previous_context = ""
        if previous_expansions:
            previous_context = "\nPREVIOUS EVENTS (for context and continuity):\n"
            for prev_exp in previous_expansions[-3:]:  # Only use last 3 for context
                prev_parent = prev_exp.get('parent_event', {})
                prev_id = prev_parent.get('event_id', 'unknown')
                prev_action = prev_parent.get('action', 'unknown')
                prev_goal = prev_exp.get('parent_goal', '')
                
                previous_context += f"\n{prev_id}: {prev_action}\n"
                previous_context += f"  Goal: {prev_goal}\n"
                previous_context += f"  Sub-events generated:\n"
                
                for sub in prev_exp.get('sub_events', [])[:3]:  # Show first 3 sub-events
                    sub_id = sub.get('sub_event_id', '')
                    sub_action = sub.get('action', '')
                    sub_desc = sub.get('description', '')
                    previous_context += f"    - {sub_id}: {sub_action} - {sub_desc}\n"
            
            previous_context += "\nEnsure the current event's sub-events build upon and reference these previous events where appropriate.\n"
        
        # Format third-party names for prompt
        third_party_list = ', '.join(third_party_names) if third_party_names else 'None available'
        
        prompt = f"""You are an expert at expanding high-level events into detailed, realistic sub-event sequences for VAWG (Violence Against Women and Girls) scenarios.

CONTEXT:
- Scenario ID: {graph_metadata.get('scenario_id', 'unknown')}
- Victim: {victim_name}
- Perpetrator: {perp_name}
- Third-Party Names Available: {third_party_list}
- Crime Types in Scenario: {', '.join(crime_types)}
- Total Duration of Abuse: {time_duration}
{previous_context}
PARENT EVENT TO EXPAND:
- Event ID: {event_id}
- Action: {action}
- Agent: {agent.get('name', 'unknown')} (role: {agent.get('role', 'unknown')})
- Patient: {patient.get('name', 'unknown')} (type: {patient.get('type', 'unknown')})
- Time: {time_info}
- Location: {location_info}
- Crime Type Hints: {', '.join(crime_hints) if crime_hints else 'None'}
- Text Span: "{text_span}"
- Description: "{description}"
- Event Position: {event_index + 1}/{total_events} ({event_index / max(1, total_events - 1) * 100:.0f}% through timeline)

TASK: Decompose this event into 2-4 realistic sub-events that build up to achieve the parent event's outcome.

**TARGET SUB-EVENT COUNT: {target_subevent_count}** (based on timeline position and violence level)

REQUIREMENTS:
1. **Goal Definition**: First identify the concrete outcome/goal of the parent event
2. **Sub-event Generation**: Create exactly {target_subevent_count} sub-events that realistically lead to this outcome
   - Target: {target_subevent_count} sub-events based on escalation level
   - Escalation 0 (normal): 1-3 sub-events
   - Escalation 1 (subtle control): 2-4 sub-events
   - Escalation 2-3 (manipulation/intimidation): 2-4 sub-events
   - Escalation 4 (severe violence): 3-6 sub-events
   - Each sub-event should be specific and actionable
   - Follow a logical temporal/causal progression
   - **AGENT FOCUS**: Prioritize perpetrator as agent in sub-events. Include third-party interactions where appropriate for realism (victim seeking support, friends checking in). Most sub-events should involve direct perpetrator actions.
3. **Timestamps**: Timestamps will be assigned later in the pipeline with escalation-aware rules
   - Parent event timestamp: {event_base_timestamp}
   - Note: Final timestamps will respect escalation milestones and ensure sub-events are commonly >1 day apart
4. **Locations**: Assign specific locations to each sub-event:
   - Online: Use "online" for all digital/social media interactions
   - Physical: Specific places like "victim's home", "perpetrator's apartment", "local library", "shopping center", "workplace", "school", "coffee shop", "park", "street near victim's home"
5. **Crime Type Tagging**: Mark crime_type (from parent or null) for each sub-event, following parent event's crime_type or its subset
6. **Relations**: Define edges between sub-events using: TEMPORAL_BEFORE, CAUSES, ENABLES, ESCALATES_TO, EVIDENCE_OF
7. **Persona Consistency**: Ensure agent roles match personas and event feasibility
8. **Name Assignment**: 
   - For victim/perpetrator: Use the exact names provided ({victim_name}, {perp_name})
   - For third-party roles: Select from available names ({third_party_list})
   - NEVER use "unknown" as a name if a specific role can be assigned
   - Only use "unknown" for abstract objects or organizations, not for people

OUTPUT SCHEMA (JSON only):
{{
  "parent_event_id": "{event_id}",
  "parent_goal": "<1 sentence describing the concrete outcome of the parent event>",
  "sub_events": [
    {{
      "sub_event_id": "{event_id}.1",
      "location": "<specific location string>",
      "location_type": "online|physical",
      "agent": {{
        "name": "<name>",
        "role": "victim|perpetrator|third_party|unknown"
      }},
      "action": "<verb or short action phrase>",
      "patient": {{
        "name": "<name or unknown>",
        "type": "person|org|object|unknown"
      }},
      "description": "<1-2 sentences describing this specific sub-event>",
      "crime_type": "<crime type from parent or null>"
    }}
  ],
  "sub_relations": [
    {{
      "src_sub_event_id": "{event_id}.1",
      "dst_sub_event_id": "{event_id}.2",
      "relation_type": "TEMPORAL_BEFORE|CAUSES|ENABLES|ESCALATES_TO|EVIDENCE_OF",
      "confidence": 0.0,
      "evidence": "<brief justification>"
    }}
  ]
}}

EXAMPLES OF SUB-EVENT DECOMPOSITION:

Example 1 - Coercive Control Event:
Parent: "Nathan monitors Emily's phone and social media"
Sub-events:
1. Nathan demands Emily's phone password (online, evening)
2. Emily reluctantly shares password (online, 30 min later)
3. Nathan secretly installs tracking app on Emily's phone (victim's home, next morning)
4. Nathan checks Emily's messages daily (perpetrator's apartment, ongoing over 2 weeks)
5. Nathan confronts Emily about a male friend's message (victim's home, 2 weeks later)

Example 2 - Escalation Event:
Parent: "Relationship dynamic shifts to volatility"
Sub-events:
1. James becomes irritable over small issues (victim's home, afternoon)
2. Sarah questions his mood changes (victim's home, evening)
3. James raises voice and dismisses her concerns (victim's home, same evening)
4. Sarah withdraws and avoids conversation (victim's home, next day)

IMPORTANT GUIDELINES:
- Sub-events should be GROUNDED in realistic behaviors seen in VAWG cases
- Online interactions should use location "online" (no platform specifics needed)
- Physical locations should be specific (not just "home" but "victim's home" or "perpetrator's apartment")
- Time progression should be realistic (minutes for chat exchanges, hours/days for behavioral patterns)
- Maintain consistency with parent event's crime_type and personas
- Sub-event IDs must follow pattern: ParentID.1, ParentID.2, etc.
- **CONTINUITY**: If previous events were provided, ensure logical continuity and reference earlier outcomes when relevant

NOW EXPAND THE PARENT EVENT ABOVE.

JSON OUTPUT:"""
        
        try:
            response = self.backend.generate(
                prompt=prompt,
                max_tokens=3072,
                temperature=0.4
            ).strip()
            
            # Clean response
            response = response.replace('```json', '').replace('```', '').strip()

            # Strip <think>...</think> reasoning chains emitted by reasoning models
            # (e.g. DeepSeek-R1/V3-Flash, Qwen3) — the actual JSON answer comes AFTER
            # the closing </think> tag; extracting from inside the thinking section
            # returns a wrong/partial JSON that lacks sub_events/parent_goal keys.
            import re as _re
            response = _re.sub(r'<think>.*?</think>', '', response, flags=_re.DOTALL).strip()

            # Parse JSON — use balanced-brace extraction to handle extra trailing
            # text that weaker models (e.g. LLaMA-8B) append after the JSON block
            expansion = _extract_first_json(response)
            if expansion is None:
                logger.error(f"Could not extract JSON for event {event_id}; skipping expansion.")
                return {
                    'parent_event': event,
                    'parent_goal': '',
                    'sub_events': [],
                    'sub_relations': []
                }

            # Validate: if required keys are missing the model likely returned an
            # error body or the wrong JSON fragment — log and treat as failure.
            if 'sub_events' not in expansion:
                logger.warning(
                    f"Parsed JSON for {event_id} missing 'sub_events' key. "
                    f"Keys found: {list(expansion.keys())}. "
                    f"Response snippet: {response[:300]!r}"
                )
                return {
                    'parent_event': event,
                    'parent_goal': '',
                    'sub_events': [],
                    'sub_relations': []
                }

            # Validate and post-process
            sub_events = expansion.get('sub_events', [])
            sub_relations = expansion.get('sub_relations', [])
            
            logger.info(f"  Expanded {event_id} into {len(sub_events)} sub-events with {len(sub_relations)} relations")
            
            # Note: Timestamps will be assigned later in generate_chat_scripts.py
            # with escalation-aware rules (20%/40% milestones, >1 day sub-event spacing)
            
            self.stats['total_subevents_generated'] += len(sub_events)
            
            return {
                'parent_event': event,
                'parent_goal': expansion.get('parent_goal', ''),
                'sub_events': sub_events,
                'sub_relations': sub_relations
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse expansion JSON for {event_id}: {e}")
            logger.error(f"Response: {response[:500]}")
            return {
                'parent_event': event,
                'parent_goal': '',
                'sub_events': [],
                'sub_relations': []
            }
        except Exception as e:
            logger.error(f"Error expanding event {event_id}: {e}")
            return {
                'parent_event': event,
                'parent_goal': '',
                'sub_events': [],
                'sub_relations': []
            }
    
    def expand_graph(self, graph: Dict[str, Any]) -> Dict[str, Any]:
        """
        Expand entire event graph hierarchically.
        
        Args:
            graph: Input event graph
        
        Returns:
            Expanded event graph with hierarchical structure
        """
        scenario_id = graph.get('scenario_id', 'unknown')
        logger.info(f"Expanding event graph for scenario {scenario_id}")
        
        events = graph.get('events', [])
        relations = graph.get('relations', [])
        metadata = graph.get('metadata', {})
        
        if not events:
            logger.warning(f"  No events to expand in scenario {scenario_id}")
            return None
        
        # Extract time_duration from metadata
        time_duration = metadata.get('time_duration', None)
        
        # Extract timestamps from events (now generated by LLM in build_event_graph)
        event_timestamps = [event.get('timestamp', 'unknown') for event in events]
        logger.info(f"  Using LLM-generated event timestamps: {event_timestamps[0]} to {event_timestamps[-1]}")
        
        # Expand each event
        expanded_events = []
        for i, event in enumerate(events):
            logger.info(f"  Expanding event {i+1}/{len(events)}: {event.get('event_id')}")
            
            # Use the event's own timestamp (generated by LLM in build_event_graph)
            event_timestamp = event.get('timestamp', 'unknown')
            
            expansion = self.expand_event_to_subevents(
                event=event,
                event_base_timestamp=event_timestamp,
                graph_metadata={
                    'scenario_id': scenario_id,
                    'crime_types': metadata.get('crime_types', []),
                    'time_duration': time_duration,
                    'victim_name': graph.get('victim_name', 'victim'),
                    'perpetrator_name': graph.get('perpetrator_name', 'perpetrator'),
                    'third_party_names': graph.get('third_party_names', [])
                },
                event_index=i,
                total_events=len(events),
                previous_expansions=expanded_events  # Pass previous expansions for context
            )
            
            expanded_events.append(expansion)
            self.stats['total_events_expanded'] += 1
        
        # Build expanded graph structure
        expanded_graph = {
            'scenario_id': scenario_id,
            'victim_name': graph.get('victim_name'),
            'perpetrator_name': graph.get('perpetrator_name'),
            'third_party_names': graph.get('third_party_names', []),
            'original_events': events,
            'original_relations': relations,
            'expanded_events': expanded_events,
            'timeline_span_days': self._calculate_span_days(event_timestamps),
            'metadata': {
                **metadata,
                'expansion_timestamp': datetime.now().isoformat(),
                'event_timestamps': event_timestamps,
                'timestamp_generation': 'LLM-based contextual'
            }
        }
        
        self.stats['total_graphs_processed'] += 1
        return expanded_graph
    
    def _calculate_span_days(self, timestamps: List[str]) -> int:
        """Calculate span in days between first and last timestamp."""
        if len(timestamps) < 2:
            return 0
        
        try:
            first = datetime.fromisoformat(timestamps[0])
            last = datetime.fromisoformat(timestamps[-1])
            return (last - first).days
        except:
            return 0
    
    def save_expanded_graph(self, expanded_graph: Dict[str, Any], output_path: Path) -> None:
        """
        Save expanded event graph to JSON file.
        
        Args:
            expanded_graph: Expanded graph structure
            output_path: Path to save JSON file
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(expanded_graph, f, indent=2, ensure_ascii=False)
        
        logger.info(f"  Saved expanded graph to {output_path.name}")
    
    def process_graph_file(
        self,
        input_path: Path,
        output_dir: Path
    ) -> bool:
        """
        Process a single event graph file and generate expanded version.
        
        Args:
            input_path: Path to input event graph JSON
            output_dir: Directory to save output expanded graph
        
        Returns:
            True if successful, False otherwise
        """
        try:
            graph = self.load_event_graph(input_path)
            expanded_graph = self.expand_graph(graph)
            
            if expanded_graph:
                output_path = output_dir / input_path.name
                self.save_expanded_graph(expanded_graph, output_path)
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
        Process all event graph files in a directory.
        
        Args:
            input_dir: Directory containing event graph JSON files
            output_dir: Directory to save expanded graphs
        
        Returns:
            Statistics dictionary
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        _REPORT_NAMES = {"event_graph_report.json", "expansion_report.json", "generation_report.json"}
        graph_files = sorted(f for f in input_dir.glob("*.json") if f.name not in _REPORT_NAMES)
        logger.info(f"Found {len(graph_files)} event graphs to expand")
        logger.info(f"Output directory: {output_dir}")
        logger.info("=" * 80)
        
        for i, filepath in enumerate(graph_files, 1):
            logger.info(f"\n[{i}/{len(graph_files)}] Processing {filepath.name}")
            self.process_graph_file(filepath, output_dir)
            
            # Log progress
            if i % 5 == 0 or i == len(graph_files):
                logger.info(f"\nProgress: {i}/{len(graph_files)} completed")
                logger.info(f"Total events expanded: {self.stats['total_events_expanded']}")
                logger.info(f"Total sub-events generated: {self.stats['total_subevents_generated']}")
                logger.info(f"Failed: {self.stats['failed']}")
        
        logger.info("\n" + "=" * 80)
        logger.info("EVENT GRAPH EXPANSION COMPLETE")
        logger.info(f"Total graphs processed: {self.stats['total_graphs_processed']}")
        logger.info(f"Total events expanded: {self.stats['total_events_expanded']}")
        logger.info(f"Total sub-events generated: {self.stats['total_subevents_generated']}")
        logger.info(f"Average sub-events per event: {self.stats['total_subevents_generated'] / max(1, self.stats['total_events_expanded']):.1f}")
        logger.info(f"Failed: {self.stats['failed']}")
        
        return self.stats


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Expand event graphs with configurable LLM backends"
    )
    parser.add_argument(
        "--model",
        choices=["gpt52", "deepseek_v4_flash", "gemini_flash_lite", "qwen35_35b", "llama3_8b"],
        default="gpt52",
        help="Preset model alias to use (default: gpt52)"
    )
    parser.add_argument(
        "--model-config",
        type=str,
        help="Custom YAML config path. Overrides --model when provided."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Input directory containing event graph JSON files (default: ./example_graph)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for expanded graphs (default: ./expanded_graphs/run_TIMESTAMP)"
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Paths
    base_dir = Path(__file__).parent.parent.parent
    
    # Model preset configs
    preset_configs = {
        "gpt52": base_dir / "RAG_kb_construct" / "configs" / "gpt52_openai.yaml",
        "deepseek_v4_flash": base_dir / "RAG_kb_construct" / "configs" / "deepseek_v4_flash_openai.yaml",
        "gemini_flash_lite": base_dir / "RAG_kb_construct" / "configs" / "gemini_flash_lite_openai.yaml",
        "qwen35_35b": base_dir / "RAG_kb_construct" / "configs" / "qwen35_35b_vllm.yaml",
        "llama3_8b": base_dir / "RAG_kb_construct" / "configs" / "llama3_8b_vllm.yaml",
    }
    
    # Determine model config
    if args.model_config:
        model_config = Path(args.model_config)
    else:
        model_config = preset_configs[args.model]
    
    # Determine input directory
    if args.input_dir:
        input_dir = Path(args.input_dir)
    else:
        input_dir = Path(__file__).parent / "example_graph"
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent / "expanded_graphs"
    
    # Verify paths
    if not model_config.exists():
        logger.error(f"Model config not found: {model_config}")
        sys.exit(1)
    
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)
    
    logger.info(f"Model: {args.model if not args.model_config else 'custom'}")
    logger.info(f"Config: {model_config}")
    logger.info(f"Input: {input_dir}")
    logger.info(f"Output: {output_dir}")
    
    # Initialize expander
    expander = EventGraphExpander(model_config_path=model_config)
    
    # Process all graphs
    stats = expander.process_directory(input_dir, output_dir)
    
    # Save processing report
    report = {
        "timestamp": datetime.now().isoformat(),
        "input_directory": str(input_dir),
        "output_directory": str(output_dir),
        "statistics": stats
    }
    
    report_file = output_dir / "expansion_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"\nReport saved to {report_file}")


if __name__ == "__main__":
    main()
