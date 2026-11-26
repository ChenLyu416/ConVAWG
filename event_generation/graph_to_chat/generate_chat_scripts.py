"""
Generate Ready-to-Use Chat Scripts from Expanded Event Graphs

This script converts expanded event graphs into chat scripts and generates actual conversations using LLM.
It processes two types of events:
1. Online events: Directly converted into chat scripts with detailed content guidance
2. Real-world events: Expanded with 1-3 additional online-chat events to cover the information flow

All events are processed in temporal order to ensure coherent information flow.
Actual conversations are generated using LLM with context from previous events.
"""

import json
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
import random
import re

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "RAG_kb_construct"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "persona_generate"))

try:
    from llm_structurer.llm_backends import HuggingFaceBackend, VLLMBackend
    from llm_structurer.config import Config
    print("Successfully imported LLM backends")
except ImportError as e:
    print(f"Warning: Could not import backends ({e}), will create simple implementation")
    HuggingFaceBackend = None
    VLLMBackend = None
    Config = None


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


class ChatScriptGenerator:
    """Generates chat scripts from expanded event graphs and generates actual conversations using LLM."""

    # ── Fallback name pools (mirrors build_event_graph._TP_NAMES_*) ─────────────────
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

    @staticmethod
    def _is_placeholder_name(name: str) -> bool:
        """Return True if *name* is a known placeholder that should be replaced."""
        if not name:
            return True
        lower = name.strip().lower()
        return lower in ("unknown", "n/a", "none", "", "unnamed") or lower.startswith("third_party_")

    @classmethod
    def _generate_fallback_name(cls, used: set, gender: str = "") -> str:
        """Pick a fresh third-party name from the gender-appropriate pool."""
        pool = cls._TP_NAMES_FEMALE if "female" in gender.lower() else cls._TP_NAMES_MALE
        available = [n for n in pool if n not in used]
        if not available:
            alt = cls._TP_NAMES_MALE if "female" in gender.lower() else cls._TP_NAMES_FEMALE
            available = [n for n in alt if n not in used] or pool
        chosen = random.choice(available)
        used.add(chosen)
        return chosen

    @classmethod
    def _sanitize_third_party_names(
        cls, names: List[str], reserved: set
    ) -> List[str]:
        """Replace any placeholder entries in *names* with generated names."""
        out = []
        used = set(reserved)
        for name in names:
            if cls._is_placeholder_name(name):
                name = cls._generate_fallback_name(used)
            used.add(name)
            out.append(name)
        return out
    
    def __init__(self, output_dir: Path, config_path: Path = None, model_name: str = None, use_llm: bool = True):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_llm = use_llm
        self.config_path = config_path
        self.model_name = model_name
        self.config = None
        
        # Track escalation level across events
        self.escalation_level = 0
        self.previous_escalation_level = 0  # Track previous level for realistic patterns
        self.high_escalation_event_occurred = False  # Track if a high escalation event just occurred
        
        # Track timeline progression for escalation calculation
        self.total_events = 0
        self.current_event_index = 0
        
        # Track escalation milestones for timeline assignment
        self.first_esc2_index = None
        self.first_esc3_index = None
        
        # Initialize LLM backend if requested
        self.llm_backend = None
        if use_llm:
            self._initialize_llm()
        
        # Template for messaging (only enabled chat type)
        self.chat_templates = {
            "messaging": "Text messaging or instant messaging"
        }
    
    def _initialize_llm(self):
        """Initialize the LLM backend based on config or model name."""
        try:
            if self.config_path:
                # Use config file
                if not Config:
                    print("  ✗ Config class not available")
                    self.use_llm = False
                    return
                
                self.config = Config(str(self.config_path))
                backend_type = self.config.llm_backend.lower()
                model_name = self.config.model_name
                print(f"Initializing LLM from config: {model_name} (backend: {backend_type})")
                
                if backend_type == "huggingface":
                    if not HuggingFaceBackend:
                        print("  ✗ HuggingFace backend not available")
                        self.use_llm = False
                        return
                    cuda_visible = self.config.get('llm.cuda_visible_devices', None)
                    if cuda_visible:
                        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible)
                    self.llm_backend = HuggingFaceBackend(
                        model_name=model_name,
                        load_in_8bit=self.config.get('llm.load_in_8bit', False),
                        device_map=self.config.get('llm.device_map', 'auto'),
                        trust_remote_code=self.config.get('llm.trust_remote_code', True),
                    )
                elif backend_type == "openai":
                    from llm_structurer.llm_backends import OpenAIBackend
                    api_key = self.config.get('llm.api_key')
                    if not api_key:
                        # Use model-specific env vars before falling back to OPENAI_API_KEY
                        _base_url = self.config.get('llm.base_url') or ''
                        _model_name = model_name or ''
                        if 'deepseek' in _base_url.lower() or 'deepseek' in _model_name.lower():
                            api_key = os.environ.get('DEEPSEEK_API_KEY') or os.environ.get('OPENAI_API_KEY')
                        elif 'generativelanguage' in _base_url.lower() or 'gemini' in _model_name.lower():
                            api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('OPENAI_API_KEY')
                        else:
                            try:
                                api_key = self.config.get_openai_api_key()
                            except ValueError as e:
                                print(f"  ✗ {e}")
                                print(f"  ✗ Please set OPENAI_API_KEY environment variable")
                                self.use_llm = False
                                return
                    if not api_key:
                        print(f"  ✗ API key not found for model {model_name}")
                        print(f"  ✗ Please set OPENAI_API_KEY / DEEPSEEK_API_KEY / GEMINI_API_KEY environment variable")
                        self.use_llm = False
                        return
                    self.llm_backend = OpenAIBackend(
                        model_name=model_name,
                        api_key=api_key,
                        base_url=self.config.get('llm.base_url')
                    )
                elif backend_type == "vllm":
                    if not VLLMBackend:
                        print("  ✗ vLLM backend not available")
                        self.use_llm = False
                        return
                    self.llm_backend = VLLMBackend(
                        model_name=model_name,
                        base_url=self.config.get('llm.base_url', None),
                        api_key=self.config.get('llm.api_key', None),
                    )
                else:
                    print(f"  ✗ Unsupported backend: {backend_type}")
                    self.use_llm = False
                    return
                
                self.model_name = model_name
                print("  ✓ LLM backend initialized successfully")
            
            elif self.model_name:
                # Legacy: direct model name (assume HuggingFace)
                print(f"Initializing LLM (HuggingFace): {self.model_name}")
                if not HuggingFaceBackend:
                    print("  ✗ HuggingFace backend not available")
                    self.use_llm = False
                    return
                
                self.llm_backend = HuggingFaceBackend(
                    model_name=self.model_name,
                    load_in_8bit=False,
                    device_map="auto"
                )
                print("  ✓ LLM backend initialized successfully")
            else:
                print("  ✗ No config or model name provided")
                self.use_llm = False
                
        except Exception as e:
            print(f"  ✗ Failed to initialize LLM: {e}")
            import traceback
            traceback.print_exc()
            print("  Will only generate prompts without actual LLM generation")
            self.use_llm = False
    
    def generate_summary(self, prompt: str, max_tokens: int = 4000) -> Optional[str]:
        """
        Generate dialogue script/summary using LLM.
        
        Args:
            prompt: The generation prompt
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated script/summary text or None if generation fails
        """
        if not self.use_llm or not self.llm_backend:
            return None
        
        try:
            response = self.llm_backend.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response
        except Exception as e:
            print(f"      Warning: Summary generation failed: {e}")
            return None
        
    def identify_event_type(self, sub_event: Dict[str, Any]) -> str:
        """
        Identify if a sub-event is online or real-world.
        
        Returns:
            'online' or 'physical'
        """
        location_type = sub_event.get("location_type", "").lower()
        location = sub_event.get("location", "").lower()
        
        if location_type == "online" or location == "online":
            return "online"
        return "physical"
    
    def infer_chat_platform(self, sub_event: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Infer the most appropriate chat platform based on event context."""
        # Always return messaging as the only enabled chat type
        return "messaging"
    
    def generate_online_chat_script(self, sub_event: Dict[str, Any], 
                                   context: Dict[str, Any],
                                   previous_conversations: List[str]) -> Dict[str, Any]:
        """
        Generate a chat script for an online event.
        
        Args:
            sub_event: The online sub-event
            context: Context from parent event and scenario
            previous_conversations: Actual generated summaries from previous events for coherence
            
        Returns:
            Chat script dictionary with generated summary
        """
        platform = self.infer_chat_platform(sub_event, context)
        
        # Calculate escalation level for this event
        escalation_level = self._calculate_escalation_level(sub_event, context)
        
        # Create the generation prompt for summary
        generation_prompt = self._create_summary_prompt(
            sub_event, context, previous_conversations, escalation_level
        )
        
        # Generate the conversation summary
        conversation_summary = None
        if self.use_llm:
            print(f"    Generating summary for {sub_event['sub_event_id']}...")
            conversation_summary = self.generate_summary(generation_prompt)
            if conversation_summary:
                print(f"      ✓ Generated summary ({len(conversation_summary)} characters)")
            else:
                print(f"      ✗ Generation failed")
        
        # Check if conversation involves both victim and perpetrator (not third party)
        sender_role = sub_event["agent"]["role"]
        receiver_role = context.get("receiver_role", "unknown")
        involves_third_party = sender_role == "third_party" or receiver_role == "third_party"
        
        # Generate role-specific style notes and red flags only for victim-perpetrator conversations
        if not involves_third_party:
            perpetrator_style_notes = self._generate_perpetrator_style_notes(sub_event, context, escalation_level)
            victim_style_notes = self._generate_victim_style_notes(sub_event, context, escalation_level)
            style_notes_dict = {
                "perpetrator": perpetrator_style_notes,
                "victim": victim_style_notes
            }
            
            perpetrator_red_flags = self._identify_perpetrator_red_flags(sub_event, context)
            victim_red_flags = self._identify_victim_red_flags(sub_event, context)
            red_flags_dict = {
                "perpetrator": perpetrator_red_flags,
                "victim": victim_red_flags
            }
        else:
            # For third-party conversations, use generic style notes
            style_notes_dict = {
                "general": ["Use realistic messaging language (contractions, informal grammar)", 
                           "Natural conversation flow appropriate to context"]
            }
            red_flags_dict = {}
        
        chat_script = {
            "event_id": sub_event["sub_event_id"],
            "timestamp": sub_event["timestamp"],
            "event_type": "online_chat",
            "platform": platform,
            "platform_description": self.chat_templates.get(platform, "Online communication"),
            "participants": {
                "sender": {
                    "name": sub_event["agent"]["name"],
                    "role": sub_event["agent"]["role"]
                },
                "receiver": {
                    "name": sub_event["patient"]["name"],
                    "role": context.get("receiver_role", "unknown")
                }
            },
            "conversation_context": {
                "action": sub_event["action"],
                "description": sub_event["description"],
                "crime_type": sub_event.get("crime_type"),
                "tone": self._infer_tone(sub_event, context),
                "emotional_state": self._infer_emotional_state(sub_event, context),
                "escalation_level": escalation_level
            },
            "content_guidance": {
                "main_points": self._extract_main_points(sub_event, context),
                "suggested_length": self._suggest_message_length(sub_event, context),
                "style_notes": style_notes_dict,
                "red_flags": red_flags_dict
            },
            "conversation_summary": conversation_summary,
            "generation_success": conversation_summary is not None
        }
        
        return chat_script
    
    def _generate_expansion_timestamps_with_llm(self, physical_event: Dict[str, Any],
                                                 num_expansions: int,
                                                 expansion_types: List[str]) -> Dict[str, str]:
        """
        Generate timestamps for expanded chat events using LLM for contextual realism.
        
        Args:
            physical_event: The physical event being expanded
            num_expansions: Number of chat events to create (1-3)
            expansion_types: List of expansion types (e.g., ['planning', 'during', 'aftermath'])
        
        Returns:
            Dictionary mapping expansion type to timestamp string
        """
        if not self.use_llm or not self.llm_backend:
            # Fallback to algorithmic generation
            return self._fallback_expansion_timestamps(physical_event, expansion_types)
        
        parent_timestamp = physical_event["timestamp"]
        event_description = physical_event.get("description", "")
        action = physical_event.get("action", "")
        crime_type = physical_event.get("crime_type", "None")
        
        prompt = f"""Generate realistic timestamps for online chat events related to a physical event.

Physical Event:
- Timestamp: {parent_timestamp}
- Action: {action}
- Description: {event_description}
- Crime Type: {crime_type}

Generate {num_expansions} chat event timestamps:
{chr(10).join([f'{i+1}. {t.upper()} chat: conversation that occurs {self._get_timing_hint(t)} the physical event' for i, t in enumerate(expansion_types)])}

Rules:
- Use YYYY-MM-DD HH:MM:SS format
- PLANNING chats: hours to days BEFORE the physical event (victims may discuss concerns, perpetrators may coordinate)
- DURING chats: minutes to hours around the physical event time (real-time reactions, distress signals)
- AFTERMATH chats: hours to days AFTER the physical event (processing, follow-up, continued threats)
- Consider the event's nature: urgent vs planned, violent vs coercive
- Timestamps should reflect realistic communication patterns

Output ONLY a JSON object with this exact format:
{{
{chr(10).join([f'  "{t}": "YYYY-MM-DD HH:MM:SS"' + (',' if i < len(expansion_types)-1 else '') for i, t in enumerate(expansion_types)])}
}}"""
        
        try:
            response = self.llm_backend.generate(
                prompt=prompt,
                max_tokens=200,
                temperature=0.3
            ).strip()
            
            # Clean response
            response = response.replace('```json', '').replace('```', '').strip()
            timestamps = _extract_first_json(response)
            
            # Validate all required timestamps present
            for exp_type in expansion_types:
                if exp_type not in timestamps:
                    print(f"  Warning: Missing {exp_type} timestamp, using fallback")
                    return self._fallback_expansion_timestamps(physical_event, expansion_types)
            
            return timestamps
            
        except Exception as e:
            print(f"  Warning: LLM timestamp generation failed ({e}), using fallback")
            return self._fallback_expansion_timestamps(physical_event, expansion_types)
    
    def _get_timing_hint(self, expansion_type: str) -> str:
        """Get timing hint for prompt."""
        hints = {
            'planning': 'before',
            'aftermath': 'after'
        }
        return hints.get(expansion_type, 'around')
    
    def _fallback_expansion_timestamps(self, physical_event: Dict[str, Any],
                                       expansion_types: List[str]) -> Dict[str, str]:
        """Fallback timestamp generation using time deltas, adjusted by escalation level."""
        base_timestamp = datetime.fromisoformat(physical_event["timestamp"])
        timestamps = {}
        
        # Adjust time gaps based on escalation level
        # Low escalation (0-1): longer gaps (days to weeks)
        # Medium escalation (2): moderate gaps (hours to days)
        # High escalation (3-4): short gaps (minutes to hours)
        
        if self.escalation_level <= 1:
            planning_gap_hours = random.randint(48, 168)  # 2-7 days
            aftermath_gap_hours = random.randint(12, 72)  # 12 hours to 3 days
        elif self.escalation_level == 2:
            planning_gap_hours = random.randint(12, 48)  # 12 hours to 2 days
            aftermath_gap_hours = random.randint(4, 24)  # 4-24 hours
        elif self.escalation_level == 3:
            planning_gap_hours = random.randint(2, 12)  # 2-12 hours
            aftermath_gap_hours = random.randint(1, 6)  # 1-6 hours
        else:  # Level 4
            planning_gap_hours = random.randint(1, 4)  # 1-4 hours
            aftermath_gap_hours = random.randint(1, 3)  # 1-3 hours
        
        for exp_type in expansion_types:
            if exp_type == 'planning':
                ts = base_timestamp - timedelta(hours=planning_gap_hours)
            else:  # aftermath
                ts = base_timestamp + timedelta(hours=aftermath_gap_hours)
            timestamps[exp_type] = ts.strftime("%Y-%m-%d %H:%M:%S")
        
        return timestamps
    
    def expand_physical_event(self, sub_event: Dict[str, Any], 
                            context: Dict[str, Any],
                            previous_conversations: List[str]) -> List[Dict[str, Any]]:
        """
        Expand a physical event into 1-2 online chat events: a pre-event planning
        conversation, optionally followed by a post-event aftermath conversation.
        No 'during' expansion is used.
        """
        num_expansions = random.randint(1, 2)
        expanded_events = []
        
        # Determine which expansion types to create (planning always, aftermath if 2nd slot)
        expansion_types = ['planning']
        if num_expansions >= 2:
            expansion_types.append('aftermath')
        
        # Generate timestamps using LLM
        expansion_timestamps = self._generate_expansion_timestamps_with_llm(
            sub_event, num_expansions, expansion_types
        )
        
        # Create events with LLM-generated timestamps
        for exp_type in expansion_types:
            timestamp_str = expansion_timestamps.get(exp_type)
            if not timestamp_str:
                continue
            
            # Parse timestamp (handle both formats)
            try:
                if ' ' in timestamp_str:
                    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                else:
                    timestamp = datetime.fromisoformat(timestamp_str)
            except:
                # Fallback if parsing fails
                base_ts = datetime.fromisoformat(sub_event["timestamp"])
                if exp_type == 'planning':
                    timestamp = base_ts - timedelta(hours=12)
                else:  # aftermath
                    timestamp = base_ts + timedelta(hours=6)
            
            if exp_type == 'planning':
                event = self._create_planning_chat(
                    sub_event, context, timestamp, previous_conversations
                )
            else:  # aftermath
                event = self._create_aftermath_chat(
                    sub_event, context, timestamp, previous_conversations
                )
            
            expanded_events.append(event)
        
        return expanded_events
    
    def _create_planning_chat(self, physical_event: Dict[str, Any], 
                            context: Dict[str, Any],
                            timestamp: datetime,
                            previous_info: List[str]) -> Dict[str, Any]:
        """Create a chat event that happens before the physical event (planning)."""
        agent = physical_event["agent"]
        patient = physical_event["patient"]
        action = physical_event.get("action", "")
        
        # Generate synthetic sub-event for planning
        planning_event = {
            "sub_event_id": f"{physical_event['sub_event_id']}.pre",
            "timestamp": timestamp.isoformat(),
            "location": "online",
            "location_type": "online",
            "agent": agent,
            "action": "discusses plans",
            "patient": patient,
            "description": f"{agent['name']} discusses plans with {patient['name']} leading up to {action}",
            "crime_type": None
        }
        
        platform = self.infer_chat_platform(planning_event, context)
        escalation_level = self._calculate_escalation_level(physical_event, context)
        
        # Check if conversation involves both victim and perpetrator (not third party)
        sender_role = agent["role"]
        receiver_role = context.get("receiver_role", "unknown")
        involves_third_party = sender_role == "third_party" or receiver_role == "third_party"
        
        # Generate role-specific style notes and red flags only for victim-perpetrator conversations
        if not involves_third_party:
            perpetrator_style_notes = self._generate_perpetrator_style_notes(physical_event, context, escalation_level)
            victim_style_notes = self._generate_victim_style_notes(physical_event, context, escalation_level)
            style_notes_dict = {
                "perpetrator": perpetrator_style_notes,
                "victim": victim_style_notes
            }
            
            perpetrator_red_flags = self._identify_perpetrator_red_flags(physical_event, context)
            victim_red_flags = self._identify_victim_red_flags(physical_event, context)
            red_flags_dict = {
                "perpetrator": perpetrator_red_flags,
                "victim": victim_red_flags
            }
        else:
            # For third-party conversations, use generic style notes
            style_notes_dict = {
                "general": ["Use realistic messaging language", "Natural conversation flow"]
            }
            red_flags_dict = {}
        
        return {
            "event_id": planning_event["sub_event_id"],
            "timestamp": timestamp.isoformat(),
            "event_type": "online_chat_expansion",
            "expansion_purpose": "planning",
            "original_physical_event": physical_event["sub_event_id"],
            "platform": platform,
            "platform_description": self.chat_templates.get(platform, "Online communication"),
            "participants": {
                "sender": {
                    "name": agent["name"],
                    "role": agent["role"]
                },
                "receiver": {
                    "name": patient["name"],
                    "role": context.get("receiver_role", "unknown")
                }
            },
            "conversation_context": {
                "action": "discusses plans",
                "description": planning_event["description"],
                "purpose": "Coordinate or hint at the upcoming physical event",
                "tone": self._infer_tone(physical_event, context),
                "emotional_state": "anticipatory",
                "escalation_level": escalation_level
            },
            "content_guidance": {
                "main_points": [
                    f"Reference or suggest the upcoming activity: {action}",
                    f"Establish context for meeting at {physical_event.get('location', 'a location')}",
                    "Maintain natural conversation flow leading to the event"
                ],
                "suggested_length": "2-4 messages exchanged",
                "style_notes": style_notes_dict,
                "red_flags": red_flags_dict,
                "information_to_convey": f"The physical event ({action}) is being planned or anticipated"
            },
            "coherence_requirements": {
                "previous_context": previous_info[-3:] if previous_info else [],
                "information_to_convey": f"Plans leading to: {physical_event['description']}",
                "follow_up_hooks": [f"The actual event at {physical_event.get('location', 'the location')}"]
            }
        }
        
        # Note: Summary generation is handled in the main generate_online_chat_script
        # Expansion events don't generate their own summaries to avoid duplication
        
        return planning_event
    
    def _create_during_chat(self, physical_event: Dict[str, Any],
                          context: Dict[str, Any],
                          timestamp: datetime,
                          previous_info: List[str]) -> Dict[str, Any]:
        """Create a chat event during/immediately after physical event (distress/update)."""
        agent = physical_event["agent"]
        patient = physical_event["patient"]
        crime_type = physical_event.get("crime_type")
        
        # Determine who initiates (victim reaching out or perpetrator checking in)
        if agent["role"] == "perpetrator":
            # Perpetrator checking in or exerting control
            initiator = agent
            receiver = patient
            action = "checks in"
            description = f"{agent['name']} sends messages checking on {patient['name']}'s location or activities"
        else:
            # Victim might reach out to support
            initiator = agent
            receiver = {"name": next(
                (n for n in context.get("third_party_names", []) if not self._is_placeholder_name(n)),
                self._generate_fallback_name({context.get("victim_name", ""), context.get("perpetrator_name", "")})
            ), "role": "third_party"}
            action = "reaches out"
            description = f"{agent['name']} sends distress signals or updates to a friend"
        
        during_event = {
            "sub_event_id": f"{physical_event['sub_event_id']}.during",
            "timestamp": timestamp.isoformat(),
            "location": "online",
            "location_type": "online",
            "agent": initiator,
            "action": action,
            "patient": receiver,
            "description": description,
            "crime_type": crime_type
        }
        
        platform = "messaging"  # During events typically use quick messaging
        escalation_level = self._calculate_escalation_level(physical_event, context)
        
        # Check if conversation involves both victim and perpetrator (not third party)
        sender_role = initiator["role"]
        receiver_role = receiver.get("role", "unknown")
        involves_third_party = sender_role == "third_party" or receiver_role == "third_party"
        
        # Generate role-specific style notes and red flags only for victim-perpetrator conversations
        if not involves_third_party:
            perpetrator_style_notes = self._generate_perpetrator_style_notes(physical_event, context, escalation_level)
            victim_style_notes = self._generate_victim_style_notes(physical_event, context, escalation_level)
            style_notes_dict = {
                "perpetrator": perpetrator_style_notes,
                "victim": victim_style_notes
            }
            
            perpetrator_red_flags = self._identify_perpetrator_red_flags(physical_event, context)
            victim_red_flags = self._identify_victim_red_flags(physical_event, context)
            red_flags_dict = {
                "perpetrator": perpetrator_red_flags,
                "victim": victim_red_flags
            }
        else:
            # For third-party conversations, use generic style notes
            style_notes_dict = {
                "general": ["Brief, urgent tone", "May include typos if distressed", "Timestamps matter"]
            }
            red_flags_dict = {}
        
        return {
            "event_id": during_event["sub_event_id"],
            "timestamp": timestamp.isoformat(),
            "event_type": "online_chat_expansion",
            "expansion_purpose": "during_event",
            "original_physical_event": physical_event["sub_event_id"],
            "platform": platform,
            "platform_description": "Immediate messaging during or right after the event",
            "participants": {
                "sender": {
                    "name": initiator["name"],
                    "role": initiator["role"]
                },
                "receiver": {
                    "name": receiver["name"],
                    "role": receiver.get("role", "unknown")
                }
            },
            "conversation_context": {
                "action": action,
                "description": description,
                "purpose": "Convey immediate situation, distress, or control",
                "crime_type": crime_type,
                "tone": "urgent" if crime_type else "checking in",
                "emotional_state": "distressed" if agent["role"] == "victim" else "controlling",
                "escalation_level": escalation_level
            },
            "content_guidance": {
                "main_points": [
                    f"Reference the ongoing situation: {physical_event['description']}",
                    "Convey immediacy and emotional response",
                    "Show signs of coercive control or distress if applicable"
                ],
                "suggested_length": "1-3 brief messages",
                "style_notes": style_notes_dict,
                "red_flags": red_flags_dict
            },
            "coherence_requirements": {
                "previous_context": previous_info[-2:] if previous_info else [],
                "information_to_convey": f"Real-time update: {physical_event['description']}",
                "follow_up_hooks": ["Aftermath and reflection in later messages"]
            }
        }
        
        # Note: Summary generation is handled elsewhere to avoid duplication
        
        return during_event
    
    def _create_aftermath_chat(self, physical_event: Dict[str, Any],
                             context: Dict[str, Any],
                             timestamp: datetime,
                             previous_info: List[str]) -> Dict[str, Any]:
        """Create a chat event after physical event (reflection/continuation)."""
        agent = physical_event["agent"]
        patient = physical_event["patient"]
        action = physical_event.get("action", "")
        crime_type = physical_event.get("crime_type")
        
        aftermath_event = {
            "sub_event_id": f"{physical_event['sub_event_id']}.post",
            "timestamp": timestamp.isoformat(),
            "location": "online",
            "location_type": "online",
            "agent": agent,
            "action": "reflects on",
            "patient": patient,
            "description": f"{agent['name']} reflects on or discusses the recent event with {patient['name']}",
            "crime_type": crime_type
        }
        
        platform = self.infer_chat_platform(aftermath_event, context)
        escalation_level = self._calculate_escalation_level(physical_event, context)
        
        # Check if conversation involves both victim and perpetrator (not third party)
        sender_role = agent["role"]
        receiver_role = context.get("receiver_role", "unknown")
        involves_third_party = sender_role == "third_party" or receiver_role == "third_party"
        
        # Generate role-specific style notes and red flags only for victim-perpetrator conversations
        if not involves_third_party:
            perpetrator_style_notes = self._generate_perpetrator_style_notes(physical_event, context, escalation_level)
            victim_style_notes = self._generate_victim_style_notes(physical_event, context, escalation_level)
            style_notes_dict = {
                "perpetrator": perpetrator_style_notes,
                "victim": victim_style_notes
            }
            
            perpetrator_red_flags = self._identify_perpetrator_red_flags(physical_event, context)
            victim_red_flags = self._identify_victim_red_flags(physical_event, context)
            red_flags_dict = {
                "perpetrator": perpetrator_red_flags,
                "victim": victim_red_flags
            }
        else:
            # For third-party conversations, use generic style notes
            style_notes_dict = {
                "general": ["Reflective tone", "May show emotional processing", "Follow-up conversations"]
            }
            red_flags_dict = {}
        
        return {
            "event_id": aftermath_event["sub_event_id"],
            "timestamp": timestamp.isoformat(),
            "event_type": "online_chat_expansion",
            "expansion_purpose": "aftermath",
            "original_physical_event": physical_event["sub_event_id"],
            "platform": platform,
            "platform_description": self.chat_templates.get(platform, "Online communication"),
            "participants": {
                "sender": {
                    "name": agent["name"],
                    "role": agent["role"]
                },
                "receiver": {
                    "name": patient["name"],
                    "role": context.get("receiver_role", "unknown")
                }
            },
            "conversation_context": {
                "action": "reflects on",
                "description": aftermath_event["description"],
                "purpose": "Process and discuss what happened in the physical event",
                "crime_type": crime_type,
                "tone": self._infer_aftermath_tone(physical_event, context),
                "emotional_state": "reflective",
                "escalation_level": escalation_level
            },
            "content_guidance": {
                "main_points": [
                    f"Reference the completed event: {physical_event['description']}",
                    "Show emotional aftermath or continuation of patterns",
                    "Set up for potential escalation or resolution"
                ],
                "suggested_length": "3-5 messages exchanged",
                "style_notes": style_notes_dict,
                "red_flags": red_flags_dict
            },
            "coherence_requirements": {
                "previous_context": previous_info[-3:] if previous_info else [],
                "information_to_convey": f"Aftermath of: {physical_event['description']}",
                "follow_up_hooks": ["Ongoing pattern of behavior", "Escalation potential"]
            }
        }
        
        # Note: Summary generation is handled elsewhere to avoid duplication
        
        return aftermath_event_script
    
    def _infer_tone(self, event: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Infer the tone of the conversation based on event characteristics."""
        crime_type = event.get("crime_type")
        action = event.get("action", "").lower()
        agent_role = event.get("agent", {}).get("role", "")
        
        if crime_type:
            if agent_role == "perpetrator":
                if "control" in str(crime_type).lower() or "monitor" in action:
                    return "controlling, possessive"
                elif "harassment" in str(crime_type).lower():
                    return "aggressive, intrusive"
                elif "stalking" in str(crime_type).lower():
                    return "obsessive, persistent"
                else:
                    return "manipulative"
            else:
                return "distressed, seeking help"
        
        # Early relationship
        if context.get("time_hint") == "early":
            return "friendly, getting to know each other"
        
        # Neutral/positive actions
        if action in ["meet", "discuss", "share"]:
            return "casual, conversational"
        
        return "neutral"
    
    def _infer_emotional_state(self, event: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Infer emotional state of the sender."""
        crime_type = event.get("crime_type")
        agent_role = event.get("agent", {}).get("role", "")
        
        if crime_type:
            if agent_role == "perpetrator":
                return "controlling, calculated"
            else:
                return "anxious, fearful"
        
        if context.get("time_hint") == "early":
            return "optimistic, curious"
        
        return "neutral, engaged"
    
    def _infer_aftermath_tone(self, event: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Infer tone for aftermath conversations."""
        crime_type = event.get("crime_type")
        agent_role = event.get("agent", {}).get("role", "")
        
        if crime_type:
            if agent_role == "perpetrator":
                return "justifying, gaslighting, or sweet-talking"
            else:
                return "confused, processing, possibly seeking support"
        
        return "reflective, casual follow-up"
    
    def _extract_main_points(self, event: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
        """Extract the main points that should be conveyed in the chat."""
        points = []
        
        description = event.get("description", "")
        action = event.get("action", "")
        crime_type = event.get("crime_type")
        
        points.append(f"Action: {action}")
        points.append(f"Context: {description}")
        
        if crime_type:
            points.append(f"Crime type indicator: {crime_type}")
        
        return points
    
    def _suggest_message_length(self, event: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Suggest appropriate message length based on context."""
        action = event.get("action", "").lower()
        platform = self.infer_chat_platform(event, context)
        
        if platform == "email":
            return "Medium to long messages (paragraph format)"
        elif "post" in action:
            return "Short to medium post (50-200 words)"
        elif "message" in action or platform == "messaging":
            return "Short messages (1-3 sentences), possibly multiple exchanges"
        else:
            return "2-5 brief messages exchanged"
    
    def _generate_style_notes(self, event: Dict[str, Any], context: Dict[str, Any], escalation_level: int = 0) -> List[str]:
        """Generate style notes for LLM to follow, reflecting escalation level based on CPS VAWG guidance."""
        notes = []
        
        agent_role = event.get("agent", {}).get("role", "")
        crime_type = event.get("crime_type")
        time_hint = context.get("time_hint", "mid")
        
        # Add slang/language style notes based on sender profile
        if agent_role == "perpetrator":
            profile = context.get("perpetrator_profile")
        elif agent_role == "victim":
            profile = context.get("victim_profile")
        else:
            profile = None
        
        slang_notes = self._generate_slang_style_notes(profile, agent_role)
        notes.extend(slang_notes)
        
        # Add escalation level context (CPS VAWG framework: 0-4)
        escalation_descriptions = [
            "normal, non-threatening interaction",
            "subtle controlling behaviors beginning to emerge - building rapport disguised as care",
            "clear manipulation and control - frequent contact demands, monitoring, guilt-tripping",
            "aggressive intimidation and coercion - gaslighting, threats, stalking, boundary violations",
            "severe threats or violent escalation - explicit violence/murder threats, extortion, doxxing, extreme harm"
        ]
        notes.append(f"Escalation Level {escalation_level}: {escalation_descriptions[min(escalation_level, 4)]}")
        
        if time_hint == "early":
            notes.append("Use casual, getting-to-know-you language")
            notes.append("Show initial attraction or interest")
        
        if agent_role == "perpetrator" and crime_type:
            if escalation_level == 0:
                notes.append("Completely normal interaction - no abuse indicators")
            elif escalation_level == 1:
                notes.append("Subtly show red flags - control disguised as care or concern")
                notes.append("Build rapport through compliments, shared interests; appear supportive and helpful")
                notes.append("No obvious coercion yet - focus on creating trust and dependency")
            elif escalation_level == 2:
                notes.append("Frequent contact with expectation of immediate responses ('why didn't you reply?')")
                notes.append("Early monitoring: ask who victim is talking to, comment on their online activity or appearance")
                notes.append("Minor put-downs or guilt-tripping disguised as concern ('I worry when you go out without telling me')")
                notes.append("Request personal information or photos framed as 'just between us'")
                notes.append("Messages become more urgent and frequent; may include occasional typos when frustrated")
            elif escalation_level == 3:
                notes.append("Clear manipulative language patterns and emotional manipulation")
                notes.append("Repeated messages demanding victim's whereabouts or actions")
                notes.append("Gaslighting, blame-shifting, emotional blackmail ('If you really loved me...'; threaten self-harm)")
                notes.append("Reference or imply monitoring via social media, location-sharing, or device access")
                notes.append("Escalate sexual content or grooming (pressure for sexual messages/images)")
                notes.append("Aggressive and demanding tone; boundary violations")
                notes.append("Messages get shorter and more fragmented; increased typos, occasional ALL CAPS for emphasis")
            else:  # Level 4
                notes.append("Severe, explicit threats of violence, harm, or murder")
                notes.append("Explicit threats (threaten to share intimate images, harm victim or loved ones, reference weapons)")
                notes.append("Harassment through multiple channels: frequent calls/texts, online stalking, creating fake accounts")
                notes.append("Send or reference unsolicited obscene images; repeatedly demand sexual acts")
                notes.append("Demand control over real-world actions (stay home, quit job, stop education)")
                notes.append("Extortion, blackmail, doxxing (release personal data if demands not met); 'revenge porn' threats")
                notes.append("Describe victim's real-time location or demonstrate surveillance evidence")
                notes.append("Reference enlisting others to harass/harm victim; coordinate physical abduction or forced marriage")
                notes.append("Very short, explosive messages; frequent typos, extensive ALL CAPS, fragmented/incoherent when enraged")
        elif agent_role == "victim" and crime_type:
            if escalation_level == 0:
                notes.append("Respond naturally - normal, healthy interaction")
            elif escalation_level == 1:
                notes.append("Respond naturally, may not yet recognize concerning patterns")
                notes.append("Show slight confusion or discomfort but maintain engagement")
            elif escalation_level == 2:
                notes.append("Show stress from frequent contact demands")
                notes.append("Attempt to justify or explain actions to reassure perpetrator")
                notes.append("May express confusion about partner's concerns or behaviors")
                notes.append("Responses become more anxious and hurried")
            elif escalation_level == 3:
                notes.append("Clear stress and self-doubt; justify or excuse partner's controlling behavior")
                notes.append("Show signs of isolation from support network")
                notes.append("Express fear of upsetting partner or triggering anger")
                notes.append("May show trauma responses or emotional distress")
                notes.append("Shorter, more careful responses; occasional typos from fear and stress")
            else:  # Level 4
                notes.append("Extreme fear, trauma responses; may be too scared to respond or overly compliant")
                notes.append("High anxiety about safety - own or loved ones")
                notes.append("May send coded distress signals or reach out to authorities/support services")
                notes.append("Express immediate safety concerns; references to violence, threats, or extreme danger")
                notes.append("Signs of severe psychological impact and terror")
                notes.append("Very brief, fragmented responses or prolonged silence; increased typos from terror; possible ALL CAPS distress signals")
        
        notes.append("Use realistic messaging language (contractions, informal grammar, typos if stressed)")
        notes.append("Include natural conversation flow with greetings/closings as appropriate")
        
        return notes
    
    def _identify_red_flags(self, event: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
        """Identify red flags that should be present in the conversation based on CPS VAWG patterns."""
        red_flags = []
        
        crime_type = event.get("crime_type")
        action = event.get("action", "").lower()
        description = event.get("description", "").lower()
        
        if not crime_type:
            return red_flags
        
        crime_str = str(crime_type).lower()
        
        # Coercive control and monitoring behaviors
        if "control" in crime_str or "coercive" in crime_str:
            red_flags.extend([
                "Excessive checking in with expectation of immediate responses",
                "Monitoring online activity, social media posts, or who victim talks to",
                "Framing control as care or concern ('I worry about you')",
                "Demanding whereabouts or real-time location sharing",
                "Limiting or monitoring contact with friends/family",
                "Making decisions for the victim (what to wear, where to go)"
            ])
        
        # Stalking and cyber-stalking
        if "stalk" in crime_str:
            red_flags.extend([
                "Unwanted repeated contact across multiple platforms",
                "Mentioning victim's real-time location or activities they shouldn't know about",
                "Creating fake accounts to monitor or contact victim",
                "Demonstrating surveillance (knowing details of victim's private life)",
                "Locating personal information through online searches"
            ])
        
        # Harassment and threatening behavior
        if "harass" in crime_str or "threaten" in crime_str or "intimidat" in crime_str:
            red_flags.extend([
                "Aggressive, demanding, or threatening language",
                "Repeated unwanted contact despite victim's requests to stop",
                "Boundary violations (contacting at work, showing up uninvited)",
                "Posting false or humiliating content about victim online",
                "Explicit threats to harm victim, loved ones, or pets"
            ])
        
        # Economic/financial abuse
        if "economic" in crime_str or "financial" in crime_str:
            red_flags.extend([
                "Financial control disguised as help or partnership",
                "Restricting access to money, bank accounts, or financial information",
                "Demanding accountability for spending",
                "Sabotaging victim's employment or education",
                "Threatening immigration or financial repercussions"
            ])
        
        # Sexual exploitation and grooming
        if "sexual" in crime_str or "groom" in crime_str:
            red_flags.extend([
                "Pressuring for sexual messages, images, or acts",
                "Normalizing sexual content in conversation gradually",
                "Ignoring or dismissing consent boundaries",
                "Sending unsolicited obscene or extreme pornographic images",
                "Threatening to share intimate images ('revenge porn')",
                "Using shared intimate material as leverage or blackmail"
            ])
        
        # Honour-based abuse and forced marriage
        if "honour" in crime_str or "forced marriage" in crime_str or "cultural" in crime_str:
            red_flags.extend([
                "Family pressure to conform to 'honour' expectations",
                "Shaming victim for relationship choices or lifestyle",
                "Threats that refusal will disgrace family or trigger violence",
                "Multiple family members coordinating pressure via messages",
                "References to community reputation or family shame"
            ])
        
        # Trafficking and exploitation
        if "traffick" in crime_str or "exploit" in crime_str or "prostitution" in crime_str:
            red_flags.extend([
                "Initial promises of work, relationships, or better life followed by control",
                "Debt bondage or making victim financially dependent",
                "Threats to victim's safety or that of relatives to ensure compliance",
                "Controlling victim's movement, documents, or communication",
                "Coercion into prostitution or sexual services"
            ])
        
        # Action-based red flags
        if "monitor" in action or "check" in action:
            red_flags.extend([
                "Demanding to know whereabouts constantly",
                "Requesting passwords, device access, or social media logins",
                "Installing spyware or tracking apps on victim's devices"
            ])
        
        # Gaslighting and manipulation patterns (common across many VAWG crimes)
        if crime_type:
            red_flags.extend([
                "Gaslighting: denying things they said or did",
                "Blame-shifting (DARVO): making victim feel responsible for abuse",
                "Emotional blackmail: threatening self-harm if victim doesn't comply",
                "Love-bombing after incidents to maintain control"
            ])
        
        return red_flags
    
    @staticmethod
    def _safe_int(value: Any, default: int) -> int:
        """Safely convert value to int, return default if conversion fails."""
        if value is None:
            return default
        try:
            return int(value)
        except (ValueError, TypeError):
            return default
    
    @staticmethod
    def _normalize_text(value: Any) -> str:
        """Normalize text value to lowercase string."""
        if value is None:
            return ""
        return str(value).lower().strip()
    
    @classmethod
    def _categorize_age(cls, age: Any) -> str:
        """Categorize age into groups."""
        age_int = cls._safe_int(age, 25)
        if age_int < 18:
            return "teen"
        elif age_int < 35:
            return "young_adult"
        elif age_int < 55:
            return "adult"
        else:
            return "mature"
    
    @classmethod
    def _categorize_occupation(cls, occupation: Any) -> str:
        """Categorize occupation into broad groups."""
        occ = cls._normalize_text(occupation)
        
        # Student categories
        if any(keyword in occ for keyword in ["student", "pupil", "school", "college", "university"]):
            return "student"
        
        # Professional/corporate
        if any(keyword in occ for keyword in ["manager", "director", "executive", "lawyer", "accountant", "consultant", "engineer", "professional"]):
            return "professional"
        
        # Creative/informal
        if any(keyword in occ for keyword in ["artist", "designer", "musician", "influencer", "content creator", "streamer", "youtuber"]):
            return "creative"
        
        # Service/retail
        if any(keyword in occ for keyword in ["retail", "server", "waiter", "barista", "cashier", "sales", "customer service"]):
            return "service"
        
        # Unemployed/at home
        if any(keyword in occ for keyword in ["unemployed", "between jobs", "stay-at-home", "homemaker", "carer"]):
            return "unemployed"
        
        return "other"
    
    
    @classmethod
    def _base_slang_rate_from_age(cls, age: Any) -> float:
        """Calculate base slang rate from age category."""
        age_category = cls._categorize_age(age)
        
        rates = {
            "teen": 0.45,           # High slang usage
            "young_adult": 0.35,    # Moderate-high slang
            "adult": 0.20,          # Moderate slang
            "mature": 0.10          # Low slang
        }
        
        return rates.get(age_category, 0.20)
    
    @classmethod
    def _occupation_slang_modifier(cls, occupation: Any) -> float:
        """Get slang modifier based on occupation (additive adjustment)."""
        occ_category = cls._categorize_occupation(occupation)
        
        modifiers = {
            "student": 0.10,        # Students use more slang
            "creative": 0.05,       # Creative types slightly more casual
            "service": 0.05,        # Service workers moderately casual
            "professional": -0.10,  # Professionals more formal
            "unemployed": 0.0,      # Neutral
            "other": 0.0            # Neutral
        }
        
        return modifiers.get(occ_category, 0.0)
    
    @classmethod
    def infer_slang_rate(cls, profile: Optional[Dict[str, Any]]) -> float:
        """Calculate overall slang rate from profile (age + occupation)."""
        if not profile:
            return 0.20  # Default moderate slang
        
        age = profile.get("age", profile.get("Age", 25))
        occupation = profile.get("occupation", profile.get("Occupation", ""))
        
        base_rate = cls._base_slang_rate_from_age(age)
        modifier = cls._occupation_slang_modifier(occupation)
        
        # Combine and clamp to [0.05, 0.60]
        final_rate = max(0.05, min(0.60, base_rate + modifier))
        return final_rate
    
    @classmethod
    def infer_slang_level_label(cls, rate: float) -> str:
        """Convert slang rate to descriptive label."""
        if rate < 0.15:
            return "minimal"
        elif rate < 0.25:
            return "low"
        elif rate < 0.35:
            return "moderate"
        elif rate < 0.45:
            return "high"
        else:
            return "very_high"
    
    @classmethod
    def _generate_slang_style_notes(cls, profile: Optional[Dict[str, Any]], role: str) -> List[str]:
        """Generate slang/casual language style notes based on age and occupation."""
        slang_rate = cls.infer_slang_rate(profile)
        slang_level = cls.infer_slang_level_label(slang_rate)
        
        notes = []
        
        # Age-based characteristics
        if profile:
            age = cls._safe_int(profile.get("age", profile.get("Age", 25)), 25)
            age_cat = cls._categorize_age(age)
            
            if age_cat == "teen":
                notes.append(f"Teen language style ({age} yo): very casual, heavy slang usage")
                notes.append("Very short messages (1-2 clauses), rapid-fire replies")
                notes.append("Heavy slang: u, ur, rn, idk, wtf, fr, lmao, lmfao, ngl, lol, omg, bruh, etc.")
                notes.append("Minimal punctuation or excessive (???, !!!, ...)")
                notes.append("Inconsistent grammar and capitalization; omit period")
                notes.append("Emojis common (😂, 💀, 😭, ❤️, etc.)")
            elif age_cat == "young_adult":
                notes.append(f"Young adult style ({age} yo): casual with moderate slang")
                notes.append("Short messages (clauses, no more than 1 sentence)")
                notes.append("Frequent slang: u, ur, idk, lol, lmfao, omg, tbh, ngl, rn, etc.")
                notes.append("Informal punctuation (..., ??, !!); omit period")
                notes.append("Some grammar shortcuts, casual tone")
                notes.append("Occasional emojis for tone (😂, 😅, ❤️, etc.)")
            elif age_cat == "adult":
                notes.append(f"Adult style ({age} yo): semi-formal with some casual elements")
                notes.append("Medium and short messages (clauses or 1-2 sentences)")
                notes.append("Moderate slang: lol, ok, idk, btw, etc.")
                notes.append("More consistent grammar but still relaxed")
                notes.append("Informal punctuation (..., ??, !!); sometimes omit period")
                notes.append("Minimal emojis, mainly for clarity")
            else:  # mature
                notes.append(f"Mature style ({age} yo): more formal, minimal slang")
                notes.append("Longer, complete messages (clauses or 1-3 sentences)")
                notes.append("Slight slang: ok, thanks, idk, btw, etc.")
                notes.append("Standard punctuation, occasional ellipses for pauses")
                notes.append("Full words, proper capitalization")
                notes.append("Rare or no emojis")
        
        # Generic slang level note if no profile
        else:
            if slang_level == "minimal":
                notes.append("Formal messaging style: complete sentences, proper grammar")
                notes.append("Minimal abbreviations, standard punctuation")
            elif slang_level == "low":
                notes.append("Semi-formal style: mostly complete sentences, light casual language")
                notes.append("Light slang: ok, lol, idk, thanks, etc.")
            elif slang_level == "moderate":
                notes.append("Casual style: mix of short and medium messages")
                notes.append("Moderate slang: u, ur, idk, lol, omg, tbh, etc.")
                notes.append("Informal punctuation (..., ??)")
            elif slang_level == "high":
                notes.append("Very casual style: short messages, frequent slang")
                notes.append("High slang: u, ur, rn, idk, wtf, lol, lmao, tbh, ngl, etc.")
                notes.append("Minimal punctuation, inconsistent grammar")
            else:  # very_high
                notes.append("Extremely casual style: very short messages, heavy slang")
                notes.append("Very high slang: u, ur, rn, idk, wtf, fr, lmao, bruh, etc.")
                notes.append("Minimal punctuation or excessive (???, !!!)")
                notes.append("Inconsistent grammar and capitalization")
        
        return notes
    
    def _generate_perpetrator_style_notes(self, event: Dict[str, Any], context: Dict[str, Any], escalation_level: int) -> List[str]:
        """Generate style notes specifically for perpetrator behavior based on escalation level."""
        notes = []
        crime_type = event.get("crime_type")
        
        # Add slang/language style notes based on perpetrator profile
        perpetrator_profile = context.get("perpetrator_profile")
        slang_notes = self._generate_slang_style_notes(perpetrator_profile, "perpetrator")
        notes.extend(slang_notes)
        
        if not crime_type:
            notes.append("Use realistic messaging language")
            return notes
        
        # Check if this is a guilt/remorse phase after high escalation
        # This reflects the realistic cycle of abuse: tension-building, incident, reconciliation, calm
        is_guilt_phase = (self.previous_escalation_level >= 3 and escalation_level < self.previous_escalation_level)
        
        if is_guilt_phase:
            notes.append("GUILT/REMORSE PHASE after previous high escalation incident:")
            notes.append("Show temporary remorse, apologize profusely for previous behavior")
            notes.append("Promise to change, claim 'it will never happen again'")
            notes.append("Blame external factors (stress, alcohol, work) for previous actions")
            notes.append("Love-bombing: excessive affection, gifts, compliments to win victim back")
            notes.append("Minimize the severity of previous incidents ('I didn't mean it that way')")
            notes.append("Extract forgiveness from victim, seek reassurance of their commitment")
            notes.append("NOTE: This appears genuine but is part of the cycle of abuse pattern")
            return notes
        
        # Escalation-level based perpetrator behaviors (CPS VAWG framework)
        if escalation_level == 0:
            notes.append("Normal, friendly conversation - no abuse present yet")
        elif escalation_level == 1:
            notes.append("Subtly increase contact frequency ('Just checking in')")
            notes.append("No obvious coercion yet - focus on creating trust and dependency")
        elif escalation_level == 2:
            notes.append("Frequent contact with expectation of immediate responses ('why didn't you reply?')")
            notes.append("Early monitoring: ask who victim is talking to, comment on their online activity or appearance")
            notes.append("Minor put-downs or guilt-tripping disguised as concern ('I worry when you go out without telling me')")
            notes.append("Request personal information or photos framed as 'just between us'")
        elif escalation_level == 3:
            notes.append("Clear manipulative language patterns and emotional manipulation")
            notes.append("Repeated messages demanding victim's whereabouts or actions")
            notes.append("Gaslighting, blame-shifting, emotional blackmail ('If you really loved me...'; threaten self-harm)")
            notes.append("Reference or imply monitoring via social media, location-sharing, or device access")
            notes.append("Escalate sexual content or grooming (pressure for sexual messages/images)")
            notes.append("Aggressive and demanding tone; boundary violations")
        else:  # Level 4
            notes.append("Severe, explicit threats of violence, harm, or murder")
            notes.append("Explicit threats (threaten to share intimate images, harm victim or loved ones, reference weapons)")
            notes.append("Harassment through multiple channels: frequent calls/texts, online stalking, creating fake accounts")
            notes.append("Send or reference unsolicited obscene images; repeatedly demand sexual acts")
            notes.append("Demand control over real-world actions (stay home, quit job, stop education)")
            notes.append("Extortion, blackmail, doxxing (release personal data if demands not met); 'revenge porn' threats")
            notes.append("Describe victim's real-time location or demonstrate surveillance evidence")
            notes.append("Reference enlisting others to harass/harm victim; coordinate physical abduction or forced marriage")
        
        notes.append("Use realistic messaging language (contractions, informal grammar)")
        notes.append("Include natural conversation flow with greetings/closings as appropriate")
        
        return notes
    
    def _generate_victim_style_notes(self, event: Dict[str, Any], context: Dict[str, Any], escalation_level: int) -> List[str]:
        """Generate style notes specifically for victim responses based on escalation level."""
        notes = []
        crime_type = event.get("crime_type")
        
        # Add slang/language style notes based on victim profile
        victim_profile = context.get("victim_profile")
        slang_notes = self._generate_slang_style_notes(victim_profile, "victim")
        notes.extend(slang_notes)
        
        if not crime_type:
            notes.append("Use realistic messaging language")
            return notes
        
        # Check if this is a reconciliation phase (perpetrator showing guilt)
        is_reconciliation_phase = (self.previous_escalation_level >= 3 and escalation_level < self.previous_escalation_level)
        
        if is_reconciliation_phase:
            notes.append("RECONCILIATION PHASE - victim responding to perpetrator's remorse:")
            notes.append("May feel relieved that 'things are getting better'")
            notes.append("Show cautious hope, wanting to believe the apologies")
            notes.append("Express forgiveness, downplay the severity of previous incidents")
            notes.append("May feel responsible for perpetrator's emotional state")
            notes.append("Attempt to maintain peace and avoid triggering anger again")
            notes.append("Could express love and recommitment to the relationship")
            notes.append("NOTE: Victim may not yet recognize this as part of an abusive cycle")
            return notes
        
        # Escalation-level based victim responses (CPS VAWG framework)
        if escalation_level == 0:
            notes.append("Respond naturally - normal, healthy interaction")
        elif escalation_level == 1:
            notes.append("Respond naturally, may not yet recognize concerning patterns")
            notes.append("Show slight confusion or discomfort but maintain engagement")
        elif escalation_level == 2:
            notes.append("Show stress from frequent contact demands")
            notes.append("Attempt to justify or explain actions to reassure perpetrator")
            notes.append("May express confusion about partner's concerns or behaviors")
        elif escalation_level == 3:
            notes.append("Clear stress and self-doubt; justify or excuse partner's controlling behavior")
            notes.append("Show signs of isolation from support network")
            notes.append("Express fear of upsetting partner or triggering anger")
            notes.append("May show trauma responses or emotional distress")
        else:  # Level 4
            notes.append("Extreme fear, trauma responses; may be too scared to respond or overly compliant")
            notes.append("High anxiety about safety - own or loved ones")
            notes.append("May send coded distress signals or reach out to authorities/support services")
            notes.append("Express immediate safety concerns; references to violence, threats, or extreme danger")
            notes.append("Signs of severe psychological impact and terror")
        
        notes.append("Use realistic messaging language (typos if stressed)")
        notes.append("Include natural conversation flow")
        
        return notes
    
    def _identify_perpetrator_red_flags(self, event: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
        """Identify red flags for perpetrator behavior based on CPS VAWG patterns."""
        red_flags = []
        
        crime_type = event.get("crime_type")
        action = event.get("action", "").lower()
        description = event.get("description", "").lower()
        
        if not crime_type:
            return red_flags
        
        crime_str = str(crime_type).lower()
        
        # Coercive control and monitoring behaviors
        if "control" in crime_str or "coercive" in crime_str:
            red_flags.extend([
                "Excessive checking in with expectation of immediate responses",
                "Monitoring online activity, social media posts, or who victim talks to",
                "Framing control as care or concern ('I worry about you')",
                "Demanding whereabouts or real-time location sharing",
                "Limiting or monitoring contact with friends/family",
                "Making decisions for the victim (what to wear, where to go)"
            ])
        
        # Stalking and cyber-stalking
        if "stalk" in crime_str:
            red_flags.extend([
                "Unwanted repeated contact across multiple platforms",
                "Mentioning victim's real-time location or activities they shouldn't know about",
                "Creating fake accounts to monitor or contact victim",
                "Demonstrating surveillance (knowing details of victim's private life)",
                "Locating personal information through online searches"
            ])
        
        # Harassment and threatening behavior
        if "harass" in crime_str or "threaten" in crime_str or "intimidat" in crime_str:
            red_flags.extend([
                "Aggressive, demanding, or threatening language",
                "Repeated unwanted contact despite victim's requests to stop",
                "Boundary violations (contacting at work, showing up uninvited)",
                "Posting false or humiliating content about victim online",
                "Explicit threats to harm victim, loved ones, or pets"
            ])
        
        # Economic/financial abuse
        if "economic" in crime_str or "financial" in crime_str:
            red_flags.extend([
                "Financial control disguised as help or partnership",
                "Restricting access to money, bank accounts, or financial information",
                "Demanding accountability for spending",
                "Sabotaging victim's employment or education",
                "Threatening immigration or financial repercussions"
            ])
        
        # Sexual exploitation and grooming
        if "sexual" in crime_str or "groom" in crime_str:
            red_flags.extend([
                "Pressuring for sexual messages, images, or acts",
                "Normalizing sexual content in conversation gradually",
                "Ignoring or dismissing consent boundaries",
                "Sending unsolicited obscene or extreme pornographic images",
                "Threatening to share intimate images ('revenge porn')",
                "Using shared intimate material as leverage or blackmail"
            ])
        
        # Honour-based abuse and forced marriage
        if "honour" in crime_str or "forced marriage" in crime_str or "cultural" in crime_str:
            red_flags.extend([
                "Family pressure to conform to 'honour' expectations",
                "Shaming victim for relationship choices or lifestyle",
                "Threats that refusal will disgrace family or trigger violence",
                "Multiple family members coordinating pressure via messages",
                "References to community reputation or family shame"
            ])
        
        # Trafficking and exploitation
        if "traffick" in crime_str or "exploit" in crime_str or "prostitution" in crime_str:
            red_flags.extend([
                "Initial promises of work, relationships, or better life followed by control",
                "Debt bondage or making victim financially dependent",
                "Threats to victim's safety or that of relatives to ensure compliance",
                "Controlling victim's movement, documents, or communication",
                "Coercion into prostitution or sexual services"
            ])
        
        # Action-based red flags
        if "monitor" in action or "check" in action:
            red_flags.extend([
                "Demanding to know whereabouts constantly",
                "Requesting passwords, device access, or social media logins",
                "Installing spyware or tracking apps on victim's devices"
            ])
        
        # Gaslighting and manipulation patterns (common across many VAWG crimes)
        if crime_type:
            red_flags.extend([
                "Gaslighting: denying things they said or did",
                "Blame-shifting (DARVO): making victim feel responsible for abuse",
                "Emotional blackmail: threatening self-harm if victim doesn't comply",
                "Love-bombing after incidents to maintain control"
            ])
        
        return red_flags
    
    def _identify_victim_red_flags(self, event: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
        """Identify red flags for victim behavior/responses based on VAWG context."""
        red_flags = []
        
        crime_type = event.get("crime_type")
        
        if not crime_type:
            return red_flags
        
        crime_str = str(crime_type).lower()
        
        # Victim response patterns that indicate abuse impact
        red_flags.extend([
            "Show signs of distress, fear, or anxiety in responses",
            "Attempt to placate or appease perpetrator to avoid conflict",
            "Express self-blame or justify perpetrator's behavior",
            "Show confusion about relationship dynamics or partner's expectations"
        ])
        
        # Context-specific victim patterns
        if "control" in crime_str or "coercive" in crime_str:
            red_flags.extend([
                "Over-explain whereabouts or actions preemptively",
                "Show signs of isolation from friends/family",
                "Express need for permission for basic activities"
            ])
        
        if "stalk" in crime_str or "harass" in crime_str:
            red_flags.extend([
                "Express feeling watched or followed",
                "Attempt to set boundaries that are repeatedly violated",
                "Show fear about perpetrator's unexpected knowledge of their activities"
            ])
        
        if any(word in crime_str for word in ["sexual", "groom", "exploit"]):
            red_flags.extend([
                "Show discomfort but compliance with sexual requests",
                "Express concerns about intimate content being shared",
                "Indicate pressure to engage in unwanted sexual activities"
            ])
        
        return red_flags
    
    def _identify_key_information(self, event: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
        """Identify key information that must be conveyed for narrative coherence."""
        key_info = []
        
        key_info.append(event.get("description", ""))
        
        action = event.get("action", "")
        key_info.append(f"Action taken: {action}")
        
        crime_type = event.get("crime_type")
        if crime_type:
            key_info.append(f"Abuse type: {crime_type}")
        
        return key_info
    
    def _generate_follow_up_hooks(self, event: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
        """Generate hooks that can be picked up by subsequent events."""
        hooks = []
        
        crime_type = event.get("crime_type")
        if crime_type:
            hooks.append("Pattern of behavior established")
            hooks.append("Potential for escalation")
        
        action = event.get("action", "")
        if action:
            hooks.append(f"Continuation of {action} in future interactions")
        
        hooks.append("Relationship dynamic continues to develop")
        
        return hooks
    
    def _calculate_base_escalation_level(self, event: Dict[str, Any], context: Dict[str, Any]) -> int:
        """Calculate base escalation/violence level for this event (0-4) as guidance with timeline progression."""
        crime_type = event.get("crime_type")
        action = event.get("action", "").lower()
        description = event.get("description", "").lower()
        
        # Calculate timeline position (0.0 = start, 1.0 = end)
        timeline_position = 0.5  # Default to middle
        if self.total_events > 0:
            timeline_position = self.current_event_index / self.total_events
        
        # No crime type = level 0
        if not crime_type:
            return 0
        
        crime_str = str(crime_type).lower()
        
        # Determine POTENTIAL level from crime type (not final)
        potential_level = 1  # Default for any crime type
        
        if "coercive control" in crime_str or "monitoring" in crime_str:
            potential_level = 2  # Clear manipulation and control
        
        if "harassment" in crime_str or "stalking" in crime_str:
            potential_level = 2  # Start lower, can escalate
            if "repeated" in description or "persistent" in description or "unwanted" in description:
                potential_level = 3
        
        if "threat" in crime_str or "intimidation" in crime_str:
            potential_level = 3  # Aggressive intimidation
        
        if "assault" in crime_str or "violence" in crime_str or "physical" in crime_str:
            potential_level = 4
        
        # Add more triggers for level 4
        if any(keyword in description for keyword in ["severe", "extreme", "force", "hurt", "harm", "attack", "violent", "threatening"]):
            potential_level = 4
        
        if any(keyword in action for keyword in ["threaten", "force", "assault", "attack", "hurt", "harm"]):
            potential_level = 4
        
        # Apply timeline-based modulation to create progression
        # Early events (0-0.3): Cap at level 2, favor 0-1
        # Mid events (0.3-0.6): Allow up to level 3
        # Late events (0.6-1.0): Allow full range up to level 4
        
        if timeline_position < 0.25:
            # Very early: favor level 0-2, balanced distribution
            if potential_level >= 3:
                level = random.choice([0, 0, 1, 1, 2, 2, 2, 2, 2, 2])  # 20% 0, 20% 1, 60% 2
            elif potential_level == 2:
                level = random.choice([0, 0, 1, 2, 2, 2])  # 33% 0, 17% 1, 50% 2
            else:
                level = random.choice([0, 0, 1, 1, 2])  # 40% 0, 40% 1, 20% 2
                
        elif timeline_position < 0.5:
            # Early-mid: favor level 2-3
            if potential_level >= 4:
                level = random.choice([1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3])  # 7% 1, 29% 2, 64% 3
            elif potential_level == 3:
                level = random.choice([1, 2, 2, 2, 3, 3, 3, 3, 3])  # 11% 1, 33% 2, 56% 3
            else:
                level = random.choice([1, 2, 2, 2, 3, 3]) if potential_level >= 2 else min(potential_level, 2)
                
        elif timeline_position < 0.75:
            # Mid-late: favor level 3, increase level 4
            if potential_level == 4:
                level = random.choice([2, 3, 3, 3, 3, 3, 4, 4, 4, 4])  # 10% 2, 50% 3, 40% 4
            elif potential_level == 3:
                level = random.choice([2, 3, 3, 3, 3, 4, 4])  # 14% 2, 57% 3, 29% 4
            else:
                level = potential_level
                
        else:
            # Late: favor level 4 heavily
            if potential_level == 4:
                level = random.choice([2, 3, 3, 4, 4, 4, 4, 4, 4, 4])  # 10% 2, 20% 3, 70% 4
            elif potential_level == 3:
                level = random.choice([2, 3, 3, 4, 4, 4, 4])  # 14% 2, 29% 3, 57% 4  
            else:
                level = potential_level
        
        # Allow realistic escalation patterns - can go up and down
        # This reflects the cycle of abuse with tension building, incident, reconciliation, calm
        
        # Track if high escalation occurs (level 3-4) for guilt/remorse patterns
        if level >= 3:
            self.high_escalation_event_occurred = True
        elif self.high_escalation_event_occurred and level < 3:
            # After high escalation, allow lower levels (guilt/remorse phase)
            self.high_escalation_event_occurred = False
        
        # Update tracking
        self.previous_escalation_level = self.escalation_level
        self.escalation_level = level
        
        return level
    
    def _infer_escalation_level_with_llm(self, event: Dict[str, Any], context: Dict[str, Any]) -> int:
        """Use LLM to infer escalation level with calculated base level as guidance."""
        # Get base calculation as guidance
        base_level = self._calculate_base_escalation_level(event, context)
        
        # If LLM is not available, just use base calculation
        if not self.use_llm or not self.llm_backend:
            return base_level
        
        # Create prompt for LLM to infer escalation level
        agent = event.get("agent", {})
        action = event.get("action", "")
        description = event.get("description", "")
        crime_type = event.get("crime_type")
        
        escalation_prompt = f"""Analyze the escalation/violence level for this event based on objective criteria.

**Event Details:**
- Action: {action}
- Description: {description}
- Crime Type: {crime_type if crime_type else "None"}
- Perpetrator Role: {agent.get('role', 'unknown')}

**Escalation Level Scale (0-4):**
- Level 0: Normal, non-threatening interaction (no abuse)
- Level 1: Subtle controlling behaviors beginning to emerge - building rapport disguised as care
- Level 2: Clear manipulation and control - frequent contact demands, monitoring, guilt-tripping
- Level 3: Aggressive intimidation and coercion - gaslighting, threats, stalking, boundary violations
- Level 4: Severe threats or violent escalation - explicit violence/murder threats, extortion, doxxing, extreme harm

**Calculated Base Level (as guidance): {base_level}**

**Context from Previous Events:**
- Previous escalation level: {self.previous_escalation_level}
- Current escalation level: {self.escalation_level}
- Timeline position: {self.current_event_index}/{self.total_events}
- Relationship stage: {context.get('time_hint', 'mid')}

**IMPORTANT - Realistic VAWG Cycle of Abuse Patterns:**
The escalation level can INCREASE or DECREASE to reflect realistic abuse dynamics:

1. **Natural Progression**: Early events start low (0-2), mid events escalate (2-3), later events may reach high levels (3-4)

2. **Cycle of Abuse Pattern**: After high escalation events (level 3-4), perpetrators often show:
   - Remorse and guilt (drop to level 0-2)
   - Apologies and promises to change
   - Love-bombing and honeymoon phase
   - Temporary calm before tension rebuilds

3. **Realistic Escalation**: The level should reflect THIS SPECIFIC event, not force monotonic increase:
   - If this event shows remorse/apology after previous violence → assign lower level (0-2)
   - If this event shows tension building → assign medium level (2-3)
   - If this event shows actual abuse → assign appropriate high level (3-4)

4. **Timeline Guidance** (not strict rules):
   - Early (0-0.3): Usually 0-2, but can have higher incidents
   - Mid (0.3-0.7): Can range 1-3, with variation
   - Late (0.7-1.0): Can reach 3-4, but also show reconciliation phases

Assign the level that accurately reflects THIS event's severity, allowing for the natural ups and downs of abusive relationships.

**Output Format:**
Respond with ONLY a single number (0-4) representing the escalation level.
Do not include any explanation or additional text."""
        
        try:
            response = self.llm_backend.generate(
                prompt=escalation_prompt,
                max_tokens=10,
                temperature=0.3  # Lower temperature for more consistent results
            )
            
            # Extract number from response
            import re
            match = re.search(r'[0-4]', response.strip())
            if match:
                inferred_level = int(match.group())
                # Ensure it's within valid range
                inferred_level = max(0, min(4, inferred_level))
                
                # Allow realistic escalation patterns with guilt/remorse after high escalation
                # After severe events (level 3-4), perpetrators often show temporary remorse
                if self.high_escalation_event_occurred and inferred_level < 3:
                    # This is likely a "honeymoon phase" or guilt period
                    # Allow the lower escalation level
                    self.high_escalation_event_occurred = False
                
                # Track if this is a high escalation event
                if inferred_level >= 3:
                    self.high_escalation_event_occurred = True
                    
                # Update escalation tracking
                self.previous_escalation_level = self.escalation_level
                self.escalation_level = inferred_level
                
                return inferred_level
            else:
                # If parsing fails, fall back to base calculation
                print(f"    Warning: Could not parse escalation level from LLM response, using base level {base_level}")
                return base_level
                
        except Exception as e:
            print(f"    Warning: LLM escalation inference failed ({e}), using base level {base_level}")
            return base_level
    
    def _calculate_escalation_level(self, event: Dict[str, Any], context: Dict[str, Any]) -> int:
        """Calculate escalation level using LLM inference with base calculation as guidance."""
        return self._infer_escalation_level_with_llm(event, context)
    
    def _create_summary_prompt(self, event: Dict[str, Any], 
                                 context: Dict[str, Any],
                                 previous_conversations: List[str],
                                 escalation_level: int) -> str:
        """Create a prompt for LLM to generate a dialogue script/summary."""
        agent = event["agent"]
        patient = event["patient"]
        action = event.get("action", "")
        description = event.get("description", "")
        crime_type = event.get("crime_type")
        timestamp = event.get("timestamp", "unknown")
        time_duration = context.get("time_duration", "unknown")
        
        prompt = f"""Generate a script/summary describing what an online messaging conversation would contain for the following scenario:

**Participants:**
- Sender: {agent['name']} ({agent['role']})
- Receiver: {patient['name']}

**Temporal Context:**
- Event Timestamp: {timestamp}
- Overall Relationship Duration: {time_duration}

**Context:**
{description}

**Action:** {action}

**Tone:** {self._infer_tone(event, context)}

**Previous Summaries:**
"""
        if previous_conversations:
            for i, conv in enumerate(previous_conversations[-3:], 1):
                prompt += f"\n--- Previous Summary {i} ---\n{conv}\n"
        else:
            prompt += "- This is an early interaction\n"
        
        prompt += f"""
**Content Requirements:**
"""
        for point in self._extract_main_points(event, context):
            prompt += f"- {point}\n"
        
        if crime_type:
            prompt += f"""
**Red Flags to Include:**
"""
            for flag in self._identify_red_flags(event, context):
                prompt += f"- {flag}\n"
        
        prompt += f"""
**Style Guidelines:**
"""
        for note in self._generate_style_notes(event, context):
            prompt += f"- {note}\n"
        
        prompt += f"""
**Output Format:**
Generate a narrative script/summary that describes what this conversation would contain, including:
- The general flow and structure of the conversation
- Key messages and topics that would be discussed
- The emotional dynamics and tone throughout
- Specific phrases or types of language that would be used
- How the conversation would build on previous summaries
- Red flags or concerning patterns that would appear

DO NOT generate the actual conversation with individual messages. Instead, provide a detailed description/script of what the conversation would include based on the scenario and guidelines above.
"""
        
        return prompt
    
    def _create_expansion_prompt(self, physical_event: Dict[str, Any],
                               context: Dict[str, Any],
                               expansion_type: str,
                               platform: str,
                               previous_conversations: List[str]) -> str:
        """Create prompt for expanded events from physical events with context from previous summaries."""
        agent = physical_event["agent"]
        patient = physical_event["patient"]
        action = physical_event.get("action", "")
        description = physical_event.get("description", "")
        
        timing_desc = {
            "planning": "BEFORE the physical event (planning/coordination)",
            "during": "DURING or immediately after the physical event",
            "aftermath": "AFTER the physical event (reflection/discussion)"
        }
        
        prompt = f"""Generate a realistic {platform} conversation that happens {timing_desc[expansion_type]}.

**Physical Event to Reference:**
{description}

**Participants:**
- Sender: {agent['name']} ({agent['role']})
- Receiver: {patient['name']}

**Purpose:** {expansion_type.capitalize()} conversation that conveys information about the physical event through online chat.

**Platform:** {self.chat_templates.get(platform, platform)}

**Previous Summaries:**
"""
        if previous_conversations:
            for i, conv in enumerate(previous_conversations[-3:], 1):
                prompt += f"\n--- Previous Summary {i} ---\n{conv}\n"
        else:
            prompt += "- This is an early interaction\n"
        
        prompt += f"""
**Requirements:**
- The conversation should naturally reference or hint at the physical event
- Maintain character consistency and relationship dynamics based on previous summaries
- The online chat should cover the key information from: {action}
"""
        
        if expansion_type == "planning":
            prompt += "- Show coordination or anticipation of the upcoming event\n"
        elif expansion_type == "during":
            prompt += "- Convey immediacy and real-time updates\n"
        else:  # aftermath
            prompt += "- Reflect on what happened and its implications\n"
        
        prompt += """
**Output Format:**
Generate a natural conversation that would realistically occur via this platform.
Make sure this conversation builds naturally on the previous summaries shown above.
"""
        
        return prompt
    
    def _parse_time_duration_to_days(self, time_duration: str) -> int:
        """Parse time_duration string to total days."""
        time_duration = time_duration.lower().strip()
        
        # Parse patterns like "18 months", "2 years", "6-12 months", etc.
        import re
        
        # Handle ranges by taking the midpoint
        range_match = re.search(r'(\d+)[-–](\d+)\s*(month|year)', time_duration)
        if range_match:
            low = int(range_match.group(1))
            high = int(range_match.group(2))
            value = (low + high) / 2
            unit = range_match.group(3)
        else:
            # Single value
            match = re.search(r'(\d+(?:\.\d+)?)\s*(month|year)', time_duration)
            if match:
                value = float(match.group(1))
                unit = match.group(2)
            else:
                # Default to 12 months if unparseable
                return 365
        
        # Convert to days
        if 'year' in unit:
            return int(value * 365)
        else:  # months
            return int(value * 30)
    
    def _calculate_escalation_for_all_events(self, all_events_with_context: List[Dict[str, Any]]) -> List[int]:
        """Pre-calculate escalation levels for all events to identify milestones."""
        escalation_levels = []
        temp_escalation = 0
        temp_previous_escalation = 0
        temp_total = len(all_events_with_context)
        
        for idx, item in enumerate(all_events_with_context):
            sub_event = item["sub_event"]
            context = item["context"]
            
            # If sub_event already has escalation_level, use it directly
            if "escalation_level" in sub_event and sub_event["escalation_level"] is not None:
                escalation = sub_event["escalation_level"]
                escalation_levels.append(escalation)
                continue
            
            # Otherwise, calculate escalation level using same logic as _calculate_escalation_level
            # (simplified version for pre-calculation)
            crime_hints = context.get("crime_type_hints", [])
            description = sub_event.get("description", "").lower()
            action = sub_event.get("action", "").lower()
            
            timeline_position = idx / max(1, temp_total - 1) if temp_total > 1 else 0.0
            
            # Violence/threat keywords
            violence_keywords = ['assault', 'violence', 'physical', 'attack', 'hit', 'hurt', 'harm']
            severe_threat_keywords = ['kill', 'murder', 'weapon', 'death', 'injure', 'destroy']
            
            has_violence = any(kw in description or kw in action for kw in violence_keywords)
            has_severe_threat = any(kw in description or kw in action for kw in severe_threat_keywords)
            has_stalking = 'stalking' in crime_hints or 'stalk' in description or 'follow' in description
            has_harassment = 'harassment' in crime_hints or 'harass' in description
            has_digital = 'digital' in crime_hints
            has_coercive = 'coercive_control' in crime_hints
            
            # Determine escalation level
            if has_severe_threat or ('domestic_violence' in crime_hints and has_violence):
                escalation = 4
            elif has_violence or has_stalking or (has_harassment and timeline_position > 0.5):
                escalation = 3
            elif has_digital or has_coercive or has_harassment or timeline_position > 0.3:
                escalation = 2
            elif timeline_position > 0.1:
                escalation = 1
            else:
                escalation = 0
            
            # Apply minimum escalation progression
            if escalation < temp_previous_escalation and temp_previous_escalation >= 3:
                escalation = max(escalation, temp_previous_escalation - 2)
            elif escalation < temp_escalation:
                escalation = max(escalation, temp_escalation - 1)
            
            escalation_levels.append(escalation)
            temp_previous_escalation = temp_escalation
            temp_escalation = escalation
        
        return escalation_levels
    
    def _assign_timestamps_with_escalation_rules(self, 
                                                  all_events_with_context: List[Dict[str, Any]],
                                                  time_duration: str,
                                                  end_date: datetime = None) -> None:
        """Assign timestamps to all events based on escalation milestones and parent grouping.
        
        Strategy:
        1. Group sub-events by parent event
        2. Assign timestamps to PARENT groups with phase-based spacing (40-120 days early, 1-14 days late)
        3. Within each parent group, assign sub-events with tight clustering (1-3 days)
        
        Rules:
        - Before first escalation 2: parents spaced 40-120 days apart
        - Between esc 2 and 3: parents spaced 7-60 days apart
        - After escalation 3: parents spaced 1-14 days apart
        - Sub-events within parent: 1-3 days apart
        
        Modifies all_events_with_context in-place by updating timestamps.
        """
        if not all_events_with_context:
            return
        
        # Parse time duration to days
        total_days = self._parse_time_duration_to_days(time_duration)
        print(f"  Total timeline duration: {total_days} days")
        
        # Set end date (working backwards from 2026-03-01)
        if end_date is None:
            end_date = datetime(2026, 3, 1)
        
        # Pre-calculate escalation levels for all events
        escalation_levels = self._calculate_escalation_for_all_events(all_events_with_context)
        
        # Group sub-events by parent
        parent_groups = {}  # parent_id -> [indices]
        parent_order = []   # Ordered list of parent IDs
        
        for idx, item in enumerate(all_events_with_context):
            parent_id = item["context"]["parent_event_id"]
            if parent_id not in parent_groups:
                parent_groups[parent_id] = []
                parent_order.append(parent_id)
            parent_groups[parent_id].append(idx)
        
        num_parents = len(parent_order)
        print(f"  Grouped {len(all_events_with_context)} sub-events into {num_parents} parent groups")
        
        # Assign escalation level to each parent (max of its sub-events)
        parent_escalations = {}
        for parent_id in parent_order:
            indices = parent_groups[parent_id]
            parent_escalations[parent_id] = max(escalation_levels[i] for i in indices)
        
        # Find escalation milestones at PARENT level
        first_esc2_parent_idx = None
        first_esc3_parent_idx = None
        for p_idx, parent_id in enumerate(parent_order):
            if parent_escalations[parent_id] >= 2 and first_esc2_parent_idx is None:
                first_esc2_parent_idx = p_idx
            if parent_escalations[parent_id] >= 3 and first_esc3_parent_idx is None:
                first_esc3_parent_idx = p_idx
        
        print(f"  Parent escalation milestones: Level 2 at parent {first_esc2_parent_idx}, Level 3 at parent {first_esc3_parent_idx}")
        
        # Store for later use (use first sub-event index of each milestone parent)
        if first_esc2_parent_idx is not None:
            self.first_esc2_index = parent_groups[parent_order[first_esc2_parent_idx]][0]
        else:
            self.first_esc2_index = None
            
        if first_esc3_parent_idx is not None:
            self.first_esc3_index = parent_groups[parent_order[first_esc3_parent_idx]][0]
        else:
            self.first_esc3_index = None
        
        # Phase budgets (distributed across parent groups)
        if first_esc2_parent_idx is None:
            num_before_esc2 = num_parents // 2
            num_between = 0
            num_after_esc3 = num_parents - num_before_esc2
        elif first_esc3_parent_idx is None:
            num_before_esc2 = first_esc2_parent_idx
            num_between = 0
            num_after_esc3 = num_parents - first_esc2_parent_idx
        else:
            num_before_esc2 = first_esc2_parent_idx
            num_between = first_esc3_parent_idx - first_esc2_parent_idx
            num_after_esc3 = num_parents - first_esc3_parent_idx
        
        # Time budgets
        days_before_esc2 = total_days * 0.2
        days_between = total_days * 0.2
        days_after_esc3 = total_days * 0.6
        
        print(f"  Phase budgets: Before esc2={days_before_esc2:.0f}d ({num_before_esc2} parents), "
              f"Between={days_between:.0f}d ({num_between} parents), "
              f"After esc3={days_after_esc3:.0f}d ({num_after_esc3} parents)")
        
        # Assign timestamps to PARENT groups working backwards
        current_date = end_date
        parent_timestamps = {}  # parent_id -> datetime (start time of parent group)
        
        for p_idx in range(num_parents - 1, -1, -1):
            parent_id = parent_order[p_idx]
            parent_esc = parent_escalations[parent_id]
            sub_event_count = len(parent_groups[parent_id])
            
            # Determine gap to PREVIOUS parent based on phase
            if first_esc3_parent_idx is not None and p_idx >= first_esc3_parent_idx:
                # Late phase: tight spacing between parents (1-14 days)
                gap_days = random.uniform(1, 14)
            elif first_esc2_parent_idx is not None and p_idx >= first_esc2_parent_idx:
                # Middle phase: moderate spacing (7-60 days)
                gap_days = random.uniform(7, 60)
            else:
                # Early phase: large spacing (40-120 days)
                gap_days = random.uniform(40, 120)
            
            # Move back by gap (this is END time of current parent group)
            parent_end_time = current_date
            
            # Reserve time for sub-events within this parent (1-3 days each)
            # Sub-events will be spread across this range
            subevent_span_days = (sub_event_count - 1) * random.uniform(1, 3) if sub_event_count > 1 else 0
            parent_start_time = parent_end_time - timedelta(days=subevent_span_days)
            
            # Store parent start time
            parent_timestamps[parent_id] = parent_start_time
            
            # Move current_date back by gap for next parent
            current_date = parent_start_time - timedelta(days=gap_days)
        
        # Now assign timestamps to individual sub-events within each parent group
        for parent_id in parent_order:
            indices = parent_groups[parent_id]
            parent_start = parent_timestamps[parent_id]
            
            if len(indices) == 1:
                # Single sub-event: use parent start time
                all_events_with_context[indices[0]]["sub_event"]["timestamp"] = parent_start.strftime("%Y-%m-%dT%H:%M:%S")
            else:
                # Multiple sub-events: distribute with 1-3 day gaps
                current_sub_time = parent_start
                for sub_idx in indices:
                    all_events_with_context[sub_idx]["sub_event"]["timestamp"] = current_sub_time.strftime("%Y-%m-%dT%H:%M:%S")
                    # Move forward 1-3 days for next sub-event
                    current_sub_time += timedelta(days=random.uniform(1, 3))
        
        # Verify milestone constraints (using first sub-event of milestone parent)
        if self.first_esc2_index is not None:
            first_esc2_date = datetime.fromisoformat(all_events_with_context[self.first_esc2_index]["sub_event"]["timestamp"].replace('Z', ''))
            days_used_before_esc2 = (end_date - first_esc2_date).days
            print(f"  Days before first Esc2: {days_used_before_esc2} (target: >= {total_days * 0.2:.0f})")
        
        if self.first_esc3_index is not None:
            first_esc3_date = datetime.fromisoformat(all_events_with_context[self.first_esc3_index]["sub_event"]["timestamp"].replace('Z', ''))
            days_used_before_esc3 = (end_date - first_esc3_date).days
            print(f"  Days before first Esc3: {days_used_before_esc3} (target: >= {total_days * 0.4:.0f})")
        
        first_date = datetime.fromisoformat(all_events_with_context[0]["sub_event"]["timestamp"].replace('Z', ''))
        actual_span = (end_date - first_date).days
        print(f"  Actual timeline span: {actual_span} days (target: ~{total_days})")
        print(f"  Parent-level spacing: early phase=40-120d, mid=7-60d, late=1-14d | Sub-events within parent=1-3d")
    
    def process_event_graph(self, graph_file: Path) -> Dict[str, Any]:
        """
        Process a single event graph file and generate chat scripts.
        
        Returns:
            Dictionary containing all generated chat scripts and metadata
        """
        print(f"\nProcessing: {graph_file.name}")
        
        with open(graph_file, 'r') as f:
            graph_data = json.load(f)
        
        scenario_id = graph_data["scenario_id"]
        victim_name = graph_data["victim_name"]
        perpetrator_name = graph_data["perpetrator_name"]
        raw_third_party_names = graph_data.get("third_party_names", [])

        # ── Infer victim/perpetrator names from event data when persona names are None ──
        if not victim_name or not perpetrator_name:
            _agent_perp_counts: dict = {}   # name -> count as perpetrator agent
            _patient_when_perp: dict = {}   # name -> count as patient when agent is perpetrator
            for _exp_ev in graph_data.get("expanded_events", []):
                for _se in _exp_ev.get("sub_events", []):
                    _a = _se.get("agent", {})
                    _p = _se.get("patient", {})
                    _a_name = _a.get("name", "") or ""
                    _a_role = (_a.get("role") or "").lower()
                    _p_name = _p.get("name", "") or ""
                    if _a_name and _a_role == "perpetrator":
                        _agent_perp_counts[_a_name] = _agent_perp_counts.get(_a_name, 0) + 1
                        if _p_name:
                            _patient_when_perp[_p_name] = _patient_when_perp.get(_p_name, 0) + 1
            if not perpetrator_name and _agent_perp_counts:
                perpetrator_name = max(_agent_perp_counts, key=_agent_perp_counts.get)
                print(f"  Inferred perpetrator_name: {perpetrator_name!r} from event agents")
            if not victim_name and _patient_when_perp:
                _victim_candidates = {n: c for n, c in _patient_when_perp.items() if n != perpetrator_name}
                if _victim_candidates:
                    victim_name = max(_victim_candidates, key=_victim_candidates.get)
                    print(f"  Inferred victim_name: {victim_name!r} from event patients")

        # ── Sanitize: replace any "unknown" / placeholder names ────────────────────
        third_party_names = self._sanitize_third_party_names(
            raw_third_party_names,
            reserved={victim_name, perpetrator_name},
        )
        if third_party_names != raw_third_party_names:
            fixed = [
                f"{old!r}→{new!r}"
                for old, new in zip(raw_third_party_names, third_party_names)
                if old != new
            ]
            print(f"  ⚠ Resolved placeholder third-party names: {', '.join(fixed)}")
        
        # Extract time_duration from metadata
        metadata = graph_data.get("metadata", {})
        time_duration = metadata.get("time_duration", "unknown")
        timeline_span_days = graph_data.get("timeline_span_days", None)
        
        print(f"  Time Duration: {time_duration}")
        if timeline_span_days:
            print(f"  Timeline Span: {timeline_span_days} days (~{timeline_span_days/30:.1f} months)")
        
        # Collect all sub-events with their parent context
        all_events_with_context = []
        
        for expanded_event in graph_data.get("expanded_events", []):
            parent_event = expanded_event["parent_event"]
            sub_events = expanded_event.get("sub_events", [])
            
            parent_context = {
                "parent_event_id": parent_event["event_id"],
                "parent_description": parent_event["description"],
                "time_hint": parent_event.get("attributes", {}).get("time_hint", "mid"),
                "crime_type_hints": parent_event.get("attributes", {}).get("crime_type_hints", []),
                "victim_name": victim_name,
                "perpetrator_name": perpetrator_name,
                "third_party_names": third_party_names,
                "time_duration": time_duration,
                "timeline_span_days": timeline_span_days
            }
            
            for sub_event in sub_events:
                # ── Resolve "unknown" agent / patient names in this sub-event ────────
                _tp_fallback = third_party_names[0] if third_party_names else None
                _reserved = {victim_name, perpetrator_name}
                for _slot in ("agent", "patient"):
                    _actor = sub_event.get(_slot, {})
                    if self._is_placeholder_name(_actor.get("name", "")):
                        _role = _actor.get("role") or _actor.get("type", "")
                        if _role == "third_party":
                            if _tp_fallback:
                                _actor["name"] = _tp_fallback
                            else:
                                _actor["name"] = self._generate_fallback_name(
                                    set(_reserved)
                                )
                        elif _role in ("victim",) or (_slot == "patient" and _role not in ("perpetrator",)):
                            # Patient with no explicit role or victim role → use victim_name if available
                            if victim_name:
                                _actor["name"] = victim_name
                            elif _tp_fallback:
                                _actor["name"] = _tp_fallback
                            else:
                                _actor["name"] = self._generate_fallback_name(set(_reserved))
                        elif _role == "perpetrator":
                            if perpetrator_name:
                                _actor["name"] = perpetrator_name

                # Determine receiver role — try name match first, fall back to patient.role
                _patient = sub_event.get("patient", {})
                receiver_name = _patient.get("name", "")
                _patient_role = (_patient.get("role") or "").lower()
                if receiver_name == victim_name:
                    receiver_role = "victim"
                elif receiver_name == perpetrator_name:
                    receiver_role = "perpetrator"
                elif _patient_role in ("victim", "perpetrator"):
                    receiver_role = _patient_role
                else:
                    receiver_role = "third_party"
                
                parent_context["receiver_role"] = receiver_role
                
                all_events_with_context.append({
                    "sub_event": sub_event,
                    "context": parent_context.copy()
                })
        
        # IMPORTANT: Reassign ALL timestamps based on escalation rules
        # This overrides any timestamps from the expanded graph
        print(f"  Reassigning timestamps with escalation-based rules...")
        self._assign_timestamps_with_escalation_rules(
            all_events_with_context,
            time_duration,
            end_date=datetime(2026, 3, 1)
        )
        
        # Sort by timestamp (after reassignment)
        all_events_with_context.sort(key=lambda x: x["sub_event"]["timestamp"])

        # Cap to 20 sub-events before generating LLM calls.
        # Rank by escalation level (highest = most narratively important), keep top 20,
        # then restore temporal order so self.total_events and timeline-position
        # heuristics downstream are always consistent with the actual event count.
        max_scripts = 20
        if len(all_events_with_context) > max_scripts:
            esc_levels = self._calculate_escalation_for_all_events(all_events_with_context)
            ranked = sorted(range(len(all_events_with_context)), key=lambda i: esc_levels[i], reverse=True)
            kept = sorted(ranked[:max_scripts])
            print(f"  Capped sub-events to {max_scripts} (was {len(all_events_with_context)})")
            all_events_with_context = [all_events_with_context[i] for i in kept]

        # Generate chat scripts in temporal order
        chat_scripts = []
        previous_conversations = []  # Store summaries for context
        self.escalation_level = 0  # Reset escalation tracking for this scenario
        self.previous_escalation_level = 0  # Reset previous level tracking
        self.high_escalation_event_occurred = False  # Reset high escalation flag
        self.total_events = len(all_events_with_context)  # Set total for timeline progression
        self.current_event_index = 0  # Reset event counter
        
        for item in all_events_with_context:
            self.current_event_index += 1  # Increment for each event
            sub_event = item["sub_event"]
            context = item["context"]
            
            event_type = self.identify_event_type(sub_event)
            
            if event_type == "online":
                # Direct conversion to chat script
                chat_script = self.generate_online_chat_script(sub_event, context, previous_conversations)
                chat_scripts.append(chat_script)
                
                # Add generated summary to context for next events
                if chat_script.get("generation_success") and chat_script.get("conversation_summary"):
                    previous_conversations.append(chat_script["conversation_summary"])
                
            else:  # physical event
                # Expand into 1-3 online chat events
                expanded_chats = self.expand_physical_event(sub_event, context, previous_conversations)
                chat_scripts.extend(expanded_chats)
                
                # Add each generated summary to context
                for chat in expanded_chats:
                    if chat.get("generation_success") and chat.get("conversation_summary"):
                        previous_conversations.append(chat["conversation_summary"])
        
        # Compile output
        output_data = {
            "scenario_id": scenario_id,
            "victim_name": victim_name,
            "perpetrator_name": perpetrator_name,
            "third_party_names": third_party_names,
            "num_chat_scripts": len(chat_scripts),
            "chat_scripts": chat_scripts,
            "metadata": {
                "source_file": str(graph_file),
                "time_duration": time_duration,
                "timeline_span_days": graph_data.get("timeline_span_days"),
                "original_num_events": len(graph_data.get("expanded_events", [])),
                "num_original_sub_events": sum(len(e.get("sub_events", [])) for e in graph_data.get("expanded_events", [])),
                "num_online_events": sum(1 for script in chat_scripts if script["event_type"] == "online_chat"),
                "num_expanded_events": sum(1 for script in chat_scripts if script["event_type"] == "online_chat_expansion"),
                "generation_timestamp": datetime.now().isoformat()
            }
        }
        
        print(f"  Generated {len(chat_scripts)} chat scripts")
        print(f"    - {output_data['metadata']['num_online_events']} direct online events")
        print(f"    - {output_data['metadata']['num_expanded_events']} expanded from physical events")
        
        return output_data
    
    def save_chat_scripts(self, output_data: Dict[str, Any], run_dir: Path):
        """Save the generated chat scripts to a file."""
        output_file = run_dir / f"{output_data['scenario_id']}.json"
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"  Saved to: {output_file}")
        
        return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Generate chat scripts from expanded event graphs with LLM"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="./exampled_expanded_graphs",
        help="Directory containing expanded event graph JSON files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./chat_scripts",
        help="Base directory for output chat scripts"
    )
    parser.add_argument(
        "--files",
        type=str,
        nargs="+",
        help="Specific files to process (optional, processes all if not specified)"
    )
    parser.add_argument(
        "--model",
        choices=["gpt52", "deepseek_v4_flash", "gemini_flash_lite", "qwen35_35b", "llama3_8b"],
        help="Preset model alias to use (default: gpt52)"
    )
    parser.add_argument(
        "--model-config",
        type=str,
        help="Custom YAML config path. Overrides --model when provided."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        help="Direct HuggingFace model name (legacy, use --model or --model-config instead)"
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        default=True,
        help="Generate actual conversations using LLM (default: True)"
    )
    parser.add_argument(
        "--no-llm",
        action="store_false",
        dest="use_llm",
        help="Skip LLM generation, only create prompts"
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_base = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return
    
    # Determine config path based on model preset or custom config
    base_dir = Path(__file__).parent.parent.parent
    preset_configs = {
        "gpt52": base_dir / "RAG_kb_construct" / "configs" / "gpt52_openai.yaml",
        "deepseek_v4_flash": base_dir / "RAG_kb_construct" / "configs" / "deepseek_v4_flash_openai.yaml",
        "gemini_flash_lite": base_dir / "RAG_kb_construct" / "configs" / "gemini_flash_lite_openai.yaml",
        "qwen35_35b": base_dir / "RAG_kb_construct" / "configs" / "qwen35_35b_vllm.yaml",
        "llama3_8b": base_dir / "RAG_kb_construct" / "configs" / "llama3_8b_vllm.yaml",
    }
    
    config_path = None
    model_name = None
    model_display = "custom"
    
    if args.model_config:
        # Custom config file
        config_path = Path(args.model_config)
        if not config_path.exists():
            print(f"Error: Config file not found: {config_path}")
            return
        model_display = f"custom ({config_path.name})"
    elif args.model:
        # Model preset
        config_path = preset_configs[args.model]
        if not config_path.exists():
            print(f"Error: Preset config not found: {config_path}")
            return
        model_display = args.model
    elif args.model_name:
        # Legacy direct model name
        model_name = args.model_name
        model_display = f"HuggingFace: {model_name}"
    else:
        # Default to gpt52 (the paper backbone)
        config_path = preset_configs["gpt52"]
        if not config_path.exists():
            print(f"Error: Default config not found: {config_path}")
            return
        model_display = "gpt52 (default)"
    
    # Create output directory
    run_dir = output_base
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Chat Script Generation")
    print(f"=" * 60)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {run_dir}")
    print(f"Model: {model_display}")
    if config_path:
        print(f"Config: {config_path}")
    print(f"LLM Generation: {'Enabled' if args.use_llm else 'Disabled (prompt-only mode)'}")
    print()
    
    # Get files to process — exclude known report/metadata files
    _REPORT_NAMES = {"expansion_report.json", "event_graph_report.json", "generation_report.json"}
    if args.files:
        graph_files = [input_dir / f for f in args.files if (input_dir / f).exists()]
    else:
        graph_files = [f for f in input_dir.glob("*.json") if f.name not in _REPORT_NAMES]
    
    if not graph_files:
        print("No graph files found to process")
        return
    
    print(f"Found {len(graph_files)} files to process")
    
    # Initialize generator
    generator = ChatScriptGenerator(
        run_dir, 
        config_path=config_path,
        model_name=model_name,
        use_llm=args.use_llm
    )
    
    # Process each file
    results = []
    for graph_file in graph_files:
        try:
            output_data = generator.process_event_graph(graph_file)
            output_file = generator.save_chat_scripts(output_data, run_dir)
            results.append({
                "scenario_id": output_data["scenario_id"],
                "status": "success",
                "num_scripts": output_data["num_chat_scripts"],
                "output_file": str(output_file)
            })
        except Exception as e:
            print(f"  Error processing {graph_file.name}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "scenario_id": graph_file.stem,
                "status": "error",
                "error": str(e)
            })
    
    # Save summary report
    report = {
        "run_timestamp": datetime.now().isoformat(),
        "input_directory": str(input_dir),
        "output_directory": str(run_dir),
        "total_files": len(graph_files),
        "successful": sum(1 for r in results if r["status"] == "success"),
        "failed": sum(1 for r in results if r["status"] == "error"),
        "results": results
    }
    
    report_file = run_dir / "generation_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total files processed: {report['total_files']}")
    print(f"Successful: {report['successful']}")
    print(f"Failed: {report['failed']}")
    print(f"\nReport saved to: {report_file}")


if __name__ == "__main__":
    main()
