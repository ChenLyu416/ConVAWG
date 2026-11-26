"""
Generate Roleplay Dialogues from Chat Scripts

This script takes chat script files (generated from event graphs) and uses LLM roleplay
to generate realistic multi-turn conversations between characters. The LLM plays both
characters sequentially to create natural dialogue exchanges.
"""

import json
import os
import argparse
import sys
import random
import traceback
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "RAG_kb_construct"))

try:
    from llm_structurer.llm_backends import HuggingFaceBackend, VLLMBackend
    from llm_structurer.config import Config
    print("Successfully imported LLM backends")
except ImportError as e:
    print(f"Warning: Could not import backends ({e})")
    HuggingFaceBackend = None
    VLLMBackend = None
    Config = None


class RoleplayDialogueGenerator:
    """Generates roleplay dialogues from chat scripts using LLM."""
    
    def __init__(self, config_path: Path = None, model_name: str = None, output_dir: Path = None,
                 min_turns: int = 2, max_turns: int = 10, adaptive_length: bool = True,
                 enable_enforcer: bool = True, prompt_template_path: Path = None,
                 template_name: str = "edited_template_FINAL", bare_prompt: bool = False):
        """
        Initialize the roleplay dialogue generator.
        
        Args:
            config_path: Path to LLM config YAML (a backend config under RAG_kb_construct/configs/)
            model_name: Direct model name for HuggingFace (legacy)
            output_dir: Directory to save generated dialogues
            min_turns: Minimum number of speaking turns per character (default: 2)
            max_turns: Maximum number of speaking turns per character (default: 10)
            adaptive_length: Whether to adjust length based on escalation level (default: True)
            enable_enforcer: Whether to apply rule-based enforcement (split, slang) (default: True)
            prompt_template_path: Path to prompt template YAML (default: ./roleplay_prompt_template.yaml)
            template_name: Which template variant to use from YAML (default: "edited_template_FINAL")
            bare_prompt: If True, omit style notes and red flags from the prompt (default: False)
        """
        self.config_path = config_path
        self.model_name = model_name
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.min_turns = min_turns
        self.max_turns = max_turns
        self.adaptive_length = adaptive_length
        self.enable_enforcer = enable_enforcer
        self.template_name = template_name
        self.bare_prompt = bare_prompt
        
        self.config = None
        self.llm_backend = None
        
        # Load prompt template
        if prompt_template_path is None:
            prompt_template_path = Path(__file__).parent / "roleplay_prompt_template.yaml"
        self.prompt_template = self._load_prompt_template(prompt_template_path)
        
        # Only initialize LLM if needed (not enforcer-only mode)
        # LLM is needed for generation
        self.needs_llm = True  # Always True for generation, checked in main
        
        # Don't initialize LLM yet - will be done in main() if needed
        # self._initialize_llm()
        
        # Initialize sentence embedding model for semantic splitting (lightweight)
        self.sentence_encoder = None
        if enable_enforcer:
            self._initialize_sentence_encoder()
        
        # Statistics
        self.stats = {
            "total_events": 0,
            "successful": 0,
            "failed": 0,
            "enforced": 0,
            "victim_initiated": 0,
            "perpetrator_initiated": 0
        }
    
    def _load_prompt_template(self, template_path: Path) -> Dict[str, str]:
        """Load prompt template from YAML file."""
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                templates = yaml.safe_load(f)
            print(f"✓ Loaded prompt template from: {template_path}")
            return templates
        except Exception as e:
            print(f"✗ Failed to load prompt template: {e}")
            sys.exit(1)
    
    def ensure_llm_initialized(self):
        """Initialize LLM backend if not already initialized and needed."""
        if self.llm_backend is None and self.needs_llm:
            self._initialize_llm()
    
    def _initialize_sentence_encoder(self):
        """Initialize lightweight sentence encoder for semantic splitting."""
        try:
            from sentence_transformers import SentenceTransformer
            print("Loading sentence encoder for semantic splitting...")
            # Use a small, fast model
            self.sentence_encoder = SentenceTransformer('all-MiniLM-L6-v2')
            print("✓ Sentence encoder loaded")
        except ImportError:
            print("Warning: sentence-transformers not installed. Falling back to simple splitting.")
            print("Install with: pip install sentence-transformers")
            self.sentence_encoder = None
        except Exception as e:
            print(f"Warning: Failed to load sentence encoder: {e}")
            print("Falling back to simple splitting.")
            self.sentence_encoder = None
    
    def _initialize_llm(self):
        """Initialize the LLM backend based on config or model name."""
        try:
            if self.config_path:
                # Use config file
                if not Config:
                    print("✗ Config class not available")
                    sys.exit(1)
                
                self.config = Config(str(self.config_path))
                backend_type = self.config.llm_backend.lower()
                model_name = self.config.model_name
                print(f"Initializing LLM from config: {model_name} (backend: {backend_type})")
                
                if backend_type == "huggingface":
                    if not HuggingFaceBackend:
                        print("✗ HuggingFace backend not available")
                        sys.exit(1)
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
                        api_key = self.config.get_openai_api_key()
                    self.llm_backend = OpenAIBackend(
                        model_name=model_name,
                        api_key=api_key,
                        base_url=self.config.get('llm.base_url')
                    )
                elif backend_type == "vllm":
                    if not VLLMBackend:
                        print("✗ vLLM backend not available")
                        sys.exit(1)

                    self.llm_backend = VLLMBackend(
                        model_name=model_name,
                        base_url=self.config.get('llm.base_url', None),
                        api_key=self.config.get('llm.api_key', None),
                    )
                else:
                    print(f"✗ Unsupported backend: {backend_type}")
                    sys.exit(1)
                
                self.model_name = model_name
                print("✓ LLM backend initialized successfully")
            
            elif self.model_name:
                # Legacy: direct model name (assume HuggingFace)
                print(f"Initializing LLM (HuggingFace): {self.model_name}")
                if not HuggingFaceBackend:
                    print("✗ HuggingFace backend not available")
                    sys.exit(1)
                
                self.llm_backend = HuggingFaceBackend(
                    model_name=self.model_name,
                    load_in_8bit=True,
                    device_map="auto"
                )
                print("✓ LLM backend initialized successfully")
            else:
                print("✗ No config or model name provided")
                sys.exit(1)
                
        except Exception as e:
            print(f"✗ Failed to initialize LLM: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    def _load_persona_info(self, scenario_id: str, persona_dir: Path) -> Optional[Dict[str, Any]]:
        """Load persona information from persona_processed_full_gpt5 directory.

        Search strategy:
        1. ``persona_dir / {scenario_id}.json``  (configured path, primary)
        2. Sibling directories of ``persona_dir`` — handles the common case
           where the config points to the wrong sub-folder (e.g. time_test_5
           instead of time_test_10) without failing silently.
        """
        persona_file = persona_dir / f"{scenario_id}.json"

        if persona_file.exists():
            try:
                with open(persona_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"    Error loading persona file {persona_file}: {e}")
                return None

        # ── Fallback: search sibling directories of persona_dir ──────────────
        parent = persona_dir.parent
        if parent.exists():
            for sibling in sorted(parent.iterdir()):
                if sibling == persona_dir or not sibling.is_dir():
                    continue
                candidate = sibling / f"{scenario_id}.json"
                if candidate.exists():
                    print(
                        f"    Warning: Persona file not in '{persona_dir.name}', "
                        f"found in sibling dir '{sibling.name}': {candidate.name}"
                    )
                    try:
                        with open(candidate, 'r', encoding='utf-8') as f:
                            return json.load(f)
                    except Exception as e:
                        print(f"    Error loading persona file {candidate}: {e}")
                        return None

        print(f"    Warning: Persona file not found for scenario {scenario_id} "
              f"(searched '{persona_dir}' and siblings)")
        return None
    
    @staticmethod
    def _parse_gap_hours(
        prev_ts: Optional[str], curr_ts: Optional[str]
    ) -> tuple:
        """Return (datetime_gap, gap_hours) or (None, None) if either timestamp is missing/unparseable."""
        if not prev_ts or not curr_ts:
            return None, None
        try:
            fmt_candidates = [
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%S.%f",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d",
            ]
            def _parse(ts):
                ts = ts.split("+")[0].split("Z")[0]  # strip timezone
                for fmt in fmt_candidates:
                    try:
                        return datetime.strptime(ts, fmt)
                    except ValueError:
                        continue
                raise ValueError(f"Unrecognised timestamp format: {ts!r}")

            t_prev = _parse(str(prev_ts))
            t_curr = _parse(str(curr_ts))
            gap = t_curr - t_prev
            gap_hours = gap.total_seconds() / 3600
            return gap, gap_hours
        except Exception:
            return None, None

    def _compute_time_gap_str(self, prev_ts: Optional[str], curr_ts: Optional[str]) -> str:
        """Return a plain-English description of the time gap between two ISO timestamps.

        Used to tell the LLM how much time has passed between conversations so it
        can decide whether to treat this as a seamless continuation or a fresh
        conversation that opens with a new greeting.
        """
        gap, gap_hours = self._parse_gap_hours(prev_ts, curr_ts)
        if gap_hours is None:
            return "unknown time"

        if gap_hours < 0:
            return "around the same time"
        elif gap_hours < 0.5:
            return "moments later (same conversation)"
        elif gap_hours < 2:
            return f"about {int(gap_hours * 60)} minutes later"
        elif gap_hours < 24:
            return f"about {int(gap_hours)} hours later (same day)"
        elif gap_hours < 48:
            return "the next day"
        elif gap_hours < 168:  # < 1 week
            days = int(gap_hours / 24)
            return f"{days} days later"
        elif gap_hours < 720:  # < 1 month
            weeks = int(gap_hours / 168)
            return f"about {weeks} week{'s' if weeks > 1 else ''} later"
        else:
            months = int(gap_hours / 720)
            return f"about {months} month{'s' if months > 1 else ''} later"

    def _calculate_victim_initiation_probability(self, persona_info: Dict[str, Any], current_ratio: float) -> float:
        """Calculate probability of victim initiating conversation based on persona traits.
        
        Args:
            persona_info: Persona information containing victim and perpetrator profiles
            current_ratio: Current victim initiation ratio (victim_initiated / total_conversations)
            
        Returns:
            Probability (0.0-1.0) that victim should initiate this conversation
        """
        import random
        
        # Base probability: aim for 30% (0.3) overall
        base_probability = 0.30
        
        # Adjust based on current ratio to maintain target
        # If below target, increase probability; if above, decrease
        ratio_adjustment = (base_probability - current_ratio) * 0.5
        probability = base_probability + ratio_adjustment
        
        # Ensure minimum 10% victim initiation
        probability = max(0.10, probability)
        
        # Adjust based on persona traits if available
        if persona_info:
            victim_profile = persona_info.get("victim_profile", {})
            perp_profile = persona_info.get("perpetrator_profile", {})
            
            # Victim traits that increase initiation likelihood
            victim_age = victim_profile.get("age", 30)
            if victim_age < 25:  # Younger victims may be more active
                probability += 0.05
            
            vulnerabilities = victim_profile.get("vulnerabilities", [])
            # Certain vulnerabilities reduce initiation (isolation, fear)
            isolation_keywords = ["isolated", "lonely", "socially withdrawn", "no support"]
            if any(keyword in str(v).lower() for v in vulnerabilities for keyword in isolation_keywords):
                probability -= 0.10
            
            # Perpetrator traits that affect victim initiation
            perp_characteristics = perp_profile.get("characteristics", [])
            # Controlling/monitoring perpetrators reduce victim initiation
            control_keywords = ["controlling", "monitoring", "possessive", "jealous", "stalking"]
            if any(keyword in str(c).lower() for c in perp_characteristics for keyword in control_keywords):
                probability -= 0.08
            
            # Manipulative perpetrators may get more victim-initiated contact
            manipulation_keywords = ["manipulative", "charming", "gaslighting"]
            if any(keyword in str(c).lower() for c in perp_characteristics for keyword in manipulation_keywords):
                probability += 0.05
        
        # Clamp to valid range [0.10, 0.60]
        probability = max(0.10, min(0.60, probability))
        
        return probability
    
    def _determine_conversation_initiator(self, persona_info: Dict[str, Any], event_index: int, total_events: int) -> str:
        """Determine who should initiate this conversation.
        
        Args:
            persona_info: Persona information
            event_index: Current event index (0-based)
            total_events: Total number of events in scenario
            
        Returns:
            "victim" or "perpetrator"
        """
        import random
        
        # Calculate current ratio
        total_initiated = self.stats["victim_initiated"] + self.stats["perpetrator_initiated"]
        current_ratio = self.stats["victim_initiated"] / max(1, total_initiated)
        
        # Calculate victim initiation probability
        victim_prob = self._calculate_victim_initiation_probability(persona_info, current_ratio)
        
        # Ensure minimum victim initiation in remaining events
        remaining_events = total_events - event_index
        if remaining_events > 0:
            needed_victim_initiations = max(0, int(total_events * 0.10) - self.stats["victim_initiated"])
            if needed_victim_initiations > 0 and needed_victim_initiations >= remaining_events:
                # Force victim initiation to meet minimum
                return "victim"
        
        # Random selection based on probability
        if random.random() < victim_prob:
            return "victim"
        else:
            return "perpetrator"
    
    def _create_character_profile(self, name: str, role: str, context: Dict[str, Any], persona_info: Dict[str, Any] = None) -> str:
        """Create a character profile/summary for the prompt.

        Uses persona_info when available.  The name-equality check is intentionally
        skipped for victim/perpetrator — the chat-script name and the persona-file
        name may differ across pipeline runs, but the *role* is always authoritative.
        """

        if persona_info:
            if role == "victim":
                vp = persona_info.get("victim_profile") or {}
                if vp:
                    lines = [f"{name} (victim)"]
                    demo = ", ".join(filter(None, [
                        str(vp.get('age', '')) or None,
                        vp.get('gender'),
                        vp.get('occupation'),
                    ]))
                    if demo:
                        lines.append(f"  Demographics: {demo}")
                    if vp.get('ethnicity'):
                        lines.append(f"  Ethnicity: {vp['ethnicity']}")
                    vulns = vp.get('vulnerabilities', [])
                    if vulns:
                        lines.append(f"  Vulnerabilities: {'; '.join(vulns[:4])}")
                    return "\n".join(lines)

            elif role == "perpetrator":
                pp = persona_info.get("perpetrator_profile") or {}
                if pp:
                    lines = [f"{name} (perpetrator)"]
                    demo = ", ".join(filter(None, [
                        str(pp.get('age', '')) or None,
                        pp.get('gender'),
                        pp.get('occupation'),
                    ]))
                    if demo:
                        lines.append(f"  Demographics: {demo}")
                    if pp.get('ethnicity'):
                        lines.append(f"  Ethnicity: {pp['ethnicity']}")
                    if pp.get('relationship_to_victim'):
                        lines.append(f"  Relationship to victim: {pp['relationship_to_victim']}")
                    chars = pp.get('characteristics', [])
                    if chars:
                        lines.append(f"  Characteristics: {'; '.join(chars[:3])}")
                    motivations = pp.get('motivations', [])
                    if motivations:
                        lines.append(f"  Motivations: {'; '.join(motivations[:2])}")
                    return "\n".join(lines)

            elif role == "third_party":
                for tp in (persona_info.get("third_party_actors") or []):
                    if tp.get("name") == name:
                        lines = [f"{name} (third party)"]
                        demo = ", ".join(filter(None, [
                            str(tp.get('age', '')) or None,
                            tp.get('gender'),
                            tp.get('occupation'),
                        ]))
                        if demo:
                            lines.append(f"  Demographics: {demo}")
                        if tp.get('role'):
                            lines.append(f"  Role: {tp['role']}")
                        if tp.get('relationship_to_victim'):
                            lines.append(f"  Relationship to victim: {tp['relationship_to_victim']}")
                        actions = tp.get('actions', [])
                        if actions:
                            lines.append(f"  Known actions: {'; '.join(actions[:3])}")
                        return "\n".join(lines)

        # Fallback — no persona info available
        escalation_level = context.get("escalation_level", 0)
        fallback = {
            "victim": (
                f"{name} (victim). Vulnerable, seeking connection, gradually becoming"
                f" entrapped in a controlling relationship."
                + (f" Escalation level {escalation_level}." if escalation_level > 0 else "")
            ),
            "perpetrator": (
                f"{name} (perpetrator). Exhibits manipulative and controlling behaviours,"
                f" often disguised as care and concern."
                + (f" Escalation level {escalation_level}." if escalation_level > 0 else "")
            ),
            "third_party": (
                f"{name} (third party — friend, family, or acquaintance). May provide"
                f" support or witness concerning behaviours."
            ),
        }
        return fallback.get(role, f"{name} — role: {role}.")
    
    def _calculate_violence_language_level(self, context: Dict[str, Any], sender_role: str) -> int:
        """Calculate violence/toxic language level based on escalation and sender role."""
        # Only apply to perpetrator
        if sender_role != "perpetrator":
            return 0
        
        escalation_level = context.get("escalation_level", 0)
        crime_type = str(context.get("crime_type", "")).lower()
        
        # Map escalation level directly to violence language level
        violence_level = escalation_level
        
        # Adjust based on crime type indicators
        if "threat" in crime_type or "intimidation" in crime_type:
            violence_level = max(violence_level, 3)
        elif "assault" in crime_type or "violence" in crime_type:
            violence_level = max(violence_level, 4)
        
        return min(violence_level, 4)  # Cap at 4
    
    def _infer_conversation_topic(self, context: Dict[str, Any], red_flags: List[str]) -> str:
        """Infer the conversation topic from context and red flags."""
        action = context.get("action", "").lower()
        description = context.get("description", "").lower()
        crime_type = str(context.get("crime_type", "")).lower()
        
        # Map to conversation topics
        if any(word in action or word in description for word in ["plan", "coordinate", "arrange"]):
            return "discussing plans"
        
        if any(word in action or word in description for word in ["control", "monitor", "check", "demand"]):
            return "malicious interactions"
        
        if "flirt" in action or "dating" in description:
            return "flirtation"
        
        if any(word in crime_type for word in ["harassment", "threat", "intimidation"]):
            return "arguments"
        
        if "guilt" in description or "manipulat" in description:
            return "deception"
        
        if red_flags:
            return "malicious interactions"
        
        # Default topics based on relationship stage
        escalation = context.get("escalation_level", 0)
        if escalation == 0:
            return "small talk"
        elif escalation <= 2:
            return "discussing interests"
        else:
            return "arguments"

    @staticmethod
    def _format_role_keyed_list(value: Any) -> str:
        """Format either a flat list[str] or a dict[role, list[str]] into bullet lines."""
        if value is None:
            return "- [None specified]"
        if isinstance(value, list):
            return "\n".join(f"- {item}" for item in value) if value else "- [None specified]"
        if isinstance(value, dict):
            lines: List[str] = []
            for role, items in value.items():
                if not items:
                    continue
                if isinstance(items, list):
                    for item in items:
                        lines.append(f"- ({role}) {item}")
                else:
                    lines.append(f"- ({role}) {items}")
            return "\n".join(lines) if lines else "- [None specified]"
        return f"- {value}"
    
    def _calculate_target_turns(self, context: Dict[str, Any], content_guidance: Dict[str, Any]) -> tuple[int, int]:
        """Calculate target dialogue length based on escalation and context.
        
        Returns:
            (target_turns, max_turns): Expected and maximum number of turns per character
        """
        if not self.adaptive_length:
            return (self.min_turns, self.max_turns)
        
        escalation = context.get("escalation_level", 0)
        suggested_length = content_guidance.get("suggested_length", "")
        
        # Parse suggested length if available
        if "brief" in suggested_length.lower() or "1-2" in suggested_length:
            base_turns = 2
        elif "extended" in suggested_length.lower() or "long" in suggested_length.lower():
            base_turns = 8
        else:
            # Default based on escalation level
            # Lower escalation = shorter initial exchanges
            # Higher escalation = more back-and-forth
            escalation_to_turns = {
                0: 3,  # Initial contact, brief
                1: 4,  # Early relationship building
                2: 5,  # Some tension, more discussion
                3: 6,  # Active conflict, argument
                4: 7   # Severe escalation, intense exchange
            }
            base_turns = escalation_to_turns.get(escalation, 4)
        
        # Clamp to configured bounds
        target = max(self.min_turns, min(base_turns, self.max_turns))
        max_allowed = self.max_turns
        
        return (target, max_allowed)
    
    def _create_roleplay_prompt(self, chat_script: Dict[str, Any], scenario_metadata: Dict[str, Any], persona_info: Dict[str, Any] = None, previous_context: Optional[Dict[str, Any]] = None, conversation_initiator: str = "perpetrator", all_previous_contexts: list = None) -> str:
        """Create the roleplay prompt based on chat script and the provided template.
        
        Args:
            chat_script: Current chat script event
            scenario_metadata: Overall scenario metadata
            persona_info: Optional persona information
            previous_context: Optional context from previous conversation (dialogue, topic, tone, emotion)
            conversation_initiator: Who should start this conversation ("victim" or "perpetrator")
        """
        
        # Extract scenario-level metadata
        perpetrator_name = scenario_metadata.get("perpetrator_name", "Perpetrator")
        victim_name = scenario_metadata.get("victim_name", "Victim")
        third_party_names = scenario_metadata.get("third_party_names", [])
        third_party_names_str = ", ".join(third_party_names) if third_party_names else "None"
        
        # Extract participants
        sender = chat_script["participants"]["sender"]
        receiver = chat_script["participants"]["receiver"]
        
        sender_name = sender["name"]
        sender_role = sender["role"]
        receiver_name = receiver["name"]
        receiver_role = receiver.get("role", "unknown")
        
        # Detect if this is a third-party conversation
        is_third_party_conv = (sender_role == "third_party" or receiver_role == "third_party")
        
        # Extract context
        context = chat_script.get("conversation_context", {})
        action = context.get("action", "")
        description = context.get("description", "")
        tone = context.get("tone", "neutral")
        emotional_state = context.get("emotional_state", "neutral")
        escalation_level = context.get("escalation_level", 0)
        crime_type = context.get("crime_type")
        crime_type_str = f"- Crime type: {crime_type}" if crime_type else ""
        
        # Extract guidance
        content_guidance = chat_script.get("content_guidance", {})
        main_points = content_guidance.get("main_points", [])
        style_notes = content_guidance.get("style_notes", [])
        red_flags = content_guidance.get("red_flags", [])
        suggested_length = content_guidance.get("suggested_length", "2-5 messages exchanged")
        
        # Format lists as strings
        main_points_str = "\n".join(f"- {point}" for point in main_points) if main_points else "- [No specific points]"
        style_notes_str = self._format_role_keyed_list(style_notes) if style_notes else "- [No specific style requirements]"
        red_flags_str = self._format_role_keyed_list(red_flags) if red_flags else "- [None specified]"

        # Bare-prompt mode: strip controlled style guidance and red flags entirely
        if self.bare_prompt:
            style_notes_str = ""
            red_flags_str = ""
        
        # Platform and timing
        platform = chat_script.get("platform", "messaging")
        timestamp = chat_script.get("timestamp", "")
        event_type = chat_script.get("event_type", "")
        expansion_purpose = chat_script.get("expansion_purpose", "")
        
        # Get conversation summary from chat script if available
        generation_prompt = chat_script.get("conversation_summary", "")
        if not generation_prompt:
            # Fallback: create from description and content guidance
            generation_prompt = f"""Generate a conversation where:
- Action: {action}
- Description: {description}
- Main points to cover: {', '.join(main_points) if main_points else 'general conversation'}
"""
        
        # Calculate violence/language level for perpetrator (only for perp-victim conversations)
        violence_language_level = self._calculate_violence_language_level(context, sender_role) if not is_third_party_conv else 0
        
        # Create character summaries with persona info
        sender_summary = self._create_character_profile(sender_name, sender_role, context, persona_info)
        receiver_summary = self._create_character_profile(receiver_name, receiver_role, context, persona_info)

        # Build character profiles string for prompt injection.
        # Include all participants present in this conversation event.
        # De-duplicate by name in case sender == receiver (shouldn't happen, but safe).
        profile_parts = []
        seen_names: set = set()
        for _name, _role, _profile in [
            (sender_name, sender_role, sender_summary),
            (receiver_name, receiver_role, receiver_summary),
        ]:
            if _name not in seen_names:
                profile_parts.append(_profile)
                seen_names.add(_name)
        character_profiles_str = "\n\n".join(profile_parts) if profile_parts else "[No persona profiles available]"

        # Determine conversation topic based on context
        conversation_topic = self._infer_conversation_topic(context, red_flags)
        
        # Calculate target dialogue length
        target_turns, max_allowed_turns = self._calculate_target_turns(context, content_guidance)

        # No persona-level language style control
        persona_language_style_str = "- [No persona language style specified]"
        
        # Build context continuity section if previous conversation exists
        continuity_section = ""
        if previous_context:
            prev_dialogue = previous_context.get('dialogue', '')
            prev_topic = previous_context.get('topic', '')
            prev_tone = previous_context.get('tone', '')
            prev_emotion = previous_context.get('emotional_state', '')
            prev_timestamp = previous_context.get('timestamp')
            curr_timestamp = chat_script.get('timestamp')

            # Compute human-readable time gap
            time_gap_str = self._compute_time_gap_str(prev_timestamp, curr_timestamp)

            # Truncate dialogue if too long; for large gaps, omit raw dialogue
            _, gap_hours = self._parse_gap_hours(prev_timestamp, curr_timestamp)
            if gap_hours is not None and gap_hours > 24:
                # Don't quote stale dialogue verbatim — only carry over relationship state
                prev_dialogue_truncated = "[not shown — conversation took place earlier]"
            else:
                prev_dialogue_truncated = prev_dialogue[-800:] if len(prev_dialogue) > 800 else prev_dialogue

            # Build full conversation history timeline (all events before this one).
            # Show dialogue excerpts only for prior subevents that share the same parent
            # event as the current one (e.g. E2.1, E2.2.pre → parent "E2").
            # If no prior subevent shares this parent, fall back to the last entry's excerpt.
            if all_previous_contexts:
                curr_event_id = chat_script.get("event_id", "")
                curr_parent = curr_event_id.split(".")[0] if curr_event_id else ""

                same_parent_indices = [
                    idx for idx, e in enumerate(all_previous_contexts)
                    if curr_parent and e.get("event_id", "").split(".")[0] == curr_parent
                ]
                # Fallback: if no same-parent prior subevent, show excerpt for the last entry only
                show_excerpt_for = set(same_parent_indices) if same_parent_indices else {len(all_previous_contexts) - 1}

                history_blocks = []
                for idx, entry in enumerate(all_previous_contexts):
                    block = f"  [{idx+1}] {entry['timestamp']}  —  {entry['topic']}"
                    if idx in show_excerpt_for:
                        excerpt = entry.get("excerpt", "").strip()
                        if excerpt:
                            indented = "\n".join(f"      {ln}" for ln in excerpt.splitlines())
                            block += f"\n{indented}"
                    history_blocks.append(block)
                conversation_history_str = "\n".join(history_blocks)
            else:
                conversation_history_str = "  (none)"

            continuity_section = self.prompt_template['continuity_section_template'].format(
                prev_topic=prev_topic,
                prev_tone=prev_tone,
                prev_emotion=prev_emotion,
                prev_dialogue=prev_dialogue_truncated,
                time_gap_str=time_gap_str,
                prev_timestamp=prev_timestamp or "unknown",
                curr_timestamp=curr_timestamp or "unknown",
                conversation_history=conversation_history_str,
            )
        
        # Build the prompt differently for third-party conversations
        if is_third_party_conv:
            # For third-party conversations, use actual participant names
            # Determine initiator based on sender/receiver roles
            if conversation_initiator == "victim" and sender_role == "victim":
                initiator_instruction = f"The VICTIM ({sender_name})"
                initiator_name = sender_name
            elif conversation_initiator == "victim" and receiver_role == "victim":
                initiator_instruction = f"The VICTIM ({receiver_name})"
                initiator_name = receiver_name
            else:
                # Default to sender initiating
                initiator_instruction = sender_role.upper().replace("_", " ")
                initiator_name = sender_name
            
            # Use third-party template
            template_key = 'third_party_template'
            prompt = self.prompt_template[template_key].format(
                continuity_section=continuity_section,
                initiator_instruction=initiator_instruction,
                initiator_name=initiator_name,
                generation_prompt=generation_prompt,
                sender_name=sender_name,
                sender_role=sender_role,
                receiver_name=receiver_name,
                receiver_role=receiver_role,
                victim_name=victim_name,
                perpetrator_name=perpetrator_name,
                third_party_names_str=third_party_names_str,
                event_type=event_type,
                expansion_purpose=expansion_purpose if expansion_purpose else 'direct conversation',
                timestamp=timestamp,
                emotional_state=emotional_state,
                tone=tone,
                conversation_topic=conversation_topic,
                action=action,
                description=description,
                main_points_str=main_points_str,
                style_notes_str=style_notes_str,
                suggested_length=suggested_length,
                target_turns=target_turns,
                max_allowed_turns=max_allowed_turns,
                character_profiles_str=character_profiles_str
            )
        else:
            # For perpetrator-victim conversations, use standard template
            initiator_instruction = "The VICTIM" if conversation_initiator == "victim" else "The PERPETRATOR"
            initiator_name = victim_name if conversation_initiator == "victim" else perpetrator_name
            expansion_purpose_str = expansion_purpose if expansion_purpose else 'direct conversation'
            
            # Use the selected template name
            template_key = self.template_name if self.template_name in self.prompt_template else 'edited_template_FINAL'
            prompt = self.prompt_template[template_key].format(
                continuity_section=continuity_section,
                initiator_instruction=initiator_instruction,
                initiator_name=initiator_name,
                generation_prompt=generation_prompt,
                persona_language_style_str=persona_language_style_str,
                perpetrator_name=perpetrator_name,
                victim_name=victim_name,
                third_party_names_str=third_party_names_str,
                event_type=event_type,
                expansion_purpose=expansion_purpose_str,
                timestamp=timestamp,
                emotional_state=emotional_state,
                tone=tone,
                conversation_topic=conversation_topic,
                violence_language_level=violence_language_level,
                action=action,
                description=description,
                crime_type_str=crime_type_str,
                escalation_level=escalation_level,
                main_points_str=main_points_str,
                style_notes_str=style_notes_str,
                red_flags_str=red_flags_str,
                suggested_length=suggested_length,
                target_turns=target_turns,
                max_allowed_turns=max_allowed_turns,
                character_profiles_str=character_profiles_str
            )
        
        return prompt
    
    def generate_dialogue(self, chat_script: Dict[str, Any], scenario_metadata: Dict[str, Any], persona_info: Dict[str, Any] = None, previous_context: Optional[Dict[str, Any]] = None, event_index: int = 0, total_events: int = 1, all_previous_contexts: list = None) -> Optional[Dict[str, Any]]:
        """
        Generate a roleplay dialogue for a single chat script event.
        
        Args:
            chat_script: The chat script event data
            scenario_metadata: Metadata about the overall scenario
            persona_info: Optional persona information for detailed character profiles
            previous_context: Optional context from previous dialogue for continuity
            event_index: Current event index for initiation tracking
            total_events: Total number of events in scenario
            
        Returns:
            Dictionary containing the generated dialogue and metadata
        """
        event_id = chat_script.get("event_id", "unknown")
        
        # Check if this is a third-party conversation
        sender = chat_script.get("participants", {}).get("sender", {})
        receiver = chat_script.get("participants", {}).get("receiver", {})
        sender_role = sender.get("role", "unknown")
        receiver_role = receiver.get("role", "unknown")
        is_third_party_conv = (sender_role == "third_party" or receiver_role == "third_party")
        
        # Determine who should initiate this conversation
        initiator = self._determine_conversation_initiator(persona_info, event_index, total_events)
        
        if previous_context:
            print(f"  Generating dialogue for event: {event_id} (with previous context, {initiator} initiates{', third-party' if is_third_party_conv else ''})")
        else:
            print(f"  Generating dialogue for event: {event_id} ({initiator} initiates{', third-party' if is_third_party_conv else ''})")
        
        # Ensure LLM is initialized before use
        self.ensure_llm_initialized()
        
        try:
            # Create roleplay prompt with persona info, previous context, and initiator
            prompt = self._create_roleplay_prompt(chat_script, scenario_metadata, persona_info, previous_context, initiator, all_previous_contexts=all_previous_contexts)
            
            # Generate dialogue using LLM
            print(f"    Calling LLM...")
            response = self.llm_backend.generate(
                prompt=prompt,
                max_tokens=3072,
                temperature=0.7
            )
            
            # Clean response
            response = response.strip()
            
            print(f"    ✓ Generated {len(response)} characters")
            
            # Apply rule-based enforcement if enabled (splitting + slang)
            enforced_dialogue = None
            if self.enable_enforcer:
                escalation_level = chat_script.get("conversation_context", {}).get("escalation_level", 0)
                
                # Get actual participant names for enforcement
                sender_name = sender.get("name", "Unknown")
                receiver_name = receiver.get("name", "Unknown")
                
                enforced_dialogue = self.enforce_dialogue_rules(
                    response, 
                    event_id,
                    persona_info=persona_info,
                    escalation_level=escalation_level,
                    participant_names=[sender_name, receiver_name]
                )
                self.stats["enforced"] += 1
            
            # For third-party conversations, nullify perpetrator-specific attributes
            context = chat_script.get("conversation_context", {}).copy()
            if is_third_party_conv:
                # Set perpetrator-specific attributes to null
                context["escalation_level"] = None
                # Keep crime_type as None if not directly perpetrator-involved
                if sender_role == "third_party" and receiver_role == "third_party":
                    context["crime_type"] = None
            
            result = {
                "event_id": event_id,
                "timestamp": chat_script.get("timestamp"),
                "platform": chat_script.get("platform"),
                "participants": chat_script.get("participants"),
                "conversation_context": context,
                "conversation_initiator": initiator,
                "generated_dialogue": response,
                "generation_success": True,
                "generation_timestamp": datetime.now().isoformat()
            }
            
            if enforced_dialogue:
                result["enforced_dialogue"] = enforced_dialogue
            
            # Update initiation stats
            if initiator == "victim":
                self.stats["victim_initiated"] += 1
            else:
                self.stats["perpetrator_initiated"] += 1
            
            self.stats["successful"] += 1
            return result
            
        except Exception as e:
            print(f"    ✗ Failed: {e}")
            self.stats["failed"] += 1
            return {
                "event_id": event_id,
                "generation_success": False,
                "error": str(e)
            }
    
    def enforce_dialogue_rules(self, dialogue_text: str, event_id: str = "unknown", 
                              persona_info: Dict[str, Any] = None, 
                              escalation_level: int = 0,
                              participant_names: List[str] = None) -> str:
        """
        Apply rule-based transformations to dialogue without LLM.
        
        This enforcer agent:
        1. Splits long utterances based on sentence count
        2. Adds slang/abbreviations based on persona age
        3. Inserts violence placeholders for perpetrator at high escalation
        
        Args:
            dialogue_text: The dialogue text to enforce
            event_id: Event identifier for logging
            persona_info: Persona information for age-based slang
            escalation_level: Escalation level (0-4) for violence insertion
            participant_names: List of participant names in this conversation (for flexible matching)
            
        Returns:
            Enforced dialogue string
        """
        import random
        import re
        
        print(f"    Enforcing dialogue rules for event: {event_id}")
        
        # Common slang substitutions dictionary
        slang_dict = {
            "you": "u",
            "your": "ur",
            "you're": "ur",
            "are": "r",
            "before": "b4",
            "want to": "wanna",
            "going to": "gonna",
            "got to": "gotta",
            "kind of": "kinda",
            "sort of": "sorta",
            "out of": "outta",
            "lot of": "lotta",
            "because": "bc",
            "okay": "ok",
            "oh my god": "omg",
            "laughing out loud": "lol",
            "laugh out loud": "lol",
            "I don't know": "idk",
            "don't know": "dunno",
            "to be honest": "tbh",
            "not gonna lie": "ngl",
            "right now": "rn",
            "something": "smth",
            "probably": "prob",
            "definitely": "def",
            "anyway": "anyways"
        }
        
        # Violence/profanity placeholders
        violence_placeholders = ["[PROFANITY]", "[INSULT]", "[AGGRESSIVE LANGUAGE]"]
        
        # No probability-based slang rates - slang substitution is disabled in enforcer
        # Only splitting is applied
        profiles_by_name: Dict[str, Dict[str, Any]] = {}
        if persona_info:
            victim_profile = persona_info.get("victim_profile", {}) or {}
            perp_profile = persona_info.get("perpetrator_profile", {}) or {}
            if victim_profile.get("name"):
                profiles_by_name[victim_profile["name"]] = victim_profile
            if perp_profile.get("name"):
                profiles_by_name[perp_profile["name"]] = perp_profile
            for tp in (persona_info.get("third_party_actors", []) or []):
                if isinstance(tp, dict) and tp.get("name"):
                    profiles_by_name[tp["name"]] = tp
        
        # Parse dialogue into lines
        lines = dialogue_text.strip().split('\n')
        processed_lines = []
        
        for line in lines:
            # Skip empty lines or background sections
            if not line.strip() or line.strip().startswith('Background:'):
                processed_lines.append(line)
                continue
            
            # Check if this is a dialogue line (Speaker: message)
            match = re.match(r'^([^:]+):\s*(.+)$', line.strip())
            if not match:
                processed_lines.append(line)
                continue
            
            speaker = match.group(1).strip()
            message = match.group(2).strip()

            # SPLIT MULTI-SENTENCE UTTERANCES BASED ON SEMANTIC BOUNDARIES
            # (Slang substitution removed - handled by LLM based on style notes)
            sentence_endings = re.findall(r'[.!?]+', message)
            num_sentences = max(1, len(sentence_endings))
            
            # Split if multiple sentences exist
            if num_sentences >= 2:
                # Split using semantic similarity
                parts = self._split_by_semantic_similarity(message)
                
                # Create multiple lines for same speaker
                for i, part in enumerate(parts):
                    if part.strip():
                        processed_lines.append(f"{speaker}: {part.strip()}")
            else:
                # No splitting needed, keep as is
                processed_lines.append(f"{speaker}: {message}")
        
        result = '\n'.join(processed_lines)
        print(f"      ✓ Rules enforced (split utterances)")
        return result
    
    def _split_by_semantic_similarity(self, text: str, similarity_threshold: float = 0.65, min_part_sentences: int = 2) -> List[str]:
        """
        Split text into parts based on semantic similarity between sentences.
        Sentences with high similarity stay together; low similarity creates a split.
        
        Args:
            text: Input text to split
            similarity_threshold: Cosine similarity threshold (0-1). Lower = more splits, higher = fewer splits. (default: 0.65)
            min_part_sentences: Minimum sentences per part to avoid excessive splitting (default: 2)
            
        Returns:
            List of text parts split at semantic boundaries
        """
        import re
        
        # If no encoder available, fall back to simple sentence splitting
        if self.sentence_encoder is None:
            sentences = re.split(r'([.!?]+)', text)
            parts = []
            for i in range(0, len(sentences), 2):
                if i+1 < len(sentences):
                    parts.append(sentences[i] + sentences[i+1])
                elif sentences[i].strip():
                    parts.append(sentences[i])
            return [p for p in parts if p.strip()]
        
        # Split into sentences
        sentence_pattern = re.split(r'([.!?]+)', text)
        sentences = []
        for i in range(0, len(sentence_pattern), 2):
            if i+1 < len(sentence_pattern):
                sent = sentence_pattern[i] + sentence_pattern[i+1]
                if sent.strip():
                    sentences.append(sent)
            elif sentence_pattern[i].strip():
                sentences.append(sentence_pattern[i])
        
        # If only one sentence, return as is
        if len(sentences) <= 1:
            return sentences
        
        try:
            # Get embeddings for all sentences
            embeddings = self.sentence_encoder.encode([s.strip() for s in sentences])
            
            # Calculate cosine similarity between adjacent sentences
            similarities = []
            for i in range(len(embeddings) - 1):
                # Compute cosine similarity
                sim = np.dot(embeddings[i], embeddings[i+1]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1])
                )
                similarities.append(sim)
            
            # Find split points where similarity drops significantly
            # Use adaptive threshold: only split at the lowest similarity points
            if len(similarities) <= 1:
                return sentences
            
            # Calculate mean and std of similarities
            mean_sim = np.mean(similarities)
            std_sim = np.std(similarities)
            
            # Only split at points that are significantly below average
            # AND below the absolute threshold
            adaptive_threshold = min(similarity_threshold, mean_sim - 0.5 * std_sim)
            
            splits = [0]  # Always start at beginning
            for i, sim in enumerate(similarities):
                # Only create split if:
                # 1. Similarity is below threshold
                # 2. We have at least min_part_sentences since last split
                if sim < adaptive_threshold and (i + 1 - splits[-1]) >= min_part_sentences:
                    splits.append(i + 1)
            
            splits.append(len(sentences))  # Always end at the end
            
            # If we only have 2 splits (start and end), no splitting occurred
            if len(splits) == 2:
                return sentences
            
            # Group sentences based on splits
            parts = []
            for i in range(len(splits) - 1):
                start_idx = splits[i]
                end_idx = splits[i + 1]
                part = ''.join(sentences[start_idx:end_idx]).strip()
                if part:
                    parts.append(part)
            
            # If we ended up with too many single-sentence parts, just return original
            single_sentence_parts = sum(1 for p in parts if len(re.findall(r'[.!?]+', p)) == 1)
            if single_sentence_parts > len(parts) * 0.6:  # More than 60% are single-sentence
                return sentences
            
            return parts if parts else sentences
            
        except Exception as e:
            # Fallback to simple splitting on error
            print(f"      Warning: Semantic splitting failed ({e}), using simple split")
            return sentences
    
    def _apply_slang_substitution(self, text: str, slang_dict: Dict[str, str], rate: float) -> str:
        """
        Apply slang substitutions to text based on rate.
        
        Args:
            text: Input text
            slang_dict: Dictionary of word -> slang mappings
            rate: Probability of substitution (0.0-1.0)
            
        Returns:
            Text with some words replaced with slang
        """
        import random
        import re
        
        # Sort by length (longest first) to handle multi-word phrases
        sorted_phrases = sorted(slang_dict.items(), key=lambda x: len(x[0].split()), reverse=True)
        
        result = text
        for original, slang in sorted_phrases:
            # Case-insensitive matching with word boundaries
            pattern = re.compile(r'\b' + re.escape(original) + r'\b', re.IGNORECASE)
            matches = list(pattern.finditer(result))
            
            # Apply substitution with probability
            for match in matches:
                if random.random() < rate:
                    # Preserve case of first character
                    matched_text = match.group()
                    if matched_text[0].isupper():
                        replacement = slang.capitalize()
                    else:
                        replacement = slang
                    result = result[:match.start()] + replacement + result[match.end():]
                    # Recalculate match positions after substitution
                    break
        
        return result
    
    def process_chat_script_file(self, input_file: Path, persona_dir: Path = None) -> Dict[str, Any]:
        """
        Process a single chat script file and generate dialogues for all events.
        
        Args:
            input_file: Path to chat script JSON file
            persona_dir: Directory containing persona JSON files
            
        Returns:
            Dictionary containing all generated dialogues
        """
        print(f"\nProcessing: {input_file.name}")
        
        # Load chat script file
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        scenario_id = data.get("scenario_id", "unknown")
        chat_scripts = data.get("chat_scripts", [])
        
        print(f"  Scenario ID: {scenario_id}")
        print(f"  Total events: {len(chat_scripts)}")
        
        # Load persona information if directory provided
        persona_info = None
        if persona_dir:
            persona_info = self._load_persona_info(scenario_id, persona_dir)
            if persona_info:
                print(f"  ✓ Loaded persona information for scenario {scenario_id}")
            else:
                print(f"  ⚠ No persona info found, using generic profiles")
        
        # Extract scenario metadata
        scenario_metadata = {
            "scenario_id": scenario_id,
            "victim_name": data.get("victim_name"),
            "perpetrator_name": data.get("perpetrator_name"),
            "third_party_names": data.get("third_party_names", [])
        }
        
        # Generate dialogues for each event, passing context forward
        generated_dialogues = []
        previous_context = None
        all_previous_contexts = []  # accumulates {timestamp, topic} for every prior event
        
        for i, chat_script in enumerate(chat_scripts, 1):
            print(f"\n  [{i}/{len(chat_scripts)}]")
            self.stats["total_events"] += 1
            
            dialogue = self.generate_dialogue(
                chat_script, 
                scenario_metadata, 
                persona_info, 
                previous_context,
                event_index=i-1,
                total_events=len(chat_scripts),
                all_previous_contexts=all_previous_contexts,
            )
            if dialogue and dialogue.get("generation_success"):
                generated_dialogues.append(dialogue)
                
                # Prepare context for next dialogue
                # Use enforced > generated in priority
                dialogue_text = dialogue.get("enforced_dialogue") or dialogue.get("generated_dialogue")
                topic = self._infer_conversation_topic(
                    chat_script.get("conversation_context", {}),
                    chat_script.get("content_guidance", {}).get("red_flags", [])
                )
                
                previous_context = {
                    "dialogue": dialogue_text,
                    "topic": topic,
                    "tone": chat_script.get("conversation_context", {}).get("tone", "neutral"),
                    "emotional_state": chat_script.get("conversation_context", {}).get("emotional_state", "neutral"),
                    "timestamp": chat_script.get("timestamp"),
                }
                # Append summary entry so future events can see the full history
                excerpt = (dialogue_text or "")[-400:]
                if len(dialogue_text or "") > 400:
                    excerpt = "[\u2026] " + excerpt
                all_previous_contexts.append({
                    "timestamp": chat_script.get("timestamp") or "unknown",
                    "event_id": chat_script.get("event_id", ""),
                    "topic": topic,
                    "excerpt": excerpt,
                })
            elif dialogue:
                # Keep failed dialogue but don't use as context
                generated_dialogues.append(dialogue)
        
        # Compile output with full persona information
        output_data = {
            "scenario_id": scenario_id,
            "victim_name": data.get("victim_name"),
            "perpetrator_name": data.get("perpetrator_name"),
            "third_party_names": data.get("third_party_names", []),
            "vawg_scenarios": persona_info if persona_info else {},
            "num_events": len(chat_scripts),
            "num_dialogues_generated": len([d for d in generated_dialogues if d.get("generation_success")]),
            "dialogues": generated_dialogues,
            "metadata": {
                "source_file": str(input_file),
                "generation_timestamp": datetime.now().isoformat(),
                "model": self.model_name,
                "persona_info_included": persona_info is not None
            }
        }
        
        return output_data
    
    def enforce_existing_dialogues(self, input_file: Path, persona_dir: Path = None) -> Dict[str, Any]:
        """
        Apply enforcer and aggressive agent to dialogues from an existing generated dialogues file.
        
        Args:
            input_file: Path to existing dialogues JSON file
            persona_dir: Directory containing persona JSON files for age-based slang
            
        Returns:
            Updated dialogue data with enforced and aggressive versions
        """
        print(f"\nEnforcing rules on dialogues from: {input_file.name}")
        
        # Load existing dialogues
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        scenario_id = data.get("scenario_id", "unknown")
        dialogues = data.get("dialogues", [])
        
        print(f"  Scenario ID: {scenario_id}")
        print(f"  Total dialogues: {len(dialogues)}")
        
        # Load persona information if directory provided
        persona_info = None
        if persona_dir:
            persona_info = self._load_persona_info(scenario_id, persona_dir)
        
        # Extract names from data
        perpetrator_name = data.get("perpetrator_name", "Perpetrator")
        victim_name = data.get("victim_name", "Victim")
        
        # Enforce each dialogue
        enforced_count = 0
        for i, dialogue in enumerate(dialogues, 1):
            event_id = dialogue.get("event_id", "unknown")
            
            # Skip if already failed generation
            if not dialogue.get("generation_success"):
                print(f"  [{i}/{len(dialogues)}] Skipping failed generation: {event_id}")
                continue
            
            if "enforced_dialogue" in dialogue:
                print(f"  [{i}/{len(dialogues)}] Skipping already enforced: {event_id}")
                continue
            
            print(f"\n  [{i}/{len(dialogues)}]")
            
            # Use best available version as source
            source_dialogue = dialogue.get("format_fixed_dialogue") or dialogue.get("rewritten_dialogue") or dialogue.get("generated_dialogue")
            
            if not source_dialogue:
                print(f"    ✗ No dialogue content found for: {event_id}")
                continue
            
            # Get escalation level from context
            escalation_level = dialogue.get("conversation_context", {}).get("escalation_level", 0)
            
            # Apply enforcer
            enforced = self.enforce_dialogue_rules(
                source_dialogue,
                event_id,
                persona_info=persona_info,
                escalation_level=escalation_level,
                perpetrator_name=perpetrator_name,
                victim_name=victim_name
            )
            
            dialogue["enforced_dialogue"] = enforced
            dialogue["enforcement_timestamp"] = datetime.now().isoformat()
            enforced_count += 1
            self.stats["enforced"] += 1
        
        # Update metadata
        data["num_dialogues_enforced"] = enforced_count
        if "metadata" not in data:
            data["metadata"] = {}
        data["metadata"]["enforcement_timestamp"] = datetime.now().isoformat()
        
        print(f"\n  Enforced: {enforced_count}/{len(dialogues)}")
        
        return data
    
    def save_dialogues(self, output_data: Dict[str, Any], run_dir: Path) -> Path:
        """Save generated dialogues to file."""
        scenario_id = output_data["scenario_id"]
        output_file = run_dir / f"{scenario_id}_dialogues.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n  Saved to: {output_file}")
        return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Generate roleplay dialogues from chat scripts using LLM"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        help="Path to chat script JSON file (e.g., 130194.json)"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Directory containing chat script JSON files to process"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./roleplay_output",
        help="Directory to save generated dialogues (default: ./roleplay_output)"
    )
    parser.add_argument(
        "--model",
        choices=["gpt52", "deepseek_v4_flash", "gemini_flash_lite", "qwen35_35b", "llama3_8b"],
        default="gpt52",
        help="Preset model to use (default: gpt52)"
    )
    parser.add_argument(
        "--model-config",
        type=str,
        help="Custom YAML config path. Overrides --model when provided."
    )
    parser.add_argument(
        "--persona-dir",
        type=str,
        help="Directory containing persona JSON files (default: ../Generation_Results/persona_processed_full_gpt5)"
    )
    parser.add_argument(
        "--min-turns",
        type=int,
        default=2,
        help="Minimum number of speaking turns per character (default: 2)"
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=10,
        help="Maximum number of speaking turns per character (default: 10)"
    )
    parser.add_argument(
        "--adaptive-length",
        action="store_true",
        default=True,
        help="Adjust dialogue length based on escalation level (default: True)"
    )
    parser.add_argument(
        "--no-adaptive-length",
        action="store_false",
        dest="adaptive_length",
        help="Use fixed min/max turns regardless of context"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        help="Number of GPUs for tensor parallelism (vLLM only, overrides config)"
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="1,2,3,4",
        help="Comma-separated GPU IDs to use (default: 1,2,3,4, avoiding GPU 0)"
    )
    parser.add_argument(
        "--no-enforcer",
        action="store_true",
        help="Disable rule-based enforcement (split, slang) (enforcer is enabled by default)"
    )
    parser.add_argument(
        "--enforce-existing",
        action="store_true",
        help="Apply enforcer to existing dialogue files (no LLM needed)"
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        help="Path to prompt template YAML file (default: ./roleplay_prompt_template.yaml)"
    )
    parser.add_argument(
        "--template-name",
        type=str,
        default="edited_template_FINAL",
        help="Template key in the prompt YAML (default: edited_template_FINAL, the paper setting)"
    )
    parser.add_argument(
        "--bare-prompt",
        action="store_true",
        default=False,
        help="Omit role-specific style notes and perpetrator red flags from the prompt; use only the base generation prompt"
    )
    
    args = parser.parse_args()
    
    # Handle enforce-existing mode (enforcer on existing files, no LLM)
    if args.enforce_existing:
        args.no_enforcer = False  # Ensure enforcer is enabled
        print("Running in enforce-existing mode (applying enforcer to existing files, no LLM needed)")
    
    # Validate input arguments
    if not args.input_file and not args.input_dir:
        print("Error: Either --input-file or --input-dir must be provided")
        sys.exit(1)
    
    if args.input_file and args.input_dir:
        print("Error: Cannot specify both --input-file and --input-dir")
        sys.exit(1)
    
    # Determine input files to process
    input_files = []
    if args.input_file:
        input_file = Path(args.input_file)
        if not input_file.exists():
            print(f"Error: Input file not found: {input_file}")
            sys.exit(1)
        input_files = [input_file]
    else:
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            print(f"Error: Input directory not found: {input_dir}")
            sys.exit(1)
        _REPORT_NAMES = {"generation_report.json", "expansion_report.json", "event_graph_report.json", "postprocessing_report.json"}
        input_files = sorted(f for f in input_dir.glob("*.json") if f.name not in _REPORT_NAMES)
        if not input_files:
            print(f"Error: No JSON files found in {input_dir}")
            sys.exit(1)
        print(f"Found {len(input_files)} JSON files to process")
    
    # Determine config path
    base_dir = Path(__file__).parent.parent
    preset_configs = {
        "gpt52": base_dir / "RAG_kb_construct" / "configs" / "gpt52_openai.yaml",
        "deepseek_v4_flash": base_dir / "RAG_kb_construct" / "configs" / "deepseek_v4_flash_openai.yaml",
        "gemini_flash_lite": base_dir / "RAG_kb_construct" / "configs" / "gemini_flash_lite_openai.yaml",
        "qwen35_35b": base_dir / "RAG_kb_construct" / "configs" / "qwen35_35b_vllm.yaml",
        "llama3_8b": base_dir / "RAG_kb_construct" / "configs" / "llama3_8b_vllm.yaml",
    }
    
    if args.model_config:
        config_path = Path(args.model_config)
        if not config_path.exists():
            print(f"Error: Config file not found: {config_path}")
            sys.exit(1)
        model_display = f"custom ({config_path.name})"
    else:
        config_path = preset_configs[args.model]
        if not config_path.exists():
            print(f"Error: Preset config not found: {config_path}")
            sys.exit(1)
        model_display = args.model
    
    # Create output directory
    run_dir = Path(args.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up persona directory
    if args.persona_dir:
        persona_dir = Path(args.persona_dir)
    else:
        persona_dir = base_dir / "Generation_Results" / "persona_processed_full_gpt5"
    
    if not persona_dir.exists():
        print(f"Warning: Persona directory not found: {persona_dir}")
        print("Will use generic character profiles")
        persona_dir = None
    else:
        print(f"Using persona directory: {persona_dir}")
    
    # Set GPU visibility before initializing LLM
    import os
    gpu_list = args.gpus
    num_gpus_available = len(gpu_list.split(','))
    
    # Validate tensor parallel size matches available GPUs
    if args.tensor_parallel_size:
        if args.tensor_parallel_size > num_gpus_available:
            print(f"Error: --tensor-parallel-size ({args.tensor_parallel_size}) exceeds available GPUs ({num_gpus_available})")
            print(f"Available GPUs: {gpu_list}")
            sys.exit(1)
        # If using fewer GPUs than available, adjust CUDA_VISIBLE_DEVICES
        gpus = gpu_list.split(',')[:args.tensor_parallel_size]
        gpu_list = ','.join(gpus)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    print(f"Using GPUs: {gpu_list} ({len(gpu_list.split(','))} GPUs)")
    
    print("=" * 80)
    print("Roleplay Dialogue Generation")
    print("=" * 80)
    if args.input_file:
        print(f"Input file: {input_files[0]}")
    else:
        print(f"Input directory: {args.input_dir}")
        print(f"Files to process: {len(input_files)}")
    print(f"Output directory: {run_dir}")
    print(f"Model: {model_display}")
    print(f"Config: {config_path}")
    print()
    
    # Determine if LLM is needed
    needs_llm = not (args.enforce_existing)
    if not needs_llm:
        print("Skipping LLM initialization (enforcer-only mode on existing files)")
        print()
    
    # Initialize generator with tensor parallel override if specified
    prompt_template_path = Path(args.prompt_template) if args.prompt_template else None
    generator = RoleplayDialogueGenerator(
        config_path=config_path,
        output_dir=run_dir,
        min_turns=args.min_turns,
        max_turns=args.max_turns,
        adaptive_length=args.adaptive_length,
        enable_enforcer=not args.no_enforcer,
        prompt_template_path=prompt_template_path,
        template_name=args.template_name,
        bare_prompt=args.bare_prompt
    )
    
    # Set needs_llm flag
    generator.needs_llm = needs_llm
    
    # Set tensor parallel override BEFORE initialization
    if args.tensor_parallel_size:
        generator.tensor_parallel_override = args.tensor_parallel_size
    
    # Process all files
    output_files = []
    for i, input_file in enumerate(input_files, 1):
        if len(input_files) > 1:
            print(f"\n{'='*80}")
            print(f"Processing file {i}/{len(input_files)}: {input_file.name}")
            print(f"{'='*80}")
        
        # Process based on mode
        if args.enforce_existing:
            # Enforce existing dialogues (no LLM)
            output_data = generator.enforce_existing_dialogues(input_file, persona_dir)
        else:
            # Generate new dialogues (with optional inline enforcing)
            output_data = generator.process_chat_script_file(input_file, persona_dir)
        
        # Save results
        output_file = generator.save_dialogues(output_data, run_dir)
        output_files.append(output_file)
    
    # Print summary
    print("\n" + "=" * 80)
    if args.enforce_existing:
        print("Enforcement Complete")
    else:
        print("Generation Complete")
    print("=" * 80)
    print(f"Files processed: {len(input_files)}")
    
    if not args.enforce_existing:
        print(f"Total events: {generator.stats['total_events']}")
        print(f"Successful: {generator.stats['successful']}")
        print(f"Failed: {generator.stats['failed']}")
        print(f"Success rate: {generator.stats['successful']/max(1, generator.stats['total_events'])*100:.1f}%")
        
        # Conversation initiation stats
        total_initiated = generator.stats['victim_initiated'] + generator.stats['perpetrator_initiated']
        if total_initiated > 0:
            victim_ratio = generator.stats['victim_initiated'] / total_initiated
            print(f"\nConversation initiation:")
            print(f"  Victim initiated: {generator.stats['victim_initiated']} ({victim_ratio*100:.1f}%)")
            print(f"  Perpetrator initiated: {generator.stats['perpetrator_initiated']} ({(1-victim_ratio)*100:.1f}%)")
    
    # Enforcer stats
    if not args.no_enforcer:
        print(f"Dialogues enforced: {generator.stats['enforced']}")
    
    print(f"\nOutput directory: {run_dir}")
    if len(output_files) <= 10:
        print("\nGenerated files:")
        for f in output_files:
            print(f"  - {f.name}")
    else:
        print(f"\nGenerated {len(output_files)} dialogue files")


if __name__ == "__main__":
    main()
