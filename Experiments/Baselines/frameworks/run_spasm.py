#!/usr/bin/env python3
"""
SPASM-adapted framework baseline
================================
Adaptation of SPASM (Luo & Laban, Findings of ACL 2026) on ConVAWG's shared
raw inputs. Preserved components: structured personas (schema sampling →
plausibility validation → natural-language crafting), two-agent simulation,
Egocentric Context Projection (ECP — dialogue history stored
perspective-agnostic and deterministically projected into the current
speaker's SELF/PARTNER view before every turn), and termination detection.

Adaptation: the victim agent's structured persona is built from the supplied
seed; the perpetrator (counterpart) persona is generated from the shared raw
inputs only. NO future events, NO explicit escalation trajectory. Sessions
stop via the SPASM termination agent or a hard maximum-turn cap.

Usage
-----
  # Server first: ../start_vllm_baselines.sh qwen3.5-35b   (port 8003)
  python run_spasm.py --persona-ids 65637 --min-dialogues 2 --max-dialogues 2 --max-turns 8
  python run_spasm.py --num-personas 10 --seed 42 --auto-judge
  python run_spasm.py --dry-run

Cost note: per-turn generation + periodic termination checks — ~450-700 LLM
calls/persona at the default 20-30 session budget x <=20 turns.
"""

from __future__ import annotations

import logging
import random
import zlib
from typing import Any, Dict, List, Optional, Tuple

from common.backends import BackendPool, JSONParseError
from common.io_utils import build_output, month_assignments, sample_num_dialogues, synthetic_timeline
from common.runner import build_shared_parser, resolve_backend_cfg, run_framework
from common.shared_cache import evidence_block
from common.taxonomy import normalize_vawg_type, research_system_prompt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("spasm")

FRAMEWORK = "spasm"

# ── Persona schema dimension values (SPASM: schema sampling) ────────────
EMOTIONS = (
    "anxious", "fearful", "sad", "angry", "frustrated", "confused",
    "hopeful", "calm", "ashamed", "lonely", "overwhelmed", "defensive",
)
INTENSITIES = ("mild", "moderate", "intense")
BEHAVIOUR_LEVELS = ("low", "medium", "high")
BEHAVIOUR_DIMS = ("expressiveness", "disclosure", "politeness", "assertiveness")
CHANNELS = (
    "SMS text messages", "WhatsApp chat", "Instagram DMs",
    "Facebook Messenger", "email",
)
TIMES_OF_DAY = ("morning", "afternoon", "evening", "late night")
CONTACT_GAPS = (
    "minutes since you last spoke", "a few hours since you last spoke",
    "since yesterday", "several days since you last spoke",
)


class SPASMGenerator:
    """SPASM-adapted generation: structured personas → ECP two-agent simulation."""

    def __init__(self, pool: BackendPool, args: Any) -> None:
        self.pool = pool
        self.gen_temperature = float(pool.cfg.get("temperature", 0.9))
        self.control_temperature = float(args.control_temperature)
        self.min_sessions = int(args.min_dialogues)
        self.max_sessions = int(args.max_dialogues)
        self.min_turns = int(args.min_turns)
        self.max_turns = int(args.max_turns)
        self.check_every = max(1, int(args.termination_check_every))
        self.schema_seed = int(args.schema_seed)
        self.model_label = f"{args.model}:{pool.cfg.get('model_name', '')}"
        self.system_prompt = research_system_prompt(
            "a persona-driven dialogue simulation assistant following the SPASM framework"
        )
        self.slim_system_prompt = research_system_prompt(
            "a persona-driven dialogue simulation agent (SPASM two-agent simulation)",
            include_taxonomy=False,
        )

    def params(self) -> Dict[str, Any]:
        return {
            "min_sessions": self.min_sessions,
            "max_sessions": self.max_sessions,
            "min_turns": self.min_turns,
            "max_turns": self.max_turns,
            "termination_check_every": self.check_every,
            "schema_seed": self.schema_seed,
            "gen_temperature": self.gen_temperature,
            "control_temperature": self.control_temperature,
        }

    # ── Structured personas: schema sampling → validation → crafting ────

    def _rng(self, *parts: Any) -> random.Random:
        key = ":".join(str(p) for p in parts)
        return random.Random(zlib.crc32(key.encode("utf-8")) ^ self.schema_seed)

    def sample_schema(self, role: str, shared_input: Dict[str, Any]) -> Dict[str, Any]:
        rng = self._rng(shared_input["persona_id"], role, "schema")
        schema: Dict[str, Any] = {
            "role": role,
            "emotional_state": {
                "emotion": rng.choice(EMOTIONS),
                "intensity": rng.choice(INTENSITIES),
            },
            "behaviour_pattern": {dim: rng.choice(BEHAVIOUR_LEVELS) for dim in BEHAVIOUR_DIMS},
        }
        if role == "victim":
            schema["seed_text"] = shared_input["persona_text"]
        else:
            schema["age"] = rng.randint(25, 60)
        return schema

    def _validation_prompt(self, schema: Dict[str, Any], grounding: str) -> str:
        import json as _json
        return (
            "SPASM plausibility validation. Check whether this sampled structured "
            "persona schema is internally consistent and plausible given the "
            "grounding material.\n\n"
            f"SCHEMA:\n{_json.dumps(schema, indent=2)}\n\n"
            f"GROUNDING:\n{grounding}\n\n"
            "If any field is implausible or contradicts the grounding, fix it "
            "(change as little as possible).\n"
            'Return ONLY JSON: {"plausible": <true|false>, "issues": ["<issue>", ...], '
            '"fixed_schema": {<the schema, corrected if needed>}}'
        )

    def validate_schema(self, schema: Dict[str, Any], grounding: str) -> Dict[str, Any]:
        data = self.pool.chat_json(
            self.system_prompt, self._validation_prompt(schema, grounding),
            temperature=self.control_temperature, max_tokens=1024, label="schema-validation",
        )
        if not data.get("plausible", True) and isinstance(data.get("fixed_schema"), dict):
            fixed = data["fixed_schema"]
            fixed.setdefault("role", schema["role"])
            return fixed
        return schema

    def _crafting_prompt(self, schema: Dict[str, Any], shared_input: Dict[str, Any]) -> str:
        import json as _json
        role = schema.get("role", "victim")
        if role == "victim":
            grounding = (
                f"VICTIM PERSONA SEED (stay faithful to it; invent only a first "
                f"name and light details consistent with it):\n"
                f"{shared_input['persona_text']}"
            )
            extra_fields = '"vulnerabilities": ["<string>", ...]'
        else:
            grounding = (
                "The perpetrator counterpart persona must be generated only from "
                "the victim seed and this DHR case evidence:\n"
                f"VICTIM SEED: {shared_input['persona_text']}\n\n"
                f"{evidence_block(shared_input)}"
            )
            extra_fields = '"relationship_to_victim": "<e.g. intimate_partner / ex_partner / family>"'
        return (
            "SPASM persona crafting. Convert the validated structured persona "
            "schema into a natural-language persona profile for a dialogue agent.\n\n"
            f"SCHEMA:\n{_json.dumps(schema, indent=2)}\n\n{grounding}\n\n"
            "Return ONLY JSON:\n"
            "{\n"
            '  "name": "<first name>", "age": <int>, "occupation": "<string>",\n'
            f"  {extra_fields},\n"
            '  "profile": "<4-6 sentence persona profile: who they are, their '
            "situation, how they communicate (reflect the behaviour pattern and "
            'emotional state from the schema)>"\n'
            "}"
        )

    def craft_persona(self, schema: Dict[str, Any], shared_input: Dict[str, Any]) -> Dict[str, Any]:
        crafted = self.pool.chat_json(
            self.system_prompt, self._crafting_prompt(schema, shared_input),
            temperature=self.control_temperature, max_tokens=1024, label="persona-crafting",
        )
        crafted["schema"] = schema
        return crafted

    # ── Egocentric Context Projection (deterministic, no LLM) ───────────

    @staticmethod
    def project_ecp(
        history: List[Dict[str, str]],
        self_persona: Dict[str, Any],
        partner_persona: Dict[str, Any],
        session_ctx: Dict[str, Any],
        self_role: str,
    ) -> str:
        """Project the perspective-agnostic history into the speaker's egocentric view."""
        schema = self_persona.get("schema", {})
        behaviour = schema.get("behaviour_pattern", {})
        emotion = session_ctx.get(f"{self_role}_emotion", {})
        relationship = (
            partner_persona.get("relationship_to_victim")
            or self_persona.get("relationship_to_victim")
            or "close contact"
        )
        lines = [
            f"YOUR PERSONA (SELF — {self_persona.get('name', 'you')}):",
            self_persona.get("profile", ""),
            "Behaviour pattern: " + ", ".join(f"{k} {v}" for k, v in behaviour.items()),
            f"Current emotional state: {emotion.get('emotion', 'neutral')} "
            f"({emotion.get('intensity', 'moderate')})",
            "",
            f"YOUR CONVERSATION PARTNER (PARTNER — {partner_persona.get('name', 'them')}): "
            f"your {relationship}. You know them only through your relationship and "
            "what they say in this conversation.",
            "",
            f"SCENE: you are messaging via {session_ctx['channel']}, {session_ctx['time_of_day']}; "
            f"{session_ctx['contact_gap']}.",
            "",
            f"CONVERSATION SO FAR (SELF = you, {self_persona.get('name', '')}; "
            f"PARTNER = {partner_persona.get('name', '')}):",
        ]
        if history:
            for msg in history:
                tag = "SELF" if msg["role"] == self_role else "PARTNER"
                lines.append(f"[{tag}] {msg['text']}")
        else:
            lines.append("(no messages yet — you send the first message)")
        return "\n".join(lines)

    # ── Turn generation and termination detection ───────────────────────

    def generate_turn(self, projection: str, speaker_name: str) -> str:
        prompt = (
            f"{projection}\n\n"
            f"Write ONLY the next chat message that SELF ({speaker_name}) sends — "
            "natural online chat style, stay fully in persona, react to the "
            "PARTNER's last message if there is one. No speaker label, no "
            "quotation marks, no explanation."
        )
        text = self.pool.chat_text(
            self.slim_system_prompt, prompt,
            temperature=self.gen_temperature, max_tokens=512,
        ).strip().strip('"')
        if text.lower().startswith(speaker_name.lower() + ":"):
            text = text[len(speaker_name) + 1:].strip()
        return text

    def _termination_prompt(self, history: List[Dict[str, str]],
                            names: Dict[str, str], ended: bool = False) -> str:
        window = history[-8:]
        transcript = "\n".join(f"{names[m['role']]}: {m['text']}" for m in window)
        question = (
            "The conversation has ended. "
            if ended else
            "Decide whether this conversation has reached a natural end point "
            "(goodbye, conversation dropped, resolution, refusal to continue, or "
            "one party going silent). "
        )
        return (
            "SPASM termination detection. Recent conversation:\n\n"
            f"{transcript}\n\n"
            f"{question}"
            "Also name the single taxonomy code (from the system prompt) that "
            "best describes what this conversation depicts.\n"
            'Return ONLY JSON: {"terminate": <true|false>, "reason": "<short>", '
            '"vawg_type": "<taxonomy code>"}'
        )

    def check_termination(self, history: List[Dict[str, str]], names: Dict[str, str],
                          ended: bool = False) -> Dict[str, Any]:
        return self.pool.chat_json(
            self.system_prompt, self._termination_prompt(history, names, ended=ended),
            temperature=self.control_temperature, max_tokens=512, label="termination",
        )

    # ── Session simulation ───────────────────────────────────────────────

    def _sample_session_context(self, persona_id: str, session_idx: int) -> Dict[str, Any]:
        rng = self._rng(persona_id, session_idx, "session")
        return {
            "channel": rng.choice(CHANNELS),
            "time_of_day": rng.choice(TIMES_OF_DAY),
            "contact_gap": rng.choice(CONTACT_GAPS),
            "victim_emotion": {"emotion": rng.choice(EMOTIONS), "intensity": rng.choice(INTENSITIES)},
            "perpetrator_emotion": {"emotion": rng.choice(EMOTIONS), "intensity": rng.choice(INTENSITIES)},
            "starter": rng.choice(("victim", "perpetrator")),
        }

    def run_session(
        self,
        persona_id: str,
        session_idx: int,
        victim: Dict[str, Any],
        perpetrator: Dict[str, Any],
    ) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        session_ctx = self._sample_session_context(persona_id, session_idx)
        personas = {"victim": victim, "perpetrator": perpetrator}
        names = {"victim": victim.get("name", "Victim"),
                 "perpetrator": perpetrator.get("name", "Perpetrator")}
        history: List[Dict[str, str]] = []
        current = session_ctx["starter"]
        terminated_by = "cap"
        termination: Dict[str, Any] = {}

        for turn in range(1, self.max_turns + 1):
            partner = "perpetrator" if current == "victim" else "victim"
            projection = self.project_ecp(
                history, personas[current], personas[partner], session_ctx, current,
            )
            text = self.generate_turn(projection, names[current])
            if not text:
                logger.warning("  session %d: empty turn %d — ending", session_idx, turn)
                break
            history.append({"role": current, "text": text})
            current = partner

            if turn >= self.min_turns and (turn - self.min_turns) % self.check_every == 0:
                try:
                    termination = self.check_termination(history, names)
                except JSONParseError:
                    termination = {}
                if termination.get("terminate"):
                    terminated_by = "agent"
                    break

        if not termination.get("vawg_type") and history:
            try:
                termination = self.check_termination(history, names, ended=True)
            except JSONParseError:
                termination = {}

        meta = {
            "session": session_idx,
            "session_context": session_ctx,
            "terminated_by": terminated_by,
            "termination_reason": termination.get("reason", ""),
            "num_turns": len(history),
            "vawg_type": normalize_vawg_type(str(termination.get("vawg_type", ""))),
        }
        return history, meta

    # ── Per-persona orchestration ────────────────────────────────────────

    def build_personas(self, shared_input: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        victim_schema = self.validate_schema(
            self.sample_schema("victim", shared_input),
            f"Victim persona seed: {shared_input['persona_text']}",
        )
        victim = self.craft_persona(victim_schema, shared_input)
        perp_schema = self.validate_schema(
            self.sample_schema("perpetrator", shared_input),
            evidence_block(shared_input),
        )
        perpetrator = self.craft_persona(perp_schema, shared_input)
        return victim, perpetrator

    def run_persona(self, shared_input: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        pid = shared_input["persona_id"]
        victim, perpetrator = self.build_personas(shared_input)
        names = {"victim": victim.get("name", "Victim"),
                 "perpetrator": perpetrator.get("name", "Perpetrator")}
        logger.info("  %s: personas crafted (%s / %s)", pid,
                    names["victim"], names["perpetrator"])

        target = sample_num_dialogues(pid, self.min_sessions, self.max_sessions)
        logger.info("  %s: session budget %d", pid, target)
        assignments = month_assignments(target)
        dialogues: List[Dict[str, Any]] = []
        session_metas: List[Dict[str, Any]] = []
        for s in range(1, target + 1):
            history, meta = self.run_session(pid, s, victim, perpetrator)
            session_metas.append(meta)
            if not history:
                continue
            dialogues.append({
                "dialogue_id": f"dialogue_{s}",
                "vawg_type": meta["vawg_type"],
                "timeline": synthetic_timeline(s, assignments),
                "speakers": {"victim": names["victim"],
                             "perpetrator": names["perpetrator"], "others": []},
                "turns": [{"speaker": names[m["role"]], "text": m["text"]} for m in history],
            })

        scenario_summary = {
            "victim_profile": {
                "name": victim.get("name"), "age": victim.get("age"),
                "occupation": victim.get("occupation"),
                "vulnerabilities": victim.get("vulnerabilities", []),
            },
            "perpetrator_profile": {
                "name": perpetrator.get("name"), "age": perpetrator.get("age"),
                "occupation": perpetrator.get("occupation"),
                "relationship_to_victim": perpetrator.get("relationship_to_victim"),
            },
            "scenario_description": (
                f"{victim.get('profile', shared_input['persona_text'])} "
                f"{perpetrator.get('profile', '')}"
            ).strip(),
            "timeline_start": "2024-01-01",
        }
        record = build_output(
            persona_id=pid,
            persona_text=shared_input["persona_text"],
            framework=FRAMEWORK,
            model_label=self.model_label,
            scenario_summary=scenario_summary,
            dialogues=dialogues,
            framework_meta_stub={
                "framework": FRAMEWORK,
                "params": self.params(),
                "num_sessions": len(dialogues),
                "num_agent_terminated": sum(
                    1 for m in session_metas if m["terminated_by"] == "agent"
                ),
                "timeline_note": "synthetic, non-semantic (no escalation scheduling)",
            },
        )
        framework_meta = {
            "framework": FRAMEWORK,
            "persona_id": pid,
            "params": self.params(),
            "victim_persona": victim,
            "perpetrator_persona": perpetrator,
            "sessions": session_metas,
        }
        return record, framework_meta

    def dry_run(self, shared_input: Dict[str, Any]) -> None:
        n = sample_num_dialogues(
            shared_input["persona_id"], self.min_sessions, self.max_sessions,
        )
        t = self.max_turns
        checks = max(0, (t - self.min_turns) // self.check_every + 1)
        est = 4 + n * (t + checks + 1)
        print("=" * 70)
        print(f"[DRY RUN] SPASM — persona {shared_input['persona_id']}")
        print(f"Planned LLM calls: 2 validations + 2 craftings + {n} x "
              f"(<= {t} turns + <= {checks} termination checks + <=1 label) ≈ {est} per persona "
              f"(budget sampled from [{self.min_sessions}, {self.max_sessions}])")
        print("=" * 70)
        victim_schema = self.sample_schema("victim", shared_input)
        perp_schema = self.sample_schema("perpetrator", shared_input)
        print("\n--- SYSTEM PROMPT (control calls) ---\n")
        print(self.system_prompt)
        print("\n--- S1: SCHEMA VALIDATION (victim) ---\n")
        print(self._validation_prompt(
            victim_schema, f"Victim persona seed: {shared_input['persona_text']}",
        ))
        print("\n--- S4: PERSONA CRAFTING (perpetrator) ---\n")
        print(self._crafting_prompt(perp_schema, shared_input))
        session_ctx = self._sample_session_context(shared_input["persona_id"], 1)
        placeholder_victim = {"name": "<victim>", "profile": "<from S2>", "schema": victim_schema}
        placeholder_perp = {"name": "<perpetrator>", "profile": "<from S4>",
                            "relationship_to_victim": "<from S4>", "schema": perp_schema}
        print("\n--- T1: TURN GENERATION (ECP projection, victim speaking) ---\n")
        print(self.project_ecp([], placeholder_victim, placeholder_perp, session_ctx, "victim"))
        print("\n--- T2: TERMINATION DETECTION ---\n")
        print(self._termination_prompt(
            [{"role": "victim", "text": "<message>"}],
            {"victim": "<victim>", "perpetrator": "<perpetrator>"},
        ))


def main() -> None:
    parser = build_shared_parser("SPASM", __doc__)
    parser.add_argument("--min-turns", type=int, default=6,
                        help="Turns before termination checks begin (default: 6)")
    parser.add_argument("--max-turns", type=int, default=20,
                        help="Hard per-session turn cap (default: 20)")
    parser.add_argument("--termination-check-every", type=int, default=2,
                        help="Run the termination agent every N turns past --min-turns (default: 2)")
    parser.add_argument("--schema-seed", type=int, default=0,
                        help="Seed for deterministic persona/session schema sampling (default: 0)")
    args = parser.parse_args()
    cfg, _ = resolve_backend_cfg(args)
    generator = SPASMGenerator(BackendPool(cfg), args)
    run_framework(FRAMEWORK, args, generator)


if __name__ == "__main__":
    main()
