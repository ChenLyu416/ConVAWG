#!/usr/bin/env python3
"""
DiaSynth-adapted framework baseline
===================================
Re-implementation of DiaSynth (Suresh et al., Findings of NAACL 2025) on
ConVAWG's shared raw inputs. Original sequence preserved:

  topic → sub-topic generation → persona generation
        → conversation-characteristic reasoning (CoT) → dialogue generation

Adaptation: the topic is "online VAWG interactions involving the victim
persona" grounded in the full CPS taxonomy + retrieved DHR evidence; the
victim persona is the supplied seed (kept verbatim); the perpetrator persona
is generated from the shared raw inputs only. One dialogue per sub-topic.
NO event plan, NO escalation schedule, NO toxicity instructions.

Usage
-----
  # Server first: ../start_vllm_baselines.sh qwen3.5-35b   (port 8003)
  python run_diasynth.py --persona-ids 65637 --min-dialogues 2 --max-dialogues 2   # smoke test
  python run_diasynth.py --num-personas 10 --seed 42 --auto-judge
  python run_diasynth.py --dry-run                               # prompts only
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from common.backends import BackendPool, JSONParseError, extract_json
from common.io_utils import build_output, month_assignments, sample_num_dialogues, synthetic_timeline
from common.runner import build_shared_parser, resolve_backend_cfg, run_framework
from common.shared_cache import evidence_block
from common.taxonomy import normalize_vawg_type, research_system_prompt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("diasynth")

FRAMEWORK = "diasynth"

# The 10 conversation characteristics from the DiaSynth paper (Table 11).
CHARACTERISTICS = (
    "1. Age and Gender of both personas\n"
    "2. Familiarity Level (how familiar the speakers are with each other)\n"
    "3. Emotional States of the speakers\n"
    "4. Formality Level\n"
    "5. Duration of the Conversation\n"
    "6. Communication Medium\n"
    "7. Topic of the Conversation\n"
    "8. Location of the Conversation\n"
    "9. Agreement or Disagreement on the viewpoints of the conversation topic\n"
    "10. Incorporation of natural human dialogue characteristics (fillers, pauses, slang)"
)


class DiaSynthGenerator:
    """DiaSynth-adapted generation: subtopics → personas → CoT → dialogues."""

    def __init__(self, pool: BackendPool, args: Any) -> None:
        self.pool = pool
        self.gen_temperature = float(pool.cfg.get("temperature", 0.9))
        self.control_temperature = float(args.control_temperature)
        self.min_dialogues = int(args.min_dialogues)
        self.max_dialogues = int(args.max_dialogues)
        self.dedup_threshold = float(args.subtopic_dedup_threshold)
        self.model_label = f"{args.model}:{pool.cfg.get('model_name', '')}"
        self.system_prompt = research_system_prompt(
            "a synthetic-dialogue generation assistant following the DiaSynth framework"
        )
        self._encoder = None  # lazy SentenceTransformer for subtopic dedup

    def params(self) -> Dict[str, Any]:
        return {
            "min_dialogues": self.min_dialogues,
            "max_dialogues": self.max_dialogues,
            "gen_temperature": self.gen_temperature,
            "control_temperature": self.control_temperature,
            "subtopic_dedup_threshold": self.dedup_threshold,
        }

    # ── Stage 1: sub-topic generation ───────────────────────────────────

    def _subtopic_prompt(self, shared_input: Dict[str, Any], n: int,
                         existing: Optional[List[str]] = None) -> str:
        persona = shared_input["persona_text"]
        prompt = (
            f"TOPIC: online interactions between a victim and a perpetrator in a "
            f"VAWG situation involving the victim persona below.\n\n"
            f"VICTIM PERSONA (ID {shared_input['persona_id']}):\n{persona}\n\n"
            f"{evidence_block(shared_input)}\n\n"
            f"Generate exactly {n} sub-topics of the topic above. Each sub-topic is a "
            "short phrase naming one specific facet or situation of online abusive "
            "interaction that is plausible for this victim, grounded in the taxonomy "
            "and the case evidence. Make the sub-topics diverse: cover different "
            "crime types, relationship moments, and communication situations.\n"
        )
        if existing:
            prompt += (
                "\nThese sub-topics already exist — the new ones must be clearly "
                f"distinct from all of them:\n- " + "\n- ".join(existing) + "\n"
            )
        prompt += '\nReturn ONLY JSON: {"subtopics": ["<sub-topic>", ...]}'
        return prompt

    def _dedup(self, subtopics: List[str]) -> List[str]:
        """Embedding-based similarity dedup (paper's similarity check)."""
        unique = list(dict.fromkeys(s.strip() for s in subtopics if s.strip()))
        if len(unique) < 2:
            return unique
        try:
            if self._encoder is None:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer("all-MiniLM-L6-v2")
            import numpy as np
            emb = self._encoder.encode(unique, normalize_embeddings=True)
            kept: List[int] = []
            for i in range(len(unique)):
                if all(float(np.dot(emb[i], emb[j])) < self.dedup_threshold for j in kept):
                    kept.append(i)
            return [unique[i] for i in kept]
        except ImportError:
            logger.warning("sentence-transformers unavailable — exact-string dedup only")
            return unique

    def generate_subtopics(self, shared_input: Dict[str, Any], n: int) -> List[str]:
        data = self.pool.chat_json(
            self.system_prompt, self._subtopic_prompt(shared_input, n),
            temperature=self.control_temperature, max_tokens=2048, label="subtopics",
        )
        subtopics = self._dedup([str(s) for s in data.get("subtopics", [])])
        for _ in range(2):  # bounded top-up if dedup shrank the set
            if len(subtopics) >= n:
                break
            missing = n - len(subtopics)
            data = self.pool.chat_json(
                self.system_prompt,
                self._subtopic_prompt(shared_input, missing, existing=subtopics),
                temperature=self.control_temperature, max_tokens=1024, label="subtopics-topup",
            )
            subtopics = self._dedup(subtopics + [str(s) for s in data.get("subtopics", [])])
        return subtopics[:n]

    # ── Stage 2: persona generation ─────────────────────────────────────

    def _persona_prompt(self, shared_input: Dict[str, Any]) -> str:
        return (
            f"VICTIM PERSONA (ID {shared_input['persona_id']}):\n"
            f"{shared_input['persona_text']}\n\n"
            f"{evidence_block(shared_input)}\n\n"
            "Following the DiaSynth persona-generation step, produce the two "
            "conversation personas. The VICTIM persona must stay faithful to the "
            "seed text above (invent only a first name and fill light details "
            "consistent with it). The PERPETRATOR persona must be generated only "
            "from the victim persona and the DHR case evidence above.\n\n"
            "Return ONLY JSON:\n"
            "{\n"
            '  "victim_profile": {"name": "<first name>", "age": <int or null>, '
            '"occupation": "<string>", "vulnerabilities": ["<string>", ...]},\n'
            '  "perpetrator_profile": {"name": "<first name>", "age": <int or null>, '
            '"occupation": "<string>", "relationship_to_victim": "<e.g. intimate_partner / '
            'ex_partner / family / acquaintance>", "description": "<2-4 sentence persona '
            'description of who they are and how they communicate>"}\n'
            "}"
        )

    def generate_personas(self, shared_input: Dict[str, Any]) -> Dict[str, Any]:
        return self.pool.chat_json(
            self.system_prompt, self._persona_prompt(shared_input),
            temperature=self.control_temperature, max_tokens=1024, label="personas",
        )

    # ── Stage 3: CoT characteristics + dialogue generation ──────────────

    def _dialogue_prompt(self, shared_input: Dict[str, Any], subtopic: str,
                         profiles: Dict[str, Any]) -> str:
        victim = profiles.get("victim_profile", {})
        perp = profiles.get("perpetrator_profile", {})
        return (
            "You are an expert dialog generator. Generate a dialogue between people "
            "with the following personas.\n\n"
            f"persona1 (VICTIM, name {victim.get('name', 'the victim')}): "
            f"{shared_input['persona_text']}\n"
            f"persona2 (PERPETRATOR, name {perp.get('name', 'the perpetrator')}): "
            f"{perp.get('description', '')} "
            f"(relationship to victim: {perp.get('relationship_to_victim', 'unspecified')})\n\n"
            f"SUB-TOPIC of this conversation: {subtopic}\n\n"
            "Characteristics of the dialogue — to be understood before generating "
            f"the dialogue:\n{CHARACTERISTICS}\n\n"
            "Chain of Thought Reasoning:\n"
            "- Before generating the dialogue, reason about the values for each "
            "characteristic listed above.\n"
            "- Explain in detail how each characteristic will influence a "
            "hypothetical dialogue between persons with those characteristics and "
            "personas.\n"
            "- Ensure that the reasoning considers the interactions between "
            "different characteristics (e.g., how familiarity and emotional state "
            "might interact).\n"
            "- This reasoning and explanation must be included between <cot> and "
            "</cot> tags. Do not skip this step.\n"
            "- The dialogue generated should be based on the explanation provided.\n\n"
            "After </cot>, return ONLY this JSON (no markdown):\n"
            '{"vawg_type": "<one taxonomy code from the system prompt>", '
            '"turns": [{"speaker": "<name>", "text": "<message>"}, ...]}\n'
            "The dialogue must have 12-20 turns, read like natural online chat "
            "messages, and involve only the two personas."
        )

    def generate_dialogue(self, shared_input: Dict[str, Any], subtopic: str,
                          profiles: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        """Returns (dialogue dict with vawg_type/turns, cot reasoning text)."""
        prompt = self._dialogue_prompt(shared_input, subtopic, profiles)
        max_retries = int(self.pool.cfg.get("max_retries", 3))
        retry_delay = float(self.pool.cfg.get("retry_delay", 10.0))
        raw = ""
        for attempt in range(1, max_retries + 1):
            raw = self.pool.chat_text(
                self.system_prompt, prompt,
                temperature=self.gen_temperature, max_tokens=4096,
            )
            cot_match = re.search(r"<cot>(.*?)</cot>", raw, flags=re.DOTALL)
            cot = cot_match.group(1).strip() if cot_match else ""
            json_part = raw.split("</cot>", 1)[1] if "</cot>" in raw else raw
            try:
                data = extract_json(json_part)
                turns = [
                    {"speaker": str(t.get("speaker", "")), "text": str(t.get("text", ""))}
                    for t in data.get("turns", [])
                    if isinstance(t, dict) and t.get("text")
                ]
                if turns:
                    return (
                        {"vawg_type": normalize_vawg_type(str(data.get("vawg_type", ""))),
                         "turns": turns},
                        cot,
                    )
                raise ValueError("empty turns")
            except (ValueError, json.JSONDecodeError) as exc:
                logger.warning("Dialogue parse failed (attempt %d/%d): %s",
                               attempt, max_retries, exc)
                if attempt < max_retries:
                    time.sleep(retry_delay)
        logger.warning("Raw response (first 1000 chars):\n%s", raw[:1000])
        raise JSONParseError("Could not parse dialogue after retries")

    # ── Per-persona orchestration ────────────────────────────────────────

    def run_persona(self, shared_input: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        pid = shared_input["persona_id"]
        target = sample_num_dialogues(pid, self.min_dialogues, self.max_dialogues)
        subtopics = self.generate_subtopics(shared_input, target)
        logger.info("  %s: %d subtopics (budget %d)", pid, len(subtopics), target)
        profiles = self.generate_personas(shared_input)
        victim = profiles.get("victim_profile", {})
        perp = profiles.get("perpetrator_profile", {})

        assignments = month_assignments(len(subtopics))
        dialogues: List[Dict[str, Any]] = []
        dialogue_traces: List[Dict[str, Any]] = []
        for i, subtopic in enumerate(subtopics, 1):
            try:
                dialogue, cot = self.generate_dialogue(shared_input, subtopic, profiles)
            except JSONParseError as exc:
                logger.error("  %s dialogue %d skipped: %s", pid, i, exc)
                dialogue_traces.append({"index": i, "subtopic": subtopic, "parse_error": str(exc)})
                continue
            dialogues.append({
                "dialogue_id": f"dialogue_{i}",
                "vawg_type": dialogue["vawg_type"],
                "timeline": synthetic_timeline(i, assignments),
                "speakers": {
                    "victim": victim.get("name", "Victim"),
                    "perpetrator": perp.get("name", "Perpetrator"),
                    "others": [],
                },
                "turns": dialogue["turns"],
            })
            dialogue_traces.append({"index": i, "subtopic": subtopic, "cot": cot})

        scenario_summary = {
            "victim_profile": victim,
            "perpetrator_profile": {k: v for k, v in perp.items() if k != "description"},
            "scenario_description": (
                f"{victim.get('name', 'The victim')} — {shared_input['persona_text']} "
                f"{perp.get('description', '')} Their online conversations cover: "
                + "; ".join(subtopics[:8]) + "."
            ),
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
                "num_subtopics": len(subtopics),
                "num_dialogues": len(dialogues),
                "timeline_note": "synthetic, non-semantic (no escalation scheduling)",
            },
        )
        framework_meta = {
            "framework": FRAMEWORK,
            "persona_id": pid,
            "params": self.params(),
            "subtopics": subtopics,
            "profiles": profiles,
            "dialogues": dialogue_traces,
        }
        return record, framework_meta

    def dry_run(self, shared_input: Dict[str, Any]) -> None:
        n = sample_num_dialogues(
            shared_input["persona_id"], self.min_dialogues, self.max_dialogues,
        )
        print("=" * 70)
        print(f"[DRY RUN] DiaSynth — persona {shared_input['persona_id']}")
        print(f"Planned LLM calls: 1 subtopic + <=2 top-up + 1 personas + {n} dialogues "
              f"≈ {n + 3} per persona (budget sampled from "
              f"[{self.min_dialogues}, {self.max_dialogues}])")
        print("=" * 70)
        print("\n--- SYSTEM PROMPT (all calls) ---\n")
        print(self.system_prompt)
        print("\n--- P1: SUBTOPIC GENERATION ---\n")
        print(self._subtopic_prompt(shared_input, n))
        print("\n--- P2: PERSONA GENERATION ---\n")
        print(self._persona_prompt(shared_input))
        print("\n--- P3: DIALOGUE WITH CoT (example subtopic) ---\n")
        print(self._dialogue_prompt(
            shared_input, "<sub-topic from P1>",
            {"victim_profile": {"name": "<victim>"},
             "perpetrator_profile": {"name": "<perpetrator>", "description": "<from P2>",
                                     "relationship_to_victim": "<from P2>"}},
        ))


def main() -> None:
    parser = build_shared_parser("DiaSynth", __doc__)
    parser.add_argument("--subtopic-dedup-threshold", type=float, default=0.85,
                        help="Cosine-similarity threshold for subtopic dedup (default: 0.85)")
    args = parser.parse_args()
    cfg, _ = resolve_backend_cfg(args)
    generator = DiaSynthGenerator(BackendPool(cfg), args)
    run_framework(FRAMEWORK, args, generator)


if __name__ == "__main__":
    main()
