#!/usr/bin/env python3
"""
LLM-adapted SynDG framework baseline
====================================
Adaptation of SynDG (ACL 2023) on ConVAWG's shared raw inputs. Original
sequence preserved: knowledge pieces → dialogue flow → incremental turn-by-turn
realization (one knowledge piece per turn) → flow-level + utterance-level
filtering with a fixed evaluation rubric.

This is explicitly the **LLM-adapted** variant: the original SynDG uses a
trained T5 realization model and trained flow/utterance filter models; here
the backbone LLM performs realization and rubric scoring instead.

Usage
-----
  # Server first: ../start_vllm_baselines.sh qwen3.5-35b   (port 8003)
  python run_syndg.py --persona-ids 65637 --min-dialogues 2 --max-dialogues 2 --turns-per-dialogue 6
  python run_syndg.py --num-personas 10 --seed 42 --auto-judge
  python run_syndg.py --dry-run

Cost note: incremental realization is ~(turns+3) LLM calls per dialogue —
~380-580 calls/persona at the default 20-30 dialogue budget x 16 turns.
"""

from __future__ import annotations

import logging
from statistics import mean
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
logger = logging.getLogger("syndg")

FRAMEWORK = "llm_adapted_syndg"

FLOW_RUBRIC = ("coherence", "knowledge_grounding", "naturalness", "speaker_consistency")
UTTERANCE_RUBRIC = ("naturalness", "knowledge_faithfulness", "coherence")


class LLMAdaptedSynDGGenerator:
    """LLM-adapted SynDG: knowledge → flow → incremental realization → filters."""

    def __init__(self, pool: BackendPool, args: Any) -> None:
        self.pool = pool
        self.gen_temperature = float(pool.cfg.get("temperature", 0.9))
        self.control_temperature = float(args.control_temperature)
        self.min_dialogues = int(args.min_dialogues)
        self.max_dialogues = int(args.max_dialogues)
        self.turns_per_dialogue = int(args.turns_per_dialogue)
        self.knowledge_pool_size = int(args.knowledge_pool_size)
        self.filter_threshold = float(args.filter_threshold)
        self.max_regen = int(args.max_regen)
        self.per_turn_filter = bool(args.per_turn_filter)
        self.model_label = f"{args.model}:{pool.cfg.get('model_name', '')}"
        self.system_prompt = research_system_prompt(
            "a grounded-dialogue generation assistant following the SynDG framework "
            "(LLM-adapted variant)"
        )

    def params(self) -> Dict[str, Any]:
        return {
            "min_dialogues": self.min_dialogues,
            "max_dialogues": self.max_dialogues,
            "turns_per_dialogue": self.turns_per_dialogue,
            "knowledge_pool_size": self.knowledge_pool_size,
            "filter_threshold": self.filter_threshold,
            "max_regen": self.max_regen,
            "per_turn_filter": self.per_turn_filter,
            "gen_temperature": self.gen_temperature,
            "control_temperature": self.control_temperature,
        }

    # ── Step 1: knowledge construction ──────────────────────────────────

    def _knowledge_prompt(self, shared_input: Dict[str, Any]) -> str:
        return (
            f"VICTIM PERSONA (ID {shared_input['persona_id']}):\n"
            f"{shared_input['persona_text']}\n\n"
            f"{evidence_block(shared_input)}\n\n"
            "SynDG step 1 — knowledge construction. Convert the raw inputs above "
            "into atomic knowledge pieces for grounded dialogue generation. Each "
            "piece is ONE short declarative statement (max ~25 words) that a single "
            "chat message could convey or enact. Generalise case evidence into "
            "behavioural patterns — do not copy real names or places from the "
            "cases. Cover four sources: (a) victim persona facts, (b) the "
            "perpetrator counterpart implied by the evidence, (c) relationship and "
            "abuse dynamics seen in the evidence, (d) elements of the relevant "
            "crime-type definitions.\n"
            "Also produce the two dialogue personas: the VICTIM faithful to the "
            "seed text (invent only a first name), and the PERPETRATOR generated "
            "only from the raw inputs above.\n\n"
            "Return ONLY JSON:\n"
            "{\n"
            '  "victim_profile": {"name": "<first name>", "age": <int or null>, '
            '"occupation": "<string>", "vulnerabilities": ["<string>", ...]},\n'
            '  "perpetrator_profile": {"name": "<first name>", "age": <int or null>, '
            '"occupation": "<string>", "relationship_to_victim": "<string>", '
            '"description": "<2-4 sentences>"},\n'
            f'  "knowledge_pieces": [{{"id": "k1", "source": "persona|evidence|definition", '
            f'"text": "<statement>"}}, ...]  (at most {self.knowledge_pool_size} pieces)\n'
            "}"
        )

    def extract_knowledge(self, shared_input: Dict[str, Any]) -> Dict[str, Any]:
        data = self.pool.chat_json(
            self.system_prompt, self._knowledge_prompt(shared_input),
            temperature=self.control_temperature, max_tokens=4096, label="knowledge",
        )
        pieces = [
            {"id": str(p.get("id", f"k{i}")), "source": str(p.get("source", "")),
             "text": str(p.get("text", ""))}
            for i, p in enumerate(data.get("knowledge_pieces", []), 1)
            if isinstance(p, dict) and p.get("text")
        ][: self.knowledge_pool_size]
        data["knowledge_pieces"] = pieces
        return data

    # ── Step 2: dialogue flow construction ──────────────────────────────

    def _pieces_block(self, pieces: List[Dict[str, str]]) -> str:
        return "\n".join(f"{p['id']} [{p['source']}]: {p['text']}" for p in pieces)

    def _flow_prompt(self, knowledge: Dict[str, Any], index: int) -> str:
        victim = knowledge["victim_profile"].get("name", "Victim")
        perp = knowledge["perpetrator_profile"].get("name", "Perpetrator")
        return (
            f"DIALOGUE PERSONAS:\n- {victim} (victim)\n- {perp} (perpetrator): "
            f"{knowledge['perpetrator_profile'].get('description', '')}\n\n"
            f"KNOWLEDGE PIECES:\n{self._pieces_block(knowledge['knowledge_pieces'])}\n\n"
            f"SynDG step 2 — dialogue flow. Plan online chat conversation #{index} "
            f"of exactly {self.turns_per_dialogue} turns between {victim} and "
            f"{perp}. A dialogue flow is an ordered series of knowledge pieces: "
            "each turn realises exactly one knowledge piece (use each piece at "
            "most once within this flow). Alternate speakers naturally. Different "
            "conversations should use different subsets of pieces. Choose the one "
            "taxonomy code this conversation depicts.\n\n"
            "Return ONLY JSON:\n"
            '{"vawg_type": "<taxonomy code>", '
            '"flow": [{"turn": 1, "speaker": "<name>", "knowledge_id": "<id>"}, ...]}'
        )

    def _flow_filter_prompt(self, knowledge: Dict[str, Any],
                            flow: List[Dict[str, Any]]) -> str:
        piece_by_id = {p["id"]: p["text"] for p in knowledge["knowledge_pieces"]}
        lines = [
            f"turn {step.get('turn')}: {step.get('speaker')} — "
            f"\"{piece_by_id.get(step.get('knowledge_id'), '?')}\""
            for step in flow
        ]
        return (
            "SynDG flow-level filtering. Evaluate this planned dialogue flow with "
            "the fixed rubric below.\n\nFLOW:\n" + "\n".join(lines) + "\n\n"
            "Rubric — score each criterion 1-5:\n"
            "- coherence: the sequence forms a logically progressing conversation\n"
            "- knowledge_grounding: each step's knowledge piece plausibly belongs here\n"
            "- naturalness: the plan reads like a plausible online chat\n"
            "- speaker_consistency: each piece is assigned to the right speaker\n\n"
            'Return ONLY JSON: {"scores": {"coherence": <1-5>, "knowledge_grounding": <1-5>, '
            '"naturalness": <1-5>, "speaker_consistency": <1-5>}, "comment": "<short>"}'
        )

    def build_flow(self, knowledge: Dict[str, Any], index: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Flow construction + flow-level filter with bounded regeneration."""
        valid_ids = {p["id"] for p in knowledge["knowledge_pieces"]}
        names = {knowledge["victim_profile"].get("name", "Victim"),
                 knowledge["perpetrator_profile"].get("name", "Perpetrator")}
        attempts: List[Dict[str, Any]] = []
        best: Optional[Dict[str, Any]] = None
        for attempt in range(1, self.max_regen + 2):
            data = self.pool.chat_json(
                self.system_prompt, self._flow_prompt(knowledge, index),
                temperature=self.control_temperature, max_tokens=2048,
                label=f"flow-{index}",
            )
            flow = [
                step for step in data.get("flow", [])
                if isinstance(step, dict)
                and step.get("knowledge_id") in valid_ids
                and step.get("speaker") in names
            ]
            if not flow:
                logger.warning("  flow %d attempt %d: empty/invalid flow", index, attempt)
                continue
            scores_data = self.pool.chat_json(
                self.system_prompt, self._flow_filter_prompt(knowledge, flow),
                temperature=self.control_temperature, max_tokens=1024,
                label=f"flow-filter-{index}",
            )
            scores = {
                c: float(scores_data.get("scores", {}).get(c, 0)) for c in FLOW_RUBRIC
            }
            overall = mean(scores.values()) if scores else 0.0
            candidate = {
                "vawg_type": normalize_vawg_type(str(data.get("vawg_type", ""))),
                "flow": flow,
                "flow_scores": scores,
                "flow_overall": overall,
                "flow_attempt": attempt,
            }
            attempts.append({"attempt": attempt, "scores": scores, "overall": overall})
            if best is None or overall > best["flow_overall"]:
                best = candidate
            if overall >= self.filter_threshold:
                return candidate, attempts
        if best is None:
            raise JSONParseError(f"No valid flow for dialogue {index}")
        best["forced_accept"] = True
        return best, attempts

    # ── Step 3: incremental realization ─────────────────────────────────

    def _realize_prompt(self, knowledge: Dict[str, Any], history: List[Dict[str, str]],
                        speaker: str, piece_text: str) -> str:
        victim = knowledge["victim_profile"].get("name", "Victim")
        perp = knowledge["perpetrator_profile"].get("name", "Perpetrator")
        role = "victim" if speaker == victim else "perpetrator"
        history_block = (
            "\n".join(f"{t['speaker']}: {t['text']}" for t in history)
            if history else "(conversation start)"
        )
        return (
            f"PERSONAS:\n- {victim} (victim): {knowledge.get('persona_text', '')}\n"
            f"- {perp} (perpetrator): "
            f"{knowledge['perpetrator_profile'].get('description', '')}\n\n"
            f"CONVERSATION SO FAR:\n{history_block}\n\n"
            f"SynDG step 3 — incremental realization. Write the next turn.\n"
            f"Next speaker: {speaker} ({role}). This message must convey exactly "
            f"this knowledge piece: \"{piece_text}\"\n\n"
            f"Write ONLY the chat message text that {speaker} sends next — natural "
            "online chat style, consistent with the conversation so far. No "
            "speaker label, no quotation marks, no explanation."
        )

    def realize_turn(self, knowledge: Dict[str, Any], history: List[Dict[str, str]],
                     speaker: str, piece_text: str) -> str:
        raw = self.pool.chat_text(
            self.system_prompt,
            self._realize_prompt(knowledge, history, speaker, piece_text),
            temperature=self.gen_temperature, max_tokens=512,
        ).strip().strip('"')
        if raw.lower().startswith(speaker.lower() + ":"):
            raw = raw[len(speaker) + 1:].strip()
        return raw

    # ── Step 4: utterance-level filter ──────────────────────────────────

    def _utterance_filter_prompt(self, turns: List[Dict[str, str]],
                                 piece_texts: List[str]) -> str:
        lines = [
            f"{i}. {t['speaker']}: \"{t['text']}\"  [should convey: {piece_texts[i - 1]}]"
            for i, t in enumerate(turns, 1)
        ]
        return (
            "SynDG utterance-level filtering. Score EVERY turn of this dialogue "
            "with the fixed rubric below.\n\nDIALOGUE:\n" + "\n".join(lines) + "\n\n"
            "Rubric — score each turn 1-5 on:\n"
            "- naturalness: reads like a real chat message\n"
            "- knowledge_faithfulness: conveys the knowledge piece shown for that turn\n"
            "- coherence: fits the conversation context\n\n"
            'Return ONLY JSON: {"turns": [{"turn": 1, "naturalness": <1-5>, '
            '"knowledge_faithfulness": <1-5>, "coherence": <1-5>}, ...]} '
            "(one entry per turn, in order)"
        )

    def filter_utterances(self, turns: List[Dict[str, str]],
                          piece_texts: List[str], index: int) -> List[Dict[str, Any]]:
        data = self.pool.chat_json(
            self.system_prompt, self._utterance_filter_prompt(turns, piece_texts),
            temperature=self.control_temperature, max_tokens=2048,
            label=f"utterance-filter-{index}",
        )
        scored: List[Dict[str, Any]] = []
        for entry in data.get("turns", []):
            if not isinstance(entry, dict):
                continue
            scores = {c: float(entry.get(c, 0)) for c in UTTERANCE_RUBRIC}
            scored.append({
                "turn": int(entry.get("turn", len(scored) + 1)),
                "scores": scores,
                "avg": mean(scores.values()),
            })
        return scored

    # ── Per-dialogue orchestration ───────────────────────────────────────

    def generate_dialogue(self, knowledge: Dict[str, Any],
                          index: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        plan, flow_attempts = self.build_flow(knowledge, index)
        piece_by_id = {p["id"]: p["text"] for p in knowledge["knowledge_pieces"]}

        turns: List[Dict[str, str]] = []
        piece_texts: List[str] = []
        inline_regens = 0
        for step in plan["flow"]:
            piece_text = piece_by_id[step["knowledge_id"]]
            text = self.realize_turn(knowledge, turns, step["speaker"], piece_text)
            if self.per_turn_filter:
                for _ in range(self.max_regen):
                    scored = self.filter_utterances(
                        turns + [{"speaker": step["speaker"], "text": text}],
                        piece_texts + [piece_text], index,
                    )
                    if not scored or scored[-1]["avg"] >= self.filter_threshold:
                        break
                    inline_regens += 1
                    text = self.realize_turn(knowledge, turns, step["speaker"], piece_text)
            turns.append({"speaker": step["speaker"], "text": text})
            piece_texts.append(piece_text)

        score_rounds: List[List[Dict[str, Any]]] = []
        batch_regens = 0
        if not self.per_turn_filter and turns:
            for _ in range(self.max_regen + 1):
                scored = self.filter_utterances(turns, piece_texts, index)
                score_rounds.append(scored)
                failing = [s["turn"] for s in scored if s["avg"] < self.filter_threshold]
                if not failing or batch_regens >= self.max_regen:
                    break
                for turn_no in failing:
                    idx = turn_no - 1
                    if 0 <= idx < len(turns):
                        turns[idx]["text"] = self.realize_turn(
                            knowledge, turns[:idx], turns[idx]["speaker"], piece_texts[idx],
                        )
                batch_regens += 1

        dialogue = {"vawg_type": plan["vawg_type"], "turns": turns}
        trace = {
            "index": index,
            "flow": plan["flow"],
            "flow_attempts": flow_attempts,
            "flow_scores": plan["flow_scores"],
            "flow_overall": plan["flow_overall"],
            "forced_accept": plan.get("forced_accept", False),
            "utterance_score_rounds": score_rounds,
            "inline_regens": inline_regens,
            "batch_regens": batch_regens,
        }
        return dialogue, trace

    # ── Per-persona orchestration ────────────────────────────────────────

    def run_persona(self, shared_input: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        pid = shared_input["persona_id"]
        knowledge = self.extract_knowledge(shared_input)
        knowledge["persona_text"] = shared_input["persona_text"]
        logger.info("  %s: %d knowledge pieces", pid, len(knowledge["knowledge_pieces"]))
        victim = knowledge.get("victim_profile", {})
        perp = knowledge.get("perpetrator_profile", {})

        target = sample_num_dialogues(pid, self.min_dialogues, self.max_dialogues)
        logger.info("  %s: dialogue budget %d", pid, target)
        assignments = month_assignments(target)
        dialogues: List[Dict[str, Any]] = []
        traces: List[Dict[str, Any]] = []
        for i in range(1, target + 1):
            try:
                dialogue, trace = self.generate_dialogue(knowledge, i)
            except JSONParseError as exc:
                logger.error("  %s dialogue %d skipped: %s", pid, i, exc)
                traces.append({"index": i, "parse_error": str(exc)})
                continue
            if not dialogue["turns"]:
                traces.append({"index": i, "parse_error": "no turns realized"})
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
            traces.append(trace)

        grounding = "; ".join(
            p["text"] for p in knowledge["knowledge_pieces"][:6]
        )
        scenario_summary = {
            "victim_profile": victim,
            "perpetrator_profile": {k: v for k, v in perp.items() if k != "description"},
            "scenario_description": (
                f"{victim.get('name', 'The victim')} — {shared_input['persona_text']} "
                f"{perp.get('description', '')} Key grounding facts: {grounding}."
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
                "num_knowledge_pieces": len(knowledge["knowledge_pieces"]),
                "num_dialogues": len(dialogues),
                "num_forced_accepts": sum(1 for t in traces if t.get("forced_accept")),
                "timeline_note": "synthetic, non-semantic (no escalation scheduling)",
            },
        )
        framework_meta = {
            "framework": FRAMEWORK,
            "persona_id": pid,
            "params": self.params(),
            "knowledge_pieces": knowledge["knowledge_pieces"],
            "profiles": {"victim_profile": victim, "perpetrator_profile": perp},
            "dialogues": traces,
        }
        return record, framework_meta

    def dry_run(self, shared_input: Dict[str, Any]) -> None:
        n = sample_num_dialogues(
            shared_input["persona_id"], self.min_dialogues, self.max_dialogues,
        )
        t = self.turns_per_dialogue
        est = 1 + n * (3 + t)
        print("=" * 70)
        print(f"[DRY RUN] LLM-adapted SynDG — persona {shared_input['persona_id']}")
        print(f"Planned LLM calls: 1 knowledge + {n} x (1 flow + 1 flow-filter + "
              f"{t} realizations + >=1 utterance-filter) ≈ {est} per persona "
              f"(budget sampled from [{self.min_dialogues}, {self.max_dialogues}])")
        print("=" * 70)
        print("\n--- SYSTEM PROMPT (all calls) ---\n")
        print(self.system_prompt)
        print("\n--- K1: KNOWLEDGE EXTRACTION ---\n")
        print(self._knowledge_prompt(shared_input))
        placeholder = {
            "persona_text": shared_input["persona_text"],
            "victim_profile": {"name": "<victim>"},
            "perpetrator_profile": {"name": "<perpetrator>", "description": "<from K1>"},
            "knowledge_pieces": [{"id": "k1", "source": "persona", "text": "<piece>"}],
        }
        print("\n--- F1: FLOW CONSTRUCTION ---\n")
        print(self._flow_prompt(placeholder, 1))
        print("\n--- Q1: FLOW-LEVEL FILTER ---\n")
        print(self._flow_filter_prompt(
            placeholder, [{"turn": 1, "speaker": "<victim>", "knowledge_id": "k1"}],
        ))
        print("\n--- R1: INCREMENTAL REALIZATION ---\n")
        print(self._realize_prompt(placeholder, [], "<victim>", "<piece>"))
        print("\n--- Q2: UTTERANCE-LEVEL FILTER ---\n")
        print(self._utterance_filter_prompt(
            [{"speaker": "<victim>", "text": "<realized turn>"}], ["<piece>"],
        ))


def main() -> None:
    parser = build_shared_parser("LLM-adapted SynDG", __doc__)
    parser.add_argument("--turns-per-dialogue", type=int, default=16,
                        help="Turns per dialogue = knowledge pieces per flow (default: 16)")
    parser.add_argument("--knowledge-pool-size", type=int, default=40,
                        help="Max atomic knowledge pieces per persona (default: 40)")
    parser.add_argument("--filter-threshold", type=float, default=3.0,
                        help="Rubric pass threshold on the 1-5 scale (default: 3.0)")
    parser.add_argument("--max-regen", type=int, default=2,
                        help="Max regenerations per flow / per filter pass (default: 2)")
    parser.add_argument("--per-turn-filter", action="store_true",
                        help="Filter each utterance right after realization (slower, "
                             "closer to the original per-utterance filtering)")
    args = parser.parse_args()
    cfg, _ = resolve_backend_cfg(args)
    generator = LLMAdaptedSynDGGenerator(BackendPool(cfg), args)
    run_framework(FRAMEWORK, args, generator)


if __name__ == "__main__":
    main()
