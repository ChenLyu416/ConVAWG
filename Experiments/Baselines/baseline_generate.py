#!/usr/bin/env python3
"""
Baseline VAWG Dialogue Generator
==================================
Given a PersonaHub persona, directly prompt an LLM to write a realistic
multi-turn, multi-speaker VAWG dialogue in a single call.
No event graph, no escalation levels, no pipeline structure.

Supported models
----------------
  gpt5.2             → OpenAI API    (model id: gpt-5.2)
  deepseek-v4-flash  → DeepSeek API  (model id: deepseek-v4-flash)
  gemini-flash-lite  → Gemini API    (model id: gemini-3.1-flash-lite)
  qwen3.5-35b        → vLLM server   (model id: Qwen/Qwen3.5-35B-A3B)
  llama3.1-70b-awq   → vLLM server   (model id: hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4)

Usage
-----
  python baseline_generate.py --model gpt5.2 --num-personas 10
  python baseline_generate.py --model qwen3.5-35b --persona-ids 65637 115911 \
      --base-url http://localhost:8003/v1
  python baseline_generate.py --model llama3.1-70b-awq --num-personas 5 \
      --base-url http://localhost:8001/v1 --output-dir results/llama3.1-70b-awq
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Make LLM_judge importable regardless of working directory
_REPO_ROOT_EARLY = Path(__file__).resolve().parents[2]
_JUDGE_DIR = _REPO_ROOT_EARLY / "LLM_judge"
if str(_JUDGE_DIR) not in sys.path:
    sys.path.insert(0, str(_JUDGE_DIR))

from llm_backend import build_backend, LLMBackend  # noqa: E402

# ---------------------------------------------------------------------------
# Lazy judge imports (only loaded when --auto-judge is used)
# ---------------------------------------------------------------------------
def _import_judge():
    """Import judge module lazily so the script works without judge deps installed."""
    try:
        import yaml
        from judge import DialogueJudge, aggregate_results
        return yaml, DialogueJudge, aggregate_results
    except ImportError as exc:
        logger.error("Cannot import LLM_judge: %s — is PyYAML installed and LLM_judge present?", exc)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
PERSONA_CSV = REPO_ROOT / "persona_generate" / "VAWG_sampled_200_personas.csv"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "results"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("baseline")

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# Keys match build_backend() config format from LLM_judge/llm_backend.py
MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "gpt5.2": {
        "backend": "openai",
        "model_name": "gpt-5.2",
        "base_url": "https://api.openai.com/v1",
        "temperature": 0.9,
        "max_tokens": 32000,
    },
    "llama3.1-70b-awq": {
        # AWQ-quantized LLaMA-3.1-70B-Instruct, as reported in the paper.
        "backend": "vllm",
        "model_name": "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
        "base_url": "http://localhost:8001/v1",
        "temperature": 0.9,
        "max_tokens": 20000,
    },
    "gemini-flash-lite": {
        "backend": "gemini",
        "model_name": "gemini-3.1-flash-lite",
        "temperature": 0.9,
        "max_tokens": 32000,
    },
    "qwen3.5-35b": {
        # Served natively by vLLM >= 0.17 (qwen3_5_moe arch). Start with:
        #   ./start_vllm_baselines.sh qwen3.5-35b   (port 8003)
        "backend": "vllm",
        "model_name": "Qwen/Qwen3.5-35B-A3B",
        "base_url": "http://localhost:8003/v1",
        "temperature": 0.9,
        "max_tokens": 32000,
    },
    "deepseek-v4-flash": {
        "backend": "deepseek",
        "model_name": "deepseek-v4-flash",
        "temperature": 0.9,
        "max_tokens": 32000,
    },
}

DEFAULT_JUDGE_CONFIG = str(_JUDGE_DIR / "configs" / "judge_config.yaml")

# ---------------------------------------------------------------------------
# Judge adapter
# ---------------------------------------------------------------------------

def to_judge_format(result: dict) -> dict:
    """
    Wrap a baseline output dict into the schema expected by DialogueJudge.

    Handles both multi-dialogue outputs (result["dialogues"] list) and the
    legacy single-dialogue format (result["dialogue"] flat list).
    """
    pid     = result["persona_id"]
    summary = result.get("scenario_summary", {})
    vp      = summary.get("victim_profile", {})
    pp      = summary.get("perpetrator_profile", {})

    vawg_scenarios = {
        "persona_id":   pid,
        "persona_text": result.get("persona_text", ""),
        "crime_types":  summary.get("crime_types", [result.get("vawg_type", "")]),
        "scenario_description": summary.get("scenario_description", ""),
        "victim_profile": {
            "name":            vp.get("name", ""),
            "age":             vp.get("age"),
            "occupation":      vp.get("occupation", ""),
            "vulnerabilities": vp.get("vulnerabilities", []),
            "gender":          "female",
        },
        "perpetrator_profile": {
            "name":                   pp.get("name", ""),
            "age":                    pp.get("age"),
            "occupation":             pp.get("occupation", ""),
            "relationship_to_victim": pp.get("relationship_to_victim", "unknown"),
        },
    }

    # Build judge events list
    events: list[dict] = []

    raw_dialogues = result.get("dialogues")  # new multi-dialogue format
    if raw_dialogues and isinstance(raw_dialogues, list) and isinstance(raw_dialogues[0], dict) and "turns" in raw_dialogues[0]:
        # New format: each entry has dialogue_id, vawg_type, speakers, turns
        for d in raw_dialogues:
            turns_str = "\n".join(
                f"{t.get('speaker', 'Speaker')}: {t.get('text', '')}"
                for t in d.get("turns", [])
            )
            tl = d.get("timeline", {})
            tl_str = ""
            if tl:
                tl_str = f"  [Month {tl.get('month', '?')}, {tl.get('date', '?')}]"
            events.append({
                "event_id": d.get("dialogue_id", f"event_{len(events)+1}"),
                "conversation_context": {
                    "crime_type":  d.get("vawg_type", ""),
                    "description": f"Baseline dialogue — VAWG type: {d.get('vawg_type', 'unspecified')}{tl_str}",
                    "timeline":    tl,
                },
                "caa_rewritten_dialogue": turns_str,
            })
    else:
        # Legacy format: flat result["dialogue"] list
        turns = result.get("dialogue", [])
        turns_str = "\n".join(
            f"{t.get('speaker', 'Speaker')}: {t.get('text', '')}"
            for t in turns
        )
        events.append({
            "event_id": "baseline_event_1",
            "conversation_context": {
                "crime_type":  result.get("vawg_type", ""),
                "description": f"Baseline dialogue — VAWG type: {result.get('vawg_type', 'unspecified')}",
            },
            "caa_rewritten_dialogue": turns_str,
        })

    return {
        "scenario_id":    result["scenario_id"],
        "vawg_scenarios": vawg_scenarios,
        "dialogues":      events,
        "_baseline":      True,
        "_model":         result.get("model"),
    }


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# Full crime-type definitions (mirrors persona_generate/prompt_templates.py)
VAWG_DEFINITIONS = """\
VAWG Crime Type Definitions
============================

Domestic abuse: Abusive behaviour by one person aged 16+ toward another aged 16+
with whom they are personally connected (intimate partners or family members).
Abuse can include physical or sexual abuse, violent or threatening behaviour,
controlling or coercive behaviour, economic abuse, and psychological or emotional
abuse. It may be a single incident or an ongoing pattern.

Controlling behaviour: Acts intended to make a person subordinate or dependent —
isolating them from support, exploiting their resources for personal gain,
depriving them of the means needed for independence or escape, regulating their
everyday behaviour.

Coercive behaviour: Repeated acts of assault, threats, humiliation, intimidation,
or other abuse used to harm, punish, or frighten a victim. Often builds gradually;
individual incidents may appear innocuous but together amount to a pattern.
Includes: monitoring activity online or via devices, controlling finances, dictating
clothing/social contacts, using children to coerce.

Stalking (FOUR pattern — Fixated, Obsessive, Unwanted, Repeated): Intrusive
behaviour causing fear, serious alarm, or distress. Examples: following or
watching the victim, sending repeated unwanted messages or gifts, boasting about
knowing the victim's location, damaging property, threats of physical or sexual harm.

Cyberstalking / digital abuse: Stalking or harassment conducted through social
media, messaging apps, email, tracking software, or smart home devices. Includes
publishing private information or images online (image-based abuse).

Harassment: Repeated behaviour on at least two occasions causing another person
alarm or distress. The perpetrator knows or ought to know their conduct amounts
to harassment.

Sexual assault: Intentional, non-consensual sexual touching. Penetration is not
required; the core elements are the absence of consent and the deliberate nature
of the act.

Economic / financial abuse: Controlling access to money, property, or employment;
running up debt in the victim's name; preventing the victim from working; taking
or controlling income or benefits.

Psychological / emotional abuse: Behaviour that causes psychological harm —
humiliation, degradation, threats, isolating the victim from friends and family,
gaslighting, exploiting mental health or past trauma.

Honour-based abuse (HBA): Violence, threats, intimidation, or coercion carried
out to protect or defend perceived family/community honour. May involve multiple
family or community members acting together.

Forced marriage: One or both parties do not consent to the marriage and pressure
or abuse (physical, emotional, financial, or threats) is used to make the marriage
happen. Also includes cases where a person lacks capacity to consent.

Trafficking / exploitation: Recruiting, transporting, or harbouring a person
through coercion, deception, or abuse of power for the purpose of sexual or
labour exploitation.

Child abuse (non-sexual): Maltreatment of a person under 18 including violence,
cruelty, neglect, or causing/allowing death or serious physical harm.

Child sexual abuse: Sexual abuse of a person under 18, including physical acts,
non-contact acts (e.g. making a child view images), grooming, and online
exploitation.\
"""


SYSTEM_PROMPT = f"""\
You are a researcher studying Violence Against Women and Girls (VAWG).
Your task is to write a realistic, naturalistic conversation that depicts \
a VAWG situation grounded in the definitions below.

{VAWG_DEFINITIONS}

The conversation should feel like real people talking or texting: \
informal, believable, and grounded in the above dynamics. \
Do not use clinical or academic language inside the dialogue turns themselves — \
reserve that framing for the metadata fields only.

Respond with VALID JSON ONLY. No markdown, no code fences, no commentary."""


def build_user_prompt(
    persona_id: str,
    persona_text: str,
    min_dialogues: int = 20,
    max_dialogues: int = 30,
) -> str:
    return f"""\
PERSONA (ID: {persona_id}):
{persona_text}

This person is the VICTIM in the scenario.

Write a scenario summary (the victim, the perpetrator, their relationship, \
and a short description of the abuse situation), then write \
{min_dialogues}–{max_dialogues} separate conversations between the characters \
(at least {min_dialogues}) spanning a 6-month period. \
Do not include graphic sexual content.

Return ONLY the following JSON (no markdown, no extra text):

{{
  "scenario_id": "{persona_id}",
  "persona_id": "{persona_id}",
  "persona_text": {json.dumps(persona_text)},
  "scenario_summary": {{
    "victim_profile": {{
      "name": "<first name>",
      "age": <int or null>,
      "occupation": "<string>",
      "vulnerabilities": ["<vulnerability1>", ...]
    }},
    "perpetrator_profile": {{
      "name": "<first name>",
      "age": <int or null>,
      "occupation": "<string>",
      "relationship_to_victim": "<e.g. intimate_partner / ex_partner / family / acquaintance>"
    }},
    "crime_types": ["<vawg_type1>", "<vawg_type2>", ...],
    "scenario_description": "<100–150 word narrative describing the abuse situation>",
    "timeline_start": "<ISO date when the 6-month period begins, e.g. 2024-01-01>"
  }},
  "dialogues": [
    {{
      "dialogue_id": "dialogue_1",
      "vawg_type": "<VAWG type for this dialogue, e.g. coercive_control / stalking / economic_abuse / ...>",
      "timeline": {{
        "month": <1–6>,
        "date": "<ISO date within the 6-month period, e.g. YYYY-MM-DD>",
        "week_in_month": <1–4>
      }},
      "speakers": {{
        "victim": "<name>",
        "perpetrator": "<name>",
        "others": ["<name>", ...]
      }},
      "turns": [
        {{"speaker": "<name>", "text": "<message>"}},
        ...
      ]
    }},
    ... ({min_dialogues}–{max_dialogues} dialogue objects total)
  ]
}}"""


# LLMClient removed — baseline_generate now delegates to
# LLM_judge/llm_backend.py's build_backend() factory, which supports
# openai, vllm (server), vllm_native (in-process), gemini, and deepseek.


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

def extract_json(raw: str) -> dict:
    """Strip markdown fences and extract the first complete JSON object."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        inner, in_fence = [], False
        for line in lines:
            if line.strip().startswith("```") and not in_fence:
                in_fence = True
                continue
            if line.strip() == "```" and in_fence:
                break
            if in_fence:
                inner.append(line)
        text = "\n".join(inner).strip()

    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object in response")
    depth, end = 0, -1
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if depth == 0:
            end = i
            break
    if end == -1:
        raise ValueError("Unmatched braces in response")
    return json.loads(text[start: end + 1])


# ---------------------------------------------------------------------------
# Persona loading
# ---------------------------------------------------------------------------

def load_personas(csv_path: Path) -> list[dict]:
    personas = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            personas.append({"persona_id": row["persona_id"], "text": row["text"]})
    return personas


def select_personas(
    all_personas: list[dict],
    num: Optional[int],
    ids: Optional[list[str]],
    seed: Optional[int],
) -> list[dict]:
    if ids:
        id_set = set(ids)
        selected = [p for p in all_personas if p["persona_id"] in id_set]
        if not selected:
            raise ValueError(f"No personas found for IDs: {ids}")
        return selected
    if num:
        if seed is not None:
            random.seed(seed)
        return random.sample(all_personas, min(num, len(all_personas)))
    return all_personas


# ---------------------------------------------------------------------------
# Per-persona generation
# ---------------------------------------------------------------------------

def generate(persona: dict, client: LLMBackend, model_label: str,
             min_dialogues: int = 20, max_dialogues: int = 30) -> Optional[dict]:
    pid, text = persona["persona_id"], persona["text"]
    logger.info("Generating: %s  (%d–%d dialogues)  (%s)", pid, min_dialogues, max_dialogues, text[:60])

    try:
        raw = client.chat(SYSTEM_PROMPT, build_user_prompt(pid, text, min_dialogues, max_dialogues))
    except Exception as exc:
        logger.error("LLM call failed for %s: %s", pid, exc)
        return None

    try:
        result = extract_json(raw)
    except (ValueError, json.JSONDecodeError) as exc:
        logger.error("JSON parse failed for %s: %s", pid, exc)
        logger.warning("Raw response (first 2000 chars):\n%s", raw[:2000])
        return None

    result["model"] = model_label
    result["generated_at"] = datetime.now(timezone.utc).isoformat()
    result.setdefault("scenario_id", pid)
    result.setdefault("persona_id", pid)
    result.setdefault("persona_text", text)
    result.setdefault("scenario_summary", {})
    result.setdefault("dialogues", [])

    n = len(result.get("dialogues", []))
    logger.info("  → %d dialogues generated (requested %d–%d)", n, min_dialogues, max_dialogues)
    if n < min_dialogues:
        logger.warning("  Only %d dialogues (below minimum %d) for %s", n, min_dialogues, pid)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Baseline VAWG dialogue generator — direct single-prompt generation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model", required=True, choices=list(MODEL_REGISTRY),
                   help="gpt5.2 | deepseek-v4-flash | gemini-flash-lite | qwen3.5-35b | llama3.1-70b-awq")
    p.add_argument("--num-personas", type=int, default=None, metavar="N",
                   help="Number of personas to randomly sample")
    p.add_argument("--persona-ids", nargs="+", default=None, metavar="ID",
                   help="Specific persona IDs (overrides --num-personas)")
    p.add_argument("--persona-csv", type=Path, default=PERSONA_CSV,
                   help=f"Persona CSV path (default: {PERSONA_CSV})")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Output directory (default: results/<model>_<timestamp>)")
    p.add_argument("--base-url", type=str, default=None,
                   help="API base URL override (e.g. http://localhost:8000/v1)")
    p.add_argument("--model-id", type=str, default=None,
                   help="Override model identifier sent to the API")
    p.add_argument("--api-key", type=str, default=None,
                   help="API key (default: $OPENAI_API_KEY, or 'EMPTY' for local vLLM)")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--seed", type=int, default=None, help="Random seed for sampling")
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--retry-delay", type=float, default=10.0)
    p.add_argument("--min-dialogues", type=int, default=20, metavar="N",
                   help="Minimum number of dialogues per persona (default: 20)")
    p.add_argument("--max-dialogues", type=int, default=30, metavar="N",
                   help="Maximum number of dialogues per persona (default: 30, spanning 6 months)")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip personas whose output file already exists")
    # ── Judge options ──────────────────────────────────────────────────────
    p.add_argument("--auto-judge", action="store_true",
                   help="Automatically run the LLM judge on each generated file")
    p.add_argument("--judge-config", type=str, default=DEFAULT_JUDGE_CONFIG,
                   help=f"Path to judge_config.yaml (default: {DEFAULT_JUDGE_CONFIG})")
    p.add_argument("--judge-output-dir", type=Path, default=None,
                   help="Where to save judge result JSONs (default: <output-dir>/judge_results/)")
    p.add_argument("--judge-model", type=str, default=None,
                   help="Override the judge LLM model name (e.g. gpt-5.2)")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    cfg = dict(MODEL_REGISTRY[args.model])
    if args.base_url:            cfg["base_url"]    = args.base_url
    if args.model_id:            cfg["model_name"]  = args.model_id
    if args.temperature is not None: cfg["temperature"] = args.temperature
    if args.max_tokens  is not None: cfg["max_tokens"]  = args.max_tokens
    if args.api_key:             cfg["api_key"]     = args.api_key
    cfg["max_retries"] = args.max_retries
    cfg["retry_delay"] = args.retry_delay

    model_label = f"{args.model}:{cfg['model_name']}"

    out_dir = args.output_dir or (
        DEFAULT_OUTPUT_ROOT / f"{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output: %s", out_dir)

    all_personas = load_personas(args.persona_csv)
    logger.info("Loaded %d personas", len(all_personas))

    selected = select_personas(all_personas, args.num_personas, args.persona_ids, args.seed)
    logger.info("Selected %d personas", len(selected))

    client = build_backend(cfg)

    # ── Optionally set up the judge ───────────────────────────────────────
    judge      = None
    judge_dir  = None
    all_judge_results: list[dict] = []

    if args.auto_judge:
        yaml, DialogueJudge, aggregate_results = _import_judge()
        if not os.path.exists(args.judge_config):
            logger.error("Judge config not found: %s", args.judge_config)
            sys.exit(1)
        with open(args.judge_config, encoding="utf-8") as _f:
            judge_cfg = yaml.safe_load(_f)
        if args.judge_model:
            judge_cfg.setdefault("llm", {})["model_name"] = args.judge_model
        judge     = DialogueJudge(judge_cfg)
        judge_dir = args.judge_output_dir or (out_dir / "judge_results")
        judge_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Auto-judge enabled → results in %s", judge_dir)

    success, failed = 0, []

    for i, persona in enumerate(selected, 1):
        pid = persona["persona_id"]
        out_file = out_dir / f"{pid}_dialogue.json"

        if args.skip_existing and out_file.exists():
            logger.info("[%d/%d] Skip existing: %s", i, len(selected), out_file.name)
            success += 1
            # Load and judge if not yet judged
            if judge is not None:
                judge_file = judge_dir / f"{pid}_judge_results.json"
                if not judge_file.exists():
                    try:
                        with open(out_file, encoding="utf-8") as _f:
                            _existing = json.load(_f)
                        _adapted = to_judge_format(_existing)
                        _jr = judge.evaluate(_adapted)
                        with open(judge_file, "w", encoding="utf-8") as _f:
                            json.dump(_jr, _f, indent=2, ensure_ascii=False)
                        all_judge_results.append(_jr)
                        logger.info("  → Judge: %s", judge_file.name)
                    except Exception as _exc:
                        logger.warning("  Judge skipped for %s: %s", pid, _exc)
                else:
                    try:
                        with open(judge_file, encoding="utf-8") as _f:
                            all_judge_results.append(json.load(_f))
                    except Exception:
                        pass
            continue

        logger.info("[%d/%d] %s", i, len(selected), pid)
        result = generate(persona, client, model_label,
                          min_dialogues=args.min_dialogues, max_dialogues=args.max_dialogues)

        if result is None:
            failed.append(pid)
            continue

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        turns = sum(len(d.get("turns", [])) for d in result.get("dialogues", []))
        logger.info("  → %s  (%d dialogues, %d total turns)",
                    out_file.name, len(result.get("dialogues", [])), turns)
        success += 1

        # ── Judge this file if requested ──────────────────────────────────
        if judge is not None:
            judge_file = judge_dir / f"{pid}_judge_results.json"
            try:
                adapted     = to_judge_format(result)
                judge_result = judge.evaluate(adapted)
                with open(judge_file, "w", encoding="utf-8") as f:
                    json.dump(judge_result, f, indent=2, ensure_ascii=False)
                all_judge_results.append(judge_result)
                logger.info("  → Judge: %s", judge_file.name)
            except Exception as exc:
                logger.error("  Judge failed for %s: %s", pid, exc)

    if judge is not None:
        judge.cleanup()

    # ── Judge batch summary ───────────────────────────────────────────────
    if all_judge_results and judge is not None:
        _, _, aggregate_results = _import_judge()
        batch = aggregate_results(all_judge_results)
        batch_path = judge_dir / "batch_summary.json"
        with open(batch_path, "w", encoding="utf-8") as f:
            json.dump(batch, f, indent=2, ensure_ascii=False)
        logger.info(
            "Judge batch summary → %s  (overall avg: %s)",
            batch_path,
            batch.get("overall_average_score"),
        )

    # Summary
    logger.info("")
    logger.info("=" * 50)
    logger.info("Done  |  model: %s  |  success: %d/%d  |  failed: %d",
                model_label, success, len(selected), len(failed))
    if failed:
        logger.info("Failed IDs: %s", failed)
    logger.info("=" * 50)

    manifest = {
        "model": model_label,
        "model_name": cfg["model_name"],
        "backend": cfg.get("backend", "openai"),
        "base_url": cfg.get("base_url", ""),
        "temperature": cfg["temperature"],
        "max_tokens": cfg["max_tokens"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "num_selected": len(selected),
        "num_success": success,
        "min_dialogues_per_persona": args.min_dialogues,
        "max_dialogues_per_persona": args.max_dialogues,
        "num_failed": len(failed),
        "failed_ids": failed,
        "auto_judge": args.auto_judge,
        "judge_config": args.judge_config if args.auto_judge else None,
    }
    with open(out_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
