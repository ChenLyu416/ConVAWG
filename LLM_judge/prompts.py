"""
Prompt templates for the LLM-as-Judge dialogue evaluator.

Each builder function returns (system_prompt, user_prompt) ready to pass to
LLMBackend.chat().
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Shared system prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert evaluator for synthetic dialogue datasets about Violence Against Women and Girls (VAWG).
Your task is to assess the quality of generated chat-based conversations on specific rubric dimensions.
Respond ONLY with valid JSON — no markdown fences, no extra text.
""".strip()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scenario_block(scenario: dict[str, Any]) -> str:
    """Render a compact scenario summary for prompt context."""
    vp = scenario.get("victim_profile", {})
    pp = scenario.get("perpetrator_profile", {})

    lines = [
        "=== SCENARIO CONTEXT ===",
        f"Crime types: {', '.join(scenario.get('crime_types', []))}",
        "",
        f"Victim:      {vp.get('name','')} | age {vp.get('age','')} | {vp.get('gender','')} | "
        f"{vp.get('occupation','')}",
        f"Perpetrator: {pp.get('name','')} | age {pp.get('age','')} | {pp.get('gender','')} | "
        f"relationship: {pp.get('relationship_to_victim','')}",
        "",
        "Scenario description:",
        scenario.get("scenario_description", "(none)"),
        "",
        "Victim vulnerabilities:",
        *[f"  - {v}" for v in vp.get("vulnerabilities", [])],
        "",
        "Perpetrator characteristics:",
        *[f"  - {c}" for c in pp.get("characteristics", [])],
    ]
    persona_text = scenario.get("persona_text", "").strip()
    if persona_text:
        lines += ["", "Persona (raw):", persona_text]
    return "\n".join(lines)


def _event_block(event: dict[str, Any]) -> str:
    """Render event metadata for a single dialogue.

    Handles both pipeline format (event_id, conversation_context, participants)
    and baseline format (dialogue_id, vawg_type, timeline, speakers).
    """
    # --- IDs / timestamp ----------------------------------------------------
    event_id  = event.get("event_id") or event.get("dialogue_id", "")
    timeline  = event.get("timeline") or {}
    timestamp = event.get("timestamp") or (
        f"month {timeline.get('month','')} week {timeline.get('week_in_month','')}"
        if isinstance(timeline, dict) else str(timeline)
    )

    # --- Crime type / escalation --------------------------------------------
    ctx          = event.get("conversation_context") or {}
    crime_type   = ctx.get("crime_type") or event.get("vawg_type", "N/A")
    escalation   = ctx.get("escalation_level", "")

    # --- Speakers -----------------------------------------------------------
    participants = event.get("participants") or {}
    sender       = participants.get("sender", {})
    receiver     = participants.get("receiver", {})
    # Baseline uses 'speakers': {'victim': name, 'perpetrator': name}
    speakers     = event.get("speakers", {})
    sender_str   = (
        f"{sender.get('name','')} ({sender.get('role','')})"
        if sender
        else f"victim={speakers.get('victim', '')}  perpetrator={speakers.get('perpetrator', '')}"
    )

    lines = [
        "=== EVENT CONTEXT ===",
        f"Event ID:        {event_id}",
        f"Timestamp:       {timestamp}",
        f"Escalation level:{escalation}",
        f"Crime type:      {crime_type}",
        f"Tone:            {ctx.get('tone', '')}",
        f"Emotional state: {ctx.get('emotional_state', '')}",
        f"Action:          {ctx.get('action', '')}",
        f"Description:     {ctx.get('description', '')}",
        f"Purpose:         {ctx.get('purpose', '')}",
        f"Speakers:        {sender_str}",
    ]
    if sender and receiver:
        lines.append(f"Receiver:        {receiver.get('name','')} ({receiver.get('role','')})")
    return "\n".join(lines)


def _get_dialogue_text(event: dict[str, Any]) -> str:
    """Return caa_rewritten_dialogue if present, else enforced_dialogue.
    Also handles baseline format where dialogue is stored as a 'turns' list.
    """
    # Pipeline format: plain-text dialogue string
    text = event.get("caa_rewritten_dialogue") or event.get("enforced_dialogue") or event.get("dialogue")
    if text:
        return text
    # Baseline format: list of {"speaker": ..., "text": ...} turns
    turns = event.get("turns", [])
    if turns:
        return "\n".join(f"{t.get('speaker', '?')}: {t.get('text', '')}" for t in turns)
    return ""


# ---------------------------------------------------------------------------
# Per-dialogue prompt  (4 metrics)
# ---------------------------------------------------------------------------

PER_DIALOGUE_METRICS = ["coherence", "humanlikeness", "persona_consistency", "toxicity_realism"]

PER_DIALOGUE_RUBRIC = """\
Rate the single dialogue below on FOUR metrics.

METRICS & GUIDANCE
==================

1. coherence
   Turn-level logical consistency and conversational flow.
   - Appropriate responses to previous utterances
   - Topic continuity
   - Absence of contradictions or abrupt shifts
   Judge internal conversational logic ONLY — not scenario accuracy.

2. humanlikeness
   Naturalness and conversational authenticity.
   - Linguistic fluency, natural phrasing
   - Emotional appropriateness
    - Realistic conversational behaviour in online chatting context
    - Treat typical chat traits (short replies, mild typos, slang, imperfect grammar)
      as normal unless they significantly harm clarity
   Focus on language quality and interaction style — not domain realism.

3. persona_consistency
   Stability of character traits and speaking style.
   - Tone, vocabulary, behavioural patterns
   - Emotional reactions consistent with assigned profile
   - Alignment with character description
   Consider both speakers across all turns.

4. toxicity_realism
   Realism of toxic / abusive language (if any).
   - Emerges at contextually appropriate points
   - Progresses in intensity in a believable way
   - Aligns with relationship dynamics and situation
   Judge REALISM of timing and expression only — NOT morality or toxicity level.
   If no toxic language is present, rate "N/A" and leave explanation empty.

RATING SCALE (1-5)
==================
1 — Very Poor
    Serious problems that substantially undermine realism, consistency, or plausibility.
    Use this only when issues are frequent and clearly damaging.
2 — Weak
    Noticeable issues that reduce quality or realism.
    Some elements are acceptable, but the dialogue often feels inconsistent,
    unnatural, or implausible.
3 — Moderate
    Acceptable overall with imperfections.
    The dialogue is generally plausible, though several weaknesses or
    inconsistencies remain.
4 — Strong
    Mostly realistic and consistent with only minor issues.
    The dialogue aligns well with expectations for the evaluated metric.
5 — Excellent
    Highly realistic, coherent, and contextually appropriate.
    The dialogue strongly satisfies the evaluated metric, with only negligible
    imperfections.

IMPORTANT: Judge within context. Consider the assigned personas and relationship dynamics.
Be fair and calibrated: do not over-penalize minor stylistic issues that are common in chat.
Reserve 1-2 for substantial problems, and use 3 when quality is broadly acceptable.

OUTPUT FORMAT (JSON only, no markdown)
=======================================
{
    "coherence":           {"rating": "<1|2|3|4|5>", "explanation": "<1-3 sentences>"},
    "humanlikeness":       {"rating": "<1|2|3|4|5>", "explanation": "<1-3 sentences>"},
    "persona_consistency": {"rating": "<1|2|3|4|5>", "explanation": "<1-3 sentences>"},
    "toxicity_realism":    {"rating": "<1|2|3|4|5|N/A>", "explanation": "<1-3 sentences>"}
}
""".strip()


def build_per_dialogue_prompt(
    scenario: dict[str, Any],
    event: dict[str, Any],
    max_dialogue_chars: int = 6000,
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for per-dialogue evaluation."""
    dialogue_text = _get_dialogue_text(event)
    if len(dialogue_text) > max_dialogue_chars:
        dialogue_text = dialogue_text[:max_dialogue_chars] + "\n[... truncated ...]"

    user_prompt = "\n\n".join([
        _scenario_block(scenario),
        _event_block(event),
        "=== DIALOGUE TO EVALUATE ===",
        dialogue_text,
        PER_DIALOGUE_RUBRIC,
    ])
    return SYSTEM_PROMPT, user_prompt


# ---------------------------------------------------------------------------
# Global prompt  (2 metrics over all dialogues)
# ---------------------------------------------------------------------------

GLOBAL_METRICS = ["crime_fidelity", "scenario_realism"]

GLOBAL_RUBRIC = """\
After reading the FULL dialogue sequence and scenario summary, rate TWO global metrics.

METRICS & GUIDANCE
==================

1. crime_fidelity
   Domain-level realism of VAWG dynamics across ALL dialogues.
   - Coercion and control behaviours
   - Escalation trajectories
   - Power imbalance
   - Victim–perpetrator interaction patterns
   Focus on domain authenticity — not writing quality or scenario alignment.

2. scenario_realism
   Faithful and plausible alignment with the GIVEN SCENARIO across ALL dialogues.
   - Key elements reflected (relationship, ages, cultural background, vulnerabilities)
   - Logical consistency with described events
   - Realistic pacing and progression
   Judge alignment to this specific scenario — not general VAWG realism.

RATING SCALE (1-5)
==================
1 — Very Poor
    Serious problems that substantially undermine realism, consistency, or plausibility.
    Use this only when issues are frequent and clearly damaging.
2 — Weak
    Noticeable issues that reduce quality or realism.
    Some elements are acceptable, but the dialogue often feels inconsistent,
    unnatural, or implausible.
3 — Moderate
    Acceptable overall with imperfections.
    The dialogue is generally plausible, though several weaknesses or
    inconsistencies remain.
4 — Strong
    Mostly realistic and consistent with only minor issues.
    The dialogue aligns well with expectations for the evaluated metric.
5 — Excellent
    Highly realistic, coherent, and contextually appropriate.
    The dialogue strongly satisfies the evaluated metric, with only negligible
    imperfections.

Be fair and calibrated: avoid over-penalizing minor chat-style imperfections.
Reserve 1-2 for substantial failures, and use 3 when quality is broadly acceptable.

OUTPUT FORMAT (JSON only, no markdown)
=======================================
{
    "crime_fidelity":   {"rating": "<1|2|3|4|5>", "explanation": "<2-4 sentences>"},
    "scenario_realism": {"rating": "<1|2|3|4|5>", "explanation": "<2-4 sentences>"}
}
""".strip()


def build_global_prompt(
    scenario: dict[str, Any],
    events: list[dict[str, Any]],
    max_global_chars: int = 20000,
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for global evaluation."""
    dialogue_chunks: list[str] = []
    total_chars = 0

    for idx, event in enumerate(events, start=1):
        ctx        = event.get("conversation_context") or {}
        event_id   = event.get("event_id") or event.get("dialogue_id", "?")
        escalation = ctx.get("escalation_level") or event.get("vawg_type", "?")
        timeline   = event.get("timeline") or {}
        timestamp  = event.get("timestamp") or (
            f"month {timeline.get('month','')} week {timeline.get('week_in_month','')}"
            if isinstance(timeline, dict) else ""
        )
        text = _get_dialogue_text(event)
        header = (
            f"--- Dialogue {idx} "
            f"(Event {event_id}, "
            f"Escalation/Type {escalation}, "
            f"{timestamp}) ---"
        )
        chunk = f"{header}\n{text}"
        if total_chars + len(chunk) > max_global_chars:
            remaining = max_global_chars - total_chars
            if remaining > 200:
                dialogue_chunks.append(chunk[:remaining] + "\n[... truncated ...]")
            dialogue_chunks.append(f"\n[{len(events) - idx} more dialogues omitted due to length]")
            break
        dialogue_chunks.append(chunk)
        total_chars += len(chunk)

    dialogues_text = "\n\n".join(dialogue_chunks)

    user_prompt = "\n\n".join([
        _scenario_block(scenario),
        "=== ALL DIALOGUES (chronological) ===",
        dialogues_text,
        GLOBAL_RUBRIC,
    ])
    return SYSTEM_PROMPT, user_prompt
