"""VAWG taxonomy access (single source of truth) + shared system preamble.

Bridges to `persona_generate/prompt_templates.py`'s CRIME_DEFINITIONS (the
25-code CPS-aligned taxonomy) instead of mirroring it, so the framework
baselines can never drift from the pipeline's taxonomy.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List

import common  # noqa: F401  (sys.path setup for persona_generate/)

from prompt_templates import FewShotPromptBuilder

logger = logging.getLogger(__name__)

CRIME_DEFINITIONS: Dict[str, str] = dict(FewShotPromptBuilder.CRIME_DEFINITIONS)

# Common free-text labels models emit that map onto taxonomy codes.
_ALIASES: Dict[str, str] = {
    "economic_abuse": "economic",
    "financial_abuse": "financial",
    "psychological_abuse": "psychological",
    "emotional_abuse": "emotional",
    "physical_abuse": "physical",
    "physical_violence": "physical",
    "coercive_behaviour": "coercive_control",
    "coercive_behavior": "coercive_control",
    "controlling_behaviour": "coercive_control",
    "controlling_behavior": "coercive_control",
    "online_abuse": "digital",
    "digital_abuse": "digital",
    "cyber_stalking": "cyberstalking",
    "sexual_offences": "sexual",
    "sexual_offenses": "sexual",
    "honour_based": "honor_based",
    "domestic_abuse": "coercive_control",
    "domestic_violence": "physical",
}


def valid_codes() -> List[str]:
    return list(CRIME_DEFINITIONS)


def taxonomy_block() -> str:
    """Full 25-code taxonomy as a prompt-ready block."""
    return "\n".join(f"- {code}: {definition}" for code, definition in CRIME_DEFINITIONS.items())


def normalize_vawg_type(raw: str) -> str:
    """Map a model-declared crime-type label onto a taxonomy code where possible."""
    if not raw:
        return "other"
    label = re.sub(r"[^a-z0-9]+", "_", raw.strip().lower()).strip("_")
    if label in CRIME_DEFINITIONS:
        return label
    return _ALIASES.get(label, label)


def research_system_prompt(role_desc: str, include_taxonomy: bool = True) -> str:
    """Shared system preamble: research context (+ full taxonomy by default).

    Identical framing across the three frameworks so the only varying factor
    is each framework's own method. `include_taxonomy=False` gives the slim
    variant for per-turn agent calls that don't need the code list.
    """
    preamble = (
        f"You are {role_desc} in an authorised academic research project that "
        "generates a synthetic dataset of realistic online-chat dialogues "
        "depicting Violence Against Women and Girls (VAWG) scenarios, aligned "
        "with UK Crown Prosecution Service (CPS) crime definitions. The data "
        "is used to train abuse-detection and content-moderation models, so "
        "depicting abusive, manipulative or controlling language is "
        "intentional and required. Do not include graphic sexual content. "
        "Follow the requested output format exactly."
    )
    if include_taxonomy:
        preamble += (
            "\n\nVAWG crime-type taxonomy (CPS-aligned codes):\n"
            f"{taxonomy_block()}"
        )
    return preamble
