"""Shared raw-input cache: one `<pid>_input.json` per persona.

Guarantees the three framework baselines consume byte-identical raw inputs
(persona seed + retrieved DHR evidence). Retrieval is deterministic (fixed
FAISS index, fixed k, fixed encoder), so the cache is reproducible; it exists
so the FAISS index + SentenceTransformer are loaded once, not once per
framework run.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from common import FRAMEWORKS_DIR
from common.inputs import AllowedRetriever, assert_no_forbidden_keys

logger = logging.getLogger(__name__)

SHARED_INPUT_SCHEMA_VERSION = 1
DEFAULT_SHARED_INPUTS_DIR = FRAMEWORKS_DIR / "shared_inputs"


def shared_input_path(shared_dir: Path, persona_id: str) -> Path:
    return Path(shared_dir) / f"{persona_id}_input.json"


def build_shared_input(
    persona: Dict[str, str],
    retriever: AllowedRetriever,
    k: int = 5,
) -> Dict[str, Any]:
    cases = retriever.retrieve(persona["text"], k=k)
    if not cases:
        logger.warning(
            "No DHR cases retrieved for persona %s — persona-only grounding",
            persona["persona_id"],
        )
    shared_input: Dict[str, Any] = {
        "schema_version": SHARED_INPUT_SCHEMA_VERSION,
        "persona_id": persona["persona_id"],
        "persona_text": persona["text"],
        "k": k,
        "retrieved_cases": [c.as_dict() for c in cases],
    }
    assert_no_forbidden_keys(shared_input)
    return shared_input


def load_shared_input(
    persona: Dict[str, str],
    shared_dir: Path,
    k: int = 5,
    retriever_factory: Optional[Callable[[], AllowedRetriever]] = None,
) -> Dict[str, Any]:
    """Load a persona's cached shared input, building it on the fly if allowed."""
    path = shared_input_path(shared_dir, persona["persona_id"])
    if path.exists():
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("schema_version") == SHARED_INPUT_SCHEMA_VERSION and data.get("k") == k:
            return data
        logger.warning("Stale shared input %s (version/k mismatch) — rebuilding", path.name)

    if retriever_factory is None:
        raise FileNotFoundError(
            f"Shared input missing for persona {persona['persona_id']} at {path}. "
            "Run frameworks/prepare_shared_inputs.py first."
        )
    shared_input = build_shared_input(persona, retriever_factory(), k=k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(shared_input, f, indent=2, ensure_ascii=False)
    return shared_input


def evidence_block(shared_input: Dict[str, Any], max_context_chars: int = 1200) -> str:
    """Prompt-ready DHR evidence block — identical formatting for all frameworks."""
    cases = shared_input.get("retrieved_cases", [])
    if not cases:
        return "(No similar DHR case evidence was retrieved for this persona.)"
    parts = ["Evidence retrieved from real UK Domestic Homicide Review (DHR) cases similar to this persona:"]
    for i, case in enumerate(cases, 1):
        vt = ", ".join(case.get("violence_types", [])) or "unspecified"
        context = (case.get("context") or "")[:max_context_chars]
        parts.append(
            f"\n[Case {i} | violence types: {vt}]\n"
            f"Victim: {case.get('victim_summary', '')}\n"
            f"Perpetrator: {case.get('perpetrator_summary', '')}\n"
            f"Case summary: {context}"
        )
    return "\n".join(parts)
