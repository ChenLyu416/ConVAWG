"""Persona loading and restricted DHR-evidence retrieval.

`AllowedRetriever` is the ONLY sanctioned entry point to DHR evidence for the
framework baselines: it wraps `persona_generate.retrieval.KBCaseRetriever` and
strips `RetrievedCase.metadata`, which carries the ConVAWG event-graph data
(`full_case` / `actors` / `relationships` / `events`) that the baselines must
never see. Allowed fields only: case_id, victim_summary, perpetrator_summary,
violence_types, context, similarity_score.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from common import PERSONA_GENERATE_DIR, REPO_ROOT

# Re-exported so the framework scripts share the single-prompt baseline's
# persona-selection semantics exactly.
from baseline_generate import load_personas, select_personas  # noqa: F401

logger = logging.getLogger(__name__)

DEFAULT_PERSONA_INDEX_DIR = PERSONA_GENERATE_DIR / "outputs" / "persona_index_full"
DEFAULT_KB_HOMICIDE_DIR = REPO_ROOT / "RAG_kb_construct" / "KB_homicide"

# Key names that would smuggle ConVAWG event-graph structure into the inputs.
FORBIDDEN_KEYS = {"metadata", "events", "relationships", "actors", "full_case"}


@dataclass
class RestrictedCase:
    """A retrieved DHR case reduced to the raw-evidence fields only."""

    case_id: str
    victim_summary: str
    perpetrator_summary: str
    violence_types: List[str] = field(default_factory=list)
    context: str = ""
    similarity_score: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AllowedRetriever:
    """Restricted wrapper around KBCaseRetriever (metadata stripped)."""

    def __init__(
        self,
        persona_index_dir: Optional[Path] = None,
        kb_homicide_dir: Optional[Path] = None,
    ) -> None:
        from retrieval import KBCaseRetriever  # persona_generate/ on sys.path

        self._retriever = KBCaseRetriever(
            persona_index_dir=Path(persona_index_dir or DEFAULT_PERSONA_INDEX_DIR),
            kb_homicide_dir=Path(kb_homicide_dir or DEFAULT_KB_HOMICIDE_DIR),
        )

    def retrieve(self, persona_text: str, k: int = 5) -> List[RestrictedCase]:
        cases = self._retriever.retrieve_similar_cases(
            persona_text=persona_text, k=k, role_filter="victim",
        )
        return [
            RestrictedCase(
                case_id=c.case_id,
                victim_summary=c.victim_summary,
                perpetrator_summary=c.perpetrator_summary,
                violence_types=list(c.violence_types),
                context=c.context,
                similarity_score=float(c.similarity_score),
            )
            for c in cases
        ]


def assert_no_forbidden_keys(obj: Any, path: str = "$") -> None:
    """Recursively assert no forbidden key appears anywhere in `obj`."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in FORBIDDEN_KEYS:
                raise ValueError(f"Forbidden key '{key}' at {path}")
            assert_no_forbidden_keys(value, f"{path}.{key}")
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            assert_no_forbidden_keys(item, f"{path}[{i}]")
