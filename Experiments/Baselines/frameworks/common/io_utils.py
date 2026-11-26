"""Judge-compatible output writers, synthetic timeline helper, run manifest.

Output shape is identical to the single-prompt baseline's `<pid>_dialogue.json`
so `LLM_judge/judge.py:_normalise_input()` accepts it unchanged. Full framework
traces go into a sibling `<pid>_framework_meta.json` to keep judge inputs lean.
"""

from __future__ import annotations

import json
import logging
import random
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def sample_num_dialogues(persona_id: str, min_dialogues: int, max_dialogues: int) -> int:
    """Per-persona dialogue budget drawn uniformly from [min, max].

    Mirrors the single-prompt baseline, which asks the model for 20-30
    dialogues and lets it pick the count. Seeded by persona ID only, so
    reruns and --skip-existing resume with identical budgets.
    """
    if max_dialogues <= min_dialogues:
        return min_dialogues
    rng = random.Random(zlib.crc32(f"budget:{persona_id}".encode("utf-8")))
    return rng.randint(min_dialogues, max_dialogues)


def month_assignments(num_dialogues: int, months_total: int = 6) -> List[int]:
    """Mirror of the single-prompt baseline's month distribution (~equal per month)."""
    per_month = num_dialogues // months_total
    extra = num_dialogues % months_total
    assignments: List[int] = []
    for m in range(1, months_total + 1):
        count = per_month + (1 if m <= extra else 0)
        assignments.extend([m] * count)
    return assignments


def synthetic_timeline(index1: int, assignments: List[int], year: int = 2024) -> Dict[str, Any]:
    """Deterministic timeline for dialogue #index1 (1-based).

    Cosmetic only — satisfies the judge's optional timeline field. It carries
    no escalation semantics (recorded as such in framework_meta).
    """
    month = assignments[index1 - 1]
    position_in_month = assignments[:index1].count(month)
    week_in_month = min(position_in_month, 4)
    day = min(2 + (position_in_month - 1) * 7, 28)
    return {
        "month": month,
        "date": f"{year}-{month:02d}-{day:02d}",
        "week_in_month": week_in_month,
    }


def observed_crime_types(dialogues: List[Dict[str, Any]]) -> List[str]:
    """Ordered union of the self-declared per-dialogue vawg_type values."""
    seen: List[str] = []
    for d in dialogues:
        vt = d.get("vawg_type")
        if vt and vt not in seen:
            seen.append(vt)
    return seen


def build_output(
    *,
    persona_id: str,
    persona_text: str,
    framework: str,
    model_label: str,
    scenario_summary: Dict[str, Any],
    dialogues: List[Dict[str, Any]],
    framework_meta_stub: Dict[str, Any],
) -> Dict[str, Any]:
    scenario_summary = dict(scenario_summary)
    scenario_summary.setdefault("crime_types", observed_crime_types(dialogues))
    return {
        "scenario_id": persona_id,
        "persona_id": persona_id,
        "persona_text": persona_text,
        "framework": framework,
        "model": model_label,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scenario_summary": scenario_summary,
        "dialogues": dialogues,
        "framework_meta": framework_meta_stub,
    }


def write_outputs(
    out_dir: Path,
    persona_id: str,
    record: Dict[str, Any],
    framework_meta: Optional[Dict[str, Any]] = None,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{persona_id}_dialogue.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    if framework_meta is not None:
        meta_file = out_dir / f"{persona_id}_framework_meta.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(framework_meta, f, indent=2, ensure_ascii=False)
    return out_file


def write_manifest(
    out_dir: Path,
    *,
    framework: str,
    model_label: str,
    cfg: Dict[str, Any],
    num_selected: int,
    num_success: int,
    failed_ids: List[str],
    min_dialogues_per_persona: int,
    max_dialogues_per_persona: int,
    num_llm_calls: int,
    params: Dict[str, Any],
    auto_judge: bool = False,
    judge_config: Optional[str] = None,
) -> Path:
    manifest = {
        "framework": framework,
        "model": model_label,
        "model_name": cfg.get("model_name", ""),
        "backend": cfg.get("backend", "openai"),
        "base_url": cfg.get("base_url", ""),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "num_selected": num_selected,
        "num_success": num_success,
        "min_dialogues_per_persona": min_dialogues_per_persona,
        "max_dialogues_per_persona": max_dialogues_per_persona,
        "num_failed": len(failed_ids),
        "failed_ids": failed_ids,
        "num_llm_calls": num_llm_calls,
        "params": params,
        "auto_judge": auto_judge,
        "judge_config": judge_config if auto_judge else None,
    }
    manifest_path = Path(out_dir) / "run_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest_path
