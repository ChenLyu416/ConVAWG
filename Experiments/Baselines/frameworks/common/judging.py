"""Optional auto-judge wiring (mirrors baseline_generate.py's --auto-judge flow).

Our output records are judge-compatible as-is (`judge.py:_normalise_input()`
handles the `scenario_summary` + `dialogues[].turns` shape), so records are
passed straight to `DialogueJudge.evaluate()` with no adapter.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import common  # noqa: F401  (sys.path setup for LLM_judge/)

logger = logging.getLogger(__name__)


def _import_judge():
    try:
        import yaml
        from judge import DialogueJudge, aggregate_results
        return yaml, DialogueJudge, aggregate_results
    except ImportError as exc:
        logger.error("Cannot import LLM_judge: %s", exc)
        sys.exit(1)


def setup_judge(
    judge_config_path: str,
    judge_model: Optional[str],
    judge_output_dir: Optional[Path],
    out_dir: Path,
) -> Tuple[Any, Path]:
    yaml, DialogueJudge, _ = _import_judge()
    config_path = Path(judge_config_path)
    if not config_path.exists():
        logger.error("Judge config not found: %s", config_path)
        sys.exit(1)
    with open(config_path, encoding="utf-8") as f:
        judge_cfg = yaml.safe_load(f)
    if judge_model:
        judge_cfg.setdefault("llm", {})["model_name"] = judge_model
    judge = DialogueJudge(judge_cfg)
    judge_dir = Path(judge_output_dir) if judge_output_dir else (Path(out_dir) / "judge_results")
    judge_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Auto-judge enabled → results in %s", judge_dir)
    return judge, judge_dir


def judge_record(
    judge: Any,
    judge_dir: Path,
    persona_id: str,
    record: Dict[str, Any],
    all_results: List[Dict[str, Any]],
) -> None:
    """Judge one record, write `<pid>_judge_results.json`, append to all_results."""
    judge_file = Path(judge_dir) / f"{persona_id}_judge_results.json"
    try:
        result = judge.evaluate(record)
        with open(judge_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        all_results.append(result)
        logger.info("  → Judge: %s", judge_file.name)
    except Exception as exc:  # noqa: BLE001 - judging must not kill generation
        logger.error("  Judge failed for %s: %s", persona_id, exc)


def judge_existing(
    judge: Any,
    judge_dir: Path,
    persona_id: str,
    out_file: Path,
    all_results: List[Dict[str, Any]],
) -> None:
    """For --skip-existing: judge a previously generated file if not yet judged."""
    judge_file = Path(judge_dir) / f"{persona_id}_judge_results.json"
    if judge_file.exists():
        try:
            with open(judge_file, encoding="utf-8") as f:
                all_results.append(json.load(f))
        except Exception:  # noqa: BLE001
            pass
        return
    try:
        with open(out_file, encoding="utf-8") as f:
            record = json.load(f)
    except Exception as exc:  # noqa: BLE001
        logger.warning("  Judge skipped for %s: %s", persona_id, exc)
        return
    judge_record(judge, judge_dir, persona_id, record, all_results)


def write_batch_summary(all_results: List[Dict[str, Any]], judge_dir: Path) -> None:
    if not all_results:
        return
    _, _, aggregate_results = _import_judge()
    batch = aggregate_results(all_results)
    batch_path = Path(judge_dir) / "batch_summary.json"
    with open(batch_path, "w", encoding="utf-8") as f:
        json.dump(batch, f, indent=2, ensure_ascii=False)
    logger.info(
        "Judge batch summary → %s  (overall avg: %s)",
        batch_path, batch.get("overall_average_score"),
    )
