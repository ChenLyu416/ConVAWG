"""Shared CLI surface and orchestration loop for the framework baselines.

Each run_*.py script builds its parser with `build_shared_parser()` (adding
framework-specific knobs), constructs its generator, and calls
`run_framework()`. A generator must expose:

- ``pool`` — the BackendPool (for the LLM-call tally)
- ``run_persona(shared_input) -> tuple[record, framework_meta]``
- ``dry_run(shared_input) -> None`` — print prompts + planned call counts,
  zero LLM calls
- ``params() -> dict`` — framework knobs for the run manifest
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from tqdm import tqdm

from common import BASELINES_DIR, LLM_JUDGE_DIR
from common.backends import MODEL_REGISTRY, BackendPool
from common.inputs import AllowedRetriever, load_personas, select_personas
from common.io_utils import write_manifest, write_outputs
from common.judging import judge_existing, judge_record, setup_judge, write_batch_summary
from common.shared_cache import DEFAULT_SHARED_INPUTS_DIR, load_shared_input

logger = logging.getLogger(__name__)

PERSONA_CSV = BASELINES_DIR.parents[1] / "persona_generate" / "VAWG_sampled_200_personas.csv"
DEFAULT_OUTPUT_ROOT = BASELINES_DIR / "results"
DEFAULT_JUDGE_CONFIG = str(LLM_JUDGE_DIR / "configs" / "judge_config.yaml")


def build_shared_parser(framework: str, doc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=f"{framework} framework baseline for ConVAWG dialogue generation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=doc,
    )
    p.add_argument("--model", default="qwen3.5-35b", choices=list(MODEL_REGISTRY),
                   help="Backbone model key from MODEL_REGISTRY (default: qwen3.5-35b)")
    p.add_argument("--num-personas", type=int, default=None, metavar="N",
                   help="Number of personas to randomly sample")
    p.add_argument("--persona-ids", nargs="+", default=None, metavar="ID",
                   help="Specific persona IDs (overrides --num-personas)")
    p.add_argument("--persona-csv", type=Path, default=PERSONA_CSV,
                   help=f"Persona CSV path (default: {PERSONA_CSV})")
    p.add_argument("--seed", type=int, default=None, help="Random seed for persona sampling")
    p.add_argument("--min-dialogues", type=int, default=20, metavar="N",
                   help="Minimum dialogues (or sessions) per persona (default: 20)")
    p.add_argument("--max-dialogues", type=int, default=30, metavar="N",
                   help="Maximum dialogues (or sessions) per persona (default: 30); "
                        "each persona's budget is sampled uniformly from [min, max], "
                        "matching the single-prompt baseline's 20-30 range")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Output directory (default: results/<framework>_<model>_<timestamp>)")
    p.add_argument("--base-url", type=str, default=None,
                   help="API base URL override (e.g. http://localhost:8003/v1)")
    p.add_argument("--model-id", type=str, default=None,
                   help="Override model identifier sent to the API")
    p.add_argument("--api-key", type=str, default=None,
                   help="API key (default: $OPENAI_API_KEY, or 'EMPTY' for local vLLM)")
    p.add_argument("--temperature", type=float, default=None,
                   help="Override the registry temperature used for dialogue-content generation")
    p.add_argument("--control-temperature", type=float, default=0.1,
                   help="Temperature for planning/labeling/filter/termination calls (default: 0.1)")
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--retry-delay", type=float, default=10.0)
    p.add_argument("--k-cases", type=int, default=5,
                   help="Number of DHR cases in the shared retrieved evidence (default: 5)")
    p.add_argument("--shared-inputs-dir", type=Path, default=DEFAULT_SHARED_INPUTS_DIR,
                   help=f"Shared raw-input cache dir (default: {DEFAULT_SHARED_INPUTS_DIR})")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip personas whose output file already exists")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the prompts and planned call counts for one persona; no LLM calls")
    # ── Judge options ──────────────────────────────────────────────────────
    p.add_argument("--auto-judge", action="store_true",
                   help="Automatically run the LLM judge on each generated file")
    p.add_argument("--judge-config", type=str, default=DEFAULT_JUDGE_CONFIG,
                   help=f"Path to judge_config.yaml (default: {DEFAULT_JUDGE_CONFIG})")
    p.add_argument("--judge-output-dir", type=Path, default=None,
                   help="Where to save judge result JSONs (default: <output-dir>/judge_results/)")
    p.add_argument("--judge-model", type=str, default=None,
                   help="Override the judge LLM model name")
    p.add_argument("--verbose", action="store_true")
    return p


def resolve_backend_cfg(args: argparse.Namespace) -> Tuple[Dict[str, Any], str]:
    """MODEL_REGISTRY entry + CLI overrides (mirrors baseline_generate.main)."""
    cfg = dict(MODEL_REGISTRY[args.model])
    if args.base_url:
        cfg["base_url"] = args.base_url
    if args.model_id:
        cfg["model_name"] = args.model_id
    if args.temperature is not None:
        cfg["temperature"] = args.temperature
    if args.api_key:
        cfg["api_key"] = args.api_key
    cfg["max_retries"] = args.max_retries
    cfg["retry_delay"] = args.retry_delay
    return cfg, f"{args.model}:{cfg['model_name']}"


class _RetrieverFactory:
    """Lazily builds (and caches) the AllowedRetriever — FAISS + encoder load once."""

    def __init__(self) -> None:
        self._retriever: Optional[AllowedRetriever] = None

    def __call__(self) -> AllowedRetriever:
        if self._retriever is None:
            logger.info("Loading DHR retrieval index (FAISS + sentence-transformers)...")
            self._retriever = AllowedRetriever()
        return self._retriever


def run_framework(framework: str, args: argparse.Namespace, generator: Any) -> None:
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    all_personas = load_personas(args.persona_csv)
    selected = select_personas(all_personas, args.num_personas, args.persona_ids, args.seed)
    logger.info("Selected %d/%d personas", len(selected), len(all_personas))

    retriever_factory = _RetrieverFactory()

    if args.dry_run:
        shared_input = load_shared_input(
            selected[0], args.shared_inputs_dir, k=args.k_cases,
            retriever_factory=retriever_factory,
        )
        generator.dry_run(shared_input)
        return

    out_dir = args.output_dir or (
        DEFAULT_OUTPUT_ROOT / f"{framework}_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output: %s", out_dir)

    judge, judge_dir = (None, None)
    all_judge_results: list = []
    if args.auto_judge:
        judge, judge_dir = setup_judge(
            args.judge_config, args.judge_model, args.judge_output_dir, out_dir,
        )

    success, failed = 0, []
    for persona in tqdm(selected, desc=framework, unit="persona"):
        pid = persona["persona_id"]
        out_file = out_dir / f"{pid}_dialogue.json"

        if args.skip_existing and out_file.exists():
            logger.info("Skip existing: %s", out_file.name)
            success += 1
            if judge is not None:
                judge_existing(judge, judge_dir, pid, out_file, all_judge_results)
            continue

        try:
            shared_input = load_shared_input(
                persona, args.shared_inputs_dir, k=args.k_cases,
                retriever_factory=retriever_factory,
            )
            record, framework_meta = generator.run_persona(shared_input)
        except Exception as exc:  # noqa: BLE001 - one persona must not kill the run
            logger.error("Persona %s failed: %s", pid, exc)
            failed.append(pid)
            continue

        if not record.get("dialogues"):
            logger.error("Persona %s produced zero valid dialogues", pid)
            failed.append(pid)
            continue

        write_outputs(out_dir, pid, record, framework_meta)
        turns = sum(len(d.get("turns", [])) for d in record["dialogues"])
        logger.info("  → %s (%d dialogues, %d turns, %d LLM calls so far)",
                    out_file.name, len(record["dialogues"]), turns, generator.pool.num_calls)
        success += 1

        if judge is not None:
            judge_record(judge, judge_dir, pid, record, all_judge_results)

    generator.pool.cleanup()
    if judge is not None:
        judge.cleanup()
        write_batch_summary(all_judge_results, judge_dir)

    logger.info("=" * 50)
    logger.info("Done | %s | success: %d/%d | failed: %d | LLM calls: %d",
                framework, success, len(selected), len(failed), generator.pool.num_calls)
    if failed:
        logger.info("Failed IDs: %s", failed)

    write_manifest(
        out_dir,
        framework=framework,
        model_label=f"{args.model}:{generator.pool.cfg.get('model_name', '')}",
        cfg=generator.pool.cfg,
        num_selected=len(selected),
        num_success=success,
        failed_ids=failed,
        min_dialogues_per_persona=args.min_dialogues,
        max_dialogues_per_persona=args.max_dialogues,
        num_llm_calls=generator.pool.num_calls,
        params=generator.params(),
        auto_judge=args.auto_judge,
        judge_config=args.judge_config,
    )
    sys.exit(0 if not failed else 1)
