#!/usr/bin/env python3
"""
LLM-as-Judge CLI runner.

Usage examples
--------------
# Evaluate a single file with default config
python run_judge.py --input path/to/18015_dialogues.json

# Evaluate all *_dialogues.json files in a directory
python run_judge.py --input path/to/converted_dialogues/

# Use a custom config
python run_judge.py --input path/to/dialogues/ --config configs/judge_config.yaml

# Override model from CLI
python run_judge.py --input path/to/file.json --model gpt-5.2

# Dry-run: print prompts for the first dialogue without calling the LLM
python run_judge.py --input path/to/file.json --dry-run
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import yaml

# Ensure the LLM_judge package root is on sys.path when called directly
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from judge import DialogueJudge, aggregate_results, _normalise_input
from prompts import build_per_dialogue_prompt, build_global_prompt


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


DEFAULT_CONFIG_PATH = str(_HERE / "configs" / "judge_config.yaml")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def collect_input_files(input_path: str) -> list[str]:
    """Return list of *_dialogues.json files from a file or directory."""
    p = Path(input_path)
    if p.is_file():
        return [str(p)]
    if p.is_dir():
        files = sorted(p.rglob("*_dialogues.json"))
        if not files:
            # fallback: any .json file
            files = sorted(p.rglob("*.json"))
        return [str(f) for f in files]
    # Glob pattern
    return sorted(glob.glob(input_path, recursive=True))


def output_path_for(input_file: str, output_dir: str) -> str:
    """Derive output JSON path from input file path."""
    stem = Path(input_file).stem  # e.g. 18015_dialogues
    return str(Path(output_dir) / f"{stem}_judge_results.json")


def save_result(result: dict, out_path: str, pretty: bool = True) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2 if pretty else None, ensure_ascii=False)
    logger.info("Saved → %s", out_path)


def count_llm_error_ratings(result: dict) -> int:
    """Count hard LLM-failure ("ERROR") ratings in a judge result.

    Results containing ERROR ratings must not be saved: skip_existing would
    keep them forever and the watchdog would never re-evaluate the file.
    (PARSE_ERROR is a model-output issue, not a failed call, and is kept.)
    """
    n = 0
    for entry in result.get("per_dialogue_ratings", []):
        for metric in (entry.get("ratings") or {}).values():
            if isinstance(metric, dict) and metric.get("rating") == "ERROR":
                n += 1
    for metric in (result.get("global_ratings") or {}).values():
        if isinstance(metric, dict) and metric.get("rating") == "ERROR":
            n += 1
    return n


# ---------------------------------------------------------------------------
# Dry-run mode
# ---------------------------------------------------------------------------

def dry_run(input_file: str, cfg: dict) -> None:
    """Print prompts for the first two dialogues without calling the LLM."""
    with open(input_file, encoding="utf-8") as f:
        data = json.load(f)
    data = _normalise_input(data)  # baseline files use scenario_summary/turns, not vawg_scenarios/caa_rewritten_dialogue

    scenario = data.get("vawg_scenarios", {})
    events   = data.get("dialogues", [])

    proc_cfg = cfg.get("processing", {})
    max_d    = proc_cfg.get("max_dialogue_chars", 6000)
    max_g    = proc_cfg.get("max_global_chars", 20000)

    print("=" * 70)
    print("DRY RUN — Per-dialogue prompt for dialogue 0")
    print("=" * 70)
    if events:
        _, user_p = build_per_dialogue_prompt(scenario, events[0], max_d)
        print(user_p[:3000])
        print("[... truncated ...]" if len(user_p) > 3000 else "")

    print("\n" + "=" * 70)
    print("DRY RUN — Global prompt (first 3000 chars)")
    print("=" * 70)
    _, global_p = build_global_prompt(scenario, events, max_g)
    print(global_p[:3000])
    print("[... truncated ...]" if len(global_p) > 3000 else "")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM-as-Judge for VAWG dialogue evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to a single *_dialogues.json file, or a directory / glob to batch-process",
    )
    parser.add_argument(
        "--config", "-c", default=DEFAULT_CONFIG_PATH,
        help=f"Path to judge_config.yaml (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--output-dir", "-o", default=None,
        help="Directory for output JSON files (overrides config output.output_dir)",
    )
    parser.add_argument(
        "--model", "-m", default=None,
        help="Override the LLM model name from config (e.g. gpt-5.2)",
    )
    parser.add_argument(
        "--api-key", default=None,
        help="API key (overrides config; also read from OPENAI_API_KEY / DEEPSEEK_API_KEY env var)",
    )
    parser.add_argument(
        "--backend", default=None,
        choices=["openai", "vllm", "deepseek", "gemini", "vllm_native"],
        help="LLM backend to use (overrides config llm.backend)",
    )
    parser.add_argument(
        "--base-url", default=None,
        help="API base URL, e.g. a vLLM server (http://localhost:8000/v1). "
             "Overrides config llm.base_url.",
    )
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=None,
        help="vLLM GPU memory utilization fraction (0–1). Overrides config llm.gpu_memory_utilization.",
    )
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=None,
        help="vLLM tensor parallel size. Overrides config llm.tensor_parallel_size.",
    )
    parser.add_argument(
        "--max-num-seqs", type=int, default=None,
        help="vLLM max concurrent sequences (lowers sampler warmup memory). Overrides config llm.max_num_seqs.",
    )
    parser.add_argument(
        "--keep-llm-errors", action="store_true",
        help="Save results even when some LLM calls failed (ratings stay 'ERROR' and are "
             "excluded from all averages). Use for judges that deterministically refuse "
             "certain dialogues (e.g. Gemini PROHIBITED_CONTENT blocks), where retrying "
             "can never succeed. With skip_existing the file will not be retried later.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print prompts for first dialogue without calling the LLM",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Set logging level to DEBUG",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load config
    cfg_path = args.config
    if not os.path.exists(cfg_path):
        logger.error("Config file not found: %s", cfg_path)
        sys.exit(1)

    cfg = load_config(cfg_path)

    # CLI overrides
    if args.model:
        cfg.setdefault("llm", {})["model_name"] = args.model
    if args.api_key:
        cfg.setdefault("llm", {})["api_key"] = args.api_key
    if args.backend:
        cfg.setdefault("llm", {})["backend"] = args.backend
    if args.base_url:
        cfg.setdefault("llm", {})["base_url"] = args.base_url
    if args.gpu_memory_utilization is not None:
        cfg.setdefault("llm", {})["gpu_memory_utilization"] = args.gpu_memory_utilization
    if args.tensor_parallel_size is not None:
        cfg.setdefault("llm", {})["tensor_parallel_size"] = args.tensor_parallel_size
    if args.max_num_seqs is not None:
        cfg.setdefault("llm", {})["max_num_seqs"] = args.max_num_seqs
    if args.output_dir:
        cfg.setdefault("output", {})["output_dir"] = args.output_dir

    output_dir = cfg.get("output", {}).get("output_dir", "./results")
    pretty     = cfg.get("output", {}).get("pretty_json", True)
    skip_existing = cfg.get("processing", {}).get("skip_existing", True)

    # Collect files
    input_files = collect_input_files(args.input)
    if not input_files:
        resolved = str(Path(args.input).resolve())
        logger.error("No dialogue JSON files found at: %s", args.input)
        logger.error("Resolved path looked up: %s", resolved)
        sys.exit(1)

    logger.info("Found %d dialogue file(s) to process", len(input_files))

    # Dry run
    if args.dry_run:
        dry_run(input_files[0], cfg)
        return

    # Build judge (creates LLM backend once, reused across files)
    judge = DialogueJudge(cfg)

    errors:         list[str]  = []
    all_results:    list[dict] = []   # accumulate for batch summary
    newly_evaluated: list[str] = []   # files evaluated in this run

    for file_path in input_files:
        out_path = output_path_for(file_path, output_dir)

        if skip_existing and Path(out_path).exists():
            logger.info("Skipping (already evaluated): %s", file_path)
            # Load existing result so it contributes to this run's batch summary
            try:
                with open(out_path, encoding="utf-8") as fp:
                    all_results.append(json.load(fp))
            except Exception:
                pass
            continue

        logger.info("Evaluating: %s  [%d/%d]", file_path, len(newly_evaluated) + 1, len(input_files))
        try:
            result = judge.evaluate_file(file_path)
            n_llm_errors = count_llm_error_ratings(result)
            if n_llm_errors and args.keep_llm_errors:
                logger.warning(
                    "%d ERROR rating(s) from failed LLM calls in %s — "
                    "saving anyway (--keep-llm-errors); they are excluded from averages",
                    n_llm_errors, file_path,
                )
            elif n_llm_errors:
                logger.error(
                    "%d ERROR rating(s) from failed LLM calls in %s — "
                    "not saving, file will be retried on the next run",
                    n_llm_errors, file_path,
                )
                errors.append(file_path)
                continue
            save_result(result, out_path, pretty=pretty)
            all_results.append(result)
            newly_evaluated.append(file_path)
        except Exception as exc:
            logger.error("Failed to evaluate %s: %s", file_path, exc, exc_info=True)
            errors.append(file_path)

    judge.cleanup()

    # ------------------------------------------------------------------
    # Batch summary across all processed files
    # ------------------------------------------------------------------
    if all_results:
        batch = aggregate_results(all_results)
        batch_path = str(Path(output_dir) / "batch_summary.json")
        save_result(batch, batch_path, pretty=pretty)
        logger.info(
            "Batch summary → %s  (overall avg: %s / %s)",
            batch_path,
            batch.get("overall_average_score"),
            batch.get("overall_average_label"),
        )

    # Final report
    total     = len(input_files)
    skipped   = total - len(newly_evaluated) - len(errors)
    processed = len(newly_evaluated)

    logger.info(
        "Done. %d newly evaluated | %d skipped (loaded for summary) | %d errors",
        processed, skipped, len(errors),
    )

    if errors:
        logger.warning("Files with errors:\n  %s", "\n  ".join(errors))
        sys.exit(1)


if __name__ == "__main__":
    main()
