#!/usr/bin/env python3
"""
Run GPT-5.2, DeepSeek, and Gemini judges in parallel on the same input files.

Each judge uses its own backend (one HTTP connection pool per judge) and writes
to a sub-directory under the output root:
    <output_dir>/gpt52/
    <output_dir>/deepseek/
    <output_dir>/gemini/

Parallelism model
-----------------
- Outer parallelism: the three judges are submitted concurrently via
  ThreadPoolExecutor(max_workers=3).  Each judge thread works through the
  input files sequentially so that per-judge rate limits are respected.
- Inner parallelism (optional, --parallel-files N): within each judge,
  process up to N files concurrently using a second thread pool.  Useful
  when files are small and API rate limits allow burst requests.

Usage examples
--------------
# Run all three judges on a directory of dialogue files
python run_multi_judge_parallel.py --input path/to/converted_dialogues/

# Custom configs
python run_multi_judge_parallel.py \\
    --input path/to/files/ \\
    --config-gpt52    configs/judge_config.yaml \\
    --config-deepseek configs/judge_config_deepseek.yaml \\
    --config-gemini   configs/judge_config_gemini.yaml

# Process up to 4 files per judge concurrently
python run_multi_judge_parallel.py --input path/to/files/ --parallel-files 4

# Output to a custom root directory
python run_multi_judge_parallel.py --input path/to/files/ --output-dir ./multi_judge_results

# Dry-run (no API calls)
python run_multi_judge_parallel.py --input path/to/files/ --dry-run
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import yaml

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from judge import DialogueJudge, aggregate_results

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [%(judge_label)s]  %(message)s",
    datefmt="%H:%M:%S",
)


class _JudgeAdapter(logging.LoggerAdapter):
    """Injects judge_label into every log record."""

    def process(self, msg, kwargs):
        kwargs.setdefault("extra", {})
        kwargs["extra"]["judge_label"] = self.extra.get("judge_label", "-")
        return msg, kwargs


logger = _JudgeAdapter(logging.getLogger(__name__), {"judge_label": "main"})

# ---------------------------------------------------------------------------
# Default config paths
# ---------------------------------------------------------------------------

_DEFAULT_CONFIGS: dict[str, str] = {
    "gpt52":    str(_HERE / "configs" / "judge_config.yaml"),
    "deepseek": str(_HERE / "configs" / "judge_config_deepseek.yaml"),
    "gemini":   str(_HERE / "configs" / "judge_config_gemini.yaml"),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def collect_input_files(input_path: str) -> list[str]:
    p = Path(input_path)
    if p.is_file():
        return [str(p)]
    if p.is_dir():
        files = sorted(p.rglob("*_dialogues.json"))
        if not files:
            files = sorted(p.rglob("*.json"))
        return [str(f) for f in files]
    return sorted(glob.glob(input_path, recursive=True))


def output_path_for(input_file: str, output_dir: str, label: str) -> str:
    stem = Path(input_file).stem
    return str(Path(output_dir) / label / f"{stem}_judge_results.json")


def save_result(result: dict, out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Per-judge worker
# ---------------------------------------------------------------------------


def _run_judge_for_file(
    judge: DialogueJudge,
    file_path: str,
    out_path: str,
    skip_existing: bool,
    label: str,
) -> Optional[dict]:
    """Evaluate a single file with the given judge.  Returns the result dict or None."""
    jlog = _JudgeAdapter(logging.getLogger(__name__), {"judge_label": label})

    if skip_existing and Path(out_path).exists():
        jlog.info("Skipping (already evaluated): %s", file_path)
        try:
            with open(out_path, encoding="utf-8") as fp:
                return json.load(fp)
        except Exception:
            return None

    jlog.info("Evaluating: %s", file_path)
    try:
        result = judge.evaluate_file(file_path)
        save_result(result, out_path)
        jlog.info("Saved → %s", out_path)
        return result
    except Exception as exc:
        jlog.error("Failed on %s: %s", file_path, exc)
        return None


def _run_judge_all_files(
    label: str,
    cfg: dict,
    input_files: list[str],
    output_root: str,
    parallel_files: int,
) -> list[Optional[dict]]:
    """Build a judge backend and process all files (optionally in parallel)."""
    jlog = _JudgeAdapter(logging.getLogger(__name__), {"judge_label": label})
    jlog.info("Initialising backend: %s / %s", cfg["llm"]["backend"], cfg["llm"]["model_name"])

    judge = DialogueJudge(cfg)
    skip = cfg.get("processing", {}).get("skip_existing", True)

    pairs = [
        (f, output_path_for(f, output_root, label))
        for f in input_files
    ]

    results: list[Optional[dict]] = [None] * len(pairs)

    if parallel_files <= 1:
        for i, (fp, op) in enumerate(pairs):
            results[i] = _run_judge_for_file(judge, fp, op, skip, label)
    else:
        # Inner thread pool — process multiple files concurrently for this judge.
        # Using threads is safe because the judge uses HTTP I/O (not GPU compute).
        future_to_idx: dict = {}
        with ThreadPoolExecutor(max_workers=parallel_files) as pool:
            for i, (fp, op) in enumerate(pairs):
                fut = pool.submit(_run_judge_for_file, judge, fp, op, skip, label)
                future_to_idx[fut] = i
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                try:
                    results[idx] = fut.result()
                except Exception as exc:
                    jlog.error("Unexpected error for file index %d: %s", idx, exc)

    judge.backend.cleanup()
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run GPT-5.2, DeepSeek, and Gemini judges in parallel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Dialogue JSON file, directory, or glob pattern",
    )
    parser.add_argument(
        "--output-dir", "-o", default="./multi_judge_results",
        help="Root output directory (sub-dirs gpt52/, deepseek/, gemini/ created automatically)",
    )
    parser.add_argument(
        "--config-gpt52", default=_DEFAULT_CONFIGS["gpt52"],
        help="Config YAML for GPT-5.2 judge",
    )
    parser.add_argument(
        "--config-deepseek", default=_DEFAULT_CONFIGS["deepseek"],
        help="Config YAML for DeepSeek judge",
    )
    parser.add_argument(
        "--config-gemini", default=_DEFAULT_CONFIGS["gemini"],
        help="Config YAML for Gemini judge",
    )
    parser.add_argument(
        "--judges", nargs="+",
        choices=["gpt52", "deepseek", "gemini"],
        default=["gpt52", "deepseek", "gemini"],
        help="Which judges to run (default: all three)",
    )
    parser.add_argument(
        "--parallel-files", type=int, default=1,
        help="Number of files to process concurrently within each judge (default: 1)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print first prompt without calling the LLM",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Set logging level to DEBUG",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ---- Collect input files ----
    input_files = collect_input_files(args.input)
    if not input_files:
        logger.info("No dialogue JSON files found at: %s", args.input)
        sys.exit(1)
    logger.info("Found %d input file(s)", input_files.__len__())

    # ---- Load configs ----
    config_map = {
        "gpt52":    args.config_gpt52,
        "deepseek": args.config_deepseek,
        "gemini":   args.config_gemini,
    }
    cfgs: dict[str, dict] = {}
    for label in args.judges:
        path = config_map[label]
        if not os.path.exists(path):
            logger.info("Config not found for %s: %s", label, path)
            sys.exit(1)
        cfgs[label] = load_config(path)

    # ---- Dry run (no API calls) ----
    if args.dry_run:
        from prompts import build_per_dialogue_prompt, build_global_prompt
        with open(input_files[0], encoding="utf-8") as f:
            data = json.load(f)
        scenario = data.get("vawg_scenarios", {})
        events = data.get("dialogues", [])
        if events:
            sys_p, user_p = build_per_dialogue_prompt(scenario, events[0], 6000)
            print("=" * 70)
            print("DRY RUN — Per-dialogue prompt for dialogue 0")
            print("=" * 70)
            print(user_p[:3000])
        return

    # ---- Launch judges in parallel (one thread per judge) ----
    logger.info(
        "Launching %d judge(s) in parallel: %s",
        len(args.judges), ", ".join(args.judges),
    )

    all_results: dict[str, list] = {}

    with ThreadPoolExecutor(max_workers=len(args.judges)) as outer_pool:
        future_to_label = {
            outer_pool.submit(
                _run_judge_all_files,
                label,
                cfgs[label],
                input_files,
                args.output_dir,
                args.parallel_files,
            ): label
            for label in args.judges
        }

        for fut in as_completed(future_to_label):
            label = future_to_label[fut]
            try:
                all_results[label] = fut.result()
                n_ok = sum(1 for r in all_results[label] if r is not None)
                _JudgeAdapter(
                    logging.getLogger(__name__), {"judge_label": label}
                ).info("Finished: %d/%d succeeded", n_ok, len(input_files))
            except Exception as exc:
                _JudgeAdapter(
                    logging.getLogger(__name__), {"judge_label": label}
                ).error("Judge thread raised: %s", exc)

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("MULTI-JUDGE PARALLEL RUN SUMMARY")
    print("=" * 60)
    for label in args.judges:
        results = all_results.get(label, [])
        n_ok = sum(1 for r in results if r is not None)
        out_dir = str(Path(args.output_dir) / label)
        print(f"  {label:<12}  {n_ok}/{len(input_files)} files OK   → {out_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
