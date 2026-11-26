#!/usr/bin/env python3
"""
Standalone Detoxify evaluation script.

Runs Detoxify on all *_dialogues.json files in an input folder and writes
per-file results plus a batch summary — without needing the full LLM judge.

Output per file: {id}_detoxify_results.json
  {
    "scenario_id": "101915",
    "num_dialogues_scored": 5,
    "per_dialogue_scores": [
      {"event_id": "...", "dialogue_index": 0, "detoxify_scores": {...}},
      ...
    ],
    "summary": {
      "toxicity":       {"mean": 0.12, "max": 0.45, "min": 0.01, "num_scored": 5},
      ...
    }
  }

Output batch file: detoxify_batch_summary.json

Usage examples:
  python run_detoxify.py \\
      --input  ../data/ConVAWG/dialogues \\
      --output ../Generation_Results/detoxify_results

  python run_detoxify.py \\
      --input  ../data/ConVAWG/dialogues \\
      --output ../Generation_Results/detoxify_results \\
      --model  unbiased \\
      --device cuda:0 \\
      --skip-existing
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

# Allow running directly from LLM_judge/ without installing
sys.path.insert(0, str(Path(__file__).parent))
from detoxify_scorer import DetoxifyScorer, compute_detoxify_summary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_dialogue_text(event: Dict) -> str:
    """Extract plain-text dialogue string from a pipeline event dict."""
    text = event.get("caa_rewritten_dialogue") or event.get("enforced_dialogue")
    if text:
        return str(text)
    turns = event.get("turns", [])
    if turns:
        return "\n".join(
            f"{t.get('speaker', '?')}: {t.get('text', '')}" for t in turns
        )
    return ""


def score_file(
    input_path: Path,
    scorer: DetoxifyScorer,
) -> Dict:
    """Score all dialogues in one *_dialogues.json file."""
    with input_path.open(encoding="utf-8") as fh:
        data = json.load(fh)

    # Support both list-of-events and dict-with-events formats
    if isinstance(data, list):
        events = data
        meta: Dict = {}
    elif isinstance(data, dict):
        events = data.get("dialogues", data.get("events", []))
        meta = {k: v for k, v in data.items() if k not in ("dialogues", "events")}
    else:
        logger.warning("Unexpected JSON structure in %s — skipping.", input_path.name)
        return {}

    per_dialogue_scores: List[Dict] = []
    for idx, event in enumerate(events):
        text = _get_dialogue_text(event)
        scores = scorer.score_text(text) if text else {label: None for label in scorer.labels}
        entry: Dict = {
            "event_id":      event.get("event_id") or event.get("dialogue_id", f"idx_{idx}"),
            "dialogue_index": idx,
            "detoxify_scores": scores,
        }
        # Carry through useful metadata if present
        for key in ("escalation_level", "crime_type", "dialogue_source"):
            if key in event:
                entry[key] = event[key]
        per_dialogue_scores.append(entry)

    summary = compute_detoxify_summary(per_dialogue_scores)

    scenario_id = input_path.stem.replace("_dialogues", "")
    result = {
        "scenario_id":          scenario_id,
        "source_file":          input_path.name,
        "model_type":           scorer.model_type,
        "device":               scorer.device,
        "num_dialogues_scored": len(per_dialogue_scores),
        "per_dialogue_scores":  per_dialogue_scores,
        "summary":              summary,
    }
    # Merge top-level metadata from source file (e.g. victim_name, perpetrator_name)
    for key in ("victim_name", "perpetrator_name", "scenario_id"):
        if key in meta:
            result[key] = meta[key]

    return result


def aggregate_batch(all_results: List[Dict], model_type: str) -> Dict:
    """Compute per-label statistics across all scored files."""
    label_accum: Dict[str, List[float]] = {}

    for res in all_results:
        summary = res.get("summary") or {}
        for label, stats in summary.items():
            if stats and stats.get("mean") is not None:
                label_accum.setdefault(label, []).append(stats["mean"])

    aggregate: Dict[str, Dict] = {}
    for label, means in label_accum.items():
        aggregate[label] = {
            "mean_of_means": round(sum(means) / len(means), 4),
            "max_of_means":  round(max(means), 4),
            "min_of_means":  round(min(means), 4),
            "num_files":     len(means),
        }

    return {
        "model_type":        model_type,
        "num_files_scored":  len(all_results),
        "per_label_stats":   aggregate,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Detoxify scoring on *_dialogues.json files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_detoxify.py \\
      --input  ../data/ConVAWG/dialogues \\
      --output ../Generation_Results/detoxify_results

  python run_detoxify.py \\
      --input  ../data/ConVAWG/dialogues \\
      --output ../Generation_Results/detoxify_results \\
      --model  unbiased --device cuda:0 --skip-existing
""",
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Folder containing *_dialogues.json files.",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Folder to write *_detoxify_results.json files.",
    )
    parser.add_argument(
        "--model",
        default="original",
        choices=["original", "unbiased", "multilingual"],
        help="Detoxify model type (default: original).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help='PyTorch device string, e.g. "cpu", "cuda", "cuda:0" (default: cpu).',
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=20000,
        help="Max characters per dialogue text before truncation (default: 20000).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files whose output already exists (residual mode).",
    )
    parser.add_argument(
        "--pattern",
        default="*_dialogues.json",
        help='Input file glob pattern (default: "*_dialogues.json").',
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.is_dir():
        logger.error("Input directory does not exist: %s", input_dir)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect input files
    all_input = sorted(input_dir.glob(args.pattern))
    if not all_input:
        logger.error("No files matching '%s' found in %s", args.pattern, input_dir)
        sys.exit(1)

    if args.skip_existing:
        to_process = [
            p for p in all_input
            if not (output_dir / (p.stem.replace("_dialogues", "") + "_detoxify_results.json")).exists()
        ]
        logger.info(
            "Skip-existing mode: %d/%d files remaining.",
            len(to_process), len(all_input),
        )
    else:
        to_process = all_input

    if not to_process:
        logger.info("All files already processed. Nothing to do.")
        sys.exit(0)

    # Build scorer (lazy — model loads on first predict call)
    scorer = DetoxifyScorer(
        model_type=args.model,
        device=args.device,
        max_chars=args.max_chars,
    )

    all_results: List[Dict] = []
    failed: List[str] = []

    for input_path in tqdm(to_process, desc="Scoring dialogues", unit="file"):
        out_name = input_path.stem.replace("_dialogues", "") + "_detoxify_results.json"
        out_path = output_dir / out_name
        try:
            result = score_file(input_path, scorer)
            if not result:
                failed.append(input_path.name)
                continue
            with out_path.open("w", encoding="utf-8") as fh:
                json.dump(result, fh, indent=2, ensure_ascii=False)
            all_results.append(result)
        except Exception as exc:
            logger.error("Failed to score %s: %s", input_path.name, exc)
            failed.append(input_path.name)

    # Write batch summary
    if all_results:
        batch = aggregate_batch(all_results, args.model)
        batch["failed_files"] = failed
        batch_path = output_dir / "detoxify_batch_summary.json"
        with batch_path.open("w", encoding="utf-8") as fh:
            json.dump(batch, fh, indent=2, ensure_ascii=False)
        logger.info("Batch summary written to %s", batch_path)

    logger.info(
        "Done. Scored: %d  Failed: %d",
        len(all_results), len(failed),
    )
    if failed:
        logger.warning("Failed files: %s", failed)

    scorer.cleanup()


if __name__ == "__main__":
    main()
