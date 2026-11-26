#!/usr/bin/env python3
"""
Zero-shot LLM baseline for toxic message classification.

Prompts GPT-5.2 / Gemini / DeepSeek directly (no fine-tuning) on the same
test splits produced by build_dataset.py and build_multiclass_dataset.py.

Two tasks
---------
  binary     — toxic (1) vs. non-toxic (0)
  multiclass — 8-class behaviour type

Two settings
------------
  setting1 — target message only
  setting2 — preceding dialogue context + target message

Usage
-----
  python run_llm_baseline.py --config configs/llm_baseline_gpt52.yaml
  python run_llm_baseline.py --config configs/llm_baseline_gemini.yaml --tasks binary
  python run_llm_baseline.py --config configs/llm_baseline_deepseek.yaml --tasks multiclass --settings setting2
  python run_llm_baseline.py --aggregate_only --output_dir results_llm_baseline
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

# Reuse the shared LLM backend from LLM_judge/
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "LLM_judge"))
from llm_backend import build_backend  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Class definitions
# ---------------------------------------------------------------------------

BINARY_LABELS = {0: "non-toxic", 1: "toxic"}

MULTICLASS_LABELS: Dict[int, str] = {
    0: "non-toxic",
    1: "surveillance",
    2: "demand",
    3: "accusation",
    4: "manipulation",
    5: "ultimatum",
    6: "threat",
    7: "other",
}
MULTICLASS_NAME_LIST = [MULTICLASS_LABELS[i] for i in sorted(MULTICLASS_LABELS)]

# Definitions shown in the multiclass prompt
MULTICLASS_DEFINITIONS = """\
- non-toxic    : the message is sent by the victim, or is a neutral/benign message
- surveillance : the perpetrator monitors, tracks, or demands information about the victim's location, contacts, or activities
- demand       : the perpetrator issues a controlling instruction or requirement to the victim
- accusation   : the perpetrator accuses or blames the victim for something
- manipulation : the perpetrator uses emotional manipulation (e.g. guilt-tripping, gaslighting)
- ultimatum    : the perpetrator delivers a conditional threat ("if you … then I …")
- threat       : the perpetrator explicitly threatens violence, harm, or severe consequences
- other        : any other perpetrator behaviour (e.g. repair, refusal, escalation)"""

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

BINARY_SYSTEM = """\
You are an expert annotator for intimate partner violence research.
Your task is to classify a single message as either TOXIC or NON-TOXIC.

A message is TOXIC if it is produced by a perpetrator and contains coercive,
controlling, threatening, or abusive language toward the victim.
A message is NON-TOXIC if it is sent by the victim or contains no abusive behaviour.

Reply with EXACTLY one word: toxic  or  non-toxic.
Do not add any explanation."""

BINARY_USER_S1 = """\
Message: {text}

Classification:"""

BINARY_USER_S2 = """\
--- Dialogue context (preceding messages) ---
{context}

--- Target message to classify ---
{text}

Classification:"""

MULTICLASS_SYSTEM = """\
You are an expert annotator for intimate partner violence research.
Your task is to classify a single message into exactly one of the following behaviour categories:

{definitions}

Reply with EXACTLY one label from the list above (the word before the colon).
Do not add any explanation.""".format(definitions=MULTICLASS_DEFINITIONS)

MULTICLASS_USER_S1 = """\
Message: {text}

Category:"""

MULTICLASS_USER_S2 = """\
--- Dialogue context (preceding messages) ---
{context}

--- Target message to classify ---
{text}

Category:"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> List[Dict]:
    out: List[Dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def parse_binary(response: str) -> int:
    """Map LLM text response to 0/1 label."""
    r = response.strip().lower()
    # Allow slight variations: "toxic", "1-toxic", "label: toxic", etc.
    if re.search(r"\bnon.?toxic\b", r) or r in ("0", "non toxic"):
        return 0
    if re.search(r"\btoxic\b", r):
        return 1
    logger.warning("Unrecognised binary response %r — defaulting to non-toxic", response)
    return 0


def parse_multiclass(response: str) -> int:
    """Map LLM text response to 0-7 label."""
    r = response.strip().lower()
    # Try exact match first, then substring
    for idx, name in MULTICLASS_LABELS.items():
        if r == name:
            return idx
    for idx, name in MULTICLASS_LABELS.items():
        if name in r:
            return idx
    logger.warning("Unrecognised multiclass response %r — defaulting to non-toxic", response)
    return 0


def build_prompt(
    example: Dict,
    task: str,
    setting: str,
) -> Tuple[str, str]:
    """Return (system_prompt, user_prompt) for a single example."""
    text = example["text"]
    context = example.get("context") or ""

    if task == "binary":
        system = BINARY_SYSTEM
        user = (BINARY_USER_S2 if setting == "setting2" else BINARY_USER_S1).format(
            text=text, context=context
        )
    else:
        system = MULTICLASS_SYSTEM
        user = (MULTICLASS_USER_S2 if setting == "setting2" else MULTICLASS_USER_S1).format(
            text=text, context=context
        )
    return system, user


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------

class LLMBaseline:
    def __init__(self, args: argparse.Namespace, cfg: Dict):
        self.args = args
        self.cfg = cfg
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _run_inference(
        self,
        examples: List[Dict],
        backend,
        task: str,
        setting: str,
        checkpoint_path: Optional[Path] = None,
        checkpoint_every: int = 100,
    ) -> List[int]:
        # Resume from partial checkpoint if available
        preds: List[int] = []
        if checkpoint_path is not None and checkpoint_path.exists():
            try:
                preds = json.loads(checkpoint_path.read_text())
                logger.info(
                    "[%s/%s] Resumed from checkpoint: %d/%d done",
                    task, setting, len(preds), len(examples),
                )
            except Exception:
                preds = []

        parse_fn = parse_binary if task == "binary" else parse_multiclass
        start = len(preds)

        for i, ex in enumerate(
            tqdm(examples[start:], desc=f"{task}/{setting}", unit="msg", initial=start, total=len(examples)),
            start=start,
        ):
            system, user = build_prompt(ex, task, setting)
            for attempt in range(1, self.cfg.get("max_retries", 3) + 1):
                try:
                    response = backend.chat(system, user)
                    preds.append(parse_fn(response))
                    break
                except Exception as exc:
                    logger.warning("Attempt %d failed: %s", attempt, exc)
                    if attempt >= self.cfg.get("max_retries", 3):
                        logger.error("All retries exhausted; defaulting to 0")
                        preds.append(0)
                    else:
                        time.sleep(self.cfg.get("retry_delay", 5.0))

            # Incremental checkpoint save
            if checkpoint_path is not None and (i + 1) % checkpoint_every == 0:
                checkpoint_path.write_text(json.dumps(preds))

        return preds

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _binary_metrics(labels: List[int], preds: List[int]) -> Dict:
        return {
            "accuracy":        float(accuracy_score(labels, preds)),
            "f1_binary":       float(f1_score(labels, preds, pos_label=1, zero_division=0)),
            "f1_macro":        float(f1_score(labels, preds, average="macro", zero_division=0)),
            "f1_weighted":     float(f1_score(labels, preds, average="weighted", zero_division=0)),
            "precision_toxic": float(precision_score(labels, preds, pos_label=1, zero_division=0)),
            "recall_toxic":    float(recall_score(labels, preds, pos_label=1, zero_division=0)),
        }

    @staticmethod
    def _multiclass_metrics(labels: List[int], preds: List[int]) -> Dict:
        metrics: Dict = {
            "accuracy":    float(accuracy_score(labels, preds)),
            "f1_macro":    float(f1_score(labels, preds, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        }
        per_cls = f1_score(
            labels, preds,
            labels=list(MULTICLASS_LABELS.keys()),
            average=None,
            zero_division=0,
        )
        for idx, name in MULTICLASS_LABELS.items():
            metrics[f"f1_{name}"] = float(per_cls[idx]) if idx < len(per_cls) else 0.0
        return metrics

    # ------------------------------------------------------------------
    # Evaluate one (task, setting) pair
    # ------------------------------------------------------------------

    def evaluate(self, task: str, setting: str, backend) -> Dict:
        # Choose data directory
        data_dir = (
            Path(self.args.data_dir_multiclass) / setting
            if task == "multiclass"
            else Path(self.args.data_dir_binary) / setting
        )
        test_file = data_dir / "test.jsonl"
        if not test_file.exists():
            raise FileNotFoundError(f"Test file not found: {test_file}")

        examples = load_jsonl(test_file)
        max_samples = getattr(self.args, "max_samples", None)
        if max_samples is not None:
            examples = examples[:max_samples]
        labels = [e["label"] for e in examples]
        logger.info("[%s/%s] %d test examples loaded", task, setting, len(examples))

        # Check for cached predictions (skip_existing)
        run_dir = self.output_dir / f"{task}_{setting}"
        run_dir.mkdir(parents=True, exist_ok=True)
        preds_file = run_dir / "predictions.json"

        if self.cfg.get("skip_existing", True) and preds_file.exists():
            cached = json.loads(preds_file.read_text())
            if len(cached) == len(examples):
                logger.info("Loading cached predictions from %s", preds_file)
                preds = cached
            else:
                logger.info(
                    "Partial checkpoint (%d/%d); resuming inference",
                    len(cached), len(examples),
                )
                preds = self._run_inference(examples, backend, task, setting,
                                            checkpoint_path=preds_file)
                preds_file.write_text(json.dumps(preds))
        else:
            preds = self._run_inference(examples, backend, task, setting,
                                        checkpoint_path=preds_file)
            preds_file.write_text(json.dumps(preds))

        # Metrics
        if task == "binary":
            metrics = self._binary_metrics(labels, preds)
            target_names = ["non-toxic", "toxic"]
        else:
            metrics = self._multiclass_metrics(labels, preds)
            target_names = MULTICLASS_NAME_LIST

        report_str = classification_report(
            labels, preds, target_names=target_names, digits=4, zero_division=0,
        )
        logger.info("\n%s", report_str)

        model_name = self.cfg.get("model_name", "unknown")
        result = {
            "model":    model_name,
            "task":     task,
            "setting":  setting,
            "test_metrics": metrics,
            "classification_report": report_str,
            "n_examples": len(examples),
        }
        out_path = run_dir / "results.json"
        out_path.write_text(json.dumps(result, indent=2))
        logger.info("Results saved → %s", out_path)
        return result

    # ------------------------------------------------------------------
    # Run all requested (task, setting) pairs
    # ------------------------------------------------------------------

    def run_all(self) -> None:
        llm_cfg = self.cfg.get("llm", {})
        backend = build_backend(llm_cfg)
        all_results: List[Dict] = []

        try:
            for task in self.args.tasks:
                for setting in self.args.settings:
                    try:
                        r = self.evaluate(task, setting, backend)
                        all_results.append(r)
                    except Exception as exc:
                        import traceback
                        logger.error("FAILED task=%s setting=%s: %s", task, setting, exc)
                        traceback.print_exc()
        finally:
            backend.cleanup()

        summary_path = self.output_dir / "summary.json"
        summary_path.write_text(json.dumps(all_results, indent=2))
        logger.info("Summary saved → %s", summary_path)
        self._print_summary(all_results)

    # ------------------------------------------------------------------
    # Aggregate from existing result files
    # ------------------------------------------------------------------

    def aggregate_results(self) -> None:
        all_results: List[Dict] = []
        for f in sorted(self.output_dir.rglob("*/results.json")):
            try:
                all_results.append(json.loads(f.read_text()))
                logger.info("Loaded %s", f)
            except Exception as exc:
                logger.warning("Could not load %s: %s", f, exc)
        if not all_results:
            logger.warning("No results.json files found under %s", self.output_dir)
            return
        (self.output_dir / "summary.json").write_text(json.dumps(all_results, indent=2))
        self._print_summary(all_results)

    # ------------------------------------------------------------------
    # Pretty table
    # ------------------------------------------------------------------

    @staticmethod
    def _print_summary(all_results: List[Dict]) -> None:
        print(f"\n{'═'*100}")
        print("LLM BASELINE — ZERO-SHOT CLASSIFICATION SUMMARY")
        print(f"{'═'*100}")

        binary = [r for r in all_results if r.get("task") == "binary"]
        multi  = [r for r in all_results if r.get("task") == "multiclass"]

        if binary:
            print("\n── Binary (toxic vs. non-toxic) ──")
            hdr = f"{'Model':<30} {'Setting':<10} {'F1-bin':>8} {'Acc':>8} {'Prec':>8} {'Rec':>8}"
            print(hdr)
            print("─" * 70)
            for r in binary:
                tm = r.get("test_metrics", {})
                print(
                    f"{r['model']:<30} {r['setting']:<10}"
                    f" {tm.get('f1_binary', float('nan')):>8.4f}"
                    f" {tm.get('accuracy',  float('nan')):>8.4f}"
                    f" {tm.get('precision_toxic', float('nan')):>8.4f}"
                    f" {tm.get('recall_toxic',    float('nan')):>8.4f}"
                )

        if multi:
            print("\n── Multiclass (8 classes) ──")
            hdr = f"{'Model':<30} {'Setting':<10} {'F1-mac':>8} {'F1-wgt':>8} {'Acc':>8}"
            print(hdr)
            print("─" * 60)
            for r in multi:
                tm = r.get("test_metrics", {})
                print(
                    f"{r['model']:<30} {r['setting']:<10}"
                    f" {tm.get('f1_macro',    float('nan')):>8.4f}"
                    f" {tm.get('f1_weighted', float('nan')):>8.4f}"
                    f" {tm.get('accuracy',    float('nan')):>8.4f}"
                )

        print(f"\n{'═'*100}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zero-shot LLM baseline for toxic message classification.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str,
        help="Path to YAML config (llm backend + processing settings).",
    )
    parser.add_argument(
        "--tasks", nargs="+", default=["binary", "multiclass"],
        choices=["binary", "multiclass"],
        help="Which task(s) to run (default: both).",
    )
    parser.add_argument(
        "--settings", nargs="+", default=["setting1", "setting2"],
        choices=["setting1", "setting2"],
        help="Which input setting(s) to evaluate (default: both).",
    )
    parser.add_argument(
        "--data_dir_binary", type=str, default="data",
        help="Root dir for binary JSONL splits (default: data/).",
    )
    parser.add_argument(
        "--data_dir_multiclass", type=str, default="data_multiclass",
        help="Root dir for multiclass JSONL splits (default: data_multiclass/).",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results_llm_baseline",
        help="Directory to write per-(task,setting) results (default: results_llm_baseline/).",
    )
    parser.add_argument(
        "--aggregate_only", action="store_true",
        help="Skip inference; just collect existing results.json files and print summary.",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Limit test set to first N examples (default: use all).",
    )
    args = parser.parse_args()

    # Load config
    cfg: Dict = {}
    if args.config:
        with open(args.config, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}

    # Merge top-level processing keys into cfg for convenience
    processing_cfg = cfg.get("processing", {})
    cfg.setdefault("max_retries", processing_cfg.get("max_retries", 3))
    cfg.setdefault("retry_delay", processing_cfg.get("retry_delay", 5.0))
    cfg.setdefault("skip_existing", processing_cfg.get("skip_existing", True))
    # Expose model name at top level for summary
    cfg.setdefault("model_name", cfg.get("llm", {}).get("model_name", "unknown"))

    # Override output_dir from config if not set on CLI
    if args.output_dir == "results_llm_baseline" and "output_dir" in cfg:
        args.output_dir = cfg["output_dir"]

    baseline = LLMBaseline(args, cfg)

    if args.aggregate_only:
        baseline.aggregate_results()
    else:
        if not args.config:
            parser.error("--config is required unless --aggregate_only is set.")
        baseline.run_all()


if __name__ == "__main__":
    main()
