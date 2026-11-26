#!/usr/bin/env python3
"""
LLM zero-shot baseline for perpetrator-victim relationship classification.

Directly prompts GPT-5.2, Gemini, or DeepSeek to classify each event-level
dialogue into one of the 5 relationship categories without any fine-tuning.

Usage:
  python llm_baseline_clf.py --model_name gpt-5.2
  python llm_baseline_clf.py --model_name gemini-3.1-flash-lite
  python llm_baseline_clf.py --model_name deepseek-v4-flash
  python llm_baseline_clf.py --compare   # print comparison table across all LLM results
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[3]
JUDGE_DIR = REPO_ROOT / "LLM_judge"
sys.path.insert(0, str(JUDGE_DIR))

from llm_backend import LLMBackend, build_backend  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONFIG_PATH = Path(__file__).parent / "configs" / "llm_baselines.yaml"

SYSTEM_PROMPT = """\
You are a relationship classifier. Given a conversation between two or more people, \
identify the perpetrator-victim relationship type.

Respond with EXACTLY ONE of the following labels — nothing else:
  intimate_partner
  ex_partner
  acquaintance
  professional
  family_or_other

Definitions:
- intimate_partner : current romantic partner / boyfriend / girlfriend
- ex_partner       : former romantic partner
- acquaintance     : someone known casually (friend, neighbour, classmate, online contact)
- professional     : colleague, boss, client, or other work/service relationship
- family_or_other  : family member or a relationship that does not fit the above
"""

USER_TEMPLATE = """\
Conversation:
{dialogue}

Relationship label:"""

LABELS = [
    "intimate_partner",
    "ex_partner",
    "acquaintance",
    "professional",
    "family_or_other",
]
LABEL2ID = {lbl: idx for idx, lbl in enumerate(LABELS)}
FALLBACK_LABEL = "intimate_partner"  # used when model response is unparseable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_test_records(cfg: dict) -> List[dict]:
    test_path = REPO_ROOT / cfg["test_file"]
    records: List[dict] = []
    with open(test_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info("Loaded %d test records from %s", len(records), test_path)
    return records


def parse_label(raw: str) -> str:
    """Extract the first matching label from the model response."""
    raw_lower = raw.lower().strip()
    for lbl in LABELS:
        if lbl in raw_lower:
            return lbl
    # Secondary: partial matching (e.g. "intimate" → intimate_partner)
    partial_map = {
        "intimate": "intimate_partner",
        "ex_partner": "ex_partner",
        "ex partner": "ex_partner",
        "ex-partner": "ex_partner",
        "acquaint": "acquaintance",
        "profession": "professional",
        "family": "family_or_other",
        "other": "family_or_other",
    }
    for key, lbl in partial_map.items():
        if key in raw_lower:
            return lbl
    logger.warning("Could not parse label from response: %r  → fallback to %s", raw[:120], FALLBACK_LABEL)
    return FALLBACK_LABEL


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    labels: List[str],
) -> dict:
    label_ids = list(range(len(labels)))
    report = classification_report(
        y_true, y_pred, labels=label_ids, target_names=labels,
        output_dict=True, zero_division=0,
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "per_class": {
            lbl: {
                "f1": report[lbl]["f1-score"],
                "precision": report[lbl]["precision"],
                "recall": report[lbl]["recall"],
                "support": report[lbl]["support"],
            }
            for lbl in labels
        },
    }


def safe_model_filename(model_name: str) -> str:
    return model_name.replace("/", "__").replace(":", "_")


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------

class LLMRelationClassifier:
    def __init__(self, model_cfg: dict, global_cfg: dict) -> None:
        self.model_name: str = model_cfg["name"]
        self.labels: List[str] = global_cfg["labels"]
        self.results_dir: Path = REPO_ROOT / global_cfg["results_dir"]
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Build backend using the same factory as LLM_judge
        backend_cfg = {k: v for k, v in model_cfg.items() if k != "name"}
        self.backend: LLMBackend = build_backend(backend_cfg)

    def classify(self, records: List[dict]) -> Tuple[List[int], List[int], List[dict]]:
        """Classify all records. Returns (y_true, y_pred, detailed_rows)."""
        y_true: List[int] = []
        y_pred: List[int] = []
        rows: List[dict] = []

        for rec in tqdm(records, desc=self.model_name, unit="sample"):
            true_label = rec["label"]
            true_id = rec["label_id"]

            user_prompt = USER_TEMPLATE.format(dialogue=rec["text"])
            raw = self.backend.chat(SYSTEM_PROMPT, user_prompt)
            pred_label = parse_label(raw)
            pred_id = LABEL2ID[pred_label]

            y_true.append(true_id)
            y_pred.append(pred_id)
            rows.append(
                {
                    "scenario_id": rec.get("scenario_id", ""),
                    "event_id": rec.get("event_id", ""),
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "correct": true_label == pred_label,
                    "raw_response": raw[:200],
                }
            )

        return y_true, y_pred, rows

    def run(self, records: List[dict]) -> dict:
        logger.info(
            "=" * 70 + "\nModel: %s  (zero-shot LLM baseline)\n" + "=" * 70,
            self.model_name,
        )

        y_true, y_pred, rows = self.classify(records)
        metrics = compute_metrics(y_true, y_pred, self.labels)

        result = {
            "model_name": self.model_name,
            "type": "llm_zero_shot",
            "test_samples": len(records),
            "test_metrics": metrics,
            "predictions": rows,
        }

        out_path = self.results_dir / f"{safe_model_filename(self.model_name)}_llm_results.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info("Results saved → %s", out_path)

        # Pretty summary
        m = metrics
        logger.info(
            "\n╔══════════════════════════════════════════╗\n"
            "║  RESULTS: %-32s║\n"
            "╠══════════════════════════════════════════╣\n"
            "║  Test accuracy  : %-24.4f║\n"
            "║  Test macro-F1  : %-24.4f║\n"
            "║  Test wt'd-F1   : %-24.4f║\n"
            "║  Test precision : %-24.4f║\n"
            "║  Test recall    : %-24.4f║\n"
            "╚══════════════════════════════════════════╝",
            self.model_name[-32:],
            m["accuracy"], m["macro_f1"], m["weighted_f1"],
            m["macro_precision"], m["macro_recall"],
        )
        logger.info("  Per-class test F1:")
        for lbl in self.labels:
            pc = m["per_class"][lbl]
            logger.info(
                "    %-28s F1=%.4f  P=%.4f  R=%.4f  n=%d",
                lbl, pc["f1"], pc["precision"], pc["recall"], pc["support"],
            )

        self.backend.cleanup()
        return result


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def compare_all(results_dir: Path, labels: List[str]) -> None:
    llm_files = sorted(results_dir.glob("*_llm_results.json"))
    if not llm_files:
        logger.warning("No LLM result files found in %s", results_dir)
        return

    rows = []
    for f in llm_files:
        d = json.load(open(f))
        m = d["test_metrics"]
        rows.append(
            {
                "Model": d["model_name"],
                "Acc": m["accuracy"],
                "Prec": m["macro_precision"],
                "Rec": m["macro_recall"],
                "MacroF1": m["macro_f1"],
                "WtF1": m["weighted_f1"],
            }
        )

    header = f"{'Model':<30} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'MacroF1':>8} {'WtF1':>6}"
    sep = "-" * len(header)
    print("\nLLM Zero-Shot Baselines — Test Set\n" + sep)
    print(header)
    print(sep)
    for r in rows:
        print(
            f"{r['Model']:<30} {r['Acc']:>6.4f} {r['Prec']:>6.4f} "
            f"{r['Rec']:>6.4f} {r['MacroF1']:>8.4f} {r['WtF1']:>6.4f}"
        )
    print(sep)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM zero-shot baseline for perpetrator-victim relation classification.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python llm_baseline_clf.py --model_name gpt-5.2\n"
            "  python llm_baseline_clf.py --model_name deepseek-v4-flash\n"
            "  python llm_baseline_clf.py --model_name gemini-3.1-flash-lite\n"
            "  python llm_baseline_clf.py --compare\n"
        ),
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Name key from configs/llm_baselines.yaml (e.g. gpt-5.2).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Override path to llm_baselines.yaml.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Print comparison table across all completed LLM results and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global CONFIG_PATH
    if args.config:
        CONFIG_PATH = Path(args.config)

    cfg = load_config()
    results_dir = REPO_ROOT / cfg["results_dir"]

    if args.compare:
        compare_all(results_dir, cfg["labels"])
        return

    if not args.model_name:
        parser_err = "Specify --model_name or --compare."
        logger.error(parser_err)
        sys.exit(1)

    # Find matching model config
    model_cfg: Optional[dict] = None
    for mc in cfg["models"]:
        if mc["name"] == args.model_name:
            model_cfg = mc
            break
    if model_cfg is None:
        logger.error(
            "Model %r not found in config. Available: %s",
            args.model_name,
            [m["name"] for m in cfg["models"]],
        )
        sys.exit(1)

    records = load_test_records(cfg)
    classifier = LLMRelationClassifier(model_cfg, cfg)
    classifier.run(records)


if __name__ == "__main__":
    main()
