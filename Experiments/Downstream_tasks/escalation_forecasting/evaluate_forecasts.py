#!/usr/bin/env python3
"""
Aggregate evaluation for escalation forecasting predictions.

Computes, for one or more predictions.jsonl files (encoder or LLM output):

  Level metrics    — accuracy, macro-F1, MAE, quadratic-weighted kappa
  Jump metrics     — derived jump = (pred > cur_level): F1, precision, recall
  Trivial baselines (from the SAME gold labels, always reported):
    persistence — predict cur_level (copy the last observed level)
    majority    — predict the training majority class
    markov1     — argmax of the train-split transition matrix P(next | cur)
  Early-warning analysis (per test scenario):
    detection rate — scenarios whose first severe (level-4) event was
                     preceded by at least one severe alert (pred >= 4)
    lead events / lead days — distance between the first correct alert and
                     the first severe event
    false-alert rate — alerts in scenarios that never reach level 4

Usage
-----
  python evaluate_forecasts.py --pred results/*/predictions.jsonl
  python evaluate_forecasts.py --pred results_llm_baseline/gpt52/direct/predictions.jsonl
"""

import argparse
import json
import logging
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SEVERE_LEVEL = 4


def load_jsonl(path: Path) -> List[Dict]:
    out: List[Dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def parse_ts(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Metric blocks
# ---------------------------------------------------------------------------

def level_metrics(gold: List[int], pred: List[int]) -> Dict[str, float]:
    g, p = np.array(gold), np.array(pred)
    return {
        "accuracy": float(accuracy_score(g, p)),
        "f1_macro": float(f1_score(g, p, average="macro", zero_division=0)),
        "mae": float(np.mean(np.abs(g - p))),
        "qwk": float(cohen_kappa_score(g, p, weights="quadratic")),
    }


def jump_metrics(gold_jump: List[int], pred_jump: List[int]) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(gold_jump, pred_jump)),
        "f1_jump": float(f1_score(gold_jump, pred_jump, pos_label=1, zero_division=0)),
        "precision_jump": float(precision_score(gold_jump, pred_jump, pos_label=1, zero_division=0)),
        "recall_jump": float(recall_score(gold_jump, pred_jump, pos_label=1, zero_division=0)),
    }


def early_warning_analysis(records: List[Dict]) -> Dict:
    """Alert = predicted next level >= SEVERE_LEVEL, evaluated per scenario."""
    by_scenario: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        by_scenario[r["scenario_id"]].append(r)

    detected, missed, lead_events, lead_days = 0, 0, [], []
    false_alert_scenarios, clean_scenarios = 0, 0
    for sid, recs in by_scenario.items():
        recs = sorted(recs, key=lambda r: r["target_pos"])
        gold_severe = [i for i, r in enumerate(recs)
                       if r["gold_next_level"] >= SEVERE_LEVEL]
        alerts = [i for i, r in enumerate(recs) if r["pred"] >= SEVERE_LEVEL]
        if gold_severe:
            first_severe = gold_severe[0]
            early_alerts = [i for i in alerts if i <= first_severe]
            if early_alerts:
                detected += 1
                first_alert = early_alerts[0]
                lead_events.append(first_severe - first_alert)
                t_alert = parse_ts(recs[first_alert]["target_timestamp"])
                t_severe = parse_ts(recs[first_severe]["target_timestamp"])
                if t_alert and t_severe:
                    lead_days.append((t_severe - t_alert).total_seconds() / 86400.0)
            else:
                missed += 1
        else:
            if alerts:
                false_alert_scenarios += 1
            else:
                clean_scenarios += 1

    n_severe = detected + missed
    n_benign = false_alert_scenarios + clean_scenarios
    return {
        "n_scenarios_with_severe": n_severe,
        "detection_rate": detected / n_severe if n_severe else None,
        "mean_lead_events": float(np.mean(lead_events)) if lead_events else None,
        "median_lead_events": float(np.median(lead_events)) if lead_events else None,
        "mean_lead_days": float(np.mean(lead_days)) if lead_days else None,
        "n_scenarios_without_severe": n_benign,
        "false_alert_rate": (false_alert_scenarios / n_benign) if n_benign else None,
    }


# ---------------------------------------------------------------------------
# Trivial baselines
# ---------------------------------------------------------------------------

def markov_transition(train_examples: List[Dict]) -> Dict[int, int]:
    trans: Dict[int, Counter] = defaultdict(Counter)
    for ex in train_examples:
        trans[ex["cur_level"]][ex["next_level"]] += 1
    return {cur: counts.most_common(1)[0][0] for cur, counts in trans.items()}


def baseline_predictions(records: List[Dict], train_examples: List[Dict]) -> Dict[str, List[int]]:
    majority = Counter(e["next_level"] for e in train_examples).most_common(1)[0][0]
    markov = markov_transition(train_examples)
    return {
        "persistence": [r["cur_level"] for r in records],
        "majority": [majority] * len(records),
        "markov1": [markov.get(r["cur_level"], majority) for r in records],
    }


# ---------------------------------------------------------------------------
# Evaluation driver
# ---------------------------------------------------------------------------

def evaluate_prediction_file(path: Path, train_examples: List[Dict]) -> Dict:
    records = load_jsonl(path)
    task = records[0].get("task", "level")
    name = str(path.parent.relative_to(path.parents[2])) if len(path.parents) > 2 else str(path.parent)

    result: Dict = {"predictions": str(path), "name": name, "n": len(records), "task": task}
    gold_jump = [r["gold_jump"] for r in records]

    if task == "level":
        gold = [r["gold_next_level"] for r in records]
        pred = [r["pred"] for r in records]
        result["level"] = level_metrics(gold, pred)
        result["jump_derived"] = jump_metrics(
            gold_jump, [int(r["pred"] > r["cur_level"]) for r in records])
        result["early_warning"] = early_warning_analysis(records)
    else:  # jump-task predictions
        result["jump"] = jump_metrics(gold_jump, [r["pred"] for r in records])

    if train_examples and task == "level":
        result["baselines"] = {}
        for bname, bpred in baseline_predictions(records, train_examples).items():
            gold = [r["gold_next_level"] for r in records]
            result["baselines"][bname] = {
                **level_metrics(gold, bpred),
                "jump_f1": f1_score(
                    gold_jump,
                    [int(p > r["cur_level"]) for p, r in zip(bpred, records)],
                    pos_label=1, zero_division=0),
            }
    return result


def print_table(results: List[Dict]) -> None:
    W = 118
    print(f"\n{'═' * W}")
    print("ESCALATION FORECASTING — EVALUATION SUMMARY (next-event level)")
    print(f"{'═' * W}")
    print(f"{'System':<48} {'Acc':>7} {'F1-mac':>7} {'MAE':>7} {'QWK':>7}"
          f" {'JumpF1':>7} {'Detect':>7} {'LeadEv':>7} {'FalseA':>7}")
    print("─" * W)
    printed_baselines = False
    for res in results:
        if "level" not in res:
            continue
        lm, jm, ew = res["level"], res["jump_derived"], res["early_warning"]
        fmt = lambda v: f"{v:>7.3f}" if isinstance(v, (int, float)) else f"{'—':>7}"
        print(f"{res['name']:<48} {lm['accuracy']:>7.3f} {lm['f1_macro']:>7.3f}"
              f" {lm['mae']:>7.3f} {lm['qwk']:>7.3f} {jm['f1_jump']:>7.3f}"
              f" {fmt(ew['detection_rate'])} {fmt(ew['mean_lead_events'])}"
              f" {fmt(ew['false_alert_rate'])}")
        if not printed_baselines and "baselines" in res:
            for bname, bm in res["baselines"].items():
                print(f"{'  baseline: ' + bname:<48} {bm['accuracy']:>7.3f}"
                      f" {bm['f1_macro']:>7.3f} {bm['mae']:>7.3f} {bm['qwk']:>7.3f}"
                      f" {bm['jump_f1']:>7.3f} {'—':>7} {'—':>7} {'—':>7}")
            printed_baselines = True
    for res in results:
        if "jump" in res:
            jm = res["jump"]
            print(f"{res['name'] + ' [jump task]':<48} {jm['accuracy']:>7.3f}"
                  f" {'—':>7} {'—':>7} {'—':>7} {jm['f1_jump']:>7.3f}"
                  f" {'—':>7} {'—':>7} {'—':>7}")
    print(f"{'═' * W}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate escalation-forecast predictions with baselines and early-warning analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python evaluate_forecasts.py --pred results/*/predictions.jsonl
  python evaluate_forecasts.py --pred results_llm_baseline/gpt52/direct/predictions.jsonl --out eval_gpt52.json
""",
    )
    parser.add_argument("--pred", nargs="+", required=True,
                        help="One or more predictions.jsonl files")
    parser.add_argument("--data_dir", default="data",
                        help="Dataset dir (train.jsonl used for baselines)")
    parser.add_argument("--out", default="evaluation_summary.json")
    args = parser.parse_args()

    train_path = Path(args.data_dir) / "train.jsonl"
    train_examples = load_jsonl(train_path) if train_path.exists() else []
    if not train_examples:
        logger.warning("No train.jsonl found — trivial baselines skipped")

    results = []
    for p in args.pred:
        path = Path(p)
        if not path.exists():
            logger.warning("Missing predictions file: %s", path)
            continue
        results.append(evaluate_prediction_file(path, train_examples))

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=float)
    logger.info("Summary → %s", args.out)
    print_table(results)


if __name__ == "__main__":
    main()
