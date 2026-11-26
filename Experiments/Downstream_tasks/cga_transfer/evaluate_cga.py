#!/usr/bin/env python3
"""
Evaluate a ConVAWG-trained escalation forecaster zero-shot on CGA derailment forecasting.

Loads a sequence-classification checkpoint (a ConVAWG-trained five-class
`level` forecaster from `../escalation_forecasting/results/<model>_level/checkpoint-*`)
and runs inference on a CGA test set built by build_dataset.py. The derailment
score is P(next level >= 3): the summed softmax probability of levels 3 and 4
(`--score_classes 3 4`, the default). Predicting a high next escalation level is
used as a zero-shot proxy for
  CGA derailment = "conversation will end in a personal attack".
These are related but distinct constructs — see README.md for the caveats.

Input rendering matches escalation_forecasting/train_forecaster.py exactly:
history blocks joined by a blank line, tokenizer truncating from the LEFT so
the most recent comments are always retained.

Reported metrics (positive class = derailment)
----------------------------------------------
  * argmax (0.5-threshold): accuracy, F1, macro-F1, precision, recall
  * threshold-free: AUPRC, AUROC  (primary numbers for zero-shot transfer —
    the ConVAWG decision threshold is not calibrated for CGA's label prior)
  * dev-calibrated: same metrics at the threshold maximising F1 on the CGA
    dev split (reported separately; uses no CGA training data)
  * majority baseline computed from the CGA train split

Outputs: <output_dir>/predictions.jsonl and <output_dir>/results.json
"""

import argparse
import json
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

EVENT_SEPARATOR = "\n\n"  # must match escalation_forecasting/train_forecaster.py

# Fallback tokenizer sources when the checkpoint dir has no tokenizer files
# (train_forecaster.py checkpoints store only model weights + config).
MODEL_TYPE_TO_TOKENIZER = {
    "deberta-v2": "microsoft/deberta-v3-base",
    "longformer": "allenai/longformer-base-4096",
    "modernbert": "answerdotai/ModernBERT-base",
    "bert": "bert-base-uncased",
    "roberta": "roberta-base",
}

MODEL_TYPE_MAX_LENGTH = {
    "deberta-v2": 512,
    "longformer": 4096,
    "modernbert": 4096,
    "bert": 512,
    "roberta": 512,
}


def load_jsonl(path: Path) -> List[Dict]:
    examples: List[Dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def binary_metrics(gold: List[int], pred: List[int],
                   probs: Optional[np.ndarray] = None) -> Dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(gold, pred)),
        "f1_derail": float(f1_score(gold, pred, pos_label=1, zero_division=0)),
        "f1_macro": float(f1_score(gold, pred, average="macro", zero_division=0)),
        "precision_derail": float(precision_score(gold, pred, pos_label=1, zero_division=0)),
        "recall_derail": float(recall_score(gold, pred, pos_label=1, zero_division=0)),
    }
    if probs is not None:
        try:
            metrics["auprc"] = float(average_precision_score(gold, probs))
            metrics["auroc"] = float(roc_auc_score(gold, probs))
        except ValueError:
            pass
    return metrics


class CGATransferEvaluator:
    """Runs a jump-forecaster checkpoint on a CGA forecasting test set."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.checkpoint = Path(args.checkpoint)
        self.data_dir = Path(args.data_dir)
        if args.output_dir:
            self.output_dir = Path(args.output_dir)
        else:
            corpus = self.data_dir.name
            tag = "_".join(p for p in self.checkpoint.parts[-2:])
            self.output_dir = Path("results") / f"{tag}__{corpus}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self) -> Tuple[torch.nn.Module, "AutoTokenizer", int]:
        logger.info("Loading checkpoint: %s", self.checkpoint)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.checkpoint, dtype=torch.float32)
        model.to(self.device).eval()
        if max(self.args.score_classes) >= model.config.num_labels:
            raise ValueError(
                f"--score_classes {self.args.score_classes} out of range for "
                f"num_labels={model.config.num_labels}. Use the default 3 4 "
                f"for 5-class *_level checkpoints (P(next level >= 3)), or "
                f"--score_classes 1 for binary *_jump checkpoints.")

        model_type = model.config.model_type
        tok_source = self.args.tokenizer
        if tok_source is None:
            has_tokenizer_files = any(
                (self.checkpoint / name).exists()
                for name in ("tokenizer.json", "tokenizer_config.json",
                             "vocab.txt", "spm.model"))
            if has_tokenizer_files:
                tok_source = str(self.checkpoint)
            else:
                tok_source = MODEL_TYPE_TO_TOKENIZER.get(model_type)
                if tok_source is None:
                    raise ValueError(
                        f"No tokenizer in checkpoint and no fallback for "
                        f"model_type={model_type!r}; pass --tokenizer.")
        tokenizer = AutoTokenizer.from_pretrained(tok_source)
        tokenizer.truncation_side = "left"  # keep the most recent comments
        logger.info("Tokenizer: %s (truncation_side=left)", tok_source)

        max_length = self.args.max_length or MODEL_TYPE_MAX_LENGTH.get(model_type, 512)
        max_length = min(max_length, getattr(tokenizer, "model_max_length", max_length))
        logger.info("model_type=%s  max_length=%d  device=%s",
                    model_type, max_length, self.device)
        return model, tokenizer, max_length

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def render_history(self, example: Dict) -> str:
        blocks = example["history"]
        if self.args.history_events > 0:
            blocks = blocks[-self.args.history_events:]
        return EVENT_SEPARATOR.join(blocks)

    def predict_probs(self, model, tokenizer, max_length: int,
                      examples: List[Dict], desc: str) -> np.ndarray:
        probs: List[float] = []
        bsz = self.args.batch_size
        with torch.no_grad():
            for start in tqdm(range(0, len(examples), bsz), desc=desc, unit="batch"):
                batch = examples[start:start + bsz]
                texts = [self.render_history(e) for e in batch]
                enc = tokenizer(texts, max_length=max_length, truncation=True,
                                padding=True, return_tensors="pt").to(self.device)
                logits = model(**enc).logits.float()
                soft = torch.softmax(logits, dim=-1)
                score = soft[:, self.args.score_classes].sum(dim=-1)
                probs.extend(score.cpu().tolist())
        return np.array(probs)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def best_f1_threshold(gold: List[int], probs: np.ndarray) -> float:
        candidates = np.unique(np.round(probs, 4))
        best_t, best_f1 = 0.5, -1.0
        for t in candidates:
            f1 = f1_score(gold, (probs >= t).astype(int), pos_label=1, zero_division=0)
            if f1 > best_f1:
                best_t, best_f1 = float(t), float(f1)
        return best_t

    def majority_baseline(self, test_gold: List[int]) -> Optional[Dict]:
        train_path = self.data_dir / "train.jsonl"
        if not train_path.exists():
            return None
        train_labels = [e["jump"] for e in load_jsonl(train_path)]
        majority = Counter(train_labels).most_common(1)[0][0]
        pred = [majority] * len(test_gold)
        return {
            "majority_class": int(majority),
            "train_label_dist": dict(sorted(Counter(train_labels).items())),
            **binary_metrics(test_gold, pred),
        }

    def run(self) -> Dict:
        test_path = self.data_dir / f"{self.args.split}.jsonl"
        examples = load_jsonl(test_path)
        if self.args.limit:
            examples = examples[: self.args.limit]
        logger.info("Evaluating %d examples from %s", len(examples), test_path)

        model, tokenizer, max_length = self.load_model()
        probs = self.predict_probs(model, tokenizer, max_length, examples,
                                   desc=f"infer {self.args.split}")
        gold = [e["jump"] for e in examples]
        pred_argmax = (probs >= 0.5).astype(int)

        results: Dict = {
            "checkpoint": str(self.checkpoint),
            "data_dir": str(self.data_dir),
            "split": self.args.split,
            "n_examples": len(examples),
            "history_events": self.args.history_events,
            "max_length": max_length,
            "gold_label_dist": dict(sorted(Counter(gold).items())),
            "metrics_argmax": binary_metrics(gold, pred_argmax.tolist(), probs),
        }

        # Dev-calibrated threshold (uses no CGA training data)
        dev_path = self.data_dir / "dev.jsonl"
        pred_final = pred_argmax
        if self.args.calibrate and dev_path.exists():
            dev_examples = load_jsonl(dev_path)
            if self.args.limit:
                dev_examples = dev_examples[: self.args.limit]
            dev_probs = self.predict_probs(model, tokenizer, max_length,
                                           dev_examples, desc="infer dev")
            dev_gold = [e["jump"] for e in dev_examples]
            threshold = self.best_f1_threshold(dev_gold, dev_probs)
            pred_cal = (probs >= threshold).astype(int)
            results["dev_calibrated_threshold"] = threshold
            results["metrics_dev_calibrated"] = binary_metrics(
                gold, pred_cal.tolist(), probs)
            pred_final = pred_cal

        baseline = self.majority_baseline(gold)
        if baseline:
            results["baseline_majority"] = baseline

        with open(self.output_dir / "predictions.jsonl", "w", encoding="utf-8") as fh:
            for ex, prob, pa, pf in zip(examples, probs, pred_argmax, pred_final):
                fh.write(json.dumps({
                    "example_id": ex["example_id"],
                    "scenario_id": ex["scenario_id"],
                    "corpus": ex.get("corpus"),
                    "task": "jump",
                    "gold_jump": ex["jump"],
                    "pred": int(pa),
                    "pred_calibrated": int(pf),
                    "prob_derail": round(float(prob), 5),
                    "n_history_events": ex["n_history_events"],
                }) + "\n")

        with open(self.output_dir / "results.json", "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        logger.info("Results -> %s", self.output_dir / "results.json")
        self._print_summary(results)
        return results

    @staticmethod
    def _print_summary(results: Dict) -> None:
        W = 96
        print(f"\n{'=' * W}")
        print(f"CGA TRANSFER — {results['checkpoint']}")
        print(f"  data: {results['data_dir']} [{results['split']}]  n={results['n_examples']}")
        print(f"{'-' * W}")
        header = f"{'Setting':<24} {'Acc':>7} {'F1':>7} {'F1-mac':>7} {'Prec':>7} {'Rec':>7} {'AUPRC':>7} {'AUROC':>7}"
        print(header)
        rows = [("argmax@0.5", results["metrics_argmax"])]
        if "metrics_dev_calibrated" in results:
            rows.append((f"dev-cal@{results['dev_calibrated_threshold']:.3f}",
                         results["metrics_dev_calibrated"]))
        if "baseline_majority" in results:
            rows.append(("majority baseline", results["baseline_majority"]))
        for name, m in rows:
            print(f"{name:<24} {m.get('accuracy', float('nan')):>7.4f}"
                  f" {m.get('f1_derail', float('nan')):>7.4f}"
                  f" {m.get('f1_macro', float('nan')):>7.4f}"
                  f" {m.get('precision_derail', float('nan')):>7.4f}"
                  f" {m.get('recall_derail', float('nan')):>7.4f}"
                  f" {m.get('auprc', float('nan')):>7.4f}"
                  f" {m.get('auroc', float('nan')):>7.4f}")
        print(f"{'=' * W}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zero-shot evaluation of a jump forecaster on CGA derailment forecasting.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # ConVAWG-trained forecaster, zero-shot on CGA-wiki:
  python evaluate_cga.py \\
      --checkpoint ../escalation_forecasting/results/microsoft_deberta-v3-base_level/checkpoint-<N> \\
      --data_dir data/wiki

  # same on Reddit CMV, with dev-threshold calibration disabled:
  python evaluate_cga.py --checkpoint <ckpt> --data_dir data/cmv --no-calibrate

  # smoke test on CPU:
  python evaluate_cga.py --checkpoint <ckpt> --data_dir data/wiki --limit 8 --cpu
""",
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Fine-tuned ConVAWG level-forecaster checkpoint directory")
    parser.add_argument("--score_classes", type=int, nargs="+", default=[3, 4],
                        help="Class indices whose softmax probabilities are "
                             "summed into the derailment score (default '3 4' "
                             "for 5-class level checkpoints = P(next level >= 3); "
                             "use '1' for binary jump checkpoints)")
    parser.add_argument("--data_dir", default="data/wiki",
                        help="CGA dataset dir from build_dataset.py (default: data/wiki)")
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--tokenizer", default=None,
                        help="Tokenizer source (default: checkpoint, else inferred "
                             "from model_type)")
    parser.add_argument("--max_length", type=int, default=None,
                        help="Override tokenizer max length (default per model type)")
    parser.add_argument("--history_events", type=int, default=0,
                        help="Use only the last N history comments (0 = all)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--calibrate", action=argparse.BooleanOptionalAction, default=True,
                        help="Also report metrics at the dev-F1-optimal threshold")
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N examples (smoke tests)")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    parser.add_argument("--gpu", type=str, default=None,
                        help="CUDA_VISIBLE_DEVICES value")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    CGATransferEvaluator(args).run()


if __name__ == "__main__":
    main()
