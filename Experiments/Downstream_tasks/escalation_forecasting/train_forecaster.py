#!/usr/bin/env python3
"""
Fine-tune encoders for next-event escalation forecasting.

Two forecasting tasks (labels from build_dataset.py)
----------------------------------------------------
  level — 5-class ordinal: escalation level (0-4) of the NEXT conversation
  jump  — binary: will the next conversation escalate above the current level?

Model choices (long-context matters: history averages several thousand words)
-----------------------------------------------------------------------------
  microsoft/deberta-v3-base   — strongest short-context encoder in this repo;
                                sees only the most recent 512 tokens
  allenai/longformer-base-4096 — sparse-attention long-context encoder
  answerdotai/ModernBERT-base — modern 8k-context encoder (SOTA encoder, 2024)

The tokenizer truncates from the LEFT so the most recent conversations are
always retained (recency window, as in CGA/CRAFT forecasting).

Metrics
-------
  level — accuracy, macro-F1, MAE, quadratic-weighted kappa (QWK)
  jump  — F1, AUPRC, AUROC, precision, recall

Per-example test predictions (with scenario metadata) are written to
<run_dir>/predictions.jsonl for evaluate_forecasts.py (trivial baselines,
early-warning / lead-time analysis).
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    set_seed,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

LEVEL_CLASSES = 5
EVENT_SEPARATOR = "\n\n"

DEFAULT_MODELS: List[str] = [
    "microsoft/deberta-v3-base",
    "allenai/longformer-base-4096",
    "answerdotai/ModernBERT-base",
]

# Per-model context length / batch size / grad-accum
# (sized for a single RTX 3090 with ~11 GB free; override via --batch_size)
MODEL_PROFILES: Dict[str, Dict] = {
    "deberta": {"max_length": 512, "batch_size": 8, "grad_accum": 4,
                "precision": "fp32", "grad_ckpt": False},
    "longformer": {"max_length": 4096, "batch_size": 1, "grad_accum": 16,
                   "precision": "fp16", "grad_ckpt": True},
    "modernbert": {"max_length": 4096, "batch_size": 2, "grad_accum": 8,
                   "precision": "bf16", "grad_ckpt": True},
    "default": {"max_length": 512, "batch_size": 16, "grad_accum": 2,
                "precision": "fp16", "grad_ckpt": False},
}


def model_profile(model_name: str) -> Dict:
    low = model_name.lower()
    for key in ("deberta", "longformer", "modernbert"):
        if key in low:
            return MODEL_PROFILES[key]
    return MODEL_PROFILES["default"]


def load_jsonl(path: Path) -> List[Dict]:
    examples: List[Dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


class WeightedTrainer(Trainer):
    """Trainer with inverse-frequency class weights on the CE loss."""

    def __init__(self, class_weights: torch.Tensor, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = self.class_weights.to(logits.device, dtype=logits.dtype)
        loss = nn.CrossEntropyLoss(weight=weight)(logits, labels)
        return (loss, outputs) if return_outputs else loss


class EscalationForecaster:

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.data_dir = Path(args.data_dir)
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_classes = LEVEL_CLASSES if args.task == "level" else 2

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def _label_of(self, example: Dict) -> int:
        return example["next_level"] if self.args.task == "level" else example["jump"]

    def _load_splits(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        train = load_jsonl(self.data_dir / "train.jsonl")
        dev = load_jsonl(self.data_dir / "dev.jsonl")
        test = load_jsonl(self.data_dir / "test.jsonl")
        logger.info("train=%d  dev=%d  test=%d", len(train), len(dev), len(test))
        return train, dev, test

    def _class_weights(self, examples: List[Dict]) -> torch.Tensor:
        n = len(examples)
        counts = [sum(1 for e in examples if self._label_of(e) == c)
                  for c in range(self.num_classes)]
        weights = []
        for c, cnt in enumerate(counts):
            w = n / (self.num_classes * cnt) if cnt > 0 else 1.0
            weights.append(w)
            logger.info("  class %d  n=%6d  weight=%.4f", c, cnt, w)
        return torch.tensor(weights, dtype=torch.float32)

    def _render_history(self, example: Dict) -> str:
        blocks = example["history"]
        if self.args.history_events > 0:
            blocks = blocks[-self.args.history_events:]
        return EVENT_SEPARATOR.join(blocks)

    def _tokenize(self, examples: List[Dict], tokenizer, max_length: int) -> Dataset:
        texts = [self._render_history(e) for e in examples]
        labels = [self._label_of(e) for e in examples]
        enc = tokenizer(texts, max_length=max_length, truncation=True,
                        padding="max_length")
        enc["labels"] = labels
        return Dataset.from_dict(dict(enc))

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _compute_metrics(self, eval_pred: EvalPrediction) -> Dict[str, float]:
        logits, labels = eval_pred.predictions, eval_pred.label_ids
        preds = np.argmax(logits, axis=-1)
        if self.args.task == "level":
            return {
                "accuracy": float(accuracy_score(labels, preds)),
                "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
                "mae": float(np.mean(np.abs(preds - labels))),
                "qwk": float(cohen_kappa_score(labels, preds, weights="quadratic")),
            }
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
        metrics = {
            "accuracy": float(accuracy_score(labels, preds)),
            "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
            "f1_jump": float(f1_score(labels, preds, pos_label=1, zero_division=0)),
            "precision_jump": float(precision_score(labels, preds, pos_label=1, zero_division=0)),
            "recall_jump": float(recall_score(labels, preds, pos_label=1, zero_division=0)),
        }
        try:
            metrics["auprc"] = float(average_precision_score(labels, probs))
            metrics["auroc"] = float(roc_auc_score(labels, probs))
        except ValueError:
            pass
        return metrics

    # ------------------------------------------------------------------
    # Train / evaluate one model
    # ------------------------------------------------------------------

    def train_and_evaluate(self, model_name: str) -> Dict:
        safe = model_name.replace("/", "_")
        run_dir = self.output_dir / f"{safe}_{self.args.task}"
        run_dir.mkdir(parents=True, exist_ok=True)

        set_seed(self.args.seed)
        profile = model_profile(model_name)
        max_len = self.args.max_length or profile["max_length"]
        bsz = self.args.batch_size or profile["batch_size"]
        grad_accum = profile["grad_accum"]
        precision = profile["precision"]

        train_data, dev_data, test_data = self._load_splits()
        class_weights = self._class_weights(train_data)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.truncation_side = "left"  # keep the most recent conversations
        model_kwargs: Dict = {"num_labels": self.num_classes}
        if precision == "fp32":
            # DeBERTa-v3 produces NaN logits under fp16 (see toxic_classification)
            model_kwargs["dtype"] = torch.float32
        model = AutoModelForSequenceClassification.from_pretrained(model_name, **model_kwargs)

        logger.info("Tokenising (model=%s, max_length=%d, truncation_side=left) ...",
                    model_name, max_len)
        train_ds = self._tokenize(train_data, tokenizer, max_len)
        dev_ds = self._tokenize(dev_data, tokenizer, max_len)
        test_ds = self._tokenize(test_data, tokenizer, max_len)

        steps_per_epoch = max(1, len(train_data) // (bsz * grad_accum))
        warmup_steps = int(0.1 * steps_per_epoch * self.args.epochs)

        training_args = TrainingArguments(
            output_dir=str(run_dir),
            num_train_epochs=self.args.epochs,
            per_device_train_batch_size=bsz,
            per_device_eval_batch_size=bsz * 2,
            gradient_accumulation_steps=grad_accum,
            learning_rate=self.args.lr,
            weight_decay=0.01,
            warmup_steps=warmup_steps,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            logging_steps=50,
            report_to="none",
            fp16=(precision == "fp16" and torch.cuda.is_available()),
            bf16=(precision == "bf16" and torch.cuda.is_available()),
            seed=self.args.seed,
            dataloader_num_workers=4,
            max_steps=self.args.max_steps if self.args.max_steps else -1,
            gradient_checkpointing=profile.get("grad_ckpt", False),
        )

        trainer = WeightedTrainer(
            class_weights=class_weights,
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=dev_ds,
            compute_metrics=self._compute_metrics,
        )

        logger.info("Training %s (%s task) ...", model_name, self.args.task)
        train_result = trainer.train()

        dev_metrics = trainer.evaluate(dev_ds)
        test_pred = trainer.predict(test_ds)
        test_metrics = self._compute_metrics(
            EvalPrediction(predictions=test_pred.predictions,
                           label_ids=test_pred.label_ids))
        logger.info("Test metrics: %s", test_metrics)

        preds = np.argmax(test_pred.predictions, axis=-1)
        probs = torch.softmax(torch.tensor(test_pred.predictions), dim=-1).numpy()

        # Per-example predictions for evaluate_forecasts.py
        with open(run_dir / "predictions.jsonl", "w", encoding="utf-8") as fh:
            for ex, pred, prob in zip(test_data, preds, probs):
                fh.write(json.dumps({
                    "example_id": ex["example_id"],
                    "scenario_id": ex["scenario_id"],
                    "target_pos": ex["target_pos"],
                    "target_timestamp": ex["target_timestamp"],
                    "cur_level": ex["cur_level"],
                    "gold_next_level": ex["next_level"],
                    "gold_jump": ex["jump"],
                    "task": self.args.task,
                    "pred": int(pred),
                    "probs": [round(float(p), 5) for p in prob],
                }) + "\n")

        report_str = classification_report(test_pred.label_ids, preds, digits=4, zero_division=0)
        logger.info("\nTest Classification Report:\n%s", report_str)

        results = {
            "model": model_name,
            "task": self.args.task,
            "history_events": self.args.history_events,
            "hyperparams": {
                "epochs": self.args.epochs, "lr": self.args.lr,
                "batch_size_per_device": bsz, "grad_accum": grad_accum,
                "max_length": max_len, "seed": self.args.seed,
                "precision": precision,
            },
            "train_runtime_s": train_result.metrics.get("train_runtime"),
            "dev_metrics": {k: v for k, v in dev_metrics.items()
                            if isinstance(v, (int, float))},
            "test_metrics": test_metrics,
            "classification_report": report_str,
        }
        with open(run_dir / "results.json", "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, default=float)
        logger.info("Results saved → %s", run_dir / "results.json")

        del model, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return results

    def run_all(self) -> None:
        all_results: List[Dict] = []
        for model_name in self.args.models:
            try:
                all_results.append(self.train_and_evaluate(model_name))
            except Exception as exc:
                import traceback
                logger.error("FAILED model=%s : %s", model_name, exc)
                traceback.print_exc()
        with open(self.output_dir / f"summary_{self.args.task}.json", "w",
                  encoding="utf-8") as fh:
            json.dump(all_results, fh, indent=2, default=float)
        self._print_summary(all_results)

    def _print_summary(self, all_results: List[Dict]) -> None:
        W = 110
        print(f"\n{'═' * W}")
        print(f"ESCALATION FORECASTING — TASK: {self.args.task.upper()}")
        print(f"{'═' * W}")
        if self.args.task == "level":
            print(f"{'Model':<40} {'Acc':>8} {'F1-mac':>8} {'MAE':>8} {'QWK':>8}")
            print("─" * W)
            for r in all_results:
                tm = r["test_metrics"]
                print(f"{r['model'].split('/')[-1]:<40}"
                      f" {tm.get('accuracy', float('nan')):>8.4f}"
                      f" {tm.get('f1_macro', float('nan')):>8.4f}"
                      f" {tm.get('mae', float('nan')):>8.4f}"
                      f" {tm.get('qwk', float('nan')):>8.4f}")
        else:
            print(f"{'Model':<40} {'Acc':>8} {'F1-jmp':>8} {'AUPRC':>8} {'AUROC':>8}"
                  f" {'Prec':>8} {'Rec':>8}")
            print("─" * W)
            for r in all_results:
                tm = r["test_metrics"]
                print(f"{r['model'].split('/')[-1]:<40}"
                      f" {tm.get('accuracy', float('nan')):>8.4f}"
                      f" {tm.get('f1_jump', float('nan')):>8.4f}"
                      f" {tm.get('auprc', float('nan')):>8.4f}"
                      f" {tm.get('auroc', float('nan')):>8.4f}"
                      f" {tm.get('precision_jump', float('nan')):>8.4f}"
                      f" {tm.get('recall_jump', float('nan')):>8.4f}")
        print(f"{'═' * W}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune encoders for next-event escalation forecasting.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python train_forecaster.py --task level
  python train_forecaster.py --task jump --models answerdotai/ModernBERT-base
  python train_forecaster.py --task level --history_events 3 --output_dir results_recent3
  python train_forecaster.py --task level --max_steps 20 --models microsoft/deberta-v3-base   # smoke test
""",
    )
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--task", choices=["level", "jump"], default="level")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--history_events", type=int, default=0,
                        help="Use only the last N history events (0 = all, "
                             "left-truncated by the tokenizer)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Limit optimiser steps (smoke tests)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=str, default=None,
                        help="CUDA_VISIBLE_DEVICES value")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    EscalationForecaster(args).run_all()


if __name__ == "__main__":
    main()
