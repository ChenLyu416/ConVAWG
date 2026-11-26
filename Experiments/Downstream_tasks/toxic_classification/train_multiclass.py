#!/usr/bin/env python3
"""
Fine-tune BERT / RoBERTa / DeBERTa-v3 for multi-class toxic-type classification.

8 classes (from build_multiclass_dataset.py)
---------------------------------------------
  0  non-toxic    (victim messages)
  1  surveillance
  2  demand
  3  accusation
  4  manipulation
  5  ultimatum
  6  threat
  7  other

Two classification settings
----------------------------
  setting1 — single message text (128 tokens)
  setting2 — dialogue context + target message (512 tokens)

Class-imbalance handling
------------------------
  Per-class inverse-frequency weights in WeightedTrainer.

Metrics reported on test set
-----------------------------
  F1-macro, F1-weighted, accuracy, per-class F1
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
    classification_report,
    f1_score,
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASS_NAMES = {
    0: "non-toxic",
    1: "surveillance",
    2: "demand",
    3: "accusation",
    4: "manipulation",
    5: "ultimatum",
    6: "threat",
    7: "other",
}
NUM_CLASSES = len(CLASS_NAMES)
CLASS_NAME_LIST = [CLASS_NAMES[i] for i in range(NUM_CLASSES)]

DEFAULT_MODELS: List[str] = [
    "bert-base-uncased",
    "roberta-base",
    "microsoft/deberta-v3-base",
]

MAX_LENGTH: Dict[str, int] = {"setting1": 128, "setting2": 512}
BASE_BATCH_SIZE: Dict[str, int] = {"setting1": 32, "setting2": 16}
GRAD_ACCUM: Dict[str, int] = {"setting1": 1, "setting2": 2}


# ---------------------------------------------------------------------------
# Custom Trainer with per-class weighted cross-entropy
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> List[Dict]:
    examples: List[Dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class MulticlassToxicClassifier:

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.data_dir   = Path(args.data_dir)
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def _load_splits(self, setting: str) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        sd = self.data_dir / setting
        train = load_jsonl(sd / "train.jsonl")
        dev   = load_jsonl(sd / "dev.jsonl")
        test  = load_jsonl(sd / "test.jsonl")
        logger.info("[%s] train=%d  dev=%d  test=%d", setting, len(train), len(dev), len(test))
        return train, dev, test

    def _class_weights(self, examples: List[Dict]) -> torch.Tensor:
        """Per-class inverse-frequency weights (balanced formula)."""
        n = len(examples)
        counts = [sum(1 for e in examples if e["label"] == c) for c in range(NUM_CLASSES)]
        weights = []
        for c, cnt in enumerate(counts):
            w = n / (NUM_CLASSES * cnt) if cnt > 0 else 1.0
            weights.append(w)
            logger.info("  class %d %-14s  n=%6d  weight=%.4f", c, CLASS_NAMES[c], cnt, w)
        return torch.tensor(weights, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Tokenisation
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize_batch(
        examples: List[Dict],
        tokenizer,
        setting: str,
        max_length: int,
    ) -> Dataset:
        texts  = [e["text"] for e in examples]
        labels = [e["label"] for e in examples]

        if setting == "setting1":
            enc = tokenizer(texts, max_length=max_length,
                            truncation=True, padding="max_length")
        else:
            contexts = [e.get("context") or "" for e in examples]
            enc = tokenizer(contexts, texts, max_length=max_length,
                            truncation="only_first", padding="max_length")

        enc["labels"] = labels
        return Dataset.from_dict(dict(enc))

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
        logits, labels = eval_pred.predictions, eval_pred.label_ids
        preds = np.argmax(logits, axis=-1)
        per_class_f1 = f1_score(labels, preds, average=None,
                                labels=list(range(NUM_CLASSES)), zero_division=0)
        metrics: Dict[str, float] = {
            "accuracy":    float(accuracy_score(labels, preds)),
            "f1_macro":    float(f1_score(labels, preds, average="macro",    zero_division=0)),
            "f1_weighted": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        }
        for c, f1 in enumerate(per_class_f1):
            metrics[f"f1_{CLASS_NAMES[c]}"] = float(f1)
        return metrics

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    @staticmethod
    def _latest_checkpoint(run_dir: Path) -> Optional[Path]:
        checkpoints = sorted(run_dir.glob("checkpoint-*"),
                             key=lambda p: int(p.name.split("-")[1]))
        return checkpoints[-1] if checkpoints else None

    def train_and_evaluate(self, model_name: str, setting: str) -> Dict:
        safe    = model_name.replace("/", "_")
        run_dir = self.output_dir / f"{safe}_{setting}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Resolve resume checkpoint
        resume_ckpt: Optional[str] = None
        if getattr(self.args, "resume_from_checkpoint", None):
            arg = self.args.resume_from_checkpoint
            if arg == "latest":
                found = self._latest_checkpoint(run_dir)
                resume_ckpt = str(found) if found else None
                if resume_ckpt:
                    logger.info("Resuming from: %s", resume_ckpt)
            else:
                resume_ckpt = arg

        set_seed(self.args.seed)

        train_data, dev_data, test_data = self._load_splits(setting)
        class_weights = self._class_weights(train_data)

        max_len    = MAX_LENGTH[setting]
        bsz        = BASE_BATCH_SIZE[setting]
        if "deberta" in model_name.lower():
            bsz = max(bsz // 2, 8)
        grad_accum = GRAD_ACCUM[setting]

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # DeBERTa-v3 loads in fp16 by default in newer transformers, producing
        # NaN logits even with fp16=False in TrainingArguments; force fp32.
        model_dtype = torch.float32 if "deberta" in model_name.lower() else None
        model     = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=NUM_CLASSES,
            **(dict(dtype=model_dtype) if model_dtype else {}),
        )

        logger.info("Tokenising splits (max_length=%d) ...", max_len)
        train_ds = self._tokenize_batch(train_data, tokenizer, setting, max_len)
        dev_ds   = self._tokenize_batch(dev_data,   tokenizer, setting, max_len)
        test_ds  = self._tokenize_batch(test_data,  tokenizer, setting, max_len)

        steps_per_epoch = max(1, len(train_data) // (bsz * grad_accum))
        warmup_steps    = int(0.1 * steps_per_epoch * self.args.epochs)

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
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            logging_steps=100,
            report_to="none",
            fp16=torch.cuda.is_available() and "deberta" not in model_name.lower(),
            bf16=False,
            seed=self.args.seed,
            dataloader_num_workers=4,
        )

        trainer = WeightedTrainer(
            class_weights=class_weights,
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=dev_ds,
            compute_metrics=self._compute_metrics,
        )

        logger.info("Training ...")
        train_result = trainer.train(resume_from_checkpoint=resume_ckpt)

        dev_metrics  = trainer.evaluate(dev_ds)
        logger.info("Dev metrics: %s", dev_metrics)

        test_pred    = trainer.predict(test_ds)
        test_metrics = self._compute_metrics(
            EvalPrediction(predictions=test_pred.predictions,
                           label_ids=test_pred.label_ids)
        )
        test_loss = test_pred.metrics.get("test_loss", float("nan"))

        preds      = np.argmax(test_pred.predictions, axis=-1)
        report_str = classification_report(
            test_pred.label_ids, preds,
            target_names=CLASS_NAME_LIST, digits=4,
        )
        logger.info("\nTest Classification Report:\n%s", report_str)

        epoch_history: List[Dict] = []
        train_loss_log: List[Dict] = []
        for entry in trainer.state.log_history:
            if "loss" in entry and "eval_loss" not in entry:
                train_loss_log.append({"step": entry.get("step"),
                                       "epoch": entry.get("epoch"),
                                       "train_loss": entry["loss"]})
            if "eval_loss" in entry:
                epoch_history.append({
                    "epoch":     entry.get("epoch"),
                    "eval_loss": entry.get("eval_loss"),
                    **{k: v for k, v in entry.items()
                       if k.startswith("eval_") and k != "eval_loss"},
                })

        results = {
            "model":    model_name,
            "setting":  setting,
            "task":     "multiclass",
            "num_classes": NUM_CLASSES,
            "class_names": CLASS_NAME_LIST,
            "hyperparams": {
                "epochs": self.args.epochs,
                "lr": self.args.lr,
                "batch_size_per_device": bsz,
                "grad_accum": grad_accum,
                "max_length": max_len,
                "seed": self.args.seed,
            },
            "train_runtime_s":   train_result.metrics.get("train_runtime"),
            "train_loss_final":  train_result.metrics.get("train_loss"),
            "epoch_eval_history": epoch_history,
            "train_loss_log":    train_loss_log,
            "dev_metrics":  {k: v for k, v in dev_metrics.items()
                             if isinstance(v, (int, float))},
            "test_metrics": {**test_metrics, "test_loss": test_loss},
            "classification_report": report_str,
        }

        out_path = run_dir / "results.json"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, default=float)
        logger.info("Results saved → %s", out_path)

        del model, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return results

    # ------------------------------------------------------------------
    # Run all combinations
    # ------------------------------------------------------------------

    def run_all(self) -> None:
        all_results: List[Dict] = []
        for setting in self.args.settings:
            for model_name in self.args.models:
                try:
                    result = self.train_and_evaluate(model_name, setting)
                    all_results.append(result)
                except Exception as exc:
                    import traceback
                    logger.error("FAILED model=%s setting=%s : %s", model_name, setting, exc)
                    traceback.print_exc()
        self._save_summary(all_results)

    def aggregate_results(self) -> None:
        all_results: List[Dict] = []
        for result_file in sorted(self.output_dir.glob("*/results.json")):
            try:
                all_results.append(json.load(open(result_file)))
                logger.info("Loaded %s", result_file)
            except Exception as exc:
                logger.warning("Could not load %s: %s", result_file, exc)
        if not all_results:
            logger.warning("No results found under %s", self.output_dir)
            return
        self._save_summary(all_results)

    def _save_summary(self, all_results: List[Dict]) -> None:
        path = self.output_dir / "summary.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(all_results, fh, indent=2, default=float)
        logger.info("Summary → %s", path)
        self._print_summary_table(all_results)

    # ------------------------------------------------------------------
    # Pretty summary
    # ------------------------------------------------------------------

    def _print_summary_table(self, all_results: List[Dict]) -> None:
        W = 130
        print(f"\n{'═'*W}")
        print("MULTI-CLASS TOXIC TYPE CLASSIFICATION — EXPERIMENT SUMMARY")
        print(f"{'═'*W}")
        hdr = (f"{'Model':<38} {'Setting':<10}"
               f" {'Acc':>7} {'F1-mac':>7} {'F1-wgt':>7}"
               f" {'nontx':>7} {'surv':>7} {'dmd':>7} {'accus':>7}"
               f" {'manip':>7} {'ultim':>7} {'threat':>7} {'other':>7}"
               f" {'TrnLoss':>8}")
        print(hdr)
        print("─" * W)
        for r in all_results:
            tm  = r.get("test_metrics", {})
            short = r["model"].split("/")[-1]
            row = (
                f"{short:<38} {r['setting']:<10}"
                f" {tm.get('accuracy',        float('nan')):>7.4f}"
                f" {tm.get('f1_macro',        float('nan')):>7.4f}"
                f" {tm.get('f1_weighted',     float('nan')):>7.4f}"
                f" {tm.get('f1_non-toxic',    float('nan')):>7.4f}"
                f" {tm.get('f1_surveillance', float('nan')):>7.4f}"
                f" {tm.get('f1_demand',       float('nan')):>7.4f}"
                f" {tm.get('f1_accusation',   float('nan')):>7.4f}"
                f" {tm.get('f1_manipulation', float('nan')):>7.4f}"
                f" {tm.get('f1_ultimatum',    float('nan')):>7.4f}"
                f" {tm.get('f1_threat',       float('nan')):>7.4f}"
                f" {tm.get('f1_other',        float('nan')):>7.4f}"
                f" {r.get('train_loss_final', float('nan')):>8.4f}"
            )
            print(row)
        print(f"{'═'*W}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune encoders for multi-class toxic-type classification."
    )
    parser.add_argument("--data_dir",   default="data_multiclass")
    parser.add_argument("--output_dir", default="results_multiclass")
    parser.add_argument("--models",     nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--settings",   nargs="+", default=["setting1", "setting2"])
    parser.add_argument("--epochs",     type=int,   default=3)
    parser.add_argument("--lr",         type=float, default=2e-5)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        metavar="PATH|latest")
    parser.add_argument("--gpu",        type=str,   default=None,
                        help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument("--aggregate_only", action="store_true",
                        help="Skip training; aggregate existing results.json files.")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    clf = MulticlassToxicClassifier(args)

    if args.aggregate_only:
        clf.aggregate_results()
    else:
        clf.run_all()


if __name__ == "__main__":
    main()
