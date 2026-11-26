#!/usr/bin/env python3
"""
Fine-tune BERT / RoBERTa / DeBERTa-v3 for message-level toxic classification.

Two classification settings
----------------------------
  setting1 — single message text (128 tokens)
  setting2 — dialogue context (all other messages) + target message (512 tokens)

Models trained by default
--------------------------
  bert-base-uncased
  roberta-base
  microsoft/deberta-v3-base

Class-imbalance handling
------------------------
  Inverse-frequency class weights passed to a custom WeightedTrainer.

Metrics reported on test set
-----------------------------
  F1-binary (toxic class), F1-macro, F1-weighted, accuracy,
  precision-toxic, recall-toxic, AUC-ROC

Also saves per-epoch training loss and dev eval metrics.
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
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MODELS: List[str] = [
    "bert-base-uncased",
    "roberta-base",
    "microsoft/deberta-v3-base",
]

# Max sequence length per setting
MAX_LENGTH: Dict[str, int] = {
    "setting1": 128,
    "setting2": 512,
}

# Per-device train batch size per setting (DeBERTa gets halved automatically)
BASE_BATCH_SIZE: Dict[str, int] = {
    "setting1": 32,
    "setting2": 16,
}

# Gradient accumulation steps per setting so effective batch ≈ 32
GRAD_ACCUM: Dict[str, int] = {
    "setting1": 1,
    "setting2": 2,
}


# ---------------------------------------------------------------------------
# Custom Trainer with class-weighted cross-entropy
# ---------------------------------------------------------------------------

class WeightedTrainer(Trainer):
    """Trainer that applies inverse-frequency class weights to the CE loss."""

    def __init__(self, class_weights: torch.Tensor, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self, model, inputs, return_outputs: bool = False, **kwargs
    ):
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
# Main classifier class
# ---------------------------------------------------------------------------

class ToxicClassifier:
    """Train and evaluate toxic message classifiers across models and settings."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.data_dir = Path(args.data_dir)
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def _load_splits(
        self, setting: str
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        sd = self.data_dir / setting
        train = load_jsonl(sd / "train.jsonl")
        dev   = load_jsonl(sd / "dev.jsonl")
        test  = load_jsonl(sd / "test.jsonl")
        logger.info(
            "[%s] train=%d  dev=%d  test=%d", setting, len(train), len(dev), len(test)
        )
        return train, dev, test

    def _class_weights(self, examples: List[Dict]) -> torch.Tensor:
        n = len(examples)
        n_pos = sum(e["label"] for e in examples)
        n_neg = n - n_pos
        w_neg = n / (2.0 * n_neg) if n_neg > 0 else 1.0
        w_pos = n / (2.0 * n_pos) if n_pos > 0 else 1.0
        logger.info(
            "Class weights → non-toxic: %.3f | toxic: %.3f  (pos=%d / %d = %.1f%%)",
            w_neg, w_pos, n_pos, n, 100 * n_pos / n,
        )
        return torch.tensor([w_neg, w_pos], dtype=torch.float32)

    # ------------------------------------------------------------------
    # Tokenisation
    # ------------------------------------------------------------------

    def _tokenize_batch(
        self,
        examples: List[Dict],
        tokenizer,
        setting: str,
        max_length: int,
    ) -> Dataset:
        texts = [e["text"] for e in examples]
        labels = [e["label"] for e in examples]

        if setting == "setting1":
            enc = tokenizer(
                texts,
                max_length=max_length,
                truncation=True,
                padding="max_length",
            )
        else:
            # sequence-pair: (context, target_message)
            # truncation=True trims the longest; we use 'only_first' so the
            # target message is always preserved in full.
            contexts = [e.get("context") or "" for e in examples]
            enc = tokenizer(
                contexts,
                texts,
                max_length=max_length,
                truncation="only_first",
                padding="max_length",
            )

        enc["labels"] = labels
        return Dataset.from_dict(dict(enc))

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
        logits, labels = eval_pred.predictions, eval_pred.label_ids
        preds = np.argmax(logits, axis=-1)
        probs = (
            torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1)[:, 1]
            .numpy()
        )
        metrics: Dict[str, float] = {
            "accuracy":        float(accuracy_score(labels, preds)),
            "f1_binary":       float(f1_score(labels, preds, pos_label=1, zero_division=0)),
            "f1_macro":        float(f1_score(labels, preds, average="macro", zero_division=0)),
            "f1_weighted":     float(f1_score(labels, preds, average="weighted", zero_division=0)),
            "precision_toxic": float(precision_score(labels, preds, pos_label=1, zero_division=0)),
            "recall_toxic":    float(recall_score(labels, preds, pos_label=1, zero_division=0)),
        }
        try:
            metrics["auc_roc"] = float(roc_auc_score(labels, probs))
        except ValueError:
            metrics["auc_roc"] = float("nan")
        return metrics

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @staticmethod
    def _latest_checkpoint(run_dir: Path) -> Optional[Path]:
        """Return the latest checkpoint sub-dir in run_dir, or None."""
        checkpoints = sorted(
            run_dir.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[1]),
        )
        return checkpoints[-1] if checkpoints else None

    def train_and_evaluate(self, model_name: str, setting: str) -> Dict:
        """Full train + evaluate cycle for one (model, setting) pair."""
        safe = model_name.replace("/", "_")
        run_dir = self.output_dir / f"{safe}_{setting}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Resolve checkpoint to resume from
        resume_ckpt: Optional[str] = None
        if getattr(self.args, "resume_from_checkpoint", None):
            arg = self.args.resume_from_checkpoint
            if arg == "latest":
                found = self._latest_checkpoint(run_dir)
                resume_ckpt = str(found) if found else None
                if resume_ckpt:
                    logger.info("Resuming from latest checkpoint: %s", resume_ckpt)
                else:
                    logger.info("No checkpoint found in %s — training from scratch", run_dir)
            else:
                resume_ckpt = arg
                logger.info("Resuming from checkpoint: %s", resume_ckpt)

        logger.info("\n%s", "=" * 70)
        logger.info("Model : %s", model_name)
        logger.info("Setting: %s", setting)
        logger.info("%s", "=" * 70)

        set_seed(self.args.seed)

        # ---- data -------------------------------------------------------
        train_data, dev_data, test_data = self._load_splits(setting)
        class_weights = self._class_weights(train_data)

        max_len = MAX_LENGTH[setting]
        bsz = BASE_BATCH_SIZE[setting]
        if "deberta" in model_name.lower():
            bsz = max(bsz // 2, 8)
        grad_accum = GRAD_ACCUM[setting]

        # ---- tokeniser + model -----------------------------------------
        logger.info("Loading tokeniser: %s", model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        logger.info("Loading model: %s", model_name)
        # DeBERTa-v3 loads in fp16 by default in newer transformers, producing
        # NaN logits even with fp16=False in TrainingArguments; force fp32.
        model_dtype = torch.float32 if "deberta" in model_name.lower() else None
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2,
            **(dict(dtype=model_dtype) if model_dtype else {}),
        )

        # ---- tokenise ---------------------------------------------------
        logger.info("Tokenising splits (max_length=%d) ...", max_len)
        train_ds = self._tokenize_batch(train_data, tokenizer, setting, max_len)
        dev_ds   = self._tokenize_batch(dev_data,   tokenizer, setting, max_len)
        test_ds  = self._tokenize_batch(test_data,  tokenizer, setting, max_len)

        # ---- training args ----------------------------------------------
        # Compute warmup steps from 10 % of total training steps
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
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="f1_binary",
            greater_is_better=True,
            logging_steps=100,
            report_to="none",
            # DeBERTa-v3 disentangled attention is numerically unstable in bf16/fp16;
            # use fp32 for it and fp16 for all other models.
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

        # ---- train -------------------------------------------------------
        logger.info("Training ...")
        train_result = trainer.train(resume_from_checkpoint=resume_ckpt)

        # ---- dev evaluation ---------------------------------------------
        dev_metrics = trainer.evaluate(dev_ds)
        logger.info("Dev metrics: %s", dev_metrics)

        # ---- test evaluation --------------------------------------------
        test_pred = trainer.predict(test_ds)
        test_metrics = self._compute_metrics(
            EvalPrediction(
                predictions=test_pred.predictions,
                label_ids=test_pred.label_ids,
            )
        )
        test_loss = test_pred.metrics.get("test_loss", float("nan"))

        # ---- classification report ---------------------------------------
        preds = np.argmax(test_pred.predictions, axis=-1)
        report_str = classification_report(
            test_pred.label_ids,
            preds,
            target_names=["non-toxic", "toxic"],
            digits=4,
        )
        logger.info("\nTest Classification Report:\n%s", report_str)

        # ---- training / dev loss history --------------------------------
        epoch_history: List[Dict] = []
        train_loss_log: List[Dict] = []
        for entry in trainer.state.log_history:
            if "loss" in entry and "eval_loss" not in entry:
                train_loss_log.append({
                    "step":  entry.get("step"),
                    "epoch": entry.get("epoch"),
                    "train_loss": entry["loss"],
                })
            if "eval_loss" in entry:
                epoch_history.append({
                    "epoch":     entry.get("epoch"),
                    "eval_loss": entry.get("eval_loss"),
                    **{k: v for k, v in entry.items()
                       if k.startswith("eval_") and k != "eval_loss"},
                })

        # ---- bundle results ---------------------------------------------
        results = {
            "model": model_name,
            "setting": setting,
            "hyperparams": {
                "epochs": self.args.epochs,
                "lr": self.args.lr,
                "batch_size_per_device": bsz,
                "grad_accum": grad_accum,
                "max_length": max_len,
                "seed": self.args.seed,
            },
            "train_runtime_s": train_result.metrics.get("train_runtime"),
            "train_loss_final": train_result.metrics.get("train_loss"),
            "epoch_eval_history": epoch_history,
            "train_loss_log": train_loss_log,
            "dev_metrics":  {k: v for k, v in dev_metrics.items()
                             if isinstance(v, (int, float))},
            "test_metrics": {**test_metrics, "test_loss": test_loss},
            "classification_report": report_str,
        }

        out_path = run_dir / "results.json"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, default=float)
        logger.info("Results saved → %s", out_path)

        # ---- free GPU memory -------------------------------------------
        del model
        del trainer
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
                    logger.error(
                        "FAILED  model=%s  setting=%s : %s", model_name, setting, exc
                    )
                    traceback.print_exc()

        # Save summary
        summary_path = self.output_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(all_results, fh, indent=2, default=float)
        logger.info("Summary saved → %s", summary_path)

        self._print_summary_table(all_results)

    def aggregate_results(self) -> None:
        """Collect individual results.json files from all sub-dirs and print summary."""
        all_results: List[Dict] = []
        for result_file in sorted(self.output_dir.glob("*/results.json")):
            try:
                with open(result_file, encoding="utf-8") as fh:
                    all_results.append(json.load(fh))
                logger.info("Loaded %s", result_file)
            except Exception as exc:
                logger.warning("Could not load %s: %s", result_file, exc)

        if not all_results:
            logger.warning("No results.json files found under %s", self.output_dir)
            return

        summary_path = self.output_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(all_results, fh, indent=2, default=float)
        logger.info("Summary saved → %s", summary_path)
        self._print_summary_table(all_results)

    # ------------------------------------------------------------------
    # Pretty summary
    # ------------------------------------------------------------------

    def _print_summary_table(self, all_results: List[Dict]) -> None:
        col_w = 38
        sep = "─" * 110
        print(f"\n{'═'*110}")
        print("TOXIC CLASSIFICATION — EXPERIMENT SUMMARY")
        print(f"{'═'*110}")
        header = (
            f"{'Model':<{col_w}} {'Setting':<10}"
            f" {'F1-bin':>8} {'F1-mac':>8} {'Acc':>8}"
            f" {'Prec':>8} {'Rec':>8} {'AUC-ROC':>8}"
            f" {'DevF1':>8} {'TrnLoss':>8}"
        )
        print(header)
        print(sep)

        for r in all_results:
            tm = r.get("test_metrics", {})
            dm = r.get("dev_metrics", {})
            short = r["model"].split("/")[-1]
            print(
                f"{short:<{col_w}} {r['setting']:<10}"
                f" {tm.get('f1_binary',    float('nan')):>8.4f}"
                f" {tm.get('f1_macro',     float('nan')):>8.4f}"
                f" {tm.get('accuracy',     float('nan')):>8.4f}"
                f" {tm.get('precision_toxic', float('nan')):>8.4f}"
                f" {tm.get('recall_toxic', float('nan')):>8.4f}"
                f" {tm.get('auc_roc',      float('nan')):>8.4f}"
                f" {dm.get('eval_f1_binary', float('nan')):>8.4f}"
                f" {r.get('train_loss_final', float('nan')):>8.4f}"
            )

        print(f"{'═'*110}")
        print("Metrics key: F1-bin=binary F1 (toxic class)  Prec/Rec=toxic class  AUC-ROC=ROC area")
        print(f"{'═'*110}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune BERT / RoBERTa / DeBERTa-v3 for binary toxic message "
            "classification on VAWG data."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # run all models, both settings\n"
            "  python train_classifier.py --data_dir data/ --output_dir results/\n\n"
            "  # single model, single setting\n"
            "  python train_classifier.py --models bert-base-uncased "
            "--settings setting1 --epochs 5\n"
        ),
    )
    parser.add_argument(
        "--data_dir", type=str, default="data",
        help="Directory containing setting1/ and setting2/ JSONL splits",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results",
        help="Where to save checkpoints and result JSON files",
    )
    parser.add_argument(
        "--models", nargs="+", default=DEFAULT_MODELS,
        help="HuggingFace model names to fine-tune",
    )
    parser.add_argument(
        "--settings", nargs="+", default=["setting1", "setting2"],
        choices=["setting1", "setting2"],
        help="Classification settings to run",
    )
    parser.add_argument("--epochs", type=int,   default=3, help="Training epochs")
    parser.add_argument("--lr",     type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--seed",   type=int,   default=42)
    parser.add_argument(
        "--resume_from_checkpoint", type=str, default=None, metavar="PATH|latest",
        help="Resume training from a checkpoint path or 'latest' to auto-detect.",
    )
    parser.add_argument(
        "--gpu", type=str, default=None,
        help="GPU id(s) to use, e.g. '1' or '1,2'.  Sets CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument(
        "--aggregate_only", action="store_true",
        help="Skip training; collect existing results.json files and print summary.",
    )
    args = parser.parse_args()

    # Pin GPU(s) before any CUDA initialisation
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        logger.info("CUDA_VISIBLE_DEVICES set to %s", args.gpu)

    classifier = ToxicClassifier(args)
    if args.aggregate_only:
        classifier.aggregate_results()
    else:
        classifier.run_all()


if __name__ == "__main__":
    main()
