#!/usr/bin/env python3
"""
Synthetic-to-real transfer experiments: ConVAWG <-> MentalManip.

Every run trains ONE model on a train-source spec and evaluates on BOTH test
sets (ConVAWG + MentalManip), following the generalized-cross-validation
protocol for synthetic dataset evaluation.

Train-source spec grammar
-------------------------
  convawg                    — full ConVAWG train split (synthetic)
  mentalmanip                — full MentalManip train split (real)
  mentalmanip:500            — stratified random subset of 500 real examples
  convawg+mentalmanip:500    — sequential: pretrain on ConVAWG, then
                               fine-tune on 500 real examples
                               (synthetic-augmentation regime)

--matrix runs the grid reported in the paper:
  mentalmanip | convawg+mentalmanip | mentalmanip:500 | convawg+mentalmanip:500

Headline results
----------------
  * transfer gap    — in-domain minus cross-domain macro-F1, both directions
  * augmentation Δ  — convawg+mentalmanip:k  vs  mentalmanip:k  on the
                      MentalManip test set (value of synthetic pretraining
                      in the low-resource regime)

Metrics per test set: accuracy, macro-F1, F1(manipulative), AUROC.
"""

import argparse
import gc
import json
import logging
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
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

SOURCES = ("convawg", "mentalmanip")
DEFAULT_MODEL = "microsoft/deberta-v3-base"
MAX_LENGTH = 512

# The four training settings reported in the paper
STANDARD_MATRIX = [
    "mentalmanip",
    "convawg+mentalmanip",
    "mentalmanip:500",
    "convawg+mentalmanip:500",
]


def load_jsonl(path: Path) -> List[Dict]:
    out: List[Dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


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


def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    preds = np.argmax(logits, axis=-1)
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "f1_manipulative": float(f1_score(labels, preds, pos_label=1, zero_division=0)),
        "precision": float(precision_score(labels, preds, pos_label=1, zero_division=0)),
        "recall": float(recall_score(labels, preds, pos_label=1, zero_division=0)),
    }
    try:
        metrics["auroc"] = float(roc_auc_score(labels, probs))
    except ValueError:
        pass
    return metrics


class TransferExperiment:

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.data_dir = Path(args.data_dir)
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._splits_cache: Dict[Tuple[str, str], List[Dict]] = {}

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def _split(self, source: str, name: str) -> List[Dict]:
        key = (source, name)
        if key not in self._splits_cache:
            self._splits_cache[key] = load_jsonl(self.data_dir / source / f"{name}.jsonl")
        return self._splits_cache[key]

    @staticmethod
    def _stratified_subset(examples: List[Dict], k: int, seed: int) -> List[Dict]:
        by_label: Dict[int, List[Dict]] = defaultdict(list)
        for ex in examples:
            by_label[ex["label"]].append(ex)
        rng = random.Random(seed)
        subset: List[Dict] = []
        total = len(examples)
        for label, group in sorted(by_label.items()):
            take = max(1, round(k * len(group) / total))
            rng.shuffle(group)
            subset.extend(group[:take])
        rng.shuffle(subset)
        return subset[:k]

    def _parse_stage(self, stage: str) -> Tuple[str, List[Dict]]:
        """'mentalmanip:100' -> (source, subset of train examples)."""
        if ":" in stage:
            source, k = stage.split(":", 1)
            examples = self._stratified_subset(self._split(source, "train"),
                                               int(k), self.args.seed)
        else:
            source, examples = stage, self._split(stage, "train")
        if source not in SOURCES:
            raise ValueError(f"Unknown train source: {source}")
        return source, examples

    def _tokenize(self, examples: List[Dict], tokenizer) -> Dataset:
        enc = tokenizer([e["text"] for e in examples], max_length=MAX_LENGTH,
                        truncation=True, padding="max_length")
        enc["labels"] = [e["label"] for e in examples]
        return Dataset.from_dict(dict(enc))

    @staticmethod
    def _class_weights(examples: List[Dict]) -> torch.Tensor:
        n = len(examples)
        counts = [sum(1 for e in examples if e["label"] == c) for c in (0, 1)]
        return torch.tensor(
            [n / (2 * c) if c > 0 else 1.0 for c in counts], dtype=torch.float32)

    # ------------------------------------------------------------------
    # One training stage
    # ------------------------------------------------------------------

    def _fit_stage(self, model, tokenizer, train_examples: List[Dict],
                   dev_examples: List[Dict], run_dir: Path, epochs: int) -> Trainer:
        bsz = self.args.batch_size
        train_ds = self._tokenize(train_examples, tokenizer)
        dev_ds = self._tokenize(dev_examples, tokenizer)
        steps_per_epoch = max(1, len(train_examples) // bsz)

        training_args = TrainingArguments(
            output_dir=str(run_dir / "ckpt"),
            num_train_epochs=epochs,
            per_device_train_batch_size=bsz,
            per_device_eval_batch_size=bsz * 2,
            learning_rate=self.args.lr,
            weight_decay=0.01,
            warmup_steps=int(0.1 * steps_per_epoch * epochs),
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            logging_steps=50,
            report_to="none",
            fp16=False,  # DeBERTa-v3 NaN issue; fp32 is safe for all models here
            seed=self.args.seed,
            dataloader_num_workers=4,
            max_steps=self.args.max_steps if self.args.max_steps else -1,
        )
        trainer = WeightedTrainer(
            class_weights=self._class_weights(train_examples),
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=dev_ds,
            compute_metrics=compute_metrics,
        )
        trainer.train()
        return trainer

    # ------------------------------------------------------------------
    # One full run (train spec -> evaluate on both test sets)
    # ------------------------------------------------------------------

    def run_spec(self, spec: str) -> Dict:
        safe = spec.replace("+", "_PLUS_").replace(":", "_")
        model_safe = self.args.model.replace("/", "_")
        run_dir = self.output_dir / f"{model_safe}__{safe}"
        run_dir.mkdir(parents=True, exist_ok=True)
        results_path = run_dir / "results.json"
        if results_path.exists() and not self.args.overwrite:
            logger.info("Skipping %s (results exist)", spec)
            return json.loads(results_path.read_text())

        set_seed(self.args.seed)
        tokenizer = AutoTokenizer.from_pretrained(self.args.model)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.args.model, num_labels=2, dtype=torch.float32)

        stages = spec.split("+")
        trainer: Optional[Trainer] = None
        stage_info: List[Dict] = []
        for si, stage in enumerate(stages):
            source, train_examples = self._parse_stage(stage)
            dev_examples = self._split(source, "dev")
            epochs = self.args.epochs if len(train_examples) > 1000 else self.args.few_shot_epochs
            logger.info("[%s] stage %d/%d: train on %s (%d examples, %d epochs)",
                        spec, si + 1, len(stages), stage, len(train_examples), epochs)
            if trainer is not None:
                # Free the previous stage's optimizer/trainer state before the
                # next stage allocates a new one (otherwise stage 2 OOMs on a
                # shared GPU).
                del trainer
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            trainer = self._fit_stage(model, tokenizer, train_examples,
                                      dev_examples, run_dir / f"stage{si + 1}", epochs)
            model = trainer.model
            stage_info.append({"stage": stage, "n_train": len(train_examples),
                               "epochs": epochs})

        # Evaluate on both test sets
        test_metrics: Dict[str, Dict] = {}
        for source in SOURCES:
            test_examples = self._split(source, "test")
            test_ds = self._tokenize(test_examples, tokenizer)
            pred = trainer.predict(test_ds)
            metrics = compute_metrics(EvalPrediction(
                predictions=pred.predictions, label_ids=pred.label_ids))
            test_metrics[source] = metrics
            logger.info("[%s] test on %s: %s", spec, source, metrics)

            preds = np.argmax(pred.predictions, axis=-1)
            probs = torch.softmax(torch.tensor(pred.predictions), dim=-1).numpy()[:, 1]
            with open(run_dir / f"predictions_{source}.jsonl", "w", encoding="utf-8") as fh:
                for ex, p, prob in zip(test_examples, preds, probs):
                    fh.write(json.dumps({
                        "example_id": ex["example_id"], "gold": ex["label"],
                        "pred": int(p), "prob_manipulative": round(float(prob), 5),
                    }) + "\n")

        results = {
            "spec": spec,
            "model": self.args.model,
            "stages": stage_info,
            "hyperparams": {"epochs": self.args.epochs,
                            "few_shot_epochs": self.args.few_shot_epochs,
                            "lr": self.args.lr, "batch_size": self.args.batch_size,
                            "max_length": MAX_LENGTH, "seed": self.args.seed},
            "test_metrics": test_metrics,
        }
        with open(results_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, default=float)
        logger.info("Results → %s", results_path)

        del model, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return results

    # ------------------------------------------------------------------
    # Drivers
    # ------------------------------------------------------------------

    def run_all(self, specs: List[str]) -> None:
        all_results: List[Dict] = []
        for spec in specs:
            try:
                all_results.append(self.run_spec(spec))
            except Exception as exc:
                import traceback
                logger.error("FAILED spec=%s : %s", spec, exc)
                traceback.print_exc()
        with open(self.output_dir / "summary.json", "w", encoding="utf-8") as fh:
            json.dump(all_results, fh, indent=2, default=float)
        self._print_summary(all_results)

    def aggregate_results(self) -> None:
        all_results = []
        for f in sorted(self.output_dir.glob("*/results.json")):
            all_results.append(json.loads(f.read_text()))
        self._print_summary(all_results)

    @staticmethod
    def _print_summary(all_results: List[Dict]) -> None:
        W = 108
        print(f"\n{'═' * W}")
        print("SYNTHETIC→REAL TRANSFER — SUMMARY (macro-F1 / AUROC per test set)")
        print(f"{'═' * W}")
        print(f"{'Train spec':<34} {'ConVAWG F1':>11} {'ConVAWG AUC':>12}"
              f" {'MentalManip F1':>15} {'MentalManip AUC':>16}")
        print("─" * W)
        for r in all_results:
            tm = r.get("test_metrics", {})
            cv, mm = tm.get("convawg", {}), tm.get("mentalmanip", {})
            print(f"{r['spec']:<34}"
                  f" {cv.get('f1_macro', float('nan')):>11.4f}"
                  f" {cv.get('auroc', float('nan')):>12.4f}"
                  f" {mm.get('f1_macro', float('nan')):>15.4f}"
                  f" {mm.get('auroc', float('nan')):>16.4f}")
        print(f"{'═' * W}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ConVAWG <-> MentalManip transfer / augmentation experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python train_transfer.py --matrix
  python train_transfer.py --specs convawg mentalmanip
  python train_transfer.py --specs convawg+mentalmanip:500
  python train_transfer.py --specs convawg --max_steps 20    # smoke test
  python train_transfer.py --aggregate_only
""",
    )
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--specs", nargs="+", default=None,
                        help="Train-source specs (see module docstring)")
    parser.add_argument("--matrix", action="store_true",
                        help="Run the standard 7-spec grid")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--few_shot_epochs", type=int, default=8,
                        help="Epochs for stages with <=1000 examples")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Per-device batch size (8 fits DeBERTa-v3 fp32 "
                             "at 512 tokens on ~11 GB of GPU memory)")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Limit optimiser steps per stage (smoke tests)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--aggregate_only", action="store_true")
    parser.add_argument("--gpu", type=str, default=None,
                        help="CUDA_VISIBLE_DEVICES value")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    experiment = TransferExperiment(args)
    if args.aggregate_only:
        experiment.aggregate_results()
        return
    specs = STANDARD_MATRIX if args.matrix else (args.specs or ["convawg"])
    experiment.run_all(specs)


if __name__ == "__main__":
    main()
