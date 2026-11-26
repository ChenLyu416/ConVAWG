#!/usr/bin/env python3
"""
Train and evaluate transformer-based baselines for perpetrator-victim
relationship classification on the VAWG-200 persona-relation dataset.

Supported models (select via --model_name):
  bert-base-uncased             (event-level, 512 tokens)
  roberta-base                  (event-level, 512 tokens)
  microsoft/deberta-v3-base     (event-level, 512 tokens)
  allenai/longformer-base-4096  (scenario-level, 4096 tokens)

Usage:
  python train_classifier.py --model_name bert-base-uncased
  python train_classifier.py --model_name allenai/longformer-base-4096
  python train_classifier.py --model_name bert-base-uncased --config configs/relation_clf.yaml

Outputs (per model, inside models/<safe_name>/):
  best checkpoint, tokenizer, training args

Results written to results/<safe_name>_results.json.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    PreTrainedModel,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

TASK_DIR = Path(__file__).parent
REPO_ROOT = TASK_DIR.parent.parent.parent


# ── Dataset ───────────────────────────────────────────────────────────────────


def load_jsonl(path: Path) -> List[Dict]:
    """Load a JSONL file into a list of dicts."""
    records = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


class RelationDataset(Dataset):
    """
    Tokenised dataset for relation classification.

    For Longformer, global_attention_mask is set to 1 on the [CLS] token so the
    model attends globally to the document representation token.
    """

    def __init__(
        self,
        records: List[Dict],
        tokenizer,
        max_length: int,
        is_longformer: bool = False,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_longformer = is_longformer

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        rec = self.records[idx]
        enc = self.tokenizer(
            rec["text"],
            max_length=self.max_length,
            truncation=True,
            padding=False,
        )
        item = {k: torch.tensor(v, dtype=torch.long) for k, v in enc.items()}
        item["labels"] = torch.tensor(rec["label_id"], dtype=torch.long)

        if self.is_longformer:
            # Global attention on first token ([CLS])
            gam = torch.zeros_like(item["input_ids"])
            gam[0] = 1
            item["global_attention_mask"] = gam

        return item


# ── Weighted-loss wrapper ─────────────────────────────────────────────────────


class WeightedTrainer(Trainer):
    """Trainer subclass that applies class-weighted cross-entropy loss."""

    def __init__(self, class_weights: torch.Tensor, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs: Dict,
        return_outputs: bool = False,
        **kwargs,
    ):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device)
        )
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ── Loss-logging callback ─────────────────────────────────────────────────────


class LossHistoryCallback(TrainerCallback):
    """Collects training and evaluation loss at each logging / eval step."""

    def __init__(self) -> None:
        self.train_loss_history: List[Dict] = []
        self.eval_loss_history: List[Dict] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        if "loss" in logs:
            self.train_loss_history.append(
                {"step": state.global_step, "train_loss": logs["loss"]}
            )
        if "eval_loss" in logs:
            self.eval_loss_history.append(
                {
                    "step": state.global_step,
                    "eval_loss": logs["eval_loss"],
                    "eval_macro_f1": logs.get("eval_macro_f1"),
                }
            )


# ── Metrics ───────────────────────────────────────────────────────────────────


def build_compute_metrics(id2label: Dict[int, str]):
    """Return a compute_metrics function that uses the provided label mapping."""

    def compute_metrics(eval_pred) -> Dict:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": float(accuracy_score(labels, preds)),
            "macro_f1": float(
                f1_score(labels, preds, average="macro", zero_division=0)
            ),
            "weighted_f1": float(
                f1_score(labels, preds, average="weighted", zero_division=0)
            ),
            "macro_precision": float(
                precision_score(labels, preds, average="macro", zero_division=0)
            ),
            "macro_recall": float(
                recall_score(labels, preds, average="macro", zero_division=0)
            ),
        }

    return compute_metrics


# ── Model trainer class ────────────────────────────────────────────────────────


class ModelTrainer:
    """
    Orchestrates tokenisation, training, evaluation, and result saving for one
    baseline model.
    """

    def __init__(self, model_cfg: Dict, global_cfg: Dict) -> None:
        self.model_name: str = model_cfg["name"]
        self.granularity: str = model_cfg["granularity"]  # "event" or "scenario"
        self.max_length: int = model_cfg["max_length"]
        self.batch_size: int = model_cfg["batch_size"]
        self.grad_accum: int = model_cfg.get("grad_accum", 1)
        self.lr: float = model_cfg["learning_rate"]
        self.weight_decay: float = model_cfg.get("weight_decay", 0.01)
        self.num_epochs: int = model_cfg["num_epochs"]
        self.warmup_ratio: float = model_cfg.get("warmup_ratio", 0.1)
        self.patience: int = model_cfg.get("early_stopping_patience", 3)

        train_cfg = global_cfg["training"]
        self.output_base: Path = REPO_ROOT / train_cfg["output_dir"]
        self.results_dir: Path = REPO_ROOT / train_cfg["results_dir"]
        # Per-model fp16 override takes precedence over global setting
        self.fp16: bool = model_cfg.get("fp16", train_cfg.get("fp16", True))
        self.logging_steps: int = train_cfg.get("logging_steps", 20)
        self.dataloader_workers: int = train_cfg.get("dataloader_num_workers", 2)

        data_cfg = global_cfg["data"]
        self._data_cfg = data_cfg
        self.dataset_dir: Path = REPO_ROOT / data_cfg["dataset_dir"]
        self.id2label: Dict[int, str] = {
            int(k): v for k, v in data_cfg["id2label"].items()
        }
        self.label2id: Dict[str, int] = {v: k for k, v in self.id2label.items()}
        self.num_labels: int = len(self.id2label)

        self.is_longformer: bool = "longformer" in self.model_name.lower()
        self.safe_name: str = self.model_name.replace("/", "__")
        self.model_dir: Path = self.output_base / self.safe_name

        self.output_base.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load_split(self, split_name: str) -> List[Dict]:
        prefix = f"{self.granularity}_{split_name}.jsonl"
        path = self.dataset_dir / prefix
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {path}\n"
                "Run build_dataset.py first."
            )
        records = load_jsonl(path)
        logger.info("  Loaded %d %s records from %s", len(records), split_name, prefix)
        return records

    # ── Class weights ─────────────────────────────────────────────────────────

    def _compute_class_weights(self, train_records: List[Dict]) -> torch.Tensor:
        train_labels = np.array([r["label_id"] for r in train_records])
        classes = np.arange(self.num_labels)
        weights = compute_class_weight(
            class_weight="balanced", classes=classes, y=train_labels
        )
        # Cap extreme weights to avoid destabilising training on singleton classes
        max_w = self._data_cfg.get("max_class_weight", 10.0)
        weights = np.clip(weights, a_min=None, a_max=max_w)
        logger.info("Class weights (capped at %.1f): %s", max_w,
                    dict(zip(self.id2label.values(), weights.tolist())))
        return torch.tensor(weights, dtype=torch.float32)

    # ── Training ──────────────────────────────────────────────────────────────

    def train_and_evaluate(self) -> Dict:
        """
        Full train → eval → test pipeline.
        Returns a results dict with all metrics and loss histories.
        """
        logger.info("=" * 70)
        logger.info("Model: %s  |  Granularity: %s  |  MaxLen: %d",
                    self.model_name, self.granularity, self.max_length)

        # ── 1. Load tokenizer ────────────────────────────────────────────────
        logger.info("Loading tokenizer …")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # ── 2. Load data ─────────────────────────────────────────────────────
        train_records = self._load_split("train")
        dev_records = self._load_split("dev")
        test_records = self._load_split("test")

        train_ds = RelationDataset(
            train_records, tokenizer, self.max_length, self.is_longformer
        )
        dev_ds = RelationDataset(
            dev_records, tokenizer, self.max_length, self.is_longformer
        )
        test_ds = RelationDataset(
            test_records, tokenizer, self.max_length, self.is_longformer
        )

        collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

        # ── 3. Class weights ─────────────────────────────────────────────────
        class_weights = self._compute_class_weights(train_records)

        # ── 4. Model ─────────────────────────────────────────────────────────
        logger.info("Loading model …")
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True,
        )

        # ── 5. Training args ─────────────────────────────────────────────────
        self.model_dir.mkdir(parents=True, exist_ok=True)
        # Compute warmup steps from ratio
        steps_per_epoch = max(1, len(train_ds) // (self.batch_size * self.grad_accum))
        total_steps = steps_per_epoch * self.num_epochs
        warmup_steps = max(1, int(total_steps * self.warmup_ratio))

        training_args = TrainingArguments(
            output_dir=str(self.model_dir),
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size * 2,
            gradient_accumulation_steps=self.grad_accum,
            learning_rate=self.lr,
            weight_decay=self.weight_decay,
            warmup_steps=warmup_steps,
            fp16=self.fp16 and torch.cuda.is_available(),
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=self.logging_steps,
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            dataloader_num_workers=self.dataloader_workers,
            report_to="none",
            save_total_limit=2,
            label_names=["labels"],
        )

        loss_callback = LossHistoryCallback()
        compute_metrics = build_compute_metrics(self.id2label)

        # ── 6. Trainer ────────────────────────────────────────────────────────
        trainer = WeightedTrainer(
            class_weights=class_weights,
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=dev_ds,
            processing_class=tokenizer,
            data_collator=collator,
            compute_metrics=compute_metrics,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=self.patience),
                loss_callback,
            ],
        )

        # ── 7. Train ──────────────────────────────────────────────────────────
        logger.info("Starting training …")
        train_result = trainer.train()
        logger.info("Training finished. Steps: %d", train_result.global_step)

        # Manually reload best checkpoint if load_best_model_at_end failed
        # (transformers 5.x saves safetensors but old code path seeks pytorch_model.bin)
        best_ckpt = trainer.state.best_model_checkpoint
        if best_ckpt and not (Path(best_ckpt) / "pytorch_model.bin").exists():
            safetensors_path = Path(best_ckpt) / "model.safetensors"
            if safetensors_path.exists():
                logger.info("Reloading best checkpoint from safetensors: %s", best_ckpt)
                trainer.model = trainer.model.__class__.from_pretrained(
                    best_ckpt,
                    num_labels=self.num_labels,
                    id2label=self.id2label,
                    label2id={v: k for k, v in self.id2label.items()},
                    ignore_mismatched_sizes=True,
                )
                trainer.model.to(trainer.args.device)

        # ── 8. Dev evaluation ─────────────────────────────────────────────────
        logger.info("Evaluating on dev set …")
        dev_metrics = trainer.evaluate(eval_dataset=dev_ds, metric_key_prefix="dev")
        logger.info("Dev metrics: %s", dev_metrics)

        # ── 9. Test evaluation ────────────────────────────────────────────────
        logger.info("Evaluating on test set …")
        test_output = trainer.predict(test_dataset=test_ds)
        test_preds = np.argmax(test_output.predictions, axis=-1)
        test_labels = test_output.label_ids

        label_names = [self.id2label[i] for i in range(self.num_labels)]
        clf_report = classification_report(
            test_labels,
            test_preds,
            labels=list(range(self.num_labels)),
            target_names=label_names,
            output_dict=True,
            zero_division=0,
        )

        test_metrics = {
            "accuracy": float(accuracy_score(test_labels, test_preds)),
            "macro_f1": float(f1_score(test_labels, test_preds, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(test_labels, test_preds, average="weighted", zero_division=0)),
            "macro_precision": float(precision_score(test_labels, test_preds, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(test_labels, test_preds, average="macro", zero_division=0)),
            "per_class": clf_report,
        }

        # ── 10. Collect results ───────────────────────────────────────────────
        results = {
            "model_name": self.model_name,
            "granularity": self.granularity,
            "max_length": self.max_length,
            "num_labels": self.num_labels,
            "id2label": self.id2label,
            "hyperparams": {
                "batch_size": self.batch_size,
                "grad_accum": self.grad_accum,
                "learning_rate": self.lr,
                "weight_decay": self.weight_decay,
                "num_epochs": self.num_epochs,
                "warmup_ratio": self.warmup_ratio,
                "total_train_steps": train_result.global_step,
                "best_epoch_metric": trainer.state.best_metric,
            },
            "train_samples": len(train_records),
            "dev_samples": len(dev_records),
            "test_samples": len(test_records),
            "train_loss_history": loss_callback.train_loss_history,
            "eval_loss_history": loss_callback.eval_loss_history,
            "dev_metrics": {
                k: v for k, v in dev_metrics.items() if not k.startswith("_")
            },
            "test_metrics": test_metrics,
        }

        # ── 11. Save results ──────────────────────────────────────────────────
        out_path = self.results_dir / f"{self.safe_name}_results.json"
        out_path.write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("Results saved → %s", out_path)

        # ── 12. Pretty print test results ─────────────────────────────────────
        self._print_results_table(test_metrics, dev_metrics)

        return results

    def _print_results_table(
        self, test_metrics: Dict, dev_metrics: Dict
    ) -> None:
        """Log a compact human-readable results summary."""
        logger.info("")
        logger.info("╔═══════════════════════════════════════════╗")
        logger.info("║  RESULTS: %-31s  ║", self.model_name.split("/")[-1])
        logger.info("╠═══════════════════════════════════════════╣")
        logger.info("║  Dev  loss      : %-24s ║",
                    f"{dev_metrics.get('dev_loss', 'N/A'):.4f}" if isinstance(dev_metrics.get("dev_loss"), float) else "N/A")
        logger.info("║  Dev  macro-F1  : %-24s ║",
                    f"{dev_metrics.get('dev_macro_f1', 0):.4f}")
        logger.info("╠═══════════════════════════════════════════╣")
        logger.info("║  Test accuracy  : %-24s ║",
                    f"{test_metrics['accuracy']:.4f}")
        logger.info("║  Test macro-F1  : %-24s ║",
                    f"{test_metrics['macro_f1']:.4f}")
        logger.info("║  Test wt'd-F1   : %-24s ║",
                    f"{test_metrics['weighted_f1']:.4f}")
        logger.info("║  Test precision : %-24s ║",
                    f"{test_metrics['macro_precision']:.4f}")
        logger.info("║  Test recall    : %-24s ║",
                    f"{test_metrics['macro_recall']:.4f}")
        logger.info("╚═══════════════════════════════════════════╝")
        # Per-class F1
        logger.info("  Per-class test F1:")
        for cls_name, row in test_metrics["per_class"].items():
            if isinstance(row, dict) and "f1-score" in row:
                logger.info("    %-22s  F1=%.4f  P=%.4f  R=%.4f  n=%d",
                            cls_name,
                            row["f1-score"],
                            row["precision"],
                            row["recall"],
                            int(row["support"]))


# ── Result comparator ─────────────────────────────────────────────────────────


def compare_all_results(results_dir: Path, id2label: Dict[int, str]) -> None:
    """Print a side-by-side comparison table for all saved result JSON files."""
    files = sorted(results_dir.glob("*_results.json"))
    if not files:
        logger.warning("No results found in %s", results_dir)
        return

    rows = []
    for fp in files:
        r = json.loads(fp.read_text())
        tm = r["test_metrics"]
        dm = r.get("dev_metrics", {})
        rows.append({
            "model": r["model_name"].split("/")[-1],
            "gran": r["granularity"][0].upper(),  # E or S
            "dev_loss": dm.get("dev_loss", float("nan")),
            "dev_mF1": dm.get("dev_macro_f1", float("nan")),
            "test_acc": tm["accuracy"],
            "test_mF1": tm["macro_f1"],
            "test_wF1": tm["weighted_f1"],
            "test_prec": tm["macro_precision"],
            "test_rec": tm["macro_recall"],
        })

    header = (
        f"{'Model':<28} {'G':>1}  {'DevLoss':>8}  {'DevF1':>6}  "
        f"{'TestAcc':>7}  {'TestMF1':>7}  {'TestWF1':>7}  "
        f"{'Prec':>6}  {'Recall':>6}"
    )
    sep = "-" * len(header)
    logger.info("")
    logger.info("ALL MODELS — TEST SET COMPARISON")
    logger.info(sep)
    logger.info(header)
    logger.info(sep)
    for row in rows:
        logger.info(
            "%-28s %1s  %8.4f  %6.4f  %7.4f  %7.4f  %7.4f  %6.4f  %6.4f",
            row["model"], row["gran"],
            row["dev_loss"] if not np.isnan(row["dev_loss"]) else -1.0,
            row["dev_mF1"] if not np.isnan(row["dev_mF1"]) else -1.0,
            row["test_acc"], row["test_mF1"], row["test_wF1"],
            row["test_prec"], row["test_rec"],
        )
    logger.info(sep)

    # Also save comparison JSON
    comp_path = results_dir / "comparison_table.json"
    comp_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    logger.info("Comparison table saved → %s", comp_path)


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train & evaluate baseline classifiers for persona-relation classification.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python train_classifier.py --model_name bert-base-uncased\n"
            "  python train_classifier.py --model_name allenai/longformer-base-4096\n"
            "  python train_classifier.py --compare  # print comparison across saved results\n"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "configs" / "relation_clf.yaml",
        help="Path to YAML config (default: configs/relation_clf.yaml)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Model to train (must match a name in config models list)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Print comparison table from saved result files then exit",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="CUDA_VISIBLE_DEVICES override (e.g. '0' or '0,1')",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        logger.info("CUDA_VISIBLE_DEVICES=%s", args.gpu)

    if not args.config.exists():
        logger.error("Config not found: %s", args.config)
        sys.exit(1)

    cfg = yaml.safe_load(args.config.read_text())

    # ── Compare mode ──────────────────────────────────────────────────────────
    if args.compare:
        results_dir = REPO_ROOT / cfg["training"]["results_dir"]
        id2label = {int(k): v for k, v in cfg["data"]["id2label"].items()}
        compare_all_results(results_dir, id2label)
        return

    # ── Find model config ──────────────────────────────────────────────────────
    model_cfgs: List[Dict] = cfg.get("models", [])
    if not model_cfgs:
        logger.error("No models defined in config.")
        sys.exit(1)

    if args.model_name is not None:
        matched = [m for m in model_cfgs if m["name"] == args.model_name]
        if not matched:
            available = [m["name"] for m in model_cfgs]
            logger.error(
                "Model '%s' not found in config. Available: %s",
                args.model_name, available,
            )
            sys.exit(1)
        model_cfgs = matched

    # ── Train each selected model ──────────────────────────────────────────────
    all_results = []
    for model_cfg in model_cfgs:
        trainer_obj = ModelTrainer(model_cfg, cfg)
        try:
            result = trainer_obj.train_and_evaluate()
            all_results.append(result)
        except Exception as exc:
            logger.error("Failed to train %s: %s", model_cfg["name"], exc, exc_info=True)

    # ── Final comparison (if multiple models ran) ─────────────────────────────
    if len(all_results) > 1 or args.model_name is None:
        results_dir = REPO_ROOT / cfg["training"]["results_dir"]
        id2label = {int(k): v for k, v in cfg["data"]["id2label"].items()}
        compare_all_results(results_dir, id2label)


if __name__ == "__main__":
    main()
