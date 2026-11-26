#!/usr/bin/env python3
"""
Build persona-relation classification dataset from VAWG-200 dialogue files.

Produces two JSONL dataset variants from the 200-scenario ConVAWG release:
  • event-level   (one sample per dialogue event)  → used for BERT / RoBERTa / DeBERTa
  • scenario-level (one sample per full scenario)   → used for Longformer

Label: perpetrator_profile.relationship_to_victim from each scenario's
       vawg_scenarios block.  Minority classes (< min_class_size scenarios)
       are merged into the "family_or_other" class.

Split strategy: scenarios are partitioned into train / dev / test (stratified),
then events inherit the scenario's split to prevent data leakage.

Outputs (in dataset/):
  event_train.jsonl, event_dev.jsonl, event_test.jsonl
  scenario_train.jsonl, scenario_dev.jsonl, scenario_test.jsonl
  label_info.json   — id2label mapping + class counts per split
  split_manifest.json — scenario_id → split assignment
"""

import argparse
import json
import logging
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
# Skip these raw relationship strings
_SKIP_LABELS = {"unknown", ""}


class DatasetBuilder:
    """Builds event-level and scenario-level JSONL datasets for relation classification."""

    def __init__(self, cfg: Dict) -> None:
        self.cfg = cfg
        data_cfg = cfg["data"]
        self.repo_root = Path(__file__).parent.parent.parent.parent
        self.dialogue_dirs: List[Path] = [
            self.repo_root / d for d in data_cfg["dialogue_dirs"]
        ]
        self.dataset_dir = self.repo_root / data_cfg["dataset_dir"]
        self.train_ratio = data_cfg["train_ratio"]
        self.dev_ratio = data_cfg["dev_ratio"]
        self.test_ratio = data_cfg["test_ratio"]
        self.seed = data_cfg["random_seed"]
        self.min_class_size = data_cfg.get("min_class_size", 4)

        # Raw label map from config; rare classes collapse to highest id
        raw_label_map: Dict[str, int] = data_cfg["label_map"]
        self.raw_label_map: Dict[str, int] = {k.lower(): v for k, v in raw_label_map.items()}
        self.id2label: Dict[int, str] = {
            int(k): v for k, v in data_cfg["id2label"].items()
        }
        self.num_labels = len(self.id2label)

        self.dataset_dir.mkdir(parents=True, exist_ok=True)

    # ── Data loading ──────────────────────────────────────────────────────────

    def _normalise_label(self, raw: str) -> Optional[str]:
        """Return canonical label string or None if it should be skipped."""
        raw = raw.strip().lower()
        if raw in _SKIP_LABELS:
            return None
        return self.raw_label_map.get(raw, None)

    def load_all_scenarios(self) -> Dict[str, Dict]:
        """
        Scan dialogue directories and return {scenario_id -> scenario_record}.
        Deduplicates by scenario_id (first seen wins).
        """
        scenarios: Dict[str, Dict] = {}
        for dialogue_dir in self.dialogue_dirs:
            if not dialogue_dir.exists():
                logger.warning("Dialogue directory not found: %s", dialogue_dir)
                continue
            files = sorted(dialogue_dir.glob("*_dialogues.json"))
            logger.info("Scanning %d files in %s", len(files), dialogue_dir.name)
            for f in tqdm(files, desc=dialogue_dir.name, unit="file"):
                scenario_id = f.stem.replace("_dialogues", "")
                if scenario_id in scenarios:
                    continue  # deduplicate
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                except Exception as exc:
                    logger.warning("Failed to read %s: %s", f, exc)
                    continue

                raw_rel = (
                    data.get("vawg_scenarios", {})
                    .get("perpetrator_profile", {})
                    .get("relationship_to_victim", "")
                )
                label_id = self._normalise_label(raw_rel)
                if label_id is None:
                    logger.debug("Skipping %s — unknown label %r", scenario_id, raw_rel)
                    continue

                events = [
                    ev
                    for ev in data.get("dialogues", [])
                    if isinstance(ev.get("enforced_dialogue"), str)
                    and ev["enforced_dialogue"].strip()
                ]
                if not events:
                    logger.debug("Skipping %s — no valid event dialogues", scenario_id)
                    continue

                scenarios[scenario_id] = {
                    "scenario_id": scenario_id,
                    "raw_relationship": raw_rel,
                    "label_id": label_id,
                    "label": self.id2label[label_id],
                    "events": events,
                }

        logger.info("Loaded %d unique scenarios", len(scenarios))
        return scenarios

    # ── Label collapsing ──────────────────────────────────────────────────────

    def collapse_rare_classes(self, scenarios: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Merge classes with fewer than self.min_class_size scenarios into
        'family_or_other' (the highest label_id).
        Modifies scenarios in-place and returns updated dict.
        """
        counts = Counter(s["label_id"] for s in scenarios.values())
        other_id = max(self.id2label.keys())
        for label_id, cnt in counts.items():
            if cnt < self.min_class_size:
                label_name = self.id2label[label_id]
                logger.info(
                    "Collapsing class '%s' (n=%d < min=%d) → class %d (%s)",
                    label_name, cnt, self.min_class_size, other_id, self.id2label[other_id],
                )
                for s in scenarios.values():
                    if s["label_id"] == label_id:
                        s["label_id"] = other_id
                        s["label"] = self.id2label[other_id]
        return scenarios

    # ── Splitting ─────────────────────────────────────────────────────────────

    def _stratified_split(
        self,
        scenario_ids: List[str],
        labels: List[int],
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Stratified train / dev / test split at scenario level.
        Classes with only 1 or 2 samples are force-assigned to train.
        """
        rng = random.Random(self.seed)
        by_label: Dict[int, List[str]] = defaultdict(list)
        for sid, lbl in zip(scenario_ids, labels):
            by_label[lbl].append(sid)

        train_ids, dev_ids, test_ids = [], [], []

        for lbl, ids in sorted(by_label.items()):
            ids = ids.copy()
            rng.shuffle(ids)
            n = len(ids)

            if n <= 3:
                # Too few for a meaningful held-out split — keep in train
                logger.info(
                    "Class %d (%s) has only %d samples → all in train",
                    lbl, self.id2label[lbl], n,
                )
                train_ids.extend(ids)
                continue

            n_test = max(1, round(n * self.test_ratio))
            n_dev = max(1, round(n * self.dev_ratio))
            n_train = n - n_test - n_dev
            if n_train < 1:
                n_train = 1
                n_test = max(0, n - n_train - n_dev)
                n_dev = n - n_train - n_test

            train_ids.extend(ids[:n_train])
            dev_ids.extend(ids[n_train : n_train + n_dev])
            test_ids.extend(ids[n_train + n_dev :])

        rng.shuffle(train_ids)
        rng.shuffle(dev_ids)
        rng.shuffle(test_ids)
        return train_ids, dev_ids, test_ids

    # ── JSONL writers ─────────────────────────────────────────────────────────

    def _write_event_split(
        self,
        scenarios: Dict[str, Dict],
        split_ids: List[str],
        split_name: str,
    ) -> int:
        """Write event-level JSONL for the given scenario split."""
        out_path = self.dataset_dir / f"event_{split_name}.jsonl"
        count = 0
        with out_path.open("w", encoding="utf-8") as fh:
            for sid in split_ids:
                s = scenarios[sid]
                for ev in s["events"]:
                    text = ev["enforced_dialogue"].strip()
                    if not text:
                        continue
                    record = {
                        "scenario_id": sid,
                        "event_id": ev.get("event_id", ""),
                        "text": text,
                        "label_id": s["label_id"],
                        "label": s["label"],
                        "split": split_name,
                        "raw_relationship": s["raw_relationship"],
                    }
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1
        logger.info("  %s event split → %d samples → %s", split_name, count, out_path.name)
        return count

    def _write_scenario_split(
        self,
        scenarios: Dict[str, Dict],
        split_ids: List[str],
        split_name: str,
        max_chars: int = 20_000,
    ) -> int:
        """
        Write scenario-level JSONL for the given scenario split.
        All events are concatenated; the text is capped at max_chars characters
        so that even the largest files don't produce excessively long strings.
        """
        out_path = self.dataset_dir / f"scenario_{split_name}.jsonl"
        count = 0
        with out_path.open("w", encoding="utf-8") as fh:
            for sid in split_ids:
                s = scenarios[sid]
                parts = [
                    ev["enforced_dialogue"].strip()
                    for ev in s["events"]
                    if ev["enforced_dialogue"].strip()
                ]
                full_text = "\n\n".join(parts)[:max_chars]
                if not full_text:
                    continue
                record = {
                    "scenario_id": sid,
                    "text": full_text,
                    "num_events": len(s["events"]),
                    "label_id": s["label_id"],
                    "label": s["label"],
                    "split": split_name,
                    "raw_relationship": s["raw_relationship"],
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
        logger.info("  %s scenario split → %d samples → %s", split_name, count, out_path.name)
        return count

    # ── Main build ────────────────────────────────────────────────────────────

    def build(self) -> None:
        """Full pipeline: load → collapse → split → write → stats."""
        scenarios = self.load_all_scenarios()
        scenarios = self.collapse_rare_classes(scenarios)

        # Overall class counts
        label_counts = Counter(s["label_id"] for s in scenarios.values())
        logger.info("Class distribution: %s", dict(sorted(label_counts.items())))

        scenario_ids = list(scenarios.keys())
        labels = [scenarios[s]["label_id"] for s in scenario_ids]

        train_ids, dev_ids, test_ids = self._stratified_split(scenario_ids, labels)
        logger.info(
            "Scenario split: train=%d  dev=%d  test=%d  total=%d",
            len(train_ids), len(dev_ids), len(test_ids),
            len(train_ids) + len(dev_ids) + len(test_ids),
        )

        # Write event-level splits
        logger.info("Writing event-level splits …")
        stats: Dict = {"event": {}, "scenario": {}}
        for name, ids in [("train", train_ids), ("dev", dev_ids), ("test", test_ids)]:
            cnt = self._write_event_split(scenarios, ids, name)
            stats["event"][name] = cnt

        # Write scenario-level splits
        logger.info("Writing scenario-level splits …")
        for name, ids in [("train", train_ids), ("dev", dev_ids), ("test", test_ids)]:
            cnt = self._write_scenario_split(scenarios, ids, name)
            stats["scenario"][name] = cnt

        # Per-split class counts
        split_class_counts: Dict[str, Dict] = {}
        for split_name, split_ids in [
            ("train", train_ids), ("dev", dev_ids), ("test", test_ids)
        ]:
            counter = Counter(scenarios[s]["label_id"] for s in split_ids)
            split_class_counts[split_name] = {
                self.id2label[k]: v for k, v in sorted(counter.items())
            }
        stats["class_distribution"] = split_class_counts

        # Save manifest: scenario_id → split
        manifest: Dict[str, str] = {}
        for name, ids in [("train", train_ids), ("dev", dev_ids), ("test", test_ids)]:
            for sid in ids:
                manifest[sid] = name
        manifest_path = self.dataset_dir / "split_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        logger.info("Split manifest → %s", manifest_path.name)

        # Save label info
        label_info = {
            "id2label": self.id2label,
            "label2id": {v: k for k, v in self.id2label.items()},
            "num_labels": self.num_labels,
            "raw_label_map": self.raw_label_map,
            "corpus_class_counts": {
                self.id2label[k]: v for k, v in sorted(label_counts.items())
            },
            "sample_counts": stats,
        }
        label_path = self.dataset_dir / "label_info.json"
        label_path.write_text(json.dumps(label_info, indent=2, ensure_ascii=False))
        logger.info("Label info → %s", label_path.name)

        # Summary
        logger.info("=" * 60)
        logger.info("Dataset build complete.")
        logger.info("  Classes: %d", self.num_labels)
        logger.info("  Event-level  — train: %d  dev: %d  test: %d",
                    stats["event"]["train"], stats["event"]["dev"], stats["event"]["test"])
        logger.info("  Scenario-level — train: %d  dev: %d  test: %d",
                    stats["scenario"]["train"], stats["scenario"]["dev"], stats["scenario"]["test"])
        for split_name, cc in split_class_counts.items():
            logger.info("  Class counts [%s]: %s", split_name, cc)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build persona-relation classification dataset from VAWG-200 dialogues.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python build_dataset.py\n"
            "  python build_dataset.py --config configs/relation_clf.yaml\n"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "configs" / "relation_clf.yaml",
        help="Path to YAML config file (default: configs/relation_clf.yaml)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.config.exists():
        logger.error("Config not found: %s", args.config)
        sys.exit(1)

    cfg = yaml.safe_load(args.config.read_text())
    builder = DatasetBuilder(cfg)
    builder.build()


if __name__ == "__main__":
    main()
