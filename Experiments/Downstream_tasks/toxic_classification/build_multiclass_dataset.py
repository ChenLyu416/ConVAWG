#!/usr/bin/env python3
"""
Build message-level multi-class toxic-type classification dataset from Stage-6 data.

Label scheme (8 classes)
-------------------------
  0  non-toxic    — victim messages (no behavior label assigned)
  1  surveillance — monitoring, tracking, location demands
  2  demand       — controlling demands / orders
  3  accusation   — blame, accusation, attack
  4  manipulation — emotional manipulation, coercion
  5  ultimatum    — conditional threats / ultimatums
  6  threat       — direct threats of harm
  7  other        — gaslighting, repair, request, refusal, escalation, other

Rare perpetrator labels (gaslighting, repair, request, refusal, escalation, other)
are merged into class 7 to prevent extremely imbalanced tail classes.

Split
-----
  Reuses the same scenario-level 70/15/15 split as the binary task
  (reads data/split_metadata.json) so results are directly comparable.

Outputs (JSONL)
---------------
  data_multiclass/setting1/{train,dev,test}.jsonl
  data_multiclass/setting2/{train,dev,test}.jsonl
"""

import argparse
import json
import logging
import collections
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]

# Pipeline stage-6 output (dialogues with per-message act labels); override with --data_dirs.
DEFAULT_DATA_DIRS = [
    str(REPO_ROOT / "Generation_Results" / "toxic_injected_dialogues"),
]

# Canonical label → class index
CORE_LABELS = {"surveillance": 1, "demand": 2, "accusation": 3,
               "manipulation": 4, "ultimatum": 5, "threat": 6}
OTHER_CLASS  = 7
NONTOXIC_CLASS = 0

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


def label_to_class(raw_label: Optional[str], is_perpetrator: bool) -> int:
    """Map a raw message label + role to a class index."""
    if not is_perpetrator:
        return NONTOXIC_CLASS
    if not raw_label or not isinstance(raw_label, str):
        return OTHER_CLASS
    key = raw_label.strip().lower()
    return CORE_LABELS.get(key, OTHER_CLASS)


# ---------------------------------------------------------------------------
# I/O helpers  (identical to build_dataset.py)
# ---------------------------------------------------------------------------

def load_scenario(json_path: Path) -> Optional[Dict]:
    try:
        with open(json_path, encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return data
        if isinstance(data, list):
            sid = json_path.stem.replace("_dialogues", "")
            return {"scenario_id": sid, "dialogues": data}
        return None
    except Exception as exc:
        logger.warning("Failed to load %s: %s", json_path, exc)
        return None


def save_jsonl(examples: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
    logger.info("Saved %d examples → %s", len(examples), path)


def _dialogue_lines(messages: List[Dict]) -> List[str]:
    lines = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        content = (m.get("content") or "").strip()
        if content:
            lines.append(f"{m.get('speaker', 'Unknown')}: {content}")
    return lines


# ---------------------------------------------------------------------------
# Example extraction
# ---------------------------------------------------------------------------

def extract_examples(scenario: Dict, scenario_id: str) -> Tuple[List[Dict], List[Dict]]:
    s1: List[Dict] = []
    s2: List[Dict] = []

    for dialogue in scenario.get("dialogues", []):
        if not isinstance(dialogue, dict):
            continue

        event_id = dialogue.get("event_id", "unknown")
        messages = (dialogue.get("caa_rewritten_messages")
                    or dialogue.get("labeled_messages")
                    or [])
        if not messages:
            continue

        all_lines = _dialogue_lines(messages)

        for idx, msg in enumerate(messages):
            if not isinstance(msg, dict):
                continue
            content = (msg.get("content") or "").strip()
            if not content:
                continue

            raw_label = msg.get("label")
            is_perp   = bool(msg.get("is_perpetrator", False))
            cls       = label_to_class(raw_label, is_perp)

            meta = {
                "scenario_id":   scenario_id,
                "event_id":      event_id,
                "speaker":       msg.get("speaker", ""),
                "speaker_role":  msg.get("speaker_role", ""),
                "is_perpetrator": is_perp,
                "raw_label":     (raw_label.strip().lower()
                                  if isinstance(raw_label, str) else None),
                "label":         cls,
                "class_name":    CLASS_NAMES[cls],
                # binary toxic flag kept for reference
                "toxicity_injected": bool(msg.get("toxicity_injected", False)),
            }

            s1.append({**meta, "text": content})

            other_lines = [ln for j, ln in enumerate(all_lines) if j != idx]
            s2.append({**meta, "text": content, "context": "\n".join(other_lines)})

    return s1, s2


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def log_stats(split_name: str, examples: List[Dict]) -> None:
    n = len(examples)
    if n == 0:
        logger.info("  %-8s  0 examples", split_name)
        return
    counts = collections.Counter(e["class_name"] for e in examples)
    logger.info("  %-8s  %d total", split_name, n)
    for cls_idx in range(NUM_CLASSES):
        name = CLASS_NAMES[cls_idx]
        cnt  = counts.get(name, 0)
        logger.info("    %-14s %5d  (%5.1f%%)", name, cnt, 100 * cnt / n)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build multi-class toxic-type message classification dataset."
    )
    parser.add_argument("--data_dirs", nargs="+", default=DEFAULT_DATA_DIRS)
    parser.add_argument("--output_dir",  default="data_multiclass")
    parser.add_argument("--split_meta",  default="data/split_metadata.json",
                        help="Path to binary split_metadata.json to reuse the same scenario split.")
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--dev_ratio",   type=float, default=0.15)
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # ------------------------------------------------------------------ #
    # Collect files                                                        #
    # ------------------------------------------------------------------ #
    json_files: List[Path] = []
    for d in args.data_dirs:
        found = sorted(Path(d).glob("*_dialogues.json"))
        logger.info("Found %d files in %s", len(found), d)
        json_files.extend(found)
    logger.info("Total JSON files: %d", len(json_files))

    # ------------------------------------------------------------------ #
    # Extract                                                              #
    # ------------------------------------------------------------------ #
    all_s1: Dict[str, List[Dict]] = {}
    all_s2: Dict[str, List[Dict]] = {}

    for json_path in tqdm(json_files, desc="Extracting"):
        scenario = load_scenario(json_path)
        if scenario is None:
            continue
        sid = str(scenario.get("scenario_id")
                  or json_path.stem.replace("_dialogues", ""))
        s1, s2 = extract_examples(scenario, sid)
        if s1:
            all_s1[sid] = s1
        if s2:
            all_s2[sid] = s2

    # ------------------------------------------------------------------ #
    # Split — reuse existing scenario split if available                  #
    # ------------------------------------------------------------------ #
    split_meta_path = Path(args.split_meta)
    if split_meta_path.exists():
        meta = json.load(open(split_meta_path))
        train_ids = [s for s in meta["train_scenario_ids"] if s in all_s1]
        dev_ids   = [s for s in meta["dev_scenario_ids"]   if s in all_s1]
        test_ids  = [s for s in meta["test_scenario_ids"]  if s in all_s1]
        logger.info("Reusing binary split: %d/%d/%d scenarios",
                    len(train_ids), len(dev_ids), len(test_ids))
    else:
        import random
        rng = random.Random(args.seed)
        ids = sorted(all_s1.keys())
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(n * args.train_ratio)
        n_dev   = int(n * args.dev_ratio)
        train_ids = ids[:n_train]
        dev_ids   = ids[n_train:n_train + n_dev]
        test_ids  = ids[n_train + n_dev:]
        logger.info("New split: %d/%d/%d scenarios", len(train_ids), len(dev_ids), len(test_ids))

    # ------------------------------------------------------------------ #
    # Assemble & save                                                      #
    # ------------------------------------------------------------------ #
    for setting_key, pool in [("setting1", all_s1), ("setting2", all_s2)]:
        logger.info("\n=== %s ===", setting_key)
        train_ex = [ex for sid in train_ids for ex in pool.get(sid, [])]
        dev_ex   = [ex for sid in dev_ids   for ex in pool.get(sid, [])]
        test_ex  = [ex for sid in test_ids  for ex in pool.get(sid, [])]

        log_stats("train", train_ex)
        log_stats("dev",   dev_ex)
        log_stats("test",  test_ex)

        save_jsonl(train_ex, output_dir / setting_key / "train.jsonl")
        save_jsonl(dev_ex,   output_dir / setting_key / "dev.jsonl")
        save_jsonl(test_ex,  output_dir / setting_key / "test.jsonl")

    # Save class map
    meta_out = {
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "label_map": {str(k): v for k, v in CORE_LABELS.items()},
        "other_class": OTHER_CLASS,
        "nontoxic_class": NONTOXIC_CLASS,
    }
    (output_dir / "class_metadata.json").write_text(
        json.dumps(meta_out, indent=2), encoding="utf-8"
    )
    logger.info("Saved class metadata → %s/class_metadata.json", output_dir)


if __name__ == "__main__":
    main()
