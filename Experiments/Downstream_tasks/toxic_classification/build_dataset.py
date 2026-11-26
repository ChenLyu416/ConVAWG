#!/usr/bin/env python3
"""
Build message-level binary toxic classification dataset from Stage 6 dialogue data.

Label convention
----------------
  1 (toxic)     — message has `toxicity_injected: true` (CAA-steered rewrite succeeded)
  0 (non-toxic) — all other messages

Two classification settings
----------------------------
  setting1 — single message text only
  setting2 — full event dialogue *excluding* the target message as context
              passed as the first sequence, target message as the second

Split strategy
--------------
  Stratified 70 / 15 / 15 split at the *scenario* (file) level to prevent
  any data leakage between train / dev / test.

Outputs (JSONL)
---------------
  <output_dir>/setting1/{train,dev,test}.jsonl
  <output_dir>/setting2/{train,dev,test}.jsonl
  <output_dir>/split_metadata.json
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]

# Pipeline stage-6 output (dialogues with per-message act labels); override with --data_dirs.
DEFAULT_DATA_DIRS = [
    str(REPO_ROOT / "Generation_Results" / "toxic_injected_dialogues"),
]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_scenario(json_path: Path) -> Optional[Dict]:
    """Load a scenario JSON file; normalise list vs. dict formats."""
    try:
        with open(json_path, encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return data
        if isinstance(data, list):
            sid = json_path.stem.replace("_dialogues", "")
            return {"scenario_id": sid, "dialogues": data}
        logger.warning("Unexpected top-level type in %s", json_path)
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


# ---------------------------------------------------------------------------
# Example extraction
# ---------------------------------------------------------------------------

def _dialogue_lines(messages: List[Dict]) -> List[str]:
    """Format a list of message dicts as 'Speaker: content' lines."""
    lines = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        content = (m.get("content") or "").strip()
        if content:
            lines.append(f"{m.get('speaker', 'Unknown')}: {content}")
    return lines


def extract_examples(
    scenario: Dict,
    scenario_id: str,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Return (setting1_examples, setting2_examples) for one scenario.

    setting1 fields: text, label, scenario_id, event_id, speaker,
                     speaker_role, is_perpetrator
    setting2 adds  : context  (all other messages in the event as plain text)
    """
    s1: List[Dict] = []
    s2: List[Dict] = []

    for dialogue in scenario.get("dialogues", []):
        if not isinstance(dialogue, dict):
            continue

        event_id: str = dialogue.get("event_id", "unknown")

        # Prefer CAA-rewritten messages; fall back to labeled_messages
        messages: List[Dict] = (
            dialogue.get("caa_rewritten_messages")
            or dialogue.get("labeled_messages")
            or []
        )
        if not messages:
            continue

        all_lines = _dialogue_lines(messages)

        for idx, msg in enumerate(messages):
            if not isinstance(msg, dict):
                continue
            content = (msg.get("content") or "").strip()
            if not content:
                continue

            label = 1 if msg.get("toxicity_injected") else 0

            meta = {
                "scenario_id": scenario_id,
                "event_id": event_id,
                "speaker": msg.get("speaker", ""),
                "speaker_role": msg.get("speaker_role", ""),
                "is_perpetrator": bool(msg.get("is_perpetrator", False)),
                "label": label,
            }

            # Setting 1 — text only
            s1.append({**meta, "text": content})

            # Setting 2 — context = every other message in this event
            other_lines = [ln for j, ln in enumerate(all_lines) if j != idx]
            s2.append({**meta, "text": content, "context": "\n".join(other_lines)})

    return s1, s2


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def split_scenarios(
    scenario_ids: List[str],
    train_ratio: float = 0.70,
    dev_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    rng = random.Random(seed)
    ids = sorted(scenario_ids)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * train_ratio)
    n_dev = int(n * dev_ratio)
    return ids[:n_train], ids[n_train : n_train + n_dev], ids[n_train + n_dev :]


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def log_stats(split_name: str, examples: List[Dict]) -> None:
    n = len(examples)
    if n == 0:
        logger.info("  %-8s  0 examples", split_name)
        return
    n_pos = sum(e["label"] for e in examples)
    logger.info(
        "  %-8s  %6d total | %5d toxic (%5.1f%%) | %6d non-toxic (%5.1f%%)",
        split_name, n, n_pos, 100 * n_pos / n, n - n_pos, 100 * (n - n_pos) / n,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build binary toxic message classification dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python build_dataset.py\n"
            "  python build_dataset.py --output_dir data/ --seed 0\n"
        ),
    )
    parser.add_argument(
        "--data_dirs", nargs="+", default=DEFAULT_DATA_DIRS,
        help="Folders containing *_dialogues.json Stage-6 files",
    )
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--dev_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # ------------------------------------------------------------------ #
    # Collect JSON files                                                   #
    # ------------------------------------------------------------------ #
    json_files: List[Path] = []
    for data_dir in args.data_dirs:
        p = Path(data_dir)
        found = sorted(p.glob("*_dialogues.json"))
        logger.info("Found %d files in %s", len(found), p)
        json_files.extend(found)
    logger.info("Total JSON files: %d", len(json_files))

    # ------------------------------------------------------------------ #
    # Load and extract                                                     #
    # ------------------------------------------------------------------ #
    all_s1: Dict[str, List[Dict]] = {}
    all_s2: Dict[str, List[Dict]] = {}

    for json_path in tqdm(json_files, desc="Extracting examples"):
        scenario = load_scenario(json_path)
        if scenario is None:
            continue
        sid = str(
            scenario.get("scenario_id")
            or json_path.stem.replace("_dialogues", "")
        )
        s1, s2 = extract_examples(scenario, sid)
        if s1:
            all_s1[sid] = s1
        if s2:
            all_s2[sid] = s2

    scenario_ids = sorted(all_s1.keys())
    logger.info("Scenarios with examples: %d", len(scenario_ids))

    # ------------------------------------------------------------------ #
    # Split                                                                #
    # ------------------------------------------------------------------ #
    train_ids, dev_ids, test_ids = split_scenarios(
        scenario_ids, args.train_ratio, args.dev_ratio, args.seed
    )
    logger.info(
        "Scenario split: %d train / %d dev / %d test",
        len(train_ids), len(dev_ids), len(test_ids),
    )

    # ------------------------------------------------------------------ #
    # Assemble and save each setting                                       #
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

    # ------------------------------------------------------------------ #
    # Save split metadata                                                  #
    # ------------------------------------------------------------------ #
    meta = {
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "dev_ratio": args.dev_ratio,
        "test_ratio": round(1 - args.train_ratio - args.dev_ratio, 4),
        "total_scenarios": len(scenario_ids),
        "n_train_scenarios": len(train_ids),
        "n_dev_scenarios": len(dev_ids),
        "n_test_scenarios": len(test_ids),
        "train_scenario_ids": train_ids,
        "dev_scenario_ids": dev_ids,
        "test_scenario_ids": test_ids,
        "source_dirs": [str(d) for d in args.data_dirs],
    }
    meta_path = output_dir / "split_metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    logger.info("Split metadata saved → %s", meta_path)
    logger.info("Dataset build complete.")


if __name__ == "__main__":
    main()
