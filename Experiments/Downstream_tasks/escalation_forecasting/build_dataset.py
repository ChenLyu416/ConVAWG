#!/usr/bin/env python3
"""
Build the escalation-forecasting dataset from the ConVAWG dialogues.

Task definition (conversation-derailment-style forecasting)
-----------------------------------------------------------
For every perpetrator-victim conversation event with at least
``--min-history`` preceding *labelled* events, one forecasting example is
created:

  input   — the full chronological history of conversations STRICTLY BEFORE
            the target event (perpetrator-victim and third-party alike),
            rendered as plain chat text with timestamps and participants.
            The target event's own text is never shown (pure forecasting).
  labels  — next_level    : escalation level (0-4) of the target event
            jump          : 1 if next_level > cur_level (imminent escalation)
            future_max_h3 : max escalation level over the next 3 labelled
                            events (horizon analysis)

Escalation levels follow the pipeline rubric (Appendix B.4 of the paper):
  0 normal interaction, 1 emerging control, 2 manipulation and monitoring,
  3 coercive intimidation, 4 severe threats or violent escalation.

History events do NOT carry escalation annotations in the rendered text —
models must infer risk from the raw conversations, as in the Conversations
Gone Awry / CRAFT forecasting protocol (dynamic evaluation at every prefix).

Split
-----
Scenario-level 70/15/15 (seed 42) so no scenario leaks across splits.

Outputs (JSONL)
---------------
  data/{train,dev,test}.jsonl
  data/split_manifest.json
  data/stats.json
"""

import argparse
import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIRS = [
    REPO_ROOT / "data" / "ConVAWG" / "dialogues",
]

ESCALATION_RUBRIC = {
    0: "normal interaction",
    1: "emerging control",
    2: "manipulation and monitoring",
    3: "coercive intimidation",
    4: "severe threats or violent escalation",
}


def load_scenario(json_path: Path) -> Optional[Dict]:
    try:
        with open(json_path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else None
    except Exception as exc:
        logger.warning("Failed to load %s: %s", json_path, exc)
        return None


def render_event_block(idx: int, dialogue: Dict) -> str:
    """Render one conversation event as a plain-text block."""
    parts = dialogue.get("participants") or {}
    sender = parts.get("sender", "Unknown")
    receiver = parts.get("receiver", "Unknown")
    ts = (dialogue.get("timestamp") or "unknown time").replace("T", " ")
    text = (dialogue.get("enforced_dialogue") or "").strip()
    header = f"[conversation {idx} | {ts} | {sender} -> {receiver}]"
    return f"{header}\n{text}"


def ordered_events(scenario: Dict) -> List[Dict]:
    """Return dialogue events sorted chronologically (stable on ties)."""
    dialogues = [d for d in scenario.get("dialogues", [])
                 if (d.get("enforced_dialogue") or "").strip()]
    indexed = sorted(
        enumerate(dialogues),
        key=lambda pair: (pair[1].get("timestamp") or "", pair[0]),
    )
    return [d for _, d in indexed]


def escalation_of(dialogue: Dict) -> Optional[int]:
    level = (dialogue.get("conversation_context") or {}).get("escalation_level")
    return level if isinstance(level, int) and 0 <= level <= 4 else None


class ForecastDatasetBuilder:
    """Builds prefix→next-event escalation forecasting examples."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.data_dirs = [Path(d) for d in args.data_dirs]
        self.out_dir = Path(args.out_dir)
        self.min_history = args.min_history
        self.horizon = args.horizon
        self.seed = args.seed
        self.train_ratio, self.dev_ratio = args.train_ratio, args.dev_ratio
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Example extraction
    # ------------------------------------------------------------------

    def extract_scenario_examples(self, scenario: Dict, scenario_id: str) -> List[Dict]:
        events = ordered_events(scenario)
        victim = scenario.get("victim_name", "the victim")
        perp = scenario.get("perpetrator_name", "the perpetrator")

        blocks = [render_event_block(i + 1, ev) for i, ev in enumerate(events)]
        levels = [escalation_of(ev) for ev in events]
        labelled_positions = [i for i, lv in enumerate(levels) if lv is not None]

        examples: List[Dict] = []
        for rank, pos in enumerate(labelled_positions):
            n_prior_labelled = rank  # labelled events strictly before this one
            if n_prior_labelled < self.min_history:
                continue
            cur_pos = labelled_positions[rank - 1]
            cur_level = levels[cur_pos]
            next_level = levels[pos]
            future = [levels[p] for p in labelled_positions[rank: rank + self.horizon]]
            examples.append({
                "example_id": f"{scenario_id}_t{pos}",
                "scenario_id": scenario_id,
                "victim_name": victim,
                "perpetrator_name": perp,
                "target_event_id": events[pos].get("event_id"),
                "target_pos": pos,
                "target_timestamp": events[pos].get("timestamp"),
                "n_history_events": pos,
                "n_history_labelled": n_prior_labelled,
                "history": blocks[:pos],
                "cur_level": cur_level,
                "next_level": next_level,
                "jump": int(next_level > cur_level),
                "future_max_h3": max(future),
            })
        return examples

    # ------------------------------------------------------------------
    # Split & save
    # ------------------------------------------------------------------

    def scenario_split(self, scenario_ids: List[str]) -> Dict[str, str]:
        rng = random.Random(self.seed)
        ids = sorted(scenario_ids)
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(n * self.train_ratio)
        n_dev = int(n * self.dev_ratio)
        assignment: Dict[str, str] = {}
        for i, sid in enumerate(ids):
            if i < n_train:
                assignment[sid] = "train"
            elif i < n_train + n_dev:
                assignment[sid] = "dev"
            else:
                assignment[sid] = "test"
        return assignment

    @staticmethod
    def save_jsonl(examples: List[Dict], path: Path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            for ex in examples:
                fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
        logger.info("Saved %d examples → %s", len(examples), path)

    def run(self) -> None:
        files: List[Path] = []
        for d in self.data_dirs:
            if not d.exists():
                logger.warning("Data dir not found: %s", d)
                continue
            files.extend(sorted(d.glob("*_dialogues.json")))
        logger.info("Found %d scenario files", len(files))

        by_scenario: Dict[str, List[Dict]] = {}
        for f in tqdm(files, desc="scenarios", unit="file"):
            scenario = load_scenario(f)
            if scenario is None:
                continue
            sid = str(scenario.get("scenario_id") or f.stem.replace("_dialogues", ""))
            if sid in by_scenario:
                continue  # deduplicate
            by_scenario[sid] = self.extract_scenario_examples(scenario, sid)

        assignment = self.scenario_split(list(by_scenario))
        splits: Dict[str, List[Dict]] = {"train": [], "dev": [], "test": []}
        for sid, examples in by_scenario.items():
            splits[assignment[sid]].extend(examples)

        for name, examples in splits.items():
            examples.sort(key=lambda e: (e["scenario_id"], e["target_pos"]))
            self.save_jsonl(examples, self.out_dir / f"{name}.jsonl")

        with open(self.out_dir / "split_manifest.json", "w", encoding="utf-8") as fh:
            json.dump(assignment, fh, indent=2)

        stats = {
            "n_scenarios": len(by_scenario),
            "min_history": self.min_history,
            "horizon": self.horizon,
            "seed": self.seed,
            "escalation_rubric": ESCALATION_RUBRIC,
            "splits": {},
        }
        for name, examples in splits.items():
            stats["splits"][name] = {
                "n_examples": len(examples),
                "n_scenarios": len({e["scenario_id"] for e in examples}),
                "next_level_dist": dict(sorted(Counter(e["next_level"] for e in examples).items())),
                "jump_dist": dict(sorted(Counter(e["jump"] for e in examples).items())),
            }
        with open(self.out_dir / "stats.json", "w", encoding="utf-8") as fh:
            json.dump(stats, fh, indent=2)
        logger.info("Stats: %s", json.dumps(stats["splits"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build escalation-forecasting dataset from ConVAWG dialogues.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python build_dataset.py
  python build_dataset.py --data_dirs ../../../data/ConVAWG/dialogues
  python build_dataset.py --min-history 3 --horizon 5 --out_dir data_h5
""",
    )
    parser.add_argument("--data_dirs", nargs="+", default=[str(p) for p in DEFAULT_DATA_DIRS])
    parser.add_argument("--out_dir", default="data")
    parser.add_argument("--min-history", type=int, default=2,
                        help="Minimum labelled events before a target (default 2)")
    parser.add_argument("--horizon", type=int, default=3,
                        help="Horizon (labelled events) for future_max label (default 3)")
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--dev_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ForecastDatasetBuilder(args).run()


if __name__ == "__main__":
    main()
