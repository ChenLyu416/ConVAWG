#!/usr/bin/env python3
"""
Build the synthetic-to-real transfer datasets (ConVAWG <-> MentalManip).

Shared task: dialogue-level binary detection of manipulative / abusive
conversation.

Source A — ConVAWG (synthetic, this repo)
-----------------------------------------
  One example per dialogue event of the ConVAWG release.  Label mapping from the
  generation-time escalation rubric (Appendix B.4):
    positive (1) — escalation_level >= 2  (manipulation-and-monitoring onwards)
    negative (0) — escalation_level <= 1, and third-party support dialogues
                   (no escalation annotation; hard negatives that *discuss*
                   abuse without enacting it)
  Scenario-level 70/15/15 split (seed 42) to prevent leakage.

Source B — MentalManip (real-derived; Wang et al., ACL 2024)
------------------------------------------------------------
  2,915 consensus-labelled movie dialogues (config `mentalmanip_con` on
  HuggingFace: audreyeleven/MentalManip), field `manipulative` in {0,1}.
  Stratified dialogue-level 70/15/15 split (seed 42).

Format alignment (reduces surface-domain gap so the transfer measures
content generalisation, not formatting artifacts):
  * Speakers in BOTH datasets are anonymised to Person1 / Person2 / Person3...
    in order of first appearance ("Personk: message" lines).
  * One dialogue per example, no metadata in the text.

Outputs (JSONL: {"text", "label", "source", "example_id", ...})
  data/convawg/{train,dev,test}.jsonl      + split_manifest.json
  data/mentalmanip/{train,dev,test}.jsonl
  data/stats.json
"""

import argparse
import json
import logging
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONVAWG_DIRS = [
    REPO_ROOT / "data" / "ConVAWG" / "dialogues",
]
MENTALMANIP_HF = "audreyeleven/MentalManip"

SPEAKER_LINE = re.compile(r"^\s*([^:]{1,40}):\s*(.*)$")


def anonymize_dialogue(text: str) -> Optional[str]:
    """Rewrite 'Name: msg' lines with Person1/Person2/... speaker tags."""
    mapping: Dict[str, str] = {}
    lines_out: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = SPEAKER_LINE.match(line)
        if not m:
            # continuation line — append to previous message
            if lines_out:
                lines_out[-1] += " " + line
            continue
        speaker, msg = m.group(1).strip(), m.group(2).strip()
        if speaker not in mapping:
            mapping[speaker] = f"Person{len(mapping) + 1}"
        lines_out.append(f"{mapping[speaker]}: {msg}")
    if len(mapping) < 2 or len(lines_out) < 2:
        return None
    return "\n".join(lines_out)


def save_jsonl(examples: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
    logger.info("Saved %d examples → %s", len(examples), path)


class TransferDatasetBuilder:

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.out_dir = Path(args.out_dir)
        self.seed = args.seed
        self.rng = random.Random(args.seed)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # ConVAWG side
    # ------------------------------------------------------------------

    def build_convawg(self) -> Dict[str, List[Dict]]:
        files: List[Path] = []
        for d in [Path(p) for p in self.args.convawg_dirs]:
            if not d.exists():
                logger.warning("ConVAWG dir not found: %s", d)
                continue
            files.extend(sorted(d.glob("*_dialogues.json")))
        logger.info("ConVAWG: %d scenario files", len(files))

        by_scenario: Dict[str, List[Dict]] = defaultdict(list)
        skipped = 0
        for f in tqdm(files, desc="convawg", unit="file"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("Failed to read %s: %s", f, exc)
                continue
            sid = str(data.get("scenario_id") or f.stem.replace("_dialogues", ""))
            if sid in by_scenario:
                continue
            perp = data.get("perpetrator_name")
            for i, dg in enumerate(data.get("dialogues", [])):
                raw = (dg.get("enforced_dialogue") or "").strip()
                if not raw:
                    continue
                level = (dg.get("conversation_context") or {}).get("escalation_level")
                parts = dg.get("participants") or {}
                third_party = perp not in (parts.get("sender"), parts.get("receiver"))
                if isinstance(level, int):
                    label = int(level >= self.args.pos_threshold)
                elif third_party:
                    label = 0  # support conversations: hard negatives
                else:
                    skipped += 1
                    continue
                text = anonymize_dialogue(raw)
                if text is None:
                    skipped += 1
                    continue
                by_scenario[sid].append({
                    "example_id": f"{sid}_d{i}",
                    "scenario_id": sid,
                    "source": "convawg",
                    "text": text,
                    "label": label,
                    "escalation_level": level,
                    "third_party": third_party,
                })
        logger.info("ConVAWG: %d scenarios, %d skipped dialogues",
                    len(by_scenario), skipped)

        # scenario-level split
        ids = sorted(by_scenario)
        self.rng.shuffle(ids)
        n = len(ids)
        n_train = int(n * self.args.train_ratio)
        n_dev = int(n * self.args.dev_ratio)
        assignment = {}
        for i, sid in enumerate(ids):
            assignment[sid] = ("train" if i < n_train
                               else "dev" if i < n_train + n_dev else "test")
        splits: Dict[str, List[Dict]] = {"train": [], "dev": [], "test": []}
        for sid, examples in by_scenario.items():
            splits[assignment[sid]].extend(examples)
        for name in splits:
            splits[name].sort(key=lambda e: e["example_id"])

        (self.out_dir / "convawg").mkdir(parents=True, exist_ok=True)
        with open(self.out_dir / "convawg" / "split_manifest.json", "w",
                  encoding="utf-8") as fh:
            json.dump(assignment, fh, indent=2)
        return splits

    # ------------------------------------------------------------------
    # MentalManip side
    # ------------------------------------------------------------------

    def build_mentalmanip(self) -> Dict[str, List[Dict]]:
        from datasets import load_dataset

        ds = load_dataset(MENTALMANIP_HF, self.args.mentalmanip_config)["train"]
        logger.info("MentalManip (%s): %d dialogues", self.args.mentalmanip_config, len(ds))

        examples: List[Dict] = []
        skipped = 0
        for row in tqdm(ds, desc="mentalmanip", unit="dlg"):
            text = anonymize_dialogue(row["dialogue"] or "")
            if text is None:
                skipped += 1
                continue
            examples.append({
                "example_id": f"mm_{row['id']}",
                "source": "mentalmanip",
                "text": text,
                "label": int(row["manipulative"]),
                "technique": row.get("technique"),
                "vulnerability": row.get("vulnerability"),
            })
        logger.info("MentalManip: %d usable, %d skipped", len(examples), skipped)

        # stratified split by label
        by_label: Dict[int, List[Dict]] = defaultdict(list)
        for ex in examples:
            by_label[ex["label"]].append(ex)
        splits: Dict[str, List[Dict]] = {"train": [], "dev": [], "test": []}
        for label, group in sorted(by_label.items()):
            self.rng.shuffle(group)
            n = len(group)
            n_train = int(n * self.args.train_ratio)
            n_dev = int(n * self.args.dev_ratio)
            splits["train"].extend(group[:n_train])
            splits["dev"].extend(group[n_train:n_train + n_dev])
            splits["test"].extend(group[n_train + n_dev:])
        for name in splits:
            self.rng.shuffle(splits[name])
        return splits

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self) -> None:
        stats: Dict = {"seed": self.seed, "sources": {}}
        for source, splits in (("convawg", self.build_convawg()),
                               ("mentalmanip", self.build_mentalmanip())):
            src_stats = {}
            for name, examples in splits.items():
                save_jsonl(examples, self.out_dir / source / f"{name}.jsonl")
                src_stats[name] = {
                    "n": len(examples),
                    "label_dist": dict(sorted(Counter(e["label"] for e in examples).items())),
                }
            stats["sources"][source] = src_stats
        with open(self.out_dir / "stats.json", "w", encoding="utf-8") as fh:
            json.dump(stats, fh, indent=2)
        logger.info("Stats: %s", json.dumps(stats, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build ConVAWG <-> MentalManip transfer datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python build_dataset.py
  python build_dataset.py --mentalmanip_config mentalmanip_maj
""",
    )
    parser.add_argument("--convawg_dirs", nargs="+",
                        default=[str(p) for p in DEFAULT_CONVAWG_DIRS])
    parser.add_argument("--mentalmanip_config", default="mentalmanip_con",
                        choices=["mentalmanip_con", "mentalmanip_maj"])
    parser.add_argument("--out_dir", default="data")
    parser.add_argument("--pos_threshold", type=int, default=2,
                        help="ConVAWG positive label = escalation_level >= this "
                             "(default 2; use 3 to make level<=2 negative)")
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--dev_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    TransferDatasetBuilder(args).run()


if __name__ == "__main__":
    main()
