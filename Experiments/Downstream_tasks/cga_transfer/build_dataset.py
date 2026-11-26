#!/usr/bin/env python3
"""
Build Conversations Gone Awry (CGA) derailment-forecasting test sets in the
ConVAWG escalation-forecasting format.

Purpose (external evaluation)
-----------------------------
The escalation-forecasting task trains and tests on ConVAWG's own synthetic
dialogues.  To answer "does a ConVAWG-trained forecaster transfer to *real*
conversations?", this script converts the public CGA corpora into the exact
input format of `../escalation_forecasting` so a ConVAWG-trained `jump`
forecaster can be evaluated zero-shot on real derailment forecasting:

  wiki — "conversations-gone-awry-corpus"      (Wikipedia talk pages,
          Zhang et al., ACL 2018; 4,188 conversations, balanced)
  cmv  — "conversations-gone-awry-cmv-corpus"  (Reddit ChangeMyView,
          Chang & Danescu-Niculescu-Mizil, EMNLP 2019; 6,842 conversations,
          balanced)

Corpora are downloaded directly from the ConvoKit file server and parsed with
the standard library (no `convokit` dependency — its install chain is not
needed just to read `conversations.json` + `utterances.jsonl`, and the pinned
environment must not be disturbed).

Forecasting protocol (matches the standard CGA/CRAFT setup)
-----------------------------------------------------------
  input  — the chronological history of comments, rendered one block per
           comment:  "[comment k | ts | speaker]\\n<text>"
           (the per-comment analogue of ConVAWG's
            "[conversation k | ts | sender -> receiver]" blocks).
           Wikipedia section headers are excluded.
  label  — jump = 1 if the conversation ends in a personal attack (wiki:
           `conversation_has_personal_attack`) / a moderator-removed
           rule-violating comment (cmv: `has_removed_comment`), else 0.
  The attack/removed comment itself is NEVER part of the input:
    * wiki — history is truncated strictly before the first comment with
      `comment_has_personal_attack` (in 2,088/2,094 positives that is the
      final comment anyway);
    * cmv — no per-comment marker exists; the derailing comment is the final
      comment of positive conversations, so the final comment is dropped.
  For symmetry the final comment is also dropped from civil conversations
  (both corpora), so "history length" is not a label give-away.
  Conversations with < --min_history history comments are dropped.

Output schema is field-compatible with escalation_forecasting/train_forecaster.py
(`history` list of blocks, binary `jump`; `cur_level`=0 / `next_level`=jump are
placeholders so the shared jump-task code paths work unchanged).  The corpora's
official train/val/test splits are kept (val -> dev.jsonl).

Outputs (per corpus)
--------------------
  data/<corpus>/{train,dev,test}.jsonl
  data/<corpus>/stats.json
"""

import argparse
import json
import logging
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CONVOKIT_BASE = "https://zissou.infosci.cornell.edu/convokit/datasets"

CORPORA: Dict[str, Dict] = {
    "wiki": {
        "dirname": "conversations-gone-awry-corpus",
        "label_key": "conversation_has_personal_attack",
        "attack_utt_key": "comment_has_personal_attack",  # per-comment marker
        "description": "Wikipedia talk pages (Zhang et al., ACL 2018)",
    },
    "cmv": {
        "dirname": "conversations-gone-awry-cmv-corpus",
        "label_key": "has_removed_comment",
        "attack_utt_key": None,  # no per-comment marker; attack = final comment
        "description": "Reddit ChangeMyView (Chang & DNM, EMNLP 2019)",
    },
}

SPLIT_RENAME = {"train": "train", "val": "dev", "test": "test"}


def conv_meta(record: Dict) -> Dict:
    """ConvoKit conversations.json values are either flat metadata dicts (wiki)
    or {"meta": {...}, "vectors": [...]} wrappers (cmv)."""
    meta = record.get("meta")
    return meta if isinstance(meta, dict) else record


def format_ts(epoch: Optional[float]) -> str:
    if epoch is None:
        return "unknown time"
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def iso_ts(epoch: Optional[float]) -> Optional[str]:
    if epoch is None:
        return None
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def render_comment_block(idx: int, utt: Dict) -> str:
    """Per-comment analogue of ConVAWG's conversation-event blocks."""
    speaker = utt.get("speaker") or "Unknown"
    header = f"[comment {idx} | {format_ts(utt.get('timestamp'))} | {speaker}]"
    return f"{header}\n{(utt.get('text') or '').strip()}"


class CGADatasetBuilder:
    """Downloads a CGA corpus and converts it into jump-forecasting JSONL."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.raw_dir = Path(args.raw_dir)
        self.out_dir = Path(args.out_dir)
        self.min_history = args.min_history
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Download / load
    # ------------------------------------------------------------------

    def ensure_corpus(self, corpus: str) -> Path:
        spec = CORPORA[corpus]
        corpus_dir = self.raw_dir / spec["dirname"]
        if (corpus_dir / "conversations.json").exists():
            logger.info("[%s] corpus already present: %s", corpus, corpus_dir)
            return corpus_dir

        zip_path = self.raw_dir / f"{spec['dirname']}.zip"
        if not zip_path.exists():
            url = f"{CONVOKIT_BASE}/{spec['dirname']}/{spec['dirname']}.zip"
            logger.info("[%s] downloading %s ...", corpus, url)
            with requests.get(url, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("content-length", 0))
                with open(zip_path, "wb") as fh, tqdm(
                    total=total, unit="B", unit_scale=True, desc=f"download {corpus}"
                ) as bar:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
                        bar.update(len(chunk))
        logger.info("[%s] extracting %s ...", corpus, zip_path.name)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(self.raw_dir)
        return corpus_dir

    @staticmethod
    def load_utterances(corpus_dir: Path) -> Dict[str, List[Dict]]:
        by_conv: Dict[str, List[Dict]] = defaultdict(list)
        with open(corpus_dir / "utterances.jsonl", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                utt = json.loads(line)
                by_conv[utt["conversation_id"]].append(utt)
        for utts in by_conv.values():
            utts.sort(key=lambda u: (u.get("timestamp") or 0, u["id"]))
        return by_conv

    # ------------------------------------------------------------------
    # Example extraction
    # ------------------------------------------------------------------

    def extract_example(self, corpus: str, conv_id: str, meta: Dict,
                        utts: List[Dict]) -> Optional[Dict]:
        spec = CORPORA[corpus]
        label = int(bool(meta.get(spec["label_key"])))

        comments = [
            u for u in utts
            if not u.get("meta", {}).get("is_section_header")
            and (u.get("text") or "").strip()
        ]
        if not comments:
            return None

        # History cutoff: never show the attack / removed comment.
        if label and spec["attack_utt_key"]:
            attack_positions = [
                i for i, u in enumerate(comments)
                if u.get("meta", {}).get(spec["attack_utt_key"])
            ]
            cut = attack_positions[0] if attack_positions else len(comments) - 1
        else:
            # Positive cmv conversations end with the removed comment; final
            # comment is dropped from negatives too (both corpora) for symmetry.
            cut = len(comments) - 1

        history_comments = comments[:cut]
        if len(history_comments) < self.min_history:
            return None

        withheld = comments[cut]
        return {
            "example_id": f"{corpus}_{conv_id}",
            "scenario_id": conv_id,
            "corpus": corpus,
            "pair_id": meta.get("pair_id"),
            "history": [render_comment_block(i + 1, u)
                        for i, u in enumerate(history_comments)],
            "n_history_events": len(history_comments),
            "n_speakers": len({u.get("speaker") for u in history_comments}),
            # jump-forecaster-compatible label fields (cur_level/next_level are
            # placeholders: jump == int(next_level > cur_level) holds).
            "cur_level": 0,
            "next_level": label,
            "jump": label,
            "target_pos": len(history_comments),
            "target_timestamp": iso_ts(withheld.get("timestamp")),
        }

    # ------------------------------------------------------------------
    # Build one corpus
    # ------------------------------------------------------------------

    def build_corpus(self, corpus: str) -> None:
        corpus_dir = self.ensure_corpus(corpus)
        with open(corpus_dir / "conversations.json", encoding="utf-8") as fh:
            conversations = json.load(fh)
        utterances = self.load_utterances(corpus_dir)
        logger.info("[%s] %d conversations, %d with utterances",
                    corpus, len(conversations),
                    sum(1 for c in conversations if c in utterances))

        splits: Dict[str, List[Dict]] = {"train": [], "dev": [], "test": []}
        n_dropped_short = 0
        for conv_id, record in tqdm(conversations.items(),
                                    desc=f"convert {corpus}", unit="conv"):
            meta = conv_meta(record)
            example = self.extract_example(corpus, conv_id, meta,
                                           utterances.get(conv_id, []))
            if example is None:
                n_dropped_short += 1
                continue
            split = SPLIT_RENAME.get(meta.get("split"), "train")
            splits[split].append(example)

        out_dir = self.out_dir / corpus
        out_dir.mkdir(parents=True, exist_ok=True)
        stats: Dict = {
            "corpus": corpus,
            "description": CORPORA[corpus]["description"],
            "label": "jump=1 iff conversation ends in personal attack / removed comment",
            "min_history": self.min_history,
            "n_conversations_total": len(conversations),
            "n_dropped_short_history": n_dropped_short,
            "splits": {},
        }
        for name, examples in splits.items():
            examples.sort(key=lambda e: e["scenario_id"])
            path = out_dir / f"{name}.jsonl"
            with open(path, "w", encoding="utf-8") as fh:
                for ex in examples:
                    fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
            hist_lens = [e["n_history_events"] for e in examples]
            stats["splits"][name] = {
                "n_examples": len(examples),
                "jump_dist": dict(sorted(Counter(e["jump"] for e in examples).items())),
                "mean_history_comments": (round(sum(hist_lens) / len(hist_lens), 2)
                                          if hist_lens else None),
            }
            logger.info("[%s] %s: %d examples -> %s", corpus, name, len(examples), path)

        with open(out_dir / "stats.json", "w", encoding="utf-8") as fh:
            json.dump(stats, fh, indent=2)
        logger.info("[%s] stats: %s", corpus, json.dumps(stats["splits"], indent=2))

    def run(self, corpora: List[str]) -> None:
        for corpus in corpora:
            self.build_corpus(corpus)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build CGA derailment-forecasting test sets in the ConVAWG "
                    "jump-forecasting format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python build_dataset.py                       # both corpora -> data/wiki, data/cmv
  python build_dataset.py --corpus wiki
  python build_dataset.py --corpus cmv --min_history 3
""",
    )
    parser.add_argument("--corpus", choices=["wiki", "cmv", "both"], default="both")
    parser.add_argument("--raw_dir", default="raw",
                        help="Where corpus zips are downloaded/extracted (default: raw)")
    parser.add_argument("--out_dir", default="data")
    parser.add_argument("--min_history", type=int, default=2,
                        help="Minimum history comments per example (default 2, "
                             "matching escalation_forecasting --min-history)")
    args = parser.parse_args()

    corpora = ["wiki", "cmv"] if args.corpus == "both" else [args.corpus]
    CGADatasetBuilder(args).run(corpora)


if __name__ == "__main__":
    main()
