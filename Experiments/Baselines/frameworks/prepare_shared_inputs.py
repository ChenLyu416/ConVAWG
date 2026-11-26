#!/usr/bin/env python3
"""
Prepare shared raw inputs for the framework baselines
=====================================================
Builds `shared_inputs/<pid>_input.json` per persona: the persona seed plus
k retrieved DHR evidence cases (restricted fields only — no event graphs,
actors, or relationships). All three framework baselines (DiaSynth,
LLM-adapted SynDG, SPASM) consume this cache, guaranteeing byte-identical
raw inputs across frameworks.

Usage
-----
  python prepare_shared_inputs.py                       # all 200 personas
  python prepare_shared_inputs.py --num-personas 10 --seed 42
  python prepare_shared_inputs.py --persona-ids 65637 115911
  python prepare_shared_inputs.py --self-check          # verify no forbidden keys
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from tqdm import tqdm

from common.inputs import assert_no_forbidden_keys, load_personas, select_personas
from common.runner import PERSONA_CSV, _RetrieverFactory
from common.shared_cache import (
    DEFAULT_SHARED_INPUTS_DIR,
    SHARED_INPUT_SCHEMA_VERSION,
    build_shared_input,
    shared_input_path,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("prepare_shared_inputs")


class SharedInputBuilder:
    """Builds and verifies the shared raw-input cache."""

    def __init__(self, shared_dir: Path, k: int) -> None:
        self.shared_dir = Path(shared_dir)
        self.k = k
        self._factory = _RetrieverFactory()

    def build(self, personas: list, skip_existing: bool = True) -> int:
        self.shared_dir.mkdir(parents=True, exist_ok=True)
        built = 0
        for persona in tqdm(personas, desc="shared inputs", unit="persona"):
            path = shared_input_path(self.shared_dir, persona["persona_id"])
            if skip_existing and path.exists():
                try:
                    with open(path, encoding="utf-8") as f:
                        data = json.load(f)
                    if (data.get("schema_version") == SHARED_INPUT_SCHEMA_VERSION
                            and data.get("k") == self.k):
                        continue
                except (json.JSONDecodeError, OSError):
                    pass
            shared_input = build_shared_input(persona, self._factory(), k=self.k)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(shared_input, f, indent=2, ensure_ascii=False)
            built += 1
        return built

    def self_check(self) -> int:
        """Assert every cached file is version-consistent and free of forbidden keys."""
        files = sorted(self.shared_dir.glob("*_input.json"))
        if not files:
            logger.error("No shared inputs found in %s", self.shared_dir)
            sys.exit(1)
        for path in files:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            assert_no_forbidden_keys(data, path.name)
            if data.get("schema_version") != SHARED_INPUT_SCHEMA_VERSION:
                raise ValueError(f"Stale schema_version in {path.name}")
            if not data.get("persona_text"):
                raise ValueError(f"Missing persona_text in {path.name}")
        return len(files)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build the shared raw-input cache for the framework baselines.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--num-personas", type=int, default=None, metavar="N")
    p.add_argument("--persona-ids", nargs="+", default=None, metavar="ID")
    p.add_argument("--persona-csv", type=Path, default=PERSONA_CSV)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--k-cases", type=int, default=5,
                   help="DHR cases to retrieve per persona (default: 5)")
    p.add_argument("--shared-inputs-dir", type=Path, default=DEFAULT_SHARED_INPUTS_DIR)
    p.add_argument("--rebuild", action="store_true",
                   help="Rebuild even if a valid cache file exists")
    p.add_argument("--self-check", action="store_true",
                   help="Verify existing cache files (forbidden keys, schema version) and exit")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    builder = SharedInputBuilder(args.shared_inputs_dir, k=args.k_cases)

    if args.self_check:
        n = builder.self_check()
        logger.info("Self-check OK: %d shared-input files, no forbidden keys", n)
        return

    personas = select_personas(
        load_personas(args.persona_csv), args.num_personas, args.persona_ids, args.seed,
    )
    logger.info("Building shared inputs for %d personas (k=%d) → %s",
                len(personas), args.k_cases, args.shared_inputs_dir)
    built = builder.build(personas, skip_existing=not args.rebuild)
    logger.info("Done: %d built, %d already cached", built, len(personas) - built)


if __name__ == "__main__":
    main()
