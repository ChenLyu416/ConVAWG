#!/usr/bin/env python3
"""
Regenerate the paper's scale-comparison table (tab:scale_comparison):
average dialogues per scenario, turns per dialogue, and words per dialogue
for ConVAWG and the eight baselines on the 50 matched personas.

Counting conventions (calibrated to reproduce the published ConVAWG dataset
statistics 30.9 / 14.5 / 201.6 on the full 200-scenario release):
  - a turn is a "Speaker: text" line (baseline JSON turns are rendered the
    same way before counting);
  - words are whitespace tokens of the full rendered dialogue text,
    including the speaker-name prefixes;
  - ConVAWG uses the enforced_dialogue track.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
BASELINES_DIR = REPO / "Experiments/Baselines/results"   # <setting>/<persona_id>_dialogue.json
MATCHED_PERSONAS = REPO / "Experiments/Baselines/matched_50_personas.json"
CONVAWG_DIR = REPO / "data/ConVAWG/dialogues"
OUTPUT_DIR = REPO / "analysis/outputs/scale_comparison"

# display name -> baseline setting dir under Experiments/Baselines/results
SYSTEMS: Dict[str, str] = {
    "LLaMA-3.1-70B-Instruct": "llama3.1-70b-awq",
    "Qwen3.5-35B-A3B": "qwen3.5-35b",
    "DeepSeek-V4-Flash": "deepseek-v4-flash",
    "Gemini-3.1-Flash-Lite": "gemini-flash-lite",
    "GPT-5.2": "gpt5.2",
    "DiaSynth (Qwen3.5)": "diasynth_qwen3.5-35b",
    "SynDG (Qwen3.5)": "syndg_qwen3.5-35b",
    "SPASM (Qwen3.5)": "spasm_qwen3.5-35b",
}

TURN_LINE_RE = re.compile(r"^([A-Za-z][^:]{0,40}):\s*(.*)$")


def load_matched_personas(base_dir: Path) -> List[str]:
    meta = json.loads(MATCHED_PERSONAS.read_text())
    return [str(x) for x in meta["all_50"]]


def stats_from_text_dialogues(texts: List[str]) -> Tuple[int, int, int]:
    """(n_dialogues, total_turns, total_words) from rendered dialogue strings."""
    n_turns = 0
    n_words = 0
    for text in texts:
        for line in text.split("\n"):
            if TURN_LINE_RE.match(line.strip()):
                n_turns += 1
        n_words += len(text.split())
    return len(texts), n_turns, n_words


def collect_baseline(system_dir: Path, personas: List[str]) -> Tuple[int, int, int, int]:
    """(n_scenarios, n_dialogues, total_turns, total_words) for one baseline."""
    n_scen = n_dial = n_turns = n_words = 0
    for pid in personas:
        fp = system_dir / f"{pid}_dialogue.json"
        if not fp.exists():
            print(f"  WARN: missing {fp}")
            continue
        data = json.loads(fp.read_text())
        n_scen += 1
        for dlg in data.get("dialogues", []):
            turns = dlg.get("turns") or []
            rendered = "\n".join(
                f"{t.get('speaker', '')}: {t.get('text', '')}" for t in turns
            )
            n_dial += 1
            n_turns += len(turns)
            n_words += len(rendered.split())
    return n_scen, n_dial, n_turns, n_words


def collect_convawg(conv_dir: Path, personas: Optional[List[str]]) -> Tuple[int, int, int, int]:
    n_scen = n_dial = n_turns = n_words = 0
    files = sorted(conv_dir.glob("*_dialogues.json"))
    if personas is not None:
        wanted = set(personas)
        files = [f for f in files if f.name.split("_")[0] in wanted]
    for fp in files:
        data = json.loads(fp.read_text())
        texts = [
            dlg.get("enforced_dialogue") or ""
            for dlg in data.get("dialogues", [])
        ]
        d, t, w = stats_from_text_dialogues(texts)
        n_scen += 1
        n_dial += d
        n_turns += t
        n_words += w
    return n_scen, n_dial, n_turns, n_words


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n  python compute_scale_comparison_table.py\n",
    )
    parser.add_argument(
        "--full-convawg",
        action="store_true",
        help="Also report ConVAWG over all 200 scenarios (reference row)",
    )
    args = parser.parse_args()

    personas = load_matched_personas(BASELINES_DIR)
    print(f"Matched personas: {len(personas)}")

    rows = []
    for name, subdir in SYSTEMS.items():
        s, d, t, w = collect_baseline(BASELINES_DIR / subdir, personas)
        rows.append((name, s, d, t, w))
        print(f"{name:24s} scen={s:3d} dial/scen={d / s:5.1f} "
              f"turns/dial={t / d:5.1f} words/dial={w / d:6.1f}")

    s, d, t, w = collect_convawg(CONVAWG_DIR, personas)
    rows.append(("ConVAWG", s, d, t, w))
    print(f"{'ConVAWG (50p)':24s} scen={s:3d} dial/scen={d / s:5.1f} "
          f"turns/dial={t / d:5.1f} words/dial={w / d:6.1f}")

    if args.full_convawg:
        s, d, t, w = collect_convawg(CONVAWG_DIR, None)
        print(f"{'ConVAWG (200p)':24s} scen={s:3d} dial/scen={d / s:5.1f} "
              f"turns/dial={t / d:5.1f} words/dial={w / d:6.1f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        name: {
            "n_scenarios": s,
            "avg_dialogues_per_scenario": round(d / s, 1),
            "avg_turns_per_dialogue": round(t / d, 1),
            "avg_words_per_dialogue": round(w / d, 1),
        }
        for name, s, d, t, w in rows
    }
    (OUTPUT_DIR / "scale_comparison.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    latex_lines = []
    for name, s, d, t, w in rows:
        pre = post = ""
        if name == "ConVAWG":
            latex_lines.append(r"\midrule")
            pre, post = r"\textbf{\toolname{}", "}"
            name = ""
        latex_lines.append(
            f"{pre}{name}{post} & {d / s:.1f} & {t / d:.1f} & {w / d:.1f} \\\\"
        )
    latex = "\n".join(latex_lines)
    (OUTPUT_DIR / "scale_comparison_rows.tex").write_text(latex + "\n")
    print("\nLaTeX rows:\n" + latex)
    print(f"\nWritten to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
