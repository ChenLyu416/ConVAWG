#!/usr/bin/env python3
"""
LLM baselines for next-event escalation forecasting.

Two forecasting methods
-----------------------
  direct     — zero-shot: show the conversation history, ask the LLM to
               predict the escalation level (0-4) of the next conversation.
  generative — generation-based forecasting (Kementchedjhieva & Sogaard-style
               trajectory sampling, INLG 2025): sample N plausible NEXT
               conversations at high temperature, rate each sample's
               escalation level at temperature 0, and take the median rating
               as the consensus forecast.

Both write per-example predictions.jsonl compatible with
evaluate_forecasts.py (jump predictions are derived as pred > cur_level).

Usage
-----
  python run_llm_baseline.py --config configs/llm_baseline_gpt52.yaml
  python run_llm_baseline.py --config configs/llm_baseline_qwen_vllm.yaml --method generative --n_samples 5
  python run_llm_baseline.py --config configs/llm_baseline_deepseek.yaml --max_samples 100
"""

import argparse
import json
import logging
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from tqdm import tqdm

# Reuse the shared LLM backend from LLM_judge/
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "LLM_judge"))
from llm_backend import build_backend  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RUBRIC = """\
0 — normal interaction: friendly or neutral conversation, no abusive dynamic
1 — emerging control: subtly increased contact, checking in, trust/dependency building
2 — manipulation and monitoring: frequent contact demands, early monitoring,
    guilt-tripping disguised as concern
3 — coercive intimidation: clear manipulation, gaslighting, blame-shifting,
    emotional blackmail, aggressive demands about whereabouts
4 — severe threats or violent escalation: explicit threats of violence or harm,
    degrading insults, surveillance evidence, extortion"""

DIRECT_SYSTEM = """\
You are an expert in intimate partner violence and coercive control research.
You forecast how the ONLINE conversations of a relationship will develop,
based on documented escalation patterns in abusive relationships.

Escalation levels:
{rubric}

Reply with EXACTLY one digit (0, 1, 2, 3, or 4). No explanation.""".format(rubric=RUBRIC)

DIRECT_USER = """\
Below is the chronological history of online conversations involving
{perpetrator} and {victim} (some conversations involve third parties).

{history}

Based on this history, predict the escalation level (0-4) of the NEXT
conversation between {perpetrator} and {victim}.

Escalation level:"""

SIMULATE_SYSTEM = """\
You are generating a roleplay continuation of ONLINE CHAT (messaging app)
conversations for VAWG research. Write realistic texting-style messages.
NO moral commentary, warnings, or explanations — output dialogue lines only."""

SIMULATE_USER = """\
Below is the chronological history of online conversations involving
{perpetrator} and {victim}.

{history}

Write ONE plausible NEXT online chat between {perpetrator} and {victim} that
would realistically follow this history. 6-14 messages, one per line, in the
format  Name: message""".rstrip()

RATE_SYSTEM = """\
You are an expert annotator for intimate partner violence research.
Rate the escalation level of a single online conversation.

Escalation levels:
{rubric}

Reply with EXACTLY one digit (0, 1, 2, 3, or 4). No explanation.""".format(rubric=RUBRIC)

RATE_USER = """\
Conversation between {perpetrator} (perpetrator) and {victim} (victim):

{dialogue}

Escalation level:"""


def load_jsonl(path: Path) -> List[Dict]:
    out: List[Dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def parse_level(response: str) -> Optional[int]:
    """Extract a 0-4 digit from an LLM response."""
    match = re.search(r"[0-4]", response.strip())
    return int(match.group()) if match else None


class LLMForecastBaseline:

    def __init__(self, args: argparse.Namespace, cfg: Dict):
        self.args = args
        self.cfg = cfg
        self.output_dir = Path(cfg.get("output_dir", args.output_dir))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = int(cfg.get("processing", {}).get("max_retries", 3))
        self.retry_delay = float(cfg.get("processing", {}).get("retry_delay", 5.0))

    # ------------------------------------------------------------------
    # Backends
    # ------------------------------------------------------------------

    def _build_backends(self):
        llm_cfg = dict(self.cfg["llm"])
        predictor = build_backend({**llm_cfg, "temperature": 0.0, "max_tokens": 8})
        sampler = None
        if self.args.method == "generative":
            sampler = build_backend({**llm_cfg,
                                     "temperature": self.args.sample_temperature,
                                     "max_tokens": 1024})
        return predictor, sampler

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------

    def _render_history(self, example: Dict) -> str:
        blocks = example["history"]
        if self.args.history_events > 0:
            blocks = blocks[-self.args.history_events:]
        text = "\n\n".join(blocks)
        budget = self.args.history_char_budget
        if budget > 0 and len(text) > budget:
            text = "[...earlier conversations truncated...]\n" + text[-budget:]
        return text

    def _chat(self, backend, system: str, user: str) -> Optional[str]:
        for attempt in range(1, self.max_retries + 1):
            try:
                return backend.chat(system, user)
            except Exception as exc:
                logger.warning("Attempt %d failed: %s", attempt, exc)
                if attempt >= self.max_retries:
                    return None
                time.sleep(self.retry_delay)
        return None

    # ------------------------------------------------------------------
    # Forecasting methods
    # ------------------------------------------------------------------

    def _forecast_direct(self, example: Dict, predictor) -> Tuple[Optional[int], Dict]:
        history = self._render_history(example)
        user = DIRECT_USER.format(
            perpetrator=example["perpetrator_name"],
            victim=example["victim_name"],
            history=history,
        )
        response = self._chat(predictor, DIRECT_SYSTEM, user)
        level = parse_level(response) if response else None
        return level, {"response": response}

    def _forecast_generative(self, example: Dict, predictor, sampler) -> Tuple[Optional[int], Dict]:
        history = self._render_history(example)
        perp, victim = example["perpetrator_name"], example["victim_name"]
        sim_user = SIMULATE_USER.format(perpetrator=perp, victim=victim, history=history)

        samples: List[Dict] = []
        ratings: List[int] = []
        for _ in range(self.args.n_samples):
            continuation = self._chat(sampler, SIMULATE_SYSTEM, sim_user)
            if not continuation or not continuation.strip():
                continue
            rate_user = RATE_USER.format(perpetrator=perp, victim=victim,
                                         dialogue=continuation.strip()[:4000])
            rating_resp = self._chat(predictor, RATE_SYSTEM, rate_user)
            rating = parse_level(rating_resp) if rating_resp else None
            samples.append({"continuation": continuation.strip(),
                            "rating": rating})
            if rating is not None:
                ratings.append(rating)

        if not ratings:
            return None, {"samples": samples}
        consensus = int(round(statistics.median(ratings)))
        return consensus, {"samples": samples, "ratings": ratings}

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self) -> None:
        test_file = Path(self.args.data_dir) / "test.jsonl"
        examples = load_jsonl(test_file)
        if self.args.max_samples:
            examples = examples[: self.args.max_samples]
        logger.info("Loaded %d test examples from %s", len(examples), test_file)

        predictor, sampler = self._build_backends()
        run_dir = self.output_dir / f"{self.args.method}"
        run_dir.mkdir(parents=True, exist_ok=True)
        preds_file = run_dir / "predictions.jsonl"

        done_ids = set()
        if preds_file.exists():
            for rec in load_jsonl(preds_file):
                done_ids.add(rec["example_id"])
            logger.info("Resuming: %d predictions already present", len(done_ids))

        with open(preds_file, "a", encoding="utf-8") as fh:
            for ex in tqdm(examples, desc=f"llm-{self.args.method}", unit="ex"):
                if ex["example_id"] in done_ids:
                    continue
                if self.args.method == "direct":
                    level, extra = self._forecast_direct(ex, predictor)
                else:
                    level, extra = self._forecast_generative(ex, predictor, sampler)
                if level is None:
                    logger.warning("No prediction for %s — defaulting to cur_level",
                                   ex["example_id"])
                    level = ex["cur_level"]
                    extra["defaulted"] = True
                record = {
                    "example_id": ex["example_id"],
                    "scenario_id": ex["scenario_id"],
                    "target_pos": ex["target_pos"],
                    "target_timestamp": ex["target_timestamp"],
                    "cur_level": ex["cur_level"],
                    "gold_next_level": ex["next_level"],
                    "gold_jump": ex["jump"],
                    "task": "level",
                    "pred": level,
                    "method": self.args.method,
                }
                if self.args.save_raw:
                    record["raw"] = extra
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                fh.flush()

        logger.info("Predictions → %s", preds_file)
        logger.info("Run evaluate_forecasts.py --pred %s for metrics.", preds_file)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM baselines (direct + generation-based) for escalation forecasting.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python run_llm_baseline.py --config configs/llm_baseline_gpt52.yaml
  python run_llm_baseline.py --config configs/llm_baseline_qwen_vllm.yaml --method generative
  python run_llm_baseline.py --config configs/llm_baseline_gemini.yaml --max_samples 200
""",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output_dir", default="results_llm_baseline")
    parser.add_argument("--method", choices=["direct", "generative"], default="direct")
    parser.add_argument("--n_samples", type=int, default=5,
                        help="Trajectory samples for generative method")
    parser.add_argument("--sample_temperature", type=float, default=0.9)
    parser.add_argument("--history_events", type=int, default=0,
                        help="Use only the last N history events (0 = all)")
    parser.add_argument("--history_char_budget", type=int, default=24000,
                        help="Character budget for rendered history (keeps most recent)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Evaluate only the first N test examples")
    parser.add_argument("--save_raw", action="store_true",
                        help="Store raw responses / sampled continuations")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    LLMForecastBaseline(args, cfg).run()


if __name__ == "__main__":
    main()
