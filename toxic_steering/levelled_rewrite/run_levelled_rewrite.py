#!/usr/bin/env python3
"""
Levelled Rewrite Pipeline
=========================
Combines GPT-based message labelling with CAA-steered rewriting.
Processes raw Stage 5 roleplay output (enforced_dialogue) OR pre-labeled
data directly.  The pipeline:

1. Parses ``enforced_dialogue`` strings into structured messages
2. Labels perpetrator messages using configurable backend (OpenAI API or vLLM)
3. Samples escalation-act messages for rewriting
4. Rewrites selected messages with a CAA forward-hook steered model at
   escalation-matched steering levels:

    Escalation 2  →  Level 1  mild
    Escalation 3  →  Level 2  moderate
    Escalation 4  →  Level 3  strong

The steering vector, the calibration file (layer + per-level alpha) and the
rewriting prompts are not part of the public release; see
``toxic_steering/README.md``.

Usage
-----
    # Run from the toxic_steering/ directory.
    # From raw Stage 5 output (labels automatically with the labeling backend):
    python -m levelled_rewrite.run_levelled_rewrite \
        --input  ../Generation_Results/roleplay_dialogues/ \
        --output ../Generation_Results/caa_rewritten_dialogues/ \
        --model <hf-model-id> \
        --vector-dir <dir-with-layer_N.pt> \
        --calibration <calibration.json> \
        --prompts <prompts.yaml>

    # From pre-labeled data (skips labeling):
    python -m levelled_rewrite.run_levelled_rewrite \
        --input <labeled_dir> --output ./levelled_rewrite_output/ \
        --model <hf-model-id> --vector-dir <dir> --calibration <calibration.json> \
        --skip-labeling
"""

import argparse
import json
import os
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from tqdm import tqdm

from levelled_rewrite.steered_rewriter import SteeredRewriter
from levelled_rewrite.message_labeler import (
    MessageLabeler,
    parse_dialogue,
    label_perpetrator_messages,
    ESCALATION_ACTS,
)

# ── Escalation → Steering level mapping ───────────────────────────────
ESC_TO_LEVEL = {2: 1, 3: 2, 4: 3}
TARGET_ESCALATION_LEVELS = {2, 3, 4}

# Default per-escalation-level injection ratios
DEFAULT_INJECTION_RATIOS = {2: 0.05, 3: 0.10, 4: 0.20}
REWRITE_CONTEXT_WINDOW = 10


# =====================================================================
# Helper: sample which messages to rewrite
# =====================================================================

def sample_indices(
    labeled_messages: List[Dict],
    injection_ratio: float = 0.10,
    min_per_conversation: int = 1,
) -> List[int]:
    """Return sorted indices of perpetrator escalation-act messages to rewrite."""
    eligible = [
        i for i, m in enumerate(labeled_messages)
        if m.get("is_perpetrator") and m.get("label") in ESCALATION_ACTS
    ]
    if not eligible:
        return []

    k = max(min_per_conversation, int(len(eligible) * injection_ratio))
    k = min(k, len(eligible))
    return sorted(random.sample(eligible, k))


# =====================================================================
# Format helpers
# =====================================================================

def format_dialogue(messages: List[Dict]) -> str:
    """Re-assemble a list of message dicts into ``Speaker: text`` lines."""
    return "\n".join(f"{m['speaker']}: {m['content']}" for m in messages)


def build_rewrite_context(
    messages: List[Dict],
    target_idx: int,
    total_surrounding_utterances: int = REWRITE_CONTEXT_WINDOW,
) -> str:
    """Build up to ``total_surrounding_utterances`` around ``target_idx``."""
    if not messages or target_idx < 0 or target_idx >= len(messages):
        return "[no surrounding context available]"

    half_window = total_surrounding_utterances // 2
    start = max(0, target_idx - half_window)
    end = min(len(messages), target_idx + half_window + 1)

    selected_indices = [
        idx for idx in range(start, end)
        if idx != target_idx
    ]

    while len(selected_indices) < total_surrounding_utterances:
        expanded = False

        if start > 0:
            start -= 1
            if start != target_idx:
                selected_indices.insert(0, start)
                expanded = True

        if len(selected_indices) >= total_surrounding_utterances:
            break

        if end < len(messages):
            if end != target_idx:
                selected_indices.append(end)
                expanded = True
            end += 1

        if not expanded:
            break

    selected_indices = selected_indices[:total_surrounding_utterances]
    if not selected_indices:
        return "[no surrounding context available]"

    return "\n".join(
        f"{messages[idx].get('speaker', 'Unknown')}: {messages[idx].get('content', '')}"
        for idx in selected_indices
    )


# =====================================================================
# Process a single dialogue entry
# =====================================================================

def process_dialogue(
    dialogue: Dict[str, Any],
    rewriter: SteeredRewriter,
    injection_ratios: Dict[int, float],
    min_per_conversation: int,
    max_retries: int,
    labeler: MessageLabeler = None,
    perpetrator_name: str = "Perpetrator",
    victim_name: str = "Victim",
) -> Optional[Dict[str, Any]]:
    """
    Rewrite perpetrator messages in one dialogue entry using CAA steering.

    Returns an updated copy of *dialogue* with:
      - ``caa_rewritten_dialogue``  (string)
      - ``caa_rewrite_metadata``    (dict)
      - ``caa_rewrite_stats``       (dict)
    or None if the dialogue is not eligible.
    """
    ctx = dialogue.get("conversation_context", {})
    esc_level: int = ctx.get("escalation_level") or 0

    if esc_level not in TARGET_ESCALATION_LEVELS:
        return None

    labeled = dialogue.get("labeled_messages")

    # ── If no labels yet, parse enforced_dialogue and label via GPT ────
    if not labeled:
        dialogue_text = dialogue.get("enforced_dialogue", "")
        if not dialogue_text:
            return None

        if labeler is None:
            # Cannot label without a labeler — skip
            return None

        messages = parse_dialogue(dialogue_text)
        if not messages:
            return None

        print(f"    Labeling {len(messages)} messages (esc={esc_level})...")
        labeled = label_perpetrator_messages(
            labeler, messages, perpetrator_name, victim_name, esc_level,
        )
        # Persist labels on the dialogue dict so they are saved
        dialogue["labeled_messages"] = labeled

    if not labeled:
        return None

    # ── Sample which messages to rewrite ───────────────────────────────
    ratio = injection_ratios.get(esc_level, 0.10)
    indices = sample_indices(labeled, ratio, min_per_conversation)
    if not indices:
        return None

    # ── Steering level ─────────────────────────────────────────────────
    level = ESC_TO_LEVEL[esc_level]
    alpha = rewriter.level_to_alpha[level]

    # ── Rewrite selected messages ──────────────────────────────────────
    rewritten_msgs: List[Dict] = [dict(m) for m in labeled]  # shallow copy
    success = 0
    failed = 0

    for idx in indices:
        msg = rewritten_msgs[idx]
        new_text = rewriter.rewrite(
            message=msg["content"],
            escalation_level=esc_level,
            message_type=msg.get("label", "escalation"),
            context_window=build_rewrite_context(rewritten_msgs, idx),
            max_retries=max_retries,
        )
        if new_text is not None:
            rewritten_msgs[idx] = {
                **msg,
                "original_content": msg["content"],
                "content": new_text,
                "toxicity_injected": True,
                "steering_level": level,
                "steering_alpha": alpha,
            }
            success += 1
        else:
            rewritten_msgs[idx] = {
                **msg,
                "toxicity_injected": False,
                "injection_failed": True,
            }
            failed += 1

    # ── Attach results to dialogue ─────────────────────────────────────
    result = dict(dialogue)
    result["caa_rewritten_dialogue"] = format_dialogue(rewritten_msgs)
    result["caa_rewritten_messages"] = rewritten_msgs
    result["caa_rewrite_metadata"] = {
        "method": "caa_steered_rewrite",
        "model": rewriter.model_name,
        "steering_layer": rewriter.layer,
        "steering_level": level,
        "steering_alpha": alpha,
        "escalation_level": esc_level,
        "timestamp": datetime.now().isoformat(),
    }
    result["caa_rewrite_stats"] = {
        "attempted": len(indices),
        "successful": success,
        "failed": failed,
    }
    return result


# =====================================================================
# Process a full file  (batched rewriting + parallel labeling)
# =====================================================================

def process_file(
    input_path: Path,
    output_dir: Path,
    rewriter: SteeredRewriter,
    injection_ratios: Dict[int, float],
    min_per_conversation: int,
    max_retries: int,
    labeler: MessageLabeler = None,
    batch_size: int = 8,
    label_workers: int = 4,
) -> Dict[str, int]:
    """Process a single dialogue JSON file.

    Improvements over the original sequential version:
    - Phase 1: GPT labeling runs in parallel across all dialogues.
    - Phase 2: All messages to rewrite are collected, then sent to the
      GPU in a single batched call via ``rewriter.rewrite_many()``.
    """
    print(f"\n{'─'*60}")
    print(f"Processing: {input_path.name}")

    with open(input_path) as f:
        data = json.load(f)

    dialogues = data.get("dialogues", data.get("roleplay_dialogues", []))
    perpetrator_name = data.get("perpetrator_name", "Perpetrator")
    victim_name = data.get("victim_name", "Victim")
    if not dialogues:
        print("  No dialogues found — skipping")
        return {"processed": 0, "rewritten": 0, "attempted": 0,
                "successful": 0, "failed": 0}

    stats = {"processed": 0, "rewritten": 0, "attempted": 0,
             "successful": 0, "failed": 0}

    # ── Phase 1: Parallel GPT labeling ────────────────────────────────
    if labeler:
        def _label_one(dlg):
            try:
                esc_level = dlg.get("conversation_context", {}).get("escalation_level") or 0
                if esc_level not in TARGET_ESCALATION_LEVELS:
                    return None
                if dlg.get("labeled_messages"):
                    return dlg["labeled_messages"]  # already labeled
                dialogue_text = dlg.get("enforced_dialogue", "")
                if not dialogue_text:
                    return None
                messages = parse_dialogue(dialogue_text)
                if not messages:
                    return None
                return label_perpetrator_messages(
                    labeler,
                    messages,
                    perpetrator_name,
                    victim_name,
                    esc_level,
                )
            except Exception as e:
                print(f"  ⚠ Labeling error for dialogue: {e}")
                return None

        effective_label_workers = label_workers if labeler.thread_safe else 1
        if effective_label_workers != label_workers:
            print(
                f"  Label backend '{labeler.backend}' is non-thread-safe; "
                f"using workers={effective_label_workers}."
            )
        print(
            f"  Labeling {len(dialogues)} dialogues "
            f"(workers={effective_label_workers})..."
        )
        with ThreadPoolExecutor(max_workers=effective_label_workers) as pool:
            label_results = list(pool.map(_label_one, dialogues))
        for dlg, labeled in zip(dialogues, label_results):
            if labeled is not None:
                dlg["labeled_messages"] = labeled

    # ── Phase 2: Collect all messages to rewrite ──────────────────────
    # Each entry: (dlg_idx, msg_idx, message_str, esc_level, label_str, context_window)
    pending_rewrites = []
    for dlg_idx, dlg in enumerate(dialogues):
        esc_level = dlg.get("conversation_context", {}).get("escalation_level") or 0
        if esc_level not in TARGET_ESCALATION_LEVELS:
            continue
        labeled = dlg.get("labeled_messages", [])
        ratio = injection_ratios.get(esc_level, 0.10)
        indices = sample_indices(labeled, ratio, min_per_conversation)
        for msg_idx in indices:
            msg = labeled[msg_idx]
            pending_rewrites.append(
                (
                    dlg_idx,
                    msg_idx,
                    msg["content"],
                    esc_level,
                    msg.get("label", "escalation"),
                    build_rewrite_context(labeled, msg_idx),
                )
            )

    if not pending_rewrites:
        print("  No eligible messages to rewrite — skipping")
    else:
        print(f"  Rewriting {len(pending_rewrites)} messages (batch_size={batch_size})...")

        # ── Phase 3: Batched GPU rewriting ─────────────────────────────
        requests = [
            (content, esc, mtype, context_window)
            for _, _, content, esc, mtype, context_window in pending_rewrites
        ]
        rewritten_texts = rewriter.rewrite_many(
            requests, max_retries=max_retries, batch_size=batch_size
        )

        # ── Phase 4: Write results back per dialogue ───────────────────
        dlg_rewrites: Dict[int, Dict[int, tuple]] = defaultdict(dict)
        for (dlg_idx, msg_idx, _, esc_level, _, _), new_text in zip(pending_rewrites, rewritten_texts):
            dlg_rewrites[dlg_idx][msg_idx] = (new_text, esc_level)

        for dlg_idx, msg_rewrites in dlg_rewrites.items():
            dlg = dialogues[dlg_idx]
            esc_level = next(iter(msg_rewrites.values()))[1]
            level = ESC_TO_LEVEL[esc_level]
            alpha = rewriter.level_to_alpha[level]
            labeled = dlg.get("labeled_messages", [])
            rewritten_msgs: List[Dict] = [dict(m) for m in labeled]
            success = 0
            failed = 0

            for msg_idx, (new_text, _) in msg_rewrites.items():
                msg = rewritten_msgs[msg_idx]
                if new_text is not None:
                    rewritten_msgs[msg_idx] = {
                        **msg,
                        "original_content": msg["content"],
                        "content": new_text,
                        "toxicity_injected": True,
                        "steering_level": level,
                        "steering_alpha": alpha,
                    }
                    success += 1
                else:
                    rewritten_msgs[msg_idx] = {
                        **msg,
                        "toxicity_injected": False,
                        "injection_failed": True,
                    }
                    failed += 1

            dlg["caa_rewritten_dialogue"] = format_dialogue(rewritten_msgs)
            dlg["caa_rewritten_messages"] = rewritten_msgs
            dlg["caa_rewrite_metadata"] = {
                "method": "caa_steered_rewrite",
                "model": rewriter.model_name,
                "steering_layer": rewriter.layer,
                "steering_level": level,
                "steering_alpha": alpha,
                "escalation_level": esc_level,
                "timestamp": datetime.now().isoformat(),
            }
            dlg["caa_rewrite_stats"] = {
                "attempted": len(msg_rewrites),
                "successful": success,
                "failed": failed,
            }

            stats["processed"] += 1
            stats["attempted"] += len(msg_rewrites)
            stats["successful"] += success
            stats["failed"] += failed
            if success > 0:
                stats["rewritten"] += 1

    # ── Save output ────────────────────────────────────────────────────
    out_path = output_dir / input_path.name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Saved → {out_path}")

    # ── Per-file stats ─────────────────────────────────────────────────
    stats_path = output_dir / "stats" / f"{input_path.stem}_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump({
            "file": input_path.name,
            "timestamp": datetime.now().isoformat(),
            "statistics": stats,
        }, f, indent=2)

    sr = (stats["successful"] / stats["attempted"] * 100
          if stats["attempted"] > 0 else 0)
    print(f"  Done: {stats['processed']} dialogues, "
          f"{stats['successful']}/{stats['attempted']} messages rewritten "
          f"({sr:.0f}%)")
    return stats


# =====================================================================
# CLI
# =====================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Rewrite perpetrator messages in labeled dialogues using "
            "CAA activation-steered model at escalation-matched alpha levels."
        ),
    )
    p.add_argument(
        "--input", type=Path, required=True,
        help="Labeled dialogue file or directory of JSON files.",
    )
    p.add_argument(
        "--output", type=Path, default=None,
        help="Output directory (default: ./levelled_rewrite_output/{timestamp}/).",
    )
    p.add_argument(
        "--config", type=Path, default=None,
        help="Optional YAML config (overrides defaults).",
    )
    # ── Model options ──────────────────────────────────────────────────
    p.add_argument(
        "--model", type=str, default=None,
        help="HuggingFace model name of the steered model (required; CLI or config).",
    )
    p.add_argument(
        "--vector-dir", type=Path, default=None,
        help="Directory containing the L2-normalised steering vectors "
             "(layer_{N}.pt). Required; CLI or config.",
    )
    p.add_argument(
        "--calibration", type=Path, default=None,
        help="Calibration JSON with the steering layer and per-level alphas. "
             "Required; CLI or config.",
    )
    p.add_argument(
        "--prompts", type=Path, default=None,
        help="YAML with the rewriting prompts (default: levelled_rewrite/prompts.yaml; "
             "schema in prompts.example.yaml).",
    )
    # ── Sampling ───────────────────────────────────────────────────────
    p.add_argument(
        "--injection-ratio", type=str, default=None,
        help=(
            "Per-level injection ratios as JSON dict, e.g. "
            "'{\"2\": 0.05, \"3\": 0.10, \"4\": 0.20}'. "
            "Can also be a single float applied to all levels."
        ),
    )
    p.add_argument(
        "--min-per-conversation", type=int, default=1,
        help="Minimum messages to rewrite per conversation.",
    )
    p.add_argument(
        "--max-retries", type=int, default=10,
        help="Max generation attempts per message before giving up.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility.",
    )
    p.add_argument(
        "--escalation-levels", type=int, nargs="+", default=[2, 3, 4],
        help="Which escalation levels to process (default: 2 3 4).",
    )
    p.add_argument(
        "--level-map", type=str, default=None,
        help=(
            "Escalation→CAA steering level map as JSON dict, e.g. "
            "'{\"2\": 1, \"3\": 1, \"4\": 2}'. "
            "Overrides the default {2:1, 3:2, 4:3} mapping."
        ),
    )
    p.add_argument(
        "--alpha-map", type=str, default=None,
        help=(
            "Steering level→alpha override as JSON dict, e.g. "
            "'{\"1\": 250, \"2\": 275, \"3\": 310}'. "
            "Overrides the alpha values from the calibration file for the "
            "specified levels; unspecified levels keep their calibration values."
        ),
    )
    # ── Labeling options ───────────────────────────────────────────────
    p.add_argument(
        "--skip-labeling", action="store_true", default=False,
        help="Skip labeling (input must already have labeled_messages).",
    )
    p.add_argument(
        "--label-backend", type=str, choices=["openai", "vllm"], default="openai",
        help="Backend for message labeling: openai or vllm (default: openai).",
    )
    p.add_argument(
        "--gpt-model", type=str, default="gpt-5.2",
        help="Label model for OpenAI backend (default: gpt-5.2).",
    )
    p.add_argument(
        "--qwen-model", type=str, default="Qwen/Qwen3.5-35B-A3B",
        help="Label model for vLLM backend (default: Qwen/Qwen3.5-35B-A3B).",
    )
    p.add_argument(
        "--gpt-temperature", type=float, default=0.3,
        help="Temperature for labeling generation (default: 0.3).",
    )
    p.add_argument(
        "--gpt-api-base", type=str, default=None,
        help="Custom OpenAI API base URL (used for --label-backend=openai).",
    )
    p.add_argument(
        "--vllm-dtype", type=str, default="auto",
        help="vLLM dtype for label model (default: auto).",
    )
    p.add_argument(
        "--vllm-tensor-parallel-size", type=int, default=1,
        help="vLLM tensor parallel size for label model (default: 1).",
    )
    p.add_argument(
        "--vllm-gpu-memory-utilization", type=float, default=0.90,
        help="vLLM GPU memory utilization for label model (default: 0.90).",
    )
    p.add_argument(
        "--vllm-max-model-len", type=int, default=4096,
        help="vLLM max model length for label model (default: 4096).",
    )
    p.add_argument(
        "--batch-size", type=int, default=8,
        help="Number of messages to rewrite in a single GPU batch (default: 8).",
    )
    p.add_argument(
        "--label-workers", type=int, default=4,
        help="Number of parallel threads for GPT labeling (default: 4).",
    )
    p.add_argument(
        "--gpus", type=str, default="0",
        help="Comma-separated GPU indices (sets CUDA_VISIBLE_DEVICES). Default: 0.",
    )
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    # ── Config file overrides ──────────────────────────────────────────
    if args.config and args.config.exists():
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
        if "injection_ratio" in cfg and args.injection_ratio is None:
            args.injection_ratio = cfg["injection_ratio"]
        args.min_per_conversation = cfg.get("min_per_conversation",
                                            args.min_per_conversation)
        args.max_retries = cfg.get("max_retries", args.max_retries)
        args.seed = cfg.get("seed", args.seed)
        if "escalation_levels" in cfg:
            args.escalation_levels = cfg["escalation_levels"]
        # Labeling config overrides
        args.skip_labeling = cfg.get("skip_labeling", args.skip_labeling)
        if "label_backend" in cfg:
            args.label_backend = cfg["label_backend"]
        if "gpt_model" in cfg:
            args.gpt_model = cfg["gpt_model"]
        if "qwen_model" in cfg:
            args.qwen_model = cfg["qwen_model"]
        if "gpt_temperature" in cfg:
            args.gpt_temperature = cfg["gpt_temperature"]
        if "gpt_api_base" in cfg:
            args.gpt_api_base = cfg["gpt_api_base"]
        if "vllm_dtype" in cfg:
            args.vllm_dtype = cfg["vllm_dtype"]
        if "vllm_tensor_parallel_size" in cfg:
            args.vllm_tensor_parallel_size = cfg["vllm_tensor_parallel_size"]
        if "vllm_gpu_memory_utilization" in cfg:
            args.vllm_gpu_memory_utilization = cfg["vllm_gpu_memory_utilization"]
        if "vllm_max_model_len" in cfg:
            args.vllm_max_model_len = cfg["vllm_max_model_len"]
        if "model" in cfg and args.model is None:
            args.model = cfg["model"]
        if "vector_dir" in cfg and args.vector_dir is None:
            args.vector_dir = Path(cfg["vector_dir"])
        if "calibration" in cfg and args.calibration is None:
            args.calibration = Path(cfg["calibration"])
        if "prompts" in cfg and args.prompts is None:
            args.prompts = Path(cfg["prompts"])
        if "level_map" in cfg and args.level_map is None:
            args.level_map = cfg["level_map"]
        if "alpha_map" in cfg and args.alpha_map is None:
            args.alpha_map = cfg["alpha_map"]

    # ── Steering assets are required (not shipped with the public release) ──
    missing = [
        flag for flag, value in (
            ("--model", args.model),
            ("--vector-dir", args.vector_dir),
            ("--calibration", args.calibration),
        ) if value is None
    ]
    if missing:
        raise SystemExit(
            f"Missing required steering settings: {', '.join(missing)}. "
            "The calibrated steering configuration is not part of the public "
            "release; see toxic_steering/README.md."
        )

    # ── Parse injection_ratio into per-level dict ───────────────────────
    raw_ratio = args.injection_ratio
    if raw_ratio is None:
        # Use defaults
        injection_ratios = dict(DEFAULT_INJECTION_RATIOS)
    elif isinstance(raw_ratio, dict):
        # Already a dict from YAML config
        injection_ratios = {int(k): float(v) for k, v in raw_ratio.items()}
    elif isinstance(raw_ratio, (int, float)):
        # Single float from YAML config — apply to all levels
        injection_ratios = {lvl: float(raw_ratio) for lvl in [2, 3, 4]}
    elif isinstance(raw_ratio, str):
        # CLI string — try JSON dict first, then single float
        try:
            parsed = json.loads(raw_ratio)
            if isinstance(parsed, dict):
                injection_ratios = {int(k): float(v) for k, v in parsed.items()}
            else:
                injection_ratios = {lvl: float(parsed) for lvl in [2, 3, 4]}
        except (json.JSONDecodeError, ValueError):
            injection_ratios = {lvl: float(raw_ratio) for lvl in [2, 3, 4]}
    else:
        injection_ratios = dict(DEFAULT_INJECTION_RATIOS)

    # ── Parse level_map ────────────────────────────────────────────────
    global ESC_TO_LEVEL
    raw_level_map = args.level_map
    if raw_level_map is None:
        resolved_level_map = None  # keep default
    elif isinstance(raw_level_map, dict):
        resolved_level_map = {int(k): int(v) for k, v in raw_level_map.items()}
    elif isinstance(raw_level_map, str):
        try:
            parsed = json.loads(raw_level_map)
            resolved_level_map = {int(k): int(v) for k, v in parsed.items()}
        except (json.JSONDecodeError, ValueError):
            raise ValueError(f"--level-map must be a JSON dict, got: {raw_level_map!r}")
    else:
        resolved_level_map = None
    if resolved_level_map is not None:
        ESC_TO_LEVEL = resolved_level_map
        print(f"[LevelMap] Using custom escalation→steering mapping: {ESC_TO_LEVEL}")

    # ── Parse alpha_map ────────────────────────────────────────────────
    raw_alpha_map = args.alpha_map
    if raw_alpha_map is None:
        resolved_alpha_map = None
    elif isinstance(raw_alpha_map, dict):
        resolved_alpha_map = {int(k): float(v) for k, v in raw_alpha_map.items()}
    elif isinstance(raw_alpha_map, str):
        try:
            parsed = json.loads(raw_alpha_map)
            resolved_alpha_map = {int(k): float(v) for k, v in parsed.items()}
        except (json.JSONDecodeError, ValueError):
            raise ValueError(f"--alpha-map must be a JSON dict, got: {raw_alpha_map!r}")
    else:
        resolved_alpha_map = None
    if resolved_alpha_map is not None:
        print(f"[AlphaMap] Overriding calibration alphas: {resolved_alpha_map}")

    # ── Update module-level targets ────────────────────────────────────
    global TARGET_ESCALATION_LEVELS
    TARGET_ESCALATION_LEVELS = set(args.escalation_levels)

    # ── Output directory ───────────────────────────────────────────────
    if args.output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = Path(f"levelled_rewrite_output/{ts}")
    args.output.mkdir(parents=True, exist_ok=True)

    # ── Reproducibility ────────────────────────────────────────────────
    random.seed(args.seed)

    # ── Initialise rewriter (loads model + vector) ─────────────────────
    rewriter = SteeredRewriter(
        model_name=args.model,
        vector_dir=args.vector_dir,
        calibration_path=args.calibration,
        prompts_path=args.prompts,
        seed=args.seed,
        level_map=resolved_level_map,
        alpha_map=resolved_alpha_map,
    )

    # ── Initialise labeler (unless skipping) ─────────────────────────
    labeler = None
    if not args.skip_labeling:
        label_model = args.gpt_model if args.label_backend == "openai" else args.qwen_model
        labeler = MessageLabeler(
            backend=args.label_backend,
            model=label_model,
            temperature=args.gpt_temperature,
            api_base=args.gpt_api_base,
            vllm_dtype=args.vllm_dtype,
            vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
            vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            vllm_max_model_len=args.vllm_max_model_len,
        )
        print(
            f"[Labeler] Checking backend={args.label_backend}, model={label_model}..."
        )
        if not labeler.check_health():
            print(f"ERROR: Label backend not accessible ({args.label_backend}).")
            if args.label_backend == "openai":
                print("Set API key:  export OPENAI_API_KEY='your-key-here'")
            else:
                print(
                    "Start a vLLM server first, e.g.: python -m "
                    f"vllm.entrypoints.openai.api_server --model {args.qwen_model} "
                    "--port 8000  (URL via --gpt-api-base or $VLLM_BASE_URL)"
                )
            print("Or pass --skip-labeling if input already has labeled_messages.")
            return 1
        print(
            f"[Labeler] ✓ backend={args.label_backend} ready (model: {label_model})"
        )
    else:
        print("[Labeler] Skipped — expecting pre-labeled input.")

    # ── Gather input files ─────────────────────────────────────────────
    input_path = args.input
    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        files = sorted(input_path.glob("*_dialogues.json"))
        if not files:
            files = sorted(input_path.glob("*.json"))
    else:
        print(f"ERROR: Input not found: {input_path}")
        return 1

    if not files:
        print(f"No JSON files found in {input_path}")
        return 1

    print(f"╔{'═'*58}╗")
    print(f"║  CAA-Steered Levelled Rewrite Pipeline{' '*19}║")
    print(f"╠{'═'*58}╣")
    print(f"║  Model:   {args.model:<47}║")
    print(f"║  Layer:   {rewriter.layer:<47}║")
    print(f"║  Levels:  {str(rewriter.level_to_alpha):<47}║")
    print(f"║  Esc→Lvl: 2→L1(α={rewriter.level_to_alpha[1]:.0f})  "
          f"3→L2(α={rewriter.level_to_alpha[2]:.0f})  "
          f"4→L3(α={rewriter.level_to_alpha[3]:.0f}){' '*5}║")
    if args.skip_labeling:
        label_mode = "skip (pre-labeled)"
    else:
        label_model = args.gpt_model if args.label_backend == "openai" else args.qwen_model
        label_mode = f"{args.label_backend}:{label_model}"
    print(f"║  Label:   {label_mode:<47}║")
    print(f"║  Files:   {len(files):<47}║")
    ratio_str = '  '.join(f'Esc{k}={v:.0%}' for k, v in sorted(injection_ratios.items()))
    print(f"║  Ratios:  {ratio_str:<47}║")
    print(f"║  Output:  {str(args.output)[:47]:<47}║")
    print(f"╚{'═'*58}╝")

    # ── Process files ──────────────────────────────────────────────────
    totals = {"files": 0, "processed": 0, "rewritten": 0,
              "attempted": 0, "successful": 0, "failed": 0}
    failed_files = []

    for fp in files:
        try:
            s = process_file(
                fp, args.output, rewriter,
                injection_ratios, args.min_per_conversation, args.max_retries,
                labeler=labeler,
                batch_size=args.batch_size,
                label_workers=args.label_workers,
            )
            totals["files"] += 1
            for k in ("processed", "rewritten", "attempted", "successful", "failed"):
                totals[k] += s[k]
        except Exception as e:
            print(f"\n{'!'*60}")
            print(f"ERROR processing {fp.name}: {e}")
            print(f"{'!'*60}")
            print("Continuing to next file...")
            failed_files.append({"file": fp.name, "error": str(e)})
            continue

    # ── Summary ────────────────────────────────────────────────────────
    sr = (totals["successful"] / totals["attempted"] * 100
          if totals["attempted"] > 0 else 0)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "method": "caa_steered_rewrite",
        "labeling_backend": None if args.skip_labeling else args.label_backend,
        "labeling_model": None if args.skip_labeling else (
            args.gpt_model if args.label_backend == "openai" else args.qwen_model
        ),
        "steering_layer": rewriter.layer,
        "level_alphas": rewriter.level_to_alpha,
        "escalation_level_mapping": ESC_TO_LEVEL,
        "injection_ratios": {str(k): v for k, v in injection_ratios.items()},
        "seed": args.seed,
        "statistics": totals,
        "failed_files": failed_files,
    }
    summary_path = args.output / "summary_stats.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("LEVELLED REWRITE COMPLETE")
    print(f"{'='*60}")
    print(f"  Files processed:        {totals['files']}")
    print(f"  Files with errors:      {len(failed_files)}")
    print(f"  Dialogues processed:    {totals['processed']}")
    print(f"  Dialogues rewritten:    {totals['rewritten']}")
    print(f"  Messages attempted:     {totals['attempted']}")
    print(f"  Messages successful:    {totals['successful']}")
    print(f"  Messages failed:        {totals['failed']}")
    print(f"  Success rate:           {sr:.1f}%")
    if failed_files:
        print(f"\n  Files with errors:")
        for err in failed_files:
            print(f"    - {err['file']}: {err['error']}")
    print(f"  Summary saved:          {summary_path}")
    print(f"{'='*60}")

    # Return error code if all files failed, success otherwise
    if len(failed_files) == len(files) and len(files) > 0:
        print("\n⚠ WARNING: All files failed to process!")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
