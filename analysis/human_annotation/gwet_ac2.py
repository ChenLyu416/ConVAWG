#!/usr/bin/env python3
"""
gwet_ac2.py
-----------
Compute Gwet's AC2 (weighted agreement coefficient) for:

  1. Inter-annotator agreement — among the adopted human annotators.
  1b. Humans + LLM agreement   — same, with the LLM judge added as an extra rater.
  2. Human-to-LLM agreement    — averaged human scores vs the LLM judge.

By default the six annotators of the paper's human evaluation
(ADOPTED_ANNOTATORS, pseudonymous directory names) are used; pass --all-annotators to include
every directory under the results dir.

Gwet's AC2 (Gwet 2008, Statistics in Medicine) uses a chance-correction term
that is more stable under skewed rating distributions than Cohen's κ / Fleiss' κ.
The coefficient is defined as:

    AC2 = (p_a − p_e) / (1 − p_e)

where
    p_a  = observed (weighted) agreement
    p_e  = Gwet's chance agreement
         = [1/(K−1)] · Σ_k  π_k (1−π_k) · Σ_l w_{kl}

Weight schemes
--------------
  unweighted  (AC1)  : w_{kl} = δ(k=l)
  linear      (AC2L) : w_{kl} = 1 − |k−l| / (K−1)
  quadratic   (AC2Q) : w_{kl} = 1 − ((k−l)/(K−1))²   ← default

Rating scale: 1–5 (K = 5) for all metrics.

Missing values (N/A, None, 0 on toxicity_realism) are excluded.

Usage
-----
    python gwet_ac2.py
    python gwet_ac2.py --results-dir analysis/human_annotation/annotation_results
    python gwet_ac2.py --weights linear
    python gwet_ac2.py --no-human-llm        # skip human-LLM sections
    python gwet_ac2.py --all-annotators      # use every annotator directory
    python gwet_ac2.py --exclude <annotator_dir> [<annotator_dir> ...]
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np


# ── Metric definitions ────────────────────────────────────────────────────────
DIALOGUE_METRICS = [
    "coherence",
    "humanlikeness",
    "persona_consistency",
    "toxicity_realism",
]
GLOBAL_METRICS = ["crime_fidelity", "scenario_realism"]
ALL_METRICS = DIALOGUE_METRICS + GLOBAL_METRICS

SCALE_MIN, SCALE_MAX = 1, 5
CATEGORIES = list(range(SCALE_MIN, SCALE_MAX + 1))   # [1, 2, 3, 4, 5]
K = len(CATEGORIES)

SKIP_DIR_NAMES = {"averaged"}

# Pseudonymous directory names of the six annotators of the paper's human evaluation.
ADOPTED_ANNOTATORS = ["Chogath", "Illaoi", "Ivern", "MasterYi", "Rammus", "Sylas"]


# ══════════════════════════════════════════════════════════════════════════════
# Weight matrix
# ══════════════════════════════════════════════════════════════════════════════

def build_weights(scheme: str) -> np.ndarray:
    """
    Build a K×K weight matrix for categories 1..K.

    scheme: "unweighted" | "linear" | "quadratic"
    """
    W = np.zeros((K, K))
    for i, c in enumerate(CATEGORIES):
        for j, k in enumerate(CATEGORIES):
            if scheme == "unweighted":
                W[i, j] = 1.0 if c == k else 0.0
            elif scheme == "linear":
                W[i, j] = 1.0 - abs(c - k) / (K - 1)
            elif scheme == "quadratic":
                W[i, j] = 1.0 - ((c - k) / (K - 1)) ** 2
            else:
                raise ValueError(f"Unknown weight scheme: {scheme!r}")
    return W


# ══════════════════════════════════════════════════════════════════════════════
# Data loading helpers
# ══════════════════════════════════════════════════════════════════════════════

def _is_valid(metric: str, val) -> bool:
    """Return True if val is a scoreable rating for the given metric."""
    if val is None or val == "N/A":
        return False
    if metric == "toxicity_realism" and val == 0:
        return False
    return True


def load_human_ratings(
    results_dir: Path,
    exclude: set[str] | None = None,
    include: set[str] | None = None,
) -> tuple[list[str], dict[str, dict[str, dict]]]:
    """
    Load per-user annotation JSONs.

    Parameters
    ----------
    include : if given, only annotator directories with these names are loaded
              (applied after SKIP_DIR_NAMES and exclude).

    Returns
    -------
    annotators : sorted list of annotator names
    data       : { annotator: { metric: { item_key: float } } }
                 item_key = (conv_name, dlg_idx)  for dialogue metrics
                 item_key = (conv_name, "__global__") for global metrics
    """
    exclude = exclude or set()
    user_dirs = sorted(
        d for d in results_dir.iterdir()
        if d.is_dir() and d.name not in SKIP_DIR_NAMES and d.name not in exclude
        and (include is None or d.name in include)
    )
    annotators = [d.name for d in user_dirs]
    data: dict[str, dict] = {}

    for user_dir in user_dirs:
        username = user_dir.name
        metric_items: dict[str, dict] = {m: {} for m in ALL_METRICS}

        for jf in sorted(user_dir.glob("*.json")):
            conv_data = json.loads(jf.read_text(encoding="utf-8"))
            conv_name = conv_data.get("conversation", jf.stem)

            # dialogue-level
            for dlg in conv_data.get("per_dialogue_ratings", []):
                d_idx = dlg.get("dialogue_index")
                item_key = (conv_name, d_idx)
                for m in DIALOGUE_METRICS:
                    val = dlg.get("ratings", {}).get(m, {}).get("score")
                    if not _is_valid(m, val):
                        continue
                    metric_items[m][item_key] = float(val)

            # global-level (one per conversation)
            global_key = (conv_name, "__global__")
            for m in GLOBAL_METRICS:
                raw = conv_data.get("global_ratings", {}).get(m, {}).get("rating")
                if raw is None or raw == "N/A":
                    continue
                try:
                    metric_items[m][global_key] = float(raw)
                except (ValueError, TypeError):
                    pass

        data[username] = metric_items

    return annotators, data


def load_llm_ratings(llm_dir: Path) -> dict[str, dict]:
    """
    Load LLM judge JSONs (one per conversation, named 1.json .. 10.json).

    Returns { metric: { item_key: float } }
    """
    llm: dict[str, dict] = {m: {} for m in ALL_METRICS}

    for jf in sorted(llm_dir.glob("*.json"), key=lambda p: int(p.stem)):
        conv_data = json.loads(jf.read_text(encoding="utf-8"))
        conv_name = f"Conversation {jf.stem}"

        for dlg in conv_data.get("per_dialogue_ratings", []):
            d_idx = dlg.get("dialogue_index")
            item_key = (conv_name, d_idx)
            for m in DIALOGUE_METRICS:
                raw = dlg.get("ratings", {}).get(m, {}).get("rating")
                if raw is None or raw == "N/A":
                    continue
                try:
                    llm[m][item_key] = float(raw)
                except (ValueError, TypeError):
                    pass

        global_key = (conv_name, "__global__")
        for m in GLOBAL_METRICS:
            raw = conv_data.get("global_ratings", {}).get(m, {}).get("rating")
            if raw is None or raw == "N/A":
                continue
            try:
                llm[m][global_key] = float(raw)
            except (ValueError, TypeError):
                pass

    return llm


# ══════════════════════════════════════════════════════════════════════════════
# Reliability matrix builder
# ══════════════════════════════════════════════════════════════════════════════

def build_matrix(
    annotators: list[str],
    data: dict[str, dict],
    metric: str,
) -> tuple[np.ndarray, list]:
    """
    Build a (n_raters × n_items) rating matrix, with NaN for missing values.
    Returns (matrix, item_keys).
    """
    all_items = sorted(
        {key for user in annotators for key in data[user][metric]},
        key=lambda x: (x[0], str(x[1])),
    )
    matrix = np.full((len(annotators), len(all_items)), np.nan)
    for r_idx, username in enumerate(annotators):
        for c_idx, item_key in enumerate(all_items):
            val = data[username][metric].get(item_key)
            if val is not None:
                matrix[r_idx, c_idx] = val
    return matrix, all_items


# ══════════════════════════════════════════════════════════════════════════════
# Gwet's AC2 core
# ══════════════════════════════════════════════════════════════════════════════

def _cat_idx(val: float) -> int:
    """Map a category value (1.0–5.0) to 0-based index, clamped."""
    return int(round(val)) - SCALE_MIN


def gwet_ac2(
    matrix: np.ndarray,
    weights: np.ndarray,
) -> tuple[Optional[float], dict]:
    """
    Compute Gwet's AC2 for a (n_raters × n_items) reliability matrix.

    Returns (ac2_score, diagnostics_dict).
    ac2_score is None if computation is not possible.

    Algorithm (Gwet 2008):
      1. p_a  = mean over items of mean pairwise weighted agreement
      2. π_k  = mean over items of (n_ik / r_i)   for each category k
      3. p_e  = [1/(K−1)] · Σ_k  π_k (1−π_k) · Σ_l w_{kl}
      4. AC2  = (p_a − p_e) / (1 − p_e)
    """
    n_raters, n_items = matrix.shape

    # ── Step 1: Observed agreement ────────────────────────────────────────────
    pa_sum = 0.0
    valid_item_count = 0
    n_pairs_total = 0

    for item_idx in range(n_items):
        col = matrix[:, item_idx]
        valid = col[~np.isnan(col)]
        r_i = len(valid)
        if r_i < 2:
            continue
        valid_item_count += 1

        # All unordered pairs of raters
        pair_agreements = []
        for a_val, b_val in itertools.combinations(valid, 2):
            ai = min(max(_cat_idx(a_val), 0), K - 1)
            bi = min(max(_cat_idx(b_val), 0), K - 1)
            pair_agreements.append(weights[ai, bi])

        pa_sum += sum(pair_agreements) / len(pair_agreements)
        n_pairs_total += len(pair_agreements)

    if valid_item_count < 2:
        return None, {"valid_items": valid_item_count, "n_raters": n_raters}

    p_a = pa_sum / valid_item_count

    # ── Step 2: Marginal probabilities π_k ───────────────────────────────────
    pi = np.zeros(K)
    for item_idx in range(n_items):
        col = matrix[:, item_idx]
        valid = col[~np.isnan(col)]
        r_i = len(valid)
        if r_i == 0:
            continue
        for v in valid:
            ki = min(max(_cat_idx(v), 0), K - 1)
            pi[ki] += 1.0 / r_i
    pi /= valid_item_count          # normalise by number of valid items

    # ── Step 3: Gwet's chance agreement ──────────────────────────────────────
    # p_e = (1/(K-1)) * Σ_k  π_k(1−π_k) · Σ_l w_{kl}
    p_e = 0.0
    for k_idx in range(K):
        sum_w_k = float(np.sum(weights[k_idx, :]))
        p_e += pi[k_idx] * (1.0 - pi[k_idx]) * sum_w_k
    p_e /= (K - 1)

    # ── Step 4: AC2 ───────────────────────────────────────────────────────────
    if abs(1.0 - p_e) < 1e-12:
        return None, {"p_a": p_a, "p_e": p_e, "valid_items": valid_item_count}

    ac2 = (p_a - p_e) / (1.0 - p_e)

    diag = {
        "p_a": round(p_a, 6),
        "p_e": round(p_e, 6),
        "valid_items": valid_item_count,
        "n_raters": n_raters,
        "pi": pi.tolist(),
    }
    return float(ac2), diag


def gwet_ac2_pairwise(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    weights: np.ndarray,
) -> tuple[Optional[float], dict]:
    """
    Compute Gwet's AC2 for exactly two raters given paired score arrays.

    NaN values are treated as missing and the corresponding item is excluded
    from *both* sequences.  This is the human-vs-LLM variant where rater A is
    the mean human score (possibly non-integer, rounded for π computation) and
    rater B is the LLM score.

    The weight for non-integer human averages is computed directly:
        w(h, l) = 1 − ((h − l) / (K−1))²   [quadratic]
        w(h, l) = 1 − |h − l| / (K−1)       [linear]
    using the continuous values, while π is computed from rounded integers.
    """
    assert len(scores_a) == len(scores_b)
    valid_mask = ~(np.isnan(scores_a) | np.isnan(scores_b))
    a = scores_a[valid_mask]
    b = scores_b[valid_mask]
    n = len(a)

    if n < 3:
        return None, {"valid_items": n}

    # ── Step 1: Observed agreement (continuous weights) ───────────────────────
    # Determine weight scheme from weight matrix shape/values
    # Use continuous formula matching the matrix scheme:
    # Check if weights are quadratic or linear by sampling
    scheme = _infer_scheme(weights)

    if scheme == "quadratic":
        w_fn = lambda x, y: 1.0 - ((x - y) / (K - 1)) ** 2
    elif scheme == "linear":
        w_fn = lambda x, y: 1.0 - abs(x - y) / (K - 1)
    else:  # unweighted — use rounded comparison
        w_fn = lambda x, y: 1.0 if abs(x - y) < 0.5 else 0.0

    p_a = float(np.mean([w_fn(float(ai), float(bi)) for ai, bi in zip(a, b)]))

    # ── Step 2: Marginals from rounded integer values ─────────────────────────
    a_int = np.clip(np.round(a).astype(int), SCALE_MIN, SCALE_MAX)
    b_int = np.clip(np.round(b).astype(int), SCALE_MIN, SCALE_MAX)
    all_int = np.concatenate([a_int, b_int])

    pi = np.zeros(K)
    for v in all_int:
        ki = v - SCALE_MIN
        pi[ki] += 1.0
    pi /= len(all_int)

    # ── Step 3: Gwet's chance ─────────────────────────────────────────────────
    p_e = 0.0
    for k_idx in range(K):
        sum_w_k = float(np.sum(weights[k_idx, :]))
        p_e += pi[k_idx] * (1.0 - pi[k_idx]) * sum_w_k
    p_e /= (K - 1)

    if abs(1.0 - p_e) < 1e-12:
        return None, {"p_a": p_a, "p_e": p_e, "valid_items": n}

    ac2 = (p_a - p_e) / (1.0 - p_e)

    diag = {
        "p_a": round(p_a, 6),
        "p_e": round(p_e, 6),
        "valid_items": n,
        "pi": pi.tolist(),
    }
    return float(ac2), diag


def _infer_scheme(weights: np.ndarray) -> str:
    """Heuristically identify weight scheme from the matrix."""
    w12 = weights[0, 1]   # w(1, 2) — one step apart
    w13 = weights[0, 2]   # w(1, 3) — two steps apart
    if abs(w12 - 1.0) < 1e-9:     # w_kk = 1 but checked off-diag
        return "unweighted"
    linear_w12 = 1.0 - 1.0 / (K - 1)
    quad_w12   = 1.0 - (1.0 / (K - 1)) ** 2
    if abs(w12 - linear_w12) < 1e-6:
        return "linear"
    if abs(w12 - quad_w12) < 1e-6:
        return "quadratic"
    return "quadratic"


# ══════════════════════════════════════════════════════════════════════════════
# Interpretation helpers
# ══════════════════════════════════════════════════════════════════════════════

def interpret(ac2: Optional[float]) -> str:
    """Landis & Koch / Gwet benchmark labels."""
    if ac2 is None:
        return "n/a"
    if ac2 < 0.00:
        return "poor (< chance)"
    if ac2 < 0.20:
        return "slight"
    if ac2 < 0.40:
        return "fair"
    if ac2 < 0.60:
        return "moderate"
    if ac2 < 0.67:
        return "tentative"
    if ac2 < 0.80:
        return "substantial"
    return "strong"


def fmt_ac2(v: Optional[float]) -> str:
    return f"{v:+.4f}" if v is not None else "    N/A "


# ══════════════════════════════════════════════════════════════════════════════
# Reporting
# ══════════════════════════════════════════════════════════════════════════════

COL = (26, 9, 8, 7, 18)
SEP = "─" * (sum(COL) + len(COL) * 3)
HDR = (
    f"{'Metric':<{COL[0]}}  {'AC2':>{COL[1]}}  "
    f"{'Items':>{COL[2]}}  {'Raters':>{COL[3]}}  "
    f"{'Interpretation':<{COL[4]}}"
)


def print_table(
    results: dict[str, Optional[float]],
    diags: dict[str, dict],
) -> None:
    print(SEP)
    print(HDR)
    print(SEP)
    for m in ALL_METRICS:
        ac2  = results.get(m)
        d    = diags.get(m, {})
        ni   = d.get("valid_items", 0)
        nr   = d.get("n_raters", 0)
        tag  = ("  [dialogue]" if m in DIALOGUE_METRICS else "  [global]  ")
        print(
            f"  {m:<{COL[0]-2}}  {fmt_ac2(ac2):>{COL[1]}}  "
            f"{ni:>{COL[2]}}  {nr:>{COL[3]}}  "
            f"{interpret(ac2):<{COL[4]}}"
            f"{tag}"
        )
    print(SEP)


def print_summary(results: dict[str, Optional[float]], diags: dict[str, dict]) -> None:
    valid = [(m, v) for m, v in results.items() if v is not None]
    if not valid:
        print("  No valid metrics.\n")
        return
    overall = float(np.mean([v for _, v in valid]))
    # item-weighted
    weights_sum = sum(diags.get(m, {}).get("valid_items", 0) for m, _ in valid)
    weighted = (
        sum(v * diags.get(m, {}).get("valid_items", 0) for m, v in valid) / weights_sum
        if weights_sum > 0 else None
    )
    print(
        f"\n  {'Unweighted mean AC2':<{COL[0]}}  {fmt_ac2(overall):>{COL[1]}}"
        f"   ({len(valid)} metrics)  →  {interpret(overall)}"
    )
    print(
        f"  {'Item-weighted mean AC2':<{COL[0]}}  {fmt_ac2(weighted):>{COL[1]}}"
        f"   (weighted by #items)  →  {interpret(weighted)}"
    )
    print(
        "\n  Benchmark (Gwet 2008):  AC2 ≥ 0.800 → reliable;  "
        "0.667–0.800 → tentative;  < 0.667 → unreliable\n"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Human-LLM agreement
# ══════════════════════════════════════════════════════════════════════════════

def compute_human_llm_ac2(
    annotators: list[str],
    human_data: dict[str, dict],
    llm_data: dict[str, dict],
    weights_mat: np.ndarray,
) -> tuple[dict[str, Optional[float]], dict[str, dict]]:
    """
    For each metric, average all human scores per item and compare with LLM.
    Returns (results, diags) dicts keyed by metric name.
    """
    results: dict[str, Optional[float]] = {}
    diags: dict[str, dict] = {}

    for m in ALL_METRICS:
        # Union of item keys where LLM has a score
        llm_items = llm_data[m]
        if not llm_items:
            results[m] = None
            diags[m] = {"valid_items": 0}
            continue

        all_keys = sorted(llm_items.keys(), key=lambda x: (x[0], str(x[1])))
        h_scores = []
        l_scores = []

        for key in all_keys:
            llm_val = llm_items.get(key)
            if llm_val is None:
                continue
            # Average valid human scores for this item
            human_vals = [
                human_data[u][m][key]
                for u in annotators
                if key in human_data[u][m]
            ]
            if not human_vals:
                continue
            h_scores.append(float(np.mean(human_vals)))
            l_scores.append(float(llm_val))

        if len(h_scores) < 3:
            results[m] = None
            diags[m] = {"valid_items": len(h_scores)}
            continue

        h_arr = np.array(h_scores)
        l_arr = np.array(l_scores)
        ac2, d = gwet_ac2_pairwise(h_arr, l_arr, weights_mat)
        d["n_raters"] = 2
        results[m] = ac2
        diags[m] = d

    return results, diags


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Gwet's AC2 agreement for annotation result files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-dir", "-d",
        default="analysis/human_annotation/annotation_results",
        help="Folder with per-annotator sub-directories (default: analysis/human_annotation/annotation_results).",
    )
    parser.add_argument(
        "--llm-dir",
        default="analysis/human_annotation/llm_result",
        help="Folder with LLM judge JSON files (default: analysis/human_annotation/llm_result).",
    )
    parser.add_argument(
        "--weights", "-w",
        default="quadratic",
        choices=["unweighted", "linear", "quadratic"],
        help="Weight scheme for AC2 (default: quadratic, i.e. interval scale).",
    )
    parser.add_argument(
        "--exclude", nargs="*", default=[],
        metavar="NAME",
        help="Annotator names to exclude (space-separated).",
    )
    parser.add_argument(
        "--all-annotators", action="store_true",
        help="Use every annotator directory instead of the adopted 6-annotator "
             f"subset ({', '.join(ADOPTED_ANNOTATORS)}).",
    )
    parser.add_argument(
        "--no-human-llm", action="store_true",
        help="Skip the human-vs-LLM agreement sections.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    llm_dir     = Path(args.llm_dir)

    if not results_dir.is_dir():
        import sys
        sys.exit(f"[Error] Results directory not found: {results_dir}")

    weights_mat = build_weights(args.weights)
    exclude_set = set(args.exclude)

    scheme_label = {
        "unweighted": "AC1 (unweighted)",
        "linear":     "AC2-linear (ordinal)",
        "quadratic":  "AC2-quadratic (interval)",
    }[args.weights]

    # ── Load data ─────────────────────────────────────────────────────────────
    include_set = None if args.all_annotators else set(ADOPTED_ANNOTATORS)
    annotators, human_data = load_human_ratings(
        results_dir, exclude=exclude_set, include=include_set
    )

    if args.exclude:
        print(f"Excluded    : {', '.join(sorted(exclude_set))}")

    print(f"\nGwet's {scheme_label}")
    print(f"Results dir : {results_dir.resolve()}")
    subset_label_ann = "all" if args.all_annotators else "adopted subset"
    print(f"Annotators  : {len(annotators)} ({subset_label_ann})  →  {', '.join(annotators)}\n")

    # ══════════════════════════════════════════════════════════════════════════
    # Section 1: Inter-annotator agreement
    # ══════════════════════════════════════════════════════════════════════════
    print("=" * (sum(COL) + len(COL) * 3))
    print("  SECTION 1 — Inter-Annotator Agreement")
    print("=" * (sum(COL) + len(COL) * 3))

    ia_results: dict[str, Optional[float]] = {}
    ia_diags: dict[str, dict] = {}

    for m in ALL_METRICS:
        matrix, _ = build_matrix(annotators, human_data, m)
        # Drop items rated by fewer than 2 annotators
        rated = np.sum(~np.isnan(matrix), axis=0)
        matrix_f = matrix[:, rated >= 2]
        ac2, d = gwet_ac2(matrix_f, weights_mat)
        d["n_raters"] = len(annotators)
        ia_results[m] = ac2
        ia_diags[m] = d

    print_table(ia_results, ia_diags)
    print_summary(ia_results, ia_diags)

    # ══════════════════════════════════════════════════════════════════════════
    # Sections 1b & 2: agreement involving the LLM judge
    # ══════════════════════════════════════════════════════════════════════════
    if not args.no_human_llm:
        if not llm_dir.is_dir():
            print(f"[Warning] LLM dir not found ({llm_dir}); skipping human-LLM sections.")
        else:
            llm_data = load_llm_ratings(llm_dir)

            # ── Section 1b: humans + LLM as one extra rater ──────────────────
            print("=" * (sum(COL) + len(COL) * 3))
            print("  SECTION 1b — Inter-Rater Agreement, Humans + LLM Judge as extra rater")
            print("=" * (sum(COL) + len(COL) * 3))
            print(f"  LLM dir : {llm_dir.resolve()}\n")

            raters_with_llm = annotators + ["LLM"]
            data_with_llm = dict(human_data)
            data_with_llm["LLM"] = llm_data

            il_results: dict[str, Optional[float]] = {}
            il_diags: dict[str, dict] = {}
            for m in ALL_METRICS:
                matrix, _ = build_matrix(raters_with_llm, data_with_llm, m)
                rated = np.sum(~np.isnan(matrix), axis=0)
                matrix_f = matrix[:, rated >= 2]
                ac2, d = gwet_ac2(matrix_f, weights_mat)
                d["n_raters"] = len(raters_with_llm)
                il_results[m] = ac2
                il_diags[m] = d

            print_table(il_results, il_diags)
            print_summary(il_results, il_diags)

            # ── Section 2: averaged humans vs LLM (pairwise) ─────────────────
            print("=" * (sum(COL) + len(COL) * 3))
            print("  SECTION 2 — Human (avg) vs LLM-Judge Agreement")
            print("=" * (sum(COL) + len(COL) * 3))
            print(f"  LLM dir : {llm_dir.resolve()}\n")

            hl_results, hl_diags = compute_human_llm_ac2(
                annotators, human_data, llm_data, weights_mat
            )
            print_table(hl_results, hl_diags)
            print_summary(hl_results, hl_diags)


if __name__ == "__main__":
    main()
