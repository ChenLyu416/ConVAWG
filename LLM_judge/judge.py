"""
Core dialogue judge.

Usage (programmatic):
    from judge import DialogueJudge
    judge = DialogueJudge(cfg)
    results = judge.evaluate_file("path/to/18015_dialogues.json")
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from llm_backend import LLMBackend, build_backend
from detoxify_scorer import (
    DetoxifyScorer,
    build_detoxify_scorer,
    compute_detoxify_summary,
    aggregate_detoxify_results,
)
from perplexity_scorer import (
    PerplexityScorer,
    build_perplexity_scorer,
    compute_perplexity_summary,
    aggregate_perplexity_results,
)
from diversity_scorer import (
    DiversityScorer,
    build_diversity_scorer,
    compute_diversity_summary,
    aggregate_diversity_results,
)
from prompts import (
    PER_DIALOGUE_METRICS,
    GLOBAL_METRICS,
    build_per_dialogue_prompt,
    build_global_prompt,
)

logger = logging.getLogger(__name__)

VALID_RATINGS = {"1", "2", "3", "4", "5", "N/A"}
LEGACY_RATING_MAP = {"Low": "1", "Medium": "3", "High": "5"}


# ---------------------------------------------------------------------------
# Baseline format normaliser
# ---------------------------------------------------------------------------

def _normalise_input(data: dict) -> dict:
    """
    Detect and convert baseline-generator output format into the standard
    judge input format.

    Baseline files use:
      - ``scenario_summary``  instead of ``vawg_scenarios``
      - ``dialogues[].turns`` (list of {speaker, text}) instead of ``caa_rewritten_dialogue``

    If the data is already in judge format (has ``vawg_scenarios`` OR dialogues
    already have ``caa_rewritten_dialogue``), it is returned unchanged.
    """
    # Already in judge format?
    if "vawg_scenarios" in data:
        return data
    dialogues = data.get("dialogues", [])
    if dialogues and "caa_rewritten_dialogue" in dialogues[0]:
        return data

    # ---- Convert baseline format ----
    summary = data.get("scenario_summary", {})
    vp = summary.get("victim_profile", {})
    pp = summary.get("perpetrator_profile", {})

    vawg_scenarios = {
        "persona_id":   data.get("persona_id", data.get("scenario_id", "")),
        "persona_text": data.get("persona_text", ""),
        "crime_types":  summary.get("crime_types", []),
        "scenario_description": summary.get("scenario_description", ""),
        "victim_profile": {
            "name":            vp.get("name", ""),
            "age":             vp.get("age"),
            "occupation":      vp.get("occupation", ""),
            "vulnerabilities": vp.get("vulnerabilities", []),
            "gender":          vp.get("gender", ""),
        },
        "perpetrator_profile": {
            "name":                   pp.get("name", ""),
            "age":                    pp.get("age"),
            "occupation":             pp.get("occupation", ""),
            "relationship_to_victim": pp.get("relationship_to_victim", "unknown"),
        },
    }

    normalised_dialogues: list[dict] = []
    for i, d in enumerate(dialogues):
        turns = d.get("turns", [])
        turns_str = "\n".join(
            f"{t.get('speaker', 'Speaker')}: {t.get('text', '')}"
            for t in turns
        )
        tl = d.get("timeline", {})
        tl_str = f"  [Month {tl.get('month', '?')}, {tl.get('date', '?')}]" if tl else ""
        normalised_dialogues.append({
            "event_id": d.get("dialogue_id", f"event_{i}"),
            "conversation_context": {
                "crime_type":  d.get("vawg_type", ""),
                "description": f"Baseline dialogue — VAWG type: {d.get('vawg_type', 'unspecified')}{tl_str}",
                "timeline":    tl,
            },
            "caa_rewritten_dialogue": turns_str,
        })

    normalised = dict(data)
    normalised["vawg_scenarios"] = vawg_scenarios
    normalised["dialogues"] = normalised_dialogues
    normalised["_baseline_normalised"] = True
    return normalised


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict:
    """
    Extract the first JSON object from an LLM reply.
    Tries direct parse first, then extracts the {...} block.
    """
    text = text.strip()
    # Strip <think>...</think> blocks (DeepSeek / Qwen chain-of-thought) before
    # attempting extraction — they contain {...} that break the greedy regex.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE)

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    logger.warning("Could not parse JSON from LLM response:\n%s", text[:500])
    return {}


def _normalise_metric(raw: Any, metric_name: str, allow_na: bool = False) -> dict:
    """
    Ensure a metric dict has 'rating' and 'explanation' keys with valid values.
    Fills defaults on parse failure.
    """
    allowed = VALID_RATINGS if allow_na else (VALID_RATINGS - {"N/A"})

    if not isinstance(raw, dict):
        return {"rating": "PARSE_ERROR", "explanation": f"Could not parse: {raw!r}"}

    rating_raw = raw.get("rating", "PARSE_ERROR")

    if isinstance(rating_raw, int) and 1 <= rating_raw <= 5:
        rating = str(rating_raw)
    elif isinstance(rating_raw, str):
        cleaned = rating_raw.strip()
        upper = cleaned.upper()
        if upper in {"N/A", "NA"}:
            rating = "N/A"
        elif cleaned in VALID_RATINGS:
            rating = cleaned
        elif cleaned in LEGACY_RATING_MAP:
            rating = LEGACY_RATING_MAP[cleaned]
        else:
            rating = cleaned
    else:
        rating = "PARSE_ERROR"

    if rating not in allowed:
        logger.warning("Unexpected rating %r for %s", rating, metric_name)
        rating = "PARSE_ERROR"

    return {
        "rating": rating,
        "explanation": raw.get("explanation", ""),
    }


def _parse_per_dialogue_response(text: str, event_id: str) -> dict:
    """Parse the LLM response for per-dialogue metrics."""
    data = _extract_json(text)
    result: dict = {}
    for metric in PER_DIALOGUE_METRICS:
        allow_na = (metric == "toxicity_realism")
        result[metric] = _normalise_metric(data.get(metric, {}), metric, allow_na=allow_na)
    return result


def _parse_global_response(text: str) -> dict:
    """Parse the LLM response for global metrics."""
    data = _extract_json(text)
    result: dict = {}
    for metric in GLOBAL_METRICS:
        result[metric] = _normalise_metric(data.get(metric, {}), metric)
    return result


# ---------------------------------------------------------------------------
# Main judge class
# ---------------------------------------------------------------------------

class DialogueJudge:
    """
    Evaluates a VAWG dialogue JSON file using an LLM judge.

    Parameters
    ----------
    cfg : dict
        Expected structure mirrors judge_config.yaml sections 'llm' and 'processing'.
    backend : LLMBackend, optional
        Provide a pre-built backend (e.g. for testing); if None, built from cfg.
    """

    def __init__(self, cfg: dict, backend: LLMBackend | None = None) -> None:
        self.llm_cfg = cfg.get("llm", cfg)            # accept flat or nested config
        self.proc_cfg = cfg.get("processing", {})
        self.backend = backend or build_backend(self.llm_cfg)
        self.detoxify: Any = build_detoxify_scorer(cfg)
        self.perplexity: Any = build_perplexity_scorer(cfg)
        self.diversity: Any = build_diversity_scorer(cfg)

        self.max_dialogue_chars: int = self.proc_cfg.get("max_dialogue_chars", 6000)
        self.max_global_chars: int   = self.proc_cfg.get("max_global_chars", 20000)
        self.skip_existing: bool     = self.proc_cfg.get("skip_existing", True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_file(self, path: str) -> dict:
        """
        Load a dialogue JSON file and return a structured evaluation result.

        Returns
        -------
        dict with keys:
          scenario_id, per_dialogue_ratings (list), global_ratings
        """
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        return self.evaluate(data)

    def evaluate(self, data: dict) -> dict:
        """
        Evaluate a dialogue dict (already parsed from JSON).

        Parameters
        ----------
        data : dict
            Contents of a *_dialogues.json file.

        Returns
        -------
        Evaluation report dict.
        """
        data = _normalise_input(data)

        scenario_id  = data.get("scenario_id", "unknown")
        # Support pipeline format (vawg_scenarios) and baseline format (scenario_summary)
        scenario_info = data.get("vawg_scenarios") or data.get("scenario_summary", {})
        events        = data.get("dialogues", [])

        logger.info("Evaluating scenario %s (%d dialogues)", scenario_id, len(events))

        # ------------------------------------------------------------------
        # Per-dialogue ratings
        # ------------------------------------------------------------------
        per_dialogue_ratings: list[dict] = []

        for idx, event in enumerate(events):
            # Support pipeline format (event_id) and baseline format (dialogue_id)
            event_id = event.get("event_id") or event.get("dialogue_id", f"event_{idx}")
            logger.info("  Dialogue %d/%d  [%s]", idx + 1, len(events), event_id)

            # Skip dialogues with no usable text content
            _has_text = (
                event.get("caa_rewritten_dialogue")
                or event.get("enforced_dialogue")
                or event.get("turns")
                or event.get("dialogue")
            )
            if not _has_text:
                logger.warning(
                    "  Skipping dialogue [%s]: no dialogue text found (all null/empty)", event_id
                )
                continue

            system_p, user_p = build_per_dialogue_prompt(
                scenario=scenario_info,
                event=event,
                max_dialogue_chars=self.max_dialogue_chars,
            )

            try:
                raw_reply = self.backend.chat(system_p, user_p)
                ratings = _parse_per_dialogue_response(raw_reply, event_id)
            except Exception as exc:
                logger.error("LLM call failed for event %s: %s", event_id, exc)
                ratings = {m: {"rating": "ERROR", "explanation": str(exc)}
                           for m in PER_DIALOGUE_METRICS}

            detoxify_scores: Any = None
            if self.detoxify is not None:
                try:
                    detoxify_scores = self.detoxify.score_dialogue(event)
                except Exception as exc:
                    logger.error("Detoxify call failed for event %s: %s", event_id, exc)
                    detoxify_scores = {label: None for label in self.detoxify.labels}

            perplexity_scores: Any = None
            if self.perplexity is not None:
                try:
                    perplexity_scores = self.perplexity.score_dialogue(event)
                except Exception as exc:
                    logger.error("Perplexity call failed for event %s: %s", event_id, exc)
                    perplexity_scores = {"perplexity": None}

            diversity_scores: Any = None
            if self.diversity is not None:
                try:
                    diversity_scores = self.diversity.score_dialogue(event)
                except Exception as exc:
                    logger.error("Diversity call failed for event %s: %s", event_id, exc)
                    diversity_scores = {"distinct_2": None, "self_bleu": None}

            # Baseline format stores timeline as dict, pipeline as string
            _timeline = event.get("timeline") or {}
            _timestamp = event.get("timestamp") or (
                f"month {_timeline.get('month','')} week {_timeline.get('week_in_month','')}"
                if isinstance(_timeline, dict) else str(_timeline)
            )
            _ctx = event.get("conversation_context") or {}
            # baseline: vawg_type; pipeline: conversation_context.crime_type
            _crime_type = _ctx.get("crime_type") or event.get("vawg_type", "")
            # determine dialogue source label
            if event.get("caa_rewritten_dialogue"):
                _source = "caa_rewritten_dialogue"
            elif event.get("enforced_dialogue"):
                _source = "enforced_dialogue"
            elif event.get("turns"):
                _source = "turns"
            else:
                _source = "unknown"
            _entry: dict = {
                "event_id":        event_id,
                "dialogue_index":  idx,
                "timestamp":       _timestamp,
                "escalation_level": _ctx.get("escalation_level"),
                "crime_type":      _crime_type,
                "dialogue_source": _source,
                "ratings": ratings,
            }
            if detoxify_scores is not None:
                _entry["detoxify_scores"] = detoxify_scores
            if perplexity_scores is not None:
                _entry["perplexity_scores"] = perplexity_scores
            if diversity_scores is not None:
                _entry["diversity_scores"] = diversity_scores
            per_dialogue_ratings.append(_entry)

        # ------------------------------------------------------------------
        # Global ratings
        # ------------------------------------------------------------------
        logger.info("  Computing global ratings for scenario %s …", scenario_id)
        system_g, user_g = build_global_prompt(
            scenario=scenario_info,
            events=events,
            max_global_chars=self.max_global_chars,
        )

        try:
            raw_global_reply = self.backend.chat(system_g, user_g)
            global_ratings = _parse_global_response(raw_global_reply)
        except Exception as exc:
            logger.error("Global LLM call failed: %s", exc)
            global_ratings = {m: {"rating": "ERROR", "explanation": str(exc)}
                              for m in GLOBAL_METRICS}

        # ------------------------------------------------------------------
        # Aggregate summary (convenience)
        # ------------------------------------------------------------------
        summary = _compute_summary(per_dialogue_ratings, global_ratings)
        detox_summary = compute_detoxify_summary(per_dialogue_ratings)
        if detox_summary is not None:
            summary["detoxify"] = detox_summary
        perplexity_summary = compute_perplexity_summary(per_dialogue_ratings)
        if perplexity_summary is not None:
            summary["perplexity"] = perplexity_summary
        diversity_summary = compute_diversity_summary(per_dialogue_ratings)
        if diversity_summary is not None:
            summary["diversity"] = diversity_summary

        # Support both pipeline format (top-level keys) and baseline format
        # (names nested inside scenario_summary.{victim,perpetrator}_profile.name)
        _ss = data.get("scenario_summary", {})
        _victim_name = (data.get("victim_name")
                        or _ss.get("victim_profile", {}).get("name", ""))
        _perp_name   = (data.get("perpetrator_name")
                        or _ss.get("perpetrator_profile", {}).get("name", ""))

        return {
            "scenario_id":           scenario_id,
            "victim_name":           _victim_name,
            "perpetrator_name":      _perp_name,
            "num_dialogues_evaluated": len(per_dialogue_ratings),
            "per_dialogue_ratings":  per_dialogue_ratings,
            "global_ratings":        global_ratings,
            "summary":               summary,
        }

    def cleanup(self) -> None:
        self.backend.cleanup()
        if self.detoxify is not None:
            self.detoxify.cleanup()
        if self.perplexity is not None:
            self.perplexity.cleanup()
        if self.diversity is not None:
            self.diversity.cleanup()


# ---------------------------------------------------------------------------
# Summary helper
# ---------------------------------------------------------------------------

# 1-5 scale (N/A / ERROR excluded from numeric averages)
# Keep legacy named labels for backward compatibility when aggregating old files.
_RATING_SCORE: dict[str, int] = {
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "Low": 1,
    "Medium": 3,
    "High": 5,
}


def _score_to_label(score: float) -> str:
    """Map a numeric average to nearest integer label on the 1-5 scale."""
    rounded = int(round(score))
    rounded = max(1, min(5, rounded))
    return str(rounded)


def _compute_summary(per_dialogue: list[dict], global_ratings: dict) -> dict:
    """
    Compute per-metric summary across all per-dialogue ratings.

    Scoring: 1..5 scale.
    N/A and ERROR entries are counted but excluded from numeric averages.
    """
    from collections import Counter

    summary: dict = {}

    for metric in PER_DIALOGUE_METRICS:
        counts: Counter = Counter()
        for entry in per_dialogue:
            r = entry["ratings"].get(metric, {}).get("rating", "ERROR")
            counts[r] += 1

        modal_rating = counts.most_common(1)[0][0] if counts else "N/A"

        # Weighted numeric average (each rating value multiplied by its count)
        total_score = sum(_RATING_SCORE[r] * n for r, n in counts.items() if r in _RATING_SCORE)
        total_rated = sum(n for r, n in counts.items() if r in _RATING_SCORE)
        avg_score   = round(total_score / total_rated, 2) if total_rated else None
        avg_label   = _score_to_label(avg_score) if avg_score is not None else "N/A"

        summary[metric] = {
            "modal_rating":    modal_rating,
            "average_score":   avg_score,
            "average_label":   avg_label,
            "counts":          dict(counts),
        }

    for metric in GLOBAL_METRICS:
        rating = global_ratings.get(metric, {}).get("rating", "N/A")
        summary[metric] = {
            "rating": rating,
            "score":  _RATING_SCORE.get(rating),  # None if N/A / ERROR
        }

    return summary


# ---------------------------------------------------------------------------
# Batch aggregation
# ---------------------------------------------------------------------------

def aggregate_results(results: list[dict]) -> dict:
    """
    Aggregate a list of per-file evaluation results into a single batch summary.

    Parameters
    ----------
    results : list[dict]
        Each element is the dict returned by DialogueJudge.evaluate().

    Returns
    -------
    dict with keys:
      num_files, scenario_ids, per_dialogue_metrics, global_metrics, overall_average_score
    """
    from collections import Counter

    if not results:
        return {}

    # ---- Per-dialogue metrics ----------------------------------------
    per_metric: dict[str, dict] = {m: {"total_score": 0, "total_rated": 0, "counts": Counter()}
                                    for m in PER_DIALOGUE_METRICS}

    for res in results:
        for entry in res.get("per_dialogue_ratings", []):
            for metric in PER_DIALOGUE_METRICS:
                rating = entry["ratings"].get(metric, {}).get("rating", "ERROR")
                per_metric[metric]["counts"][rating] += 1
                if rating in _RATING_SCORE:
                    per_metric[metric]["total_score"] += _RATING_SCORE[rating]
                    per_metric[metric]["total_rated"] += 1

    per_dialogue_summary: dict = {}
    for metric, data in per_metric.items():
        total_score = data["total_score"]
        total_rated = data["total_rated"]
        avg_score   = round(total_score / total_rated, 2) if total_rated else None
        avg_label   = _score_to_label(avg_score) if avg_score is not None else "N/A"
        counts      = dict(data["counts"])
        modal_rating = data["counts"].most_common(1)[0][0] if data["counts"] else "N/A"
        per_dialogue_summary[metric] = {
            "modal_rating":  modal_rating,
            "average_score": avg_score,
            "average_label": avg_label,
            "total_dialogues_rated": total_rated,
            "counts": counts,
        }

    # ---- Global metrics ---------------------------------------------
    global_metric_data: dict[str, dict] = {m: {"total_score": 0, "total_rated": 0, "counts": Counter()}
                                            for m in GLOBAL_METRICS}

    for res in results:
        for metric in GLOBAL_METRICS:
            rating = res.get("global_ratings", {}).get(metric, {}).get("rating", "N/A")
            global_metric_data[metric]["counts"][rating] += 1
            if rating in _RATING_SCORE:
                global_metric_data[metric]["total_score"] += _RATING_SCORE[rating]
                global_metric_data[metric]["total_rated"] += 1

    global_summary: dict = {}
    for metric, data in global_metric_data.items():
        total_score = data["total_score"]
        total_rated = data["total_rated"]
        avg_score   = round(total_score / total_rated, 2) if total_rated else None
        avg_label   = _score_to_label(avg_score) if avg_score is not None else "N/A"
        modal_rating = data["counts"].most_common(1)[0][0] if data["counts"] else "N/A"
        global_summary[metric] = {
            "modal_rating":  modal_rating,
            "average_score": avg_score,
            "average_label": avg_label,
            "total_files_rated": total_rated,
            "counts": dict(data["counts"]),
        }

    # ---- Overall average across all 6 LLM-judge metrics --------------
    all_scores = [
        per_dialogue_summary[m]["average_score"]
        for m in PER_DIALOGUE_METRICS
        if per_dialogue_summary[m]["average_score"] is not None
    ] + [
        global_summary[m]["average_score"]
        for m in GLOBAL_METRICS
        if global_summary[m]["average_score"] is not None
    ]
    overall_avg = round(sum(all_scores) / len(all_scores), 2) if all_scores else None

    batch: dict = {
        "num_files":               len(results),
        "scenario_ids":            [r.get("scenario_id") for r in results],
        "per_dialogue_metrics":    per_dialogue_summary,
        "global_metrics":          global_summary,
        "overall_average_score":   overall_avg,
        "overall_average_label":   _score_to_label(overall_avg) if overall_avg is not None else "N/A",
    }
    detox_agg = aggregate_detoxify_results(results)
    if detox_agg is not None:
        batch["detoxify_metrics"] = detox_agg
    perplexity_agg = aggregate_perplexity_results(results)
    if perplexity_agg is not None:
        batch["perplexity_metrics"] = perplexity_agg
    diversity_agg = aggregate_diversity_results(results)
    if diversity_agg is not None:
        batch["diversity_metrics"] = diversity_agg
    return batch
