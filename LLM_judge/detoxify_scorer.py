#!/usr/bin/env python3
"""
Detoxify scorer for dialogue-level toxicity evaluation.

Uses the `detoxify` library (local inference, no API key required) to score
each dialogue on multi-label toxicity attributes.

Available models
----------------
original      (Jigsaw Toxicity Prediction — English)
  labels: toxicity, severe_toxicity, obscene, threat, insult, identity_attack

unbiased      (Jigsaw Civil Comments — English, bias-aware)
  labels: toxicity, severe_toxicity, obscene, threat, insult, identity_attack,
          sexual_explicit

multilingual  (Civil Comments — 6 languages)
  labels: same as unbiased

Usage (standalone):
    from detoxify_scorer import DetoxifyScorer
    scorer = DetoxifyScorer(model_type="original", device="cpu")
    scores = scorer.score_text("I hate you.")
    # {"toxicity": 0.91, "severe_toxicity": 0.04, ...}

Usage (via factory):
    from detoxify_scorer import build_detoxify_scorer
    scorer = build_detoxify_scorer(full_cfg)   # None if disabled
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

SUPPORTED_MODELS = ("original", "unbiased", "multilingual")

# Expected output labels per model type
MODEL_LABELS: Dict[str, List[str]] = {
    "original": [
        "toxicity",
        "severe_toxicity",
        "obscene",
        "threat",
        "insult",
        "identity_attack",
    ],
    "unbiased": [
        "toxicity",
        "severe_toxicity",
        "obscene",
        "threat",
        "insult",
        "identity_attack",
        "sexual_explicit",
    ],
    "multilingual": [
        "toxicity",
        "severe_toxicity",
        "obscene",
        "threat",
        "insult",
        "identity_attack",
        "sexual_explicit",
    ],
}


# ---------------------------------------------------------------------------
# Dialogue-text extractor (mirrors judge.py to avoid a circular import)
# ---------------------------------------------------------------------------

def _get_dialogue_text(event: Dict) -> str:
    """Extract plain-text dialogue string from various pipeline event formats."""
    text = event.get("caa_rewritten_dialogue") or event.get("enforced_dialogue")
    if text:
        return str(text)
    turns = event.get("turns", [])
    if turns:
        return "\n".join(
            f"{t.get('speaker', '?')}: {t.get('text', '')}" for t in turns
        )
    return ""


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class DetoxifyScorer:
    """
    Local Detoxify inference wrapper for dialogue-level scoring.

    The underlying model is loaded lazily on first call to ``score_text`` to
    avoid consuming GPU/CPU memory at import time.

    Parameters
    ----------
    model_type : str
        One of "original", "unbiased", "multilingual".
    device : str
        PyTorch device string, e.g. "cpu", "cuda", "cuda:0".
    max_chars : int
        Maximum characters of text to pass per call; longer text is truncated.
    """

    def __init__(
        self,
        model_type: str = "original",
        device: str = "cpu",
        max_chars: int = 20000,
    ) -> None:
        if model_type not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported Detoxify model_type {model_type!r}. "
                f"Choose from: {SUPPORTED_MODELS}"
            )
        # Detoxify passes the device string directly to PyTorch — multi-GPU
        # specs like "cuda:1,2,3,4" are not valid.  Pick the first index.
        if "," in device:
            first = device.split(",")[0].strip()
            logger.warning(
                "Detoxify does not support multi-GPU device string %r. "
                "Using first device: %r",
                device, first,
            )
            device = first
        self.model_type = model_type
        self.device = device
        self.max_chars = max_chars
        self._model: Optional[object] = None   # loaded lazily
        self._load_failed: bool = False        # avoid retrying after a permanent failure
        self.labels: List[str] = MODEL_LABELS[model_type]
        logger.info(
            "DetoxifyScorer configured: model=%s  device=%s", model_type, device
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        if self._load_failed:
            raise RuntimeError(
                "Detoxify model failed to load previously; skipping further attempts."
            )
        try:
            from detoxify import Detoxify
        except ImportError:
            raise ImportError(
                "detoxify is not installed. Run: pip install detoxify"
            )
        logger.info("Loading Detoxify model '%s' on device '%s' …", self.model_type, self.device)
        try:
            self._model = Detoxify(self.model_type, device=self.device)
        except Exception as exc:
            self._load_failed = True
            raise RuntimeError(f"Detoxify model load failed: {exc}") from exc
        logger.info("Detoxify model loaded.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_text(self, text: str) -> Dict[str, Optional[float]]:
        """
        Score a plain-text string with Detoxify.

        Parameters
        ----------
        text : str
            Raw dialogue text to score.

        Returns
        -------
        dict mapping label name to float score [0.0, 1.0], or None on error.
        """
        empty: Dict[str, Optional[float]] = {label: None for label in self.labels}

        if not text.strip():
            return empty

        if len(text) > self.max_chars:
            text = text[: self.max_chars]

        self._ensure_model()
        try:
            raw: Dict = self._model.predict(text)  # type: ignore[union-attr]
            return {
                label: float(raw[label]) if label in raw else None
                for label in self.labels
            }
        except Exception as exc:
            logger.error("Detoxify prediction error: %s", exc)
            return empty

    def score_dialogue(self, event: Dict) -> Dict[str, Optional[float]]:
        """
        Extract dialogue text from a pipeline event dict and score it.

        Handles pipeline format (``caa_rewritten_dialogue`` / ``enforced_dialogue``)
        and baseline format (``turns`` list).

        Parameters
        ----------
        event : dict
            A single dialogue entry from a *_dialogues.json file.

        Returns
        -------
        dict mapping label name to score (0.0–1.0) or None.
        """
        text = _get_dialogue_text(event)
        if not text:
            logger.debug(
                "Empty dialogue for event %s — skipping Detoxify scoring.",
                event.get("event_id") or event.get("dialogue_id", "?"),
            )
            return {label: None for label in self.labels}
        return self.score_text(text)

    def cleanup(self) -> None:
        """Release model weights from memory."""
        if self._model is not None:
            del self._model
            self._model = None
            try:
                import gc
                import torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            logger.info("DetoxifyScorer model released.")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_detoxify_scorer(cfg: dict) -> Optional[DetoxifyScorer]:
    """
    Instantiate a ``DetoxifyScorer`` from a top-level config dict.

    Reads ``cfg["detoxify"]``.  Returns ``None`` if disabled or the section
    is absent.

    Parameters
    ----------
    cfg : dict
        Full judge config (top-level keys: llm, processing, output,
        perspective, detoxify).
    """
    dcfg = cfg.get("detoxify", {})
    if not dcfg.get("enabled", False):
        logger.debug("Detoxify scorer disabled in config.")
        return None

    return DetoxifyScorer(
        model_type=dcfg.get("model_type", "original"),
        device=dcfg.get("device", "cpu"),
        max_chars=dcfg.get("max_chars", 20000),
    )


# ---------------------------------------------------------------------------
# Aggregation helpers (used by judge.py)
# ---------------------------------------------------------------------------

def compute_detoxify_summary(
    per_dialogue: List[Dict],
) -> Optional[Dict[str, Dict]]:
    """
    Compute per-label statistics over all per-dialogue Detoxify scores.

    Parameters
    ----------
    per_dialogue : list[dict]
        ``per_dialogue_ratings`` list from a single evaluate() result.

    Returns
    -------
    dict of {label: {mean, max, min, num_scored}} or None if no scores present.
    """
    entries = [e["detoxify_scores"] for e in per_dialogue if "detoxify_scores" in e]
    if not entries:
        return None

    all_labels = list(entries[0].keys())
    summary: Dict[str, Dict] = {}
    for label in all_labels:
        vals = [e[label] for e in entries if e.get(label) is not None]
        if vals:
            summary[label] = {
                "mean":       round(sum(vals) / len(vals), 4),
                "max":        round(max(vals), 4),
                "min":        round(min(vals), 4),
                "num_scored": len(vals),
            }
        else:
            summary[label] = {
                "mean": None, "max": None, "min": None, "num_scored": 0,
            }
    return summary


def aggregate_detoxify_results(results: List[Dict]) -> Optional[Dict[str, Dict]]:
    """
    Aggregate Detoxify scores across multiple evaluate() result dicts.

    Parameters
    ----------
    results : list[dict]
        Each element is the dict returned by ``DialogueJudge.evaluate()``.

    Returns
    -------
    dict of {label: {mean, max, min, total_scored}} or None if no scores present.
    """
    accum: Dict[str, List[float]] = {}
    for res in results:
        for entry in res.get("per_dialogue_ratings", []):
            ds = entry.get("detoxify_scores", {})
            for label, score in ds.items():
                if score is not None:
                    accum.setdefault(label, []).append(score)

    if not accum:
        return None

    agg: Dict[str, Dict] = {}
    for label, vals in accum.items():
        agg[label] = {
            "mean":         round(sum(vals) / len(vals), 4),
            "max":          round(max(vals), 4),
            "min":          round(min(vals), 4),
            "total_scored": len(vals),
        }
    return agg
