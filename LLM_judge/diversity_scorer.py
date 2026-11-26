"""
Diversity scorer for dialogue-level lexical variety evaluation.

Implements two complementary metrics:

**Distinct-2**
    Ratio of unique bigrams to total bigrams over the full dialogue text.
    Range [0, 1]; higher values indicate more lexically diverse output.
    Reference: Li et al. (2016) "A Diversity-Promoting Objective Function
    for Neural Conversation Models".

**Self-BLEU**
    Intra-dialogue Self-BLEU: for each turn in the dialogue, compute BLEU-2
    against all other turns in the same dialogue, then average.
    Range [0, 1]; lower values indicate more lexically diverse / less
    repetitive turn structure.  Returns ``None`` for dialogues with fewer
    than 2 non-trivial turns.
    Reference: Zhu et al. (2018) "Texygen: A Benchmarking Platform for
    Text Generation Models".

Usage (standalone):
    from diversity_scorer import DiversityScorer
    scorer = DiversityScorer()
    scores = scorer.score_text("Hey, you there?  Stop ignoring me.")
    # {"distinct_2": 0.8571, "self_bleu": None}

Usage (via factory):
    from diversity_scorer import build_diversity_scorer
    scorer = build_diversity_scorer(full_cfg)   # None if disabled
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text helpers (mirrors perplexity_scorer.py to avoid circular import)
# ---------------------------------------------------------------------------

def _get_dialogue_text(event: Dict) -> str:
    """Extract concatenated plain-text dialogue string from a pipeline event dict."""
    text = event.get("caa_rewritten_dialogue") or event.get("enforced_dialogue")
    if text:
        return str(text)
    turns = event.get("turns", [])
    if turns:
        return "\n".join(
            f"{t.get('speaker', '?')}: {t.get('text', '')}" for t in turns
        )
    return ""


def _get_dialogue_turns(event: Dict) -> List[str]:
    """
    Extract individual turn text strings from various pipeline event formats.

    For baseline-format events (``turns`` list), the ``text`` field of each
    turn is returned.  For pipeline-format events (``caa_rewritten_dialogue``
    or ``enforced_dialogue``), lines are split and speaker prefixes are
    stripped so only the utterance content is used for Self-BLEU.
    """
    # Baseline / converted format: explicit turns list
    turns = event.get("turns", [])
    if turns:
        return [t.get("text", "").strip() for t in turns if t.get("text", "").strip()]

    # Pipeline format: dialogue stored as a single string, one turn per line
    text = event.get("caa_rewritten_dialogue") or event.get("enforced_dialogue")
    if not text:
        return []

    result: List[str] = []
    for line in str(text).split("\n"):
        line = line.strip()
        if not line:
            continue
        # Strip "Speaker: " prefix if present
        if ": " in line:
            result.append(line.split(": ", 1)[1].strip())
        else:
            result.append(line)
    return result


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"\w+")


def _tokenize(text: str) -> List[str]:
    """Lowercase word tokenisation using a simple regex (no external deps)."""
    return _WORD_RE.findall(text.lower())


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class DiversityScorer:
    """
    Pure-computation diversity scorer (no model weights required).

    Computes Distinct-2 and intra-dialogue Self-BLEU (BLEU-2) for each
    dialogue.  Unlike ``PerplexityScorer`` no GPU or HuggingFace model is
    loaded — the only optional dependency is ``nltk`` (for BLEU computation).

    Parameters
    ----------
    max_chars : int
        Maximum characters of text processed per call; longer text is
        truncated.  Mirrors the ``perplexity_scorer`` convention.
    min_turn_tokens : int
        Minimum number of tokens a turn must contain to be included in the
        Self-BLEU computation.  Turns shorter than this are skipped to
        avoid degenerate BLEU values from single-word utterances.
    """

    def __init__(
        self,
        max_chars: int = 20000,
        min_turn_tokens: int = 3,
    ) -> None:
        self.max_chars = max_chars
        self.min_turn_tokens = min_turn_tokens
        self._nltk_ok: Optional[bool] = None   # checked lazily

        logger.info(
            "DiversityScorer configured: max_chars=%d  min_turn_tokens=%d",
            max_chars, min_turn_tokens,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_nltk(self) -> None:
        """Check that nltk is importable (called lazily on first Self-BLEU use)."""
        if self._nltk_ok is True:
            return
        if self._nltk_ok is False:
            raise ImportError(
                "nltk is required for Self-BLEU scoring. "
                "Run: pip install nltk"
            )
        try:
            import nltk  # noqa: F401
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction  # noqa: F401
            self._nltk_ok = True
        except ImportError:
            self._nltk_ok = False
            raise ImportError(
                "nltk is required for Self-BLEU scoring. "
                "Run: pip install nltk"
            )

    def _compute_distinct_2(self, tokens: List[str]) -> Optional[float]:
        """Distinct-2: unique bigrams / total bigrams."""
        if len(tokens) < 2:
            return None
        bigrams = list(zip(tokens[:-1], tokens[1:]))
        if not bigrams:
            return None
        return round(len(set(bigrams)) / len(bigrams), 4)

    def _compute_self_bleu(self, turns: List[str]) -> Optional[float]:
        """
        Intra-dialogue Self-BLEU (BLEU-2 with add-1 smoothing).

        For each turn i, compute BLEU-2(hypothesis=turn_i,
        references=all_other_turns), then return the mean.
        Returns ``None`` if fewer than 2 turns pass the token-length
        threshold.
        """
        self._ensure_nltk()
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

        smoother = SmoothingFunction().method1

        tokenized = [_tokenize(t) for t in turns]
        tokenized = [t for t in tokenized if len(t) >= self.min_turn_tokens]

        if len(tokenized) < 2:
            return None

        bleu_scores: List[float] = []
        for i, hyp in enumerate(tokenized):
            refs = [tokenized[j] for j in range(len(tokenized)) if j != i]
            score = sentence_bleu(
                refs,
                hyp,
                weights=(0.5, 0.5),
                smoothing_function=smoother,
            )
            bleu_scores.append(score)

        return round(sum(bleu_scores) / len(bleu_scores), 4)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_text(self, text: str) -> Dict[str, Optional[float]]:
        """
        Compute Distinct-2 for a plain-text string.

        Self-BLEU is returned as ``None`` because plain text has no
        turn structure; use ``score_dialogue`` to get Self-BLEU.

        Parameters
        ----------
        text : str
            Raw dialogue text to score.

        Returns
        -------
        dict with keys ``"distinct_2"`` and ``"self_bleu"``.
        """
        empty: Dict[str, Optional[float]] = {"distinct_2": None, "self_bleu": None}

        if not text.strip():
            return empty

        if len(text) > self.max_chars:
            text = text[: self.max_chars]

        try:
            tokens = _tokenize(text)
            distinct_2 = self._compute_distinct_2(tokens)
            return {"distinct_2": distinct_2, "self_bleu": None}
        except Exception as exc:
            logger.error("Distinct-2 scoring error: %s", exc)
            return empty

    def score_dialogue(self, event: Dict) -> Dict[str, Optional[float]]:
        """
        Extract dialogue text and turns from a pipeline event dict, then
        compute Distinct-2 and intra-dialogue Self-BLEU.

        Handles pipeline format (``caa_rewritten_dialogue`` /
        ``enforced_dialogue``) and baseline format (``turns`` list).

        Parameters
        ----------
        event : dict
            A single dialogue entry from a *_dialogues.json file.

        Returns
        -------
        dict with keys ``"distinct_2"`` and ``"self_bleu"``.
        """
        empty: Dict[str, Optional[float]] = {"distinct_2": None, "self_bleu": None}

        text = _get_dialogue_text(event)
        if not text:
            logger.debug(
                "Empty dialogue for event %s — skipping diversity scoring.",
                event.get("event_id") or event.get("dialogue_id", "?"),
            )
            return empty

        if len(text) > self.max_chars:
            text = text[: self.max_chars]

        # Distinct-2 over full concatenated text
        distinct_2: Optional[float] = None
        try:
            tokens = _tokenize(text)
            distinct_2 = self._compute_distinct_2(tokens)
        except Exception as exc:
            logger.error("Distinct-2 scoring error for event %s: %s",
                         event.get("event_id", "?"), exc)

        # Self-BLEU over individual turns
        self_bleu: Optional[float] = None
        try:
            turns = _get_dialogue_turns(event)
            self_bleu = self._compute_self_bleu(turns)
        except ImportError:
            logger.warning("nltk not available — Self-BLEU skipped.")
        except Exception as exc:
            logger.error("Self-BLEU scoring error for event %s: %s",
                         event.get("event_id", "?"), exc)

        return {"distinct_2": distinct_2, "self_bleu": self_bleu}

    def cleanup(self) -> None:
        """No-op — DiversityScorer holds no model weights."""
        logger.info("DiversityScorer cleanup (no-op).")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_diversity_scorer(cfg: dict) -> Optional[DiversityScorer]:
    """
    Instantiate a ``DiversityScorer`` from a top-level config dict.

    Reads ``cfg["diversity"]``.  Returns ``None`` if disabled or the section
    is absent.

    Parameters
    ----------
    cfg : dict
        Full judge config (top-level keys: llm, processing, output,
        perspective, detoxify, perplexity, diversity).
    """
    dcfg = cfg.get("diversity", {})
    if not dcfg.get("enabled", False):
        logger.debug("Diversity scorer disabled in config.")
        return None

    return DiversityScorer(
        max_chars=int(dcfg.get("max_chars", 20000)),
        min_turn_tokens=int(dcfg.get("min_turn_tokens", 3)),
    )


# ---------------------------------------------------------------------------
# Aggregation helpers (used by judge.py)
# ---------------------------------------------------------------------------

def compute_diversity_summary(
    per_dialogue: List[Dict],
) -> Optional[Dict[str, Dict]]:
    """
    Compute Distinct-2 and Self-BLEU statistics over all per-dialogue scores
    for one scenario.

    Parameters
    ----------
    per_dialogue : list[dict]
        ``per_dialogue_ratings`` list from a single ``evaluate()`` result.

    Returns
    -------
    dict with keys ``"distinct_2"`` and/or ``"self_bleu"``, each mapping to
    ``{mean, max, min, num_scored}``, or ``None`` if no diversity scores
    are present.
    """
    entries = [e["diversity_scores"] for e in per_dialogue if "diversity_scores" in e]
    if not entries:
        return None

    d2_vals  = [e["distinct_2"] for e in entries if e.get("distinct_2")  is not None]
    sb_vals  = [e["self_bleu"]  for e in entries if e.get("self_bleu")   is not None]

    if not d2_vals and not sb_vals:
        return None

    result: Dict[str, Dict] = {}
    if d2_vals:
        result["distinct_2"] = {
            "mean":       round(sum(d2_vals) / len(d2_vals), 4),
            "max":        round(max(d2_vals), 4),
            "min":        round(min(d2_vals), 4),
            "num_scored": len(d2_vals),
        }
    if sb_vals:
        result["self_bleu"] = {
            "mean":       round(sum(sb_vals) / len(sb_vals), 4),
            "max":        round(max(sb_vals), 4),
            "min":        round(min(sb_vals), 4),
            "num_scored": len(sb_vals),
        }
    return result


def aggregate_diversity_results(results: List[Dict]) -> Optional[Dict[str, Dict]]:
    """
    Aggregate diversity scores across multiple ``evaluate()`` result dicts
    (i.e., across all scenarios in a batch).

    Parameters
    ----------
    results : list[dict]
        Each element is the dict returned by ``DialogueJudge.evaluate()``.

    Returns
    -------
    dict with keys ``"distinct_2"`` and/or ``"self_bleu"``, each mapping to
    ``{mean, max, min, total_scored}``, or ``None`` if no diversity scores
    are present.
    """
    d2_vals: List[float] = []
    sb_vals: List[float] = []

    for res in results:
        for entry in res.get("per_dialogue_ratings", []):
            ds = entry.get("diversity_scores", {})
            d2 = ds.get("distinct_2")
            sb = ds.get("self_bleu")
            if d2 is not None:
                d2_vals.append(d2)
            if sb is not None:
                sb_vals.append(sb)

    if not d2_vals and not sb_vals:
        return None

    result: Dict[str, Dict] = {}
    if d2_vals:
        result["distinct_2"] = {
            "mean":         round(sum(d2_vals) / len(d2_vals), 4),
            "max":          round(max(d2_vals), 4),
            "min":          round(min(d2_vals), 4),
            "total_scored": len(d2_vals),
        }
    if sb_vals:
        result["self_bleu"] = {
            "mean":         round(sum(sb_vals) / len(sb_vals), 4),
            "max":          round(max(sb_vals), 4),
            "min":          round(min(sb_vals), 4),
            "total_scored": len(sb_vals),
        }
    return result
