"""
Perplexity scorer for dialogue-level fluency evaluation.

Uses a HuggingFace causal language model (default: GPT-2) to compute
the perplexity of each dialogue. Lower perplexity indicates more natural,
fluent language; higher perplexity suggests unusual or incoherent text.

Perplexity is defined as exp(mean NLL loss) over the token sequence,
computed with a strided sliding window for long texts to avoid OOM errors
and edge effects from the context window boundary.

Usage (standalone):
    from perplexity_scorer import PerplexityScorer
    scorer = PerplexityScorer(model_name="gpt2", device="cpu")
    scores = scorer.score_text("Hey, you there? Stop ignoring me.")
    # {"perplexity": 42.7}

Usage (via factory):
    from perplexity_scorer import build_perplexity_scorer
    scorer = build_perplexity_scorer(full_cfg)   # None if disabled
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dialogue-text extractor (mirrors detoxify_scorer.py to avoid circular import)
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

class PerplexityScorer:
    """
    Local perplexity scorer using a HuggingFace causal language model.

    The model is loaded lazily on first call to ``score_text`` to avoid
    consuming GPU/CPU memory at import time.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier, e.g. ``"gpt2"``, ``"gpt2-large"``,
        ``"openai-community/gpt2"``.
    device : str
        PyTorch device string, e.g. ``"cpu"``, ``"cuda"``, ``"cuda:0"``.
        Detoxify convention: multi-GPU strings like "cuda:1,2,3,4" are
        automatically reduced to their first index.
    max_length : int
        Maximum token window size per stride chunk (default 512 matches
        GPT-2's native context length).
    stride : int
        Stride (in tokens) for the sliding window. Setting stride == max_length
        disables overlap (fastest); smaller strides reduce boundary effects.
    max_chars : int
        Maximum characters of text passed per call; longer text is truncated.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        device: str = "cpu",
        max_length: int = 512,
        stride: int = 256,
        max_chars: int = 20000,
    ) -> None:
        # Detoxify uses the same single-index normalisation — keep consistency
        if "," in device:
            first = device.split(",")[0].strip()
            logger.warning(
                "PerplexityScorer does not support multi-GPU device string %r. "
                "Using first device: %r",
                device, first,
            )
            device = first

        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.stride = stride
        self.max_chars = max_chars

        self._model = None       # loaded lazily
        self._tokenizer = None   # loaded lazily
        self._load_failed = False

        logger.info(
            "PerplexityScorer configured: model=%s  device=%s  "
            "max_length=%d  stride=%d",
            model_name, device, max_length, stride,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        if self._load_failed:
            raise RuntimeError(
                "PerplexityScorer model failed to load previously; skipping."
            )
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
        except ImportError:
            raise ImportError(
                "transformers and torch are required for PerplexityScorer. "
                "Run: pip install transformers torch"
            )

        logger.info(
            "Loading perplexity model '%s' on device '%s' …",
            self.model_name, self.device,
        )
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self._model.eval()
            self._model.to(self.device)
        except Exception as exc:
            self._load_failed = True
            raise RuntimeError(
                f"Failed to load perplexity model '{self.model_name}': {exc}"
            ) from exc
        logger.info("Perplexity model loaded.")

    def _compute_ppl(self, text: str) -> float:
        """
        Compute perplexity using a strided sliding window over the token sequence.

        References the HuggingFace perplexity tutorial:
        https://huggingface.co/docs/transformers/perplexity
        """
        import torch

        self._ensure_model()

        encodings = self._tokenizer(text, return_tensors="pt")
        input_ids = encodings.input_ids.to(self.device)
        seq_len = input_ids.size(1)

        if seq_len == 0:
            return float("inf")

        nlls: List[float] = []
        prev_end_loc = 0

        for begin_loc in range(0, seq_len, self.stride):
            end_loc = min(begin_loc + self.max_length, seq_len)
            trg_len = end_loc - prev_end_loc   # tokens that are newly predicted

            input_chunk = input_ids[:, begin_loc:end_loc]
            target_ids = input_chunk.clone()
            # Mask prefix tokens that were already accounted for in prior chunks
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = self._model(input_chunk, labels=target_ids)
                # outputs.loss is mean NLL over non-masked tokens
                nlls.append(outputs.loss.item())

            prev_end_loc = end_loc
            if end_loc >= seq_len:
                break

        return float(
            __import__("math").exp(sum(nlls) / len(nlls)) if nlls else float("inf")
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_text(self, text: str) -> Dict[str, Optional[float]]:
        """
        Compute perplexity for a plain-text string.

        Parameters
        ----------
        text : str
            Raw dialogue text to score.

        Returns
        -------
        dict with key ``"perplexity"`` → float, or None on error.
        """
        empty: Dict[str, Optional[float]] = {"perplexity": None}

        if not text.strip():
            return empty

        if len(text) > self.max_chars:
            text = text[: self.max_chars]

        try:
            ppl = self._compute_ppl(text)
            return {"perplexity": round(ppl, 4)}
        except Exception as exc:
            logger.error("Perplexity scoring error: %s", exc)
            return empty

    def score_dialogue(self, event: Dict) -> Dict[str, Optional[float]]:
        """
        Extract dialogue text from a pipeline event dict and compute perplexity.

        Handles pipeline format (``caa_rewritten_dialogue`` / ``enforced_dialogue``)
        and baseline format (``turns`` list).

        Parameters
        ----------
        event : dict
            A single dialogue entry from a *_dialogues.json file.

        Returns
        -------
        dict with key ``"perplexity"`` → float or None.
        """
        text = _get_dialogue_text(event)
        if not text:
            logger.debug(
                "Empty dialogue for event %s — skipping perplexity scoring.",
                event.get("event_id") or event.get("dialogue_id", "?"),
            )
            return {"perplexity": None}
        return self.score_text(text)

    def cleanup(self) -> None:
        """Release model weights from memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("PerplexityScorer model released.")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_perplexity_scorer(cfg: dict) -> Optional[PerplexityScorer]:
    """
    Instantiate a ``PerplexityScorer`` from a top-level config dict.

    Reads ``cfg["perplexity"]``. Returns ``None`` if disabled or the section
    is absent.

    Parameters
    ----------
    cfg : dict
        Full judge config (top-level keys: llm, processing, output,
        perspective, detoxify, perplexity).
    """
    pcfg = cfg.get("perplexity", {})
    if not pcfg.get("enabled", False):
        logger.debug("Perplexity scorer disabled in config.")
        return None

    return PerplexityScorer(
        model_name=pcfg.get("model_name", "gpt2"),
        device=pcfg.get("device", "cpu"),
        max_length=int(pcfg.get("max_length", 512)),
        stride=int(pcfg.get("stride", 256)),
        max_chars=int(pcfg.get("max_chars", 20000)),
    )


# ---------------------------------------------------------------------------
# Aggregation helpers (used by judge.py)
# ---------------------------------------------------------------------------

def compute_perplexity_summary(
    per_dialogue: List[Dict],
) -> Optional[Dict[str, Dict]]:
    """
    Compute perplexity statistics over all per-dialogue scores for one scenario.

    Parameters
    ----------
    per_dialogue : list[dict]
        ``per_dialogue_ratings`` list from a single ``evaluate()`` result.

    Returns
    -------
    dict with key ``"perplexity"`` → ``{mean, max, min, num_scored}``,
    or ``None`` if no perplexity scores are present.
    """
    entries = [e["perplexity_scores"] for e in per_dialogue if "perplexity_scores" in e]
    if not entries:
        return None

    vals = [e["perplexity"] for e in entries if e.get("perplexity") is not None]
    if not vals:
        return {"perplexity": {"mean": None, "max": None, "min": None, "num_scored": 0}}

    return {
        "perplexity": {
            "mean":       round(sum(vals) / len(vals), 4),
            "max":        round(max(vals), 4),
            "min":        round(min(vals), 4),
            "num_scored": len(vals),
        }
    }


def aggregate_perplexity_results(results: List[Dict]) -> Optional[Dict[str, Dict]]:
    """
    Aggregate perplexity scores across multiple ``evaluate()`` result dicts
    (i.e., across all scenarios in a batch).

    Parameters
    ----------
    results : list[dict]
        Each element is the dict returned by ``DialogueJudge.evaluate()``.

    Returns
    -------
    dict with key ``"perplexity"`` → ``{mean, max, min, total_scored}``,
    or ``None`` if no perplexity scores are present.
    """
    vals: List[float] = []
    for res in results:
        for entry in res.get("per_dialogue_ratings", []):
            ps = entry.get("perplexity_scores", {})
            score = ps.get("perplexity")
            if score is not None:
                vals.append(score)

    if not vals:
        return None

    return {
        "perplexity": {
            "mean":         round(sum(vals) / len(vals), 4),
            "max":          round(max(vals), 4),
            "min":          round(min(vals), 4),
            "total_scored": len(vals),
        }
    }
