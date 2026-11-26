#!/usr/bin/env python3
"""
Message Labeler — LLM-based perpetrator message classification.

Labels each perpetrator message with a communicative act so that the
levelled_rewrite pipeline can process raw roleplay output directly and select
only escalation acts for rewriting.
"""

import os
import re
from typing import Dict, List, Any, Optional

from openai import OpenAI


# ── Valid classification labels ────────────────────────────────────────
VALID_LABELS = {
    "request", "refusal", "escalation", "repair", "accusation",
    "demand", "threat", "ultimatum", "surveillance",
    "gaslighting", "manipulation", "other",
}

# ── Escalation-act subset (triggers toxic rewriting) ───────────────────
ESCALATION_ACTS = {
    "accusation", "demand", "threat", "ultimatum",
    "surveillance", "gaslighting", "manipulation",
}


class MessageLabeler:
    """Labels perpetrator messages via OpenAI-compatible API or local vLLM."""

    def __init__(
        self,
        backend: str = "openai",
        api_key: str = None,
        model: str = "gpt-5.2",
        temperature: float = 0.3,
        max_tokens: int = 500,
        api_base: str = None,
        vllm_dtype: str = "auto",
        vllm_tensor_parallel_size: int = 1,
        vllm_gpu_memory_utilization: float = 0.90,
        vllm_max_model_len: int = 4096,
    ):
        """Initialise message labeling backend.

        Supported backends:
        - ``openai``: OpenAI-compatible chat completions API
        - ``vllm``: local vLLM **server** (OpenAI-compatible endpoint, e.g.
          ``python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3.5-35B-A3B``).
          ``api_base`` selects the server URL (default: $VLLM_BASE_URL or
          http://localhost:8000/v1). The legacy ``vllm_*`` GPU arguments are
          accepted but ignored — configure them on the server instead.
        """
        self.backend = backend.lower().strip()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        if self.backend == "openai":
            if api_key is None or api_key == "your-api-key-here":
                api_key = os.environ.get("OPENAI_API_KEY")

            client_kwargs: Dict[str, Any] = {}
            if api_base:
                client_kwargs["base_url"] = api_base
            if api_key:
                client_kwargs["api_key"] = api_key

            self.client = OpenAI(**client_kwargs)
            self._thread_safe = True

        elif self.backend == "vllm":
            base_url = (
                api_base
                or os.environ.get("VLLM_BASE_URL")
                or "http://localhost:8000/v1"
            )
            print(
                f"[MessageLabeler] Using vLLM server at {base_url} "
                f"(model: {self.model})",
                flush=True,
            )
            self.client = OpenAI(
                api_key=api_key or os.environ.get("VLLM_API_KEY", "EMPTY"),
                base_url=base_url,
            )
            self._thread_safe = True

        else:
            raise ValueError(
                f"Unsupported label backend '{backend}'. Supported: openai, vllm"
            )

    @property
    def thread_safe(self) -> bool:
        return self._thread_safe

    # ── Health check ───────────────────────────────────────────────────

    def check_health(self) -> bool:
        """Return True if the configured labeling backend is reachable."""
        try:
            if self.backend in ("openai", "vllm"):
                self.client.models.list()
                return True

            return False
        except Exception as e:
            print(f"Label backend check failed ({self.backend}): {e}")
            return False

    # ── Prompt construction ────────────────────────────────────────────

    @staticmethod
    def create_labeling_prompt(
        message: str,
        conversation_context: str,
        escalation_level: int,
    ) -> str:
        """Create the labeling prompt."""
        return (
            "You are analyzing a domestic abuse conversation message to "
            "classify the perpetrator's communication pattern.\n\n"
            f"ESCALATION LEVEL: {escalation_level}\n\n"
            f"CONVERSATION CONTEXT:\n{conversation_context}\n\n"
            f'PERPETRATOR MESSAGE TO CLASSIFY:\n"{message}"\n\n'
            "CLASSIFICATION OPTIONS:\n"
            "- request: Asking for something (neutral to mild)\n"
            "- refusal: Saying no or resisting\n"
            "- escalation: Increasing intensity/aggression\n"
            "- repair: Attempting to fix or apologize\n"
            "- accusation: Blaming or accusing the victim (ESCALATION ACT)\n"
            "- demand: Forceful requirement or order (ESCALATION ACT)\n"
            "- threat: Explicit or implied threat of harm (ESCALATION ACT)\n"
            "- ultimatum: Conditional threat (do X or else Y) (ESCALATION ACT)\n"
            "- surveillance: Monitoring, tracking, or demanding location/info "
            "(ESCALATION ACT)\n"
            "- gaslighting: Reality distortion, making victim doubt themselves "
            "(ESCALATION ACT)\n"
            "- manipulation: Emotional manipulation or guilt-tripping "
            "(ESCALATION ACT)\n"
            "- other: None of the above\n\n"
            "Respond with ONLY the classification label (one word, lowercase). "
            "Choose the MOST DOMINANT pattern in this message."
        )

    @staticmethod
    def _normalize_label(raw_text: str) -> str:
        """Normalize model output to one valid label."""
        if not raw_text:
            return "other"
        text = raw_text.strip().lower()
        text = text.splitlines()[0].strip()
        text = re.sub(r"[^a-z_]", "", text)
        return text if text in VALID_LABELS else "other"

    # ── Single-message labeling ────────────────────────────────────────

    def label_message(
        self,
        message: str,
        conversation_context: str,
        escalation_level: int,
    ) -> Optional[str]:
        """Label a single message via configured backend."""
        prompt = self.create_labeling_prompt(
            message, conversation_context, escalation_level,
        )

        try:
            if self.backend == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_completion_tokens=self.max_tokens,
                )
                raw = response.choices[0].message.content or ""
                return self._normalize_label(raw)

            if self.backend == "vllm":
                # vLLM server: /no_think suppresses Qwen chain-of-thought
                messages = [{"role": "user", "content": prompt}]
                if "qwen" in self.model.lower():
                    messages = [
                        {"role": "system", "content": "/no_think"},
                        {"role": "user", "content": prompt},
                    ]
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                raw = response.choices[0].message.content or ""
                return self._normalize_label(raw)

            return None

        except Exception as e:
            print(f"Error labeling message ({self.backend}): {e}")
            return None


# ── Dialogue parsing helpers ──────────────────────────────────────────

def parse_dialogue(dialogue_text: str) -> List[Dict[str, str]]:
    """Parse an ``enforced_dialogue`` string into a list of message dicts.

    Each line is expected to be ``Speaker: message text``.
    Returns a list of ``{"speaker": ..., "content": ...}`` dicts.
    """
    messages: List[Dict[str, str]] = []
    for line in dialogue_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        if ":" in line:
            speaker, content = line.split(":", 1)
            messages.append({
                "speaker": speaker.strip(),
                "content": content.strip(),
            })
    return messages


def label_perpetrator_messages(
    labeler: MessageLabeler,
    messages: List[Dict[str, str]],
    perpetrator_name: str,
    victim_name: Optional[str],
    escalation_level: int,
) -> List[Dict[str, Any]]:
    """Label all perpetrator messages using the GPT labeler.

    First assigns each message speaker a role by matching speaker name to
    ``perpetrator_name`` / ``victim_name`` (case-insensitive, punctuation-
    insensitive). Then labels ONLY perpetrator utterances.

    Non-perpetrator messages receive ``label=None, is_perpetrator=False``.
    """
    def _normalize_name(name: Optional[str]) -> str:
        if not name:
            return ""
        normalized = re.sub(r"[^a-z0-9]", "", name.lower())
        return normalized

    perpetrator_norm = _normalize_name(perpetrator_name)
    victim_norm = _normalize_name(victim_name)

    def _match_role(speaker: str) -> str:
        speaker_norm = _normalize_name(speaker)
        if speaker_norm and speaker_norm == perpetrator_norm:
            return "perpetrator"
        if speaker_norm and victim_norm and speaker_norm == victim_norm:
            return "victim"
        return "unknown"

    labeled: List[Dict[str, Any]] = []

    for i, msg in enumerate(messages):
        labeled_msg: Dict[str, Any] = dict(msg)
        speaker_role = _match_role(msg.get("speaker", ""))
        labeled_msg["speaker_role"] = speaker_role

        if speaker_role == "perpetrator":
            # Build context from preceding 3 messages
            ctx_start = max(0, i - 3)
            ctx_msgs = messages[ctx_start:i]
            context = "\n".join(
                f"{m['speaker']}: {m['content']}" for m in ctx_msgs
            )

            label = labeler.label_message(
                message=msg["content"],
                conversation_context=context,
                escalation_level=escalation_level,
            )
            labeled_msg["label"] = label if label else "other"
            labeled_msg["is_perpetrator"] = True
        else:
            labeled_msg["label"] = None
            labeled_msg["is_perpetrator"] = False

        labeled.append(labeled_msg)

    return labeled
