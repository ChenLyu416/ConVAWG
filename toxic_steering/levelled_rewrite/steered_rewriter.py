#!/usr/bin/env python3
"""
Steered Rewriter — Uses CAA forward-hook steering to rewrite messages.

Loads a decoder-only chat model once, attaches a forward hook at the calibrated
layer, and generates rewrites at a specified alpha (steering strength).
The rewriter takes an original message + context and produces a rewritten
version while preserving the message intent and approximate length.

Three assets are required at runtime and are NOT part of the public release
(see toxic_steering/README.md for the request route and file formats):

  * the L2-normalised steering vector  (``<vector_dir>/layer_{N}.pt``)
  * the calibration file               (``{"layer": N, "levels": {"1": a1, ...}}``)
  * the rewriting prompts              (``prompts.yaml``; schema in
                                        ``prompts.example.yaml``)
"""

import torch
import yaml
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Dict, List

from transformers import AutoModelForCausalLM, AutoTokenizer


class SteeredRewriter:
    """Generates escalation-matched rewrites using a CAA-steered chat model."""

    REQUIRED_PROMPT_KEYS = ("system_prompt", "rewrite_prompt", "toxicity_features")
    DEFAULT_PROMPTS_PATH = Path(__file__).parent / "prompts.yaml"


    def __init__(
        self,
        model_name: str,
        vector_dir: str | Path,
        calibration_path: str | Path,
        prompts_path: Optional[str | Path] = None,
        device_map: str = "auto",
        dtype: torch.dtype | str = "auto",
        max_new_tokens: int = 512,
        temperature: float = 0.95,
        seed: int = 42,
        level_map: Optional[Dict[int, int]] = None,
        alpha_map: Optional[Dict[int, float]] = None,
    ):
        import json

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.vector_dir = Path(vector_dir)

        # ── Load rewriting prompts (withheld asset) ────────────────────
        self._load_prompts(prompts_path)

        # ── Load calibration ───────────────────────────────────────────
        with open(calibration_path) as f:
            cal = json.load(f)

        self.layer: int = cal["layer"]
        self.level_to_alpha: Dict[int, float] = {
            int(k): float(v) for k, v in cal["levels"].items()
        }
        # Apply per-level alpha overrides on top of calibration values
        if alpha_map is not None:
            overrides = {int(k): float(v) for k, v in alpha_map.items()}
            self.level_to_alpha.update(overrides)
            print(
                f"[SteeredRewriter] alpha_map override applied: {overrides}",
                flush=True,
            )
        # Steering strengths must be non-negative and non-decreasing in level
        ordered = [self.level_to_alpha[k] for k in sorted(self.level_to_alpha)]
        if any(a < 0 for a in ordered):
            raise ValueError(f"Steering alpha must be >= 0, got {self.level_to_alpha}")
        if any(a > b for a, b in zip(ordered, ordered[1:])):
            raise ValueError(
                f"Steering alphas must be non-decreasing across levels, got {self.level_to_alpha}"
            )
        # Mapping: escalation_level → steering level
        # Default: Esc 2 → Level 1, Esc 3 → Level 2, Esc 4 → Level 3
        self.esc_to_level: Dict[int, int] = (
            level_map if level_map is not None else {2: 1, 3: 2, 4: 3}
        )

        selected_dtype = self._resolve_torch_dtype(dtype)
        self.dtype = selected_dtype

        # ── Load model ─────────────────────────────────────────────────
        print(
            f"[SteeredRewriter] Loading model: {model_name} "
            f"(dtype={selected_dtype}, device_map={device_map})",
            flush=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=selected_dtype,
            device_map=device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        print(f"[SteeredRewriter] Model loaded, setting to eval mode", flush=True)
        self.model.eval()
        print(f"[SteeredRewriter] Getting input device", flush=True)
        self._input_device = next(self.model.parameters()).device
        print(f"[SteeredRewriter] Input device: {self._input_device}, loading tokenizer", flush=True)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True,
        )
        print(f"[SteeredRewriter] Tokenizer loaded", flush=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Left-padding is required for batched decoder-only generation
        self.tokenizer.padding_side = "left"

        # ── Load steering vector ───────────────────────────────────────
        print(f"[SteeredRewriter] Loading steering vector", flush=True)
        vec_path = self.vector_dir / f"layer_{self.layer}.pt"
        if not vec_path.exists():
            raise FileNotFoundError(f"Steering vector not found: {vec_path}")
        self.vector = torch.load(vec_path, weights_only=True)
        print(
            f"[SteeredRewriter] Ready — layer={self.layer}, "
            f"vector shape={tuple(self.vector.shape)}, "
            f"levels={self.level_to_alpha}",
            flush=True
        )

        # ── Reproducibility ────────────────────────────────────────────
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _load_prompts(self, prompts_path: Optional[str | Path]) -> None:
        """Load the system prompt, rewrite template and per-level guidelines."""
        path = Path(prompts_path) if prompts_path else self.DEFAULT_PROMPTS_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"Rewriting prompts not found: {path}. The Stage 4 prompts are not "
                "part of the public release; see toxic_steering/README.md and "
                "levelled_rewrite/prompts.example.yaml for the expected schema."
            )
        with open(path) as f:
            prompts = yaml.safe_load(f) or {}
        missing = [k for k in self.REQUIRED_PROMPT_KEYS if not prompts.get(k)]
        if missing:
            raise ValueError(f"{path} is missing required keys: {missing}")

        self.system_prompt: str = str(prompts["system_prompt"]).strip()
        self.rewrite_prompt: str = str(prompts["rewrite_prompt"])
        self.toxicity_features: Dict[int, List[str]] = {
            int(k): [str(x) for x in v]
            for k, v in prompts["toxicity_features"].items()
        }
        self._fallback_features: List[str] = self.toxicity_features[
            max(self.toxicity_features)
        ]

    @staticmethod
    def _resolve_torch_dtype(dtype: torch.dtype | str) -> torch.dtype:
        """Resolve requested dtype with GPU-aware fallback.

        - ``auto``: bf16 on Ampere+ GPUs (cc >= 8), else fp16 on older CUDA GPUs
        - CPU-only: fp32
        - explicit values: bf16/fp16/fp32 aliases
        """
        if isinstance(dtype, torch.dtype):
            return dtype

        key = str(dtype).strip().lower()
        aliases = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "half": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        if key in aliases:
            return aliases[key]

        if key != "auto":
            raise ValueError(f"Unsupported dtype '{dtype}'. Use auto|bf16|fp16|fp32")

        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability(0)
            return torch.bfloat16 if major >= 8 else torch.float16
        return torch.float32

    # ── Forward-hook machinery ─────────────────────────────────────────

    @contextmanager
    def _steering_hook(self, alpha: float):
        """Register a forward hook that adds alpha * v to hidden states."""
        vector = self.vector

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                h, rest = output[0], output[1:]
            else:
                h, rest = output, None
            steer = vector.to(device=h.device, dtype=h.dtype)
            h = h + alpha * steer
            return (h,) + rest if rest is not None else h

        layer_module = self.model.model.layers[self.layer]
        handle = layer_module.register_forward_hook(hook_fn)
        try:
            yield
        finally:
            handle.remove()

    # ── Chat formatting ────────────────────────────────────────────────

    def _format_prompt(self, user_content: str) -> str:
        """Wrap user_content in ChatML with the configured system prompt."""
        return (
            "<|im_start|>system\n"
            f"{self.system_prompt}\n"
            "<|im_end|>\n"
            f"<|im_start|>user\n{user_content}\n<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    # ── Core rewrite method ────────────────────────────────────────────

    def rewrite(
        self,
        message: str,
        escalation_level: int,
        message_type: str = "escalation",
        context_window: Optional[str] = None,
        max_retries: int = 10,
    ) -> Optional[str]:
        """
        Rewrite a single message using the CAA-steered model.

        Args:
            message: Original perpetrator message text.
            escalation_level: Dialogue escalation level (2, 3, or 4).
            message_type: Label type (e.g. "threat", "demand", "accusation").
            context_window: Up to 10 surrounding utterances from the dialogue.
            max_retries: Number of generation attempts before giving up.

        Returns:
            Rewritten toxic message, or None if all attempts fail validation.
        """
        level = self.esc_to_level.get(escalation_level)
        if level is None:
            raise ValueError(
                f"Unsupported escalation_level={escalation_level}. "
                f"Supported: {list(self.esc_to_level.keys())}"
            )

        alpha = self.level_to_alpha[level]
        features = self.toxicity_features.get(
            escalation_level, self._fallback_features
        )

        prompt_text = self.rewrite_prompt.format(
            msg_type=message_type,
            esc_level=escalation_level,
            orig_len=len(message),
            context_window=context_window or "[no surrounding context available]",
            message=message,
            features="\n".join(f"- {f}" for f in features),
            len_lo=max(1, len(message) - 10),
            len_hi=len(message) + 10,
        )
        formatted = self._format_prompt(prompt_text)

        for attempt in range(max_retries):
            inputs = self.tokenizer(
                formatted,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            ).to(self._input_device)

            with torch.no_grad(), self._steering_hook(alpha):
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            prompt_len = inputs.input_ids.shape[1]
            new_ids = output_ids[:, prompt_len:]
            text = self.tokenizer.decode(new_ids[0], skip_special_tokens=True).strip()

            # ── Stop-sequence truncation (matches vLLM stop config) ──
            for stop_seq in ["\n\n", "Original:", "MESSAGE:"]:
                if stop_seq in text:
                    text = text[:text.index(stop_seq)]

            # ── Clean up ─────────────────────
            text = text.strip().strip('"').strip("'").strip()

            # ── Validate ───────────────────────────────────────────────
            ok, reason = self._validate(message, text)
            if ok:
                if attempt > 0:
                    print(f"      ✓ Valid after {attempt + 1} attempts")
                return text
            else:
                if attempt < max_retries - 1:
                    print(f"      ⚠ Attempt {attempt + 1} failed: {reason}, retrying...")
                else:
                    print(f"      ✗ All {max_retries} attempts failed. Last reason: {reason}")

        return None  # all attempts failed

    @staticmethod
    def _is_gibberish(text: str) -> tuple[bool, str]:
        """Heuristic check that *text* looks like natural language.

        Returns ``(True, reason)`` if the text appears to be random /
        garbled characters rather than a coherent sentence.

        Heuristics
        ----------
        1. **Single-char token ratio** – if > 40 % of whitespace-separated
           tokens are single characters (letters, digits, symbols) the text
           is likely spaced-out gibberish
           (e.g. ``"s e s s 1 c r e g u o"``).
        2. **Alpha ratio** – natural English is mostly letters + spaces.
           If fewer than 50 % of non-space characters are alphabetic the
           output is probably garbage.
        3. **Repeated-char ratio** – if > 50 % of the text is the same
           character repeated (e.g. ``"aaaaaaaaaa"``).
        4. **Avg word length** – real words average ≥ 2.5 chars.  Heavily
           fragmented / spaced-out text has a much lower average.
        """
        stripped = text.strip()
        if not stripped:
            return True, "Empty text"

        tokens = stripped.split()
        if not tokens:
            return True, "No tokens"

        # 1. Single-character token ratio
        single_char_count = sum(1 for t in tokens if len(t) == 1)
        single_ratio = single_char_count / len(tokens)
        if single_ratio > 0.40:
            return True, (
                f"Gibberish: {single_ratio:.0%} of tokens are single chars "
                f"({single_char_count}/{len(tokens)})"
            )

        # 2. Alpha ratio among non-space characters
        non_space = stripped.replace(" ", "")
        if non_space:
            alpha_count = sum(1 for c in non_space if c.isalpha())
            alpha_ratio = alpha_count / len(non_space)
            if alpha_ratio < 0.50:
                return True, (
                    f"Gibberish: only {alpha_ratio:.0%} alphabetic chars "
                    f"({alpha_count}/{len(non_space)})"
                )

        # 3. Repeated-character dominance
        from collections import Counter
        char_counts = Counter(non_space.lower())
        if char_counts:
            most_common_count = char_counts.most_common(1)[0][1]
            if most_common_count / len(non_space) > 0.50:
                return True, (
                    f"Gibberish: single char dominates "
                    f"({most_common_count}/{len(non_space)})"
                )

        # 4. Average word length too low (fragmented output)
        avg_len = sum(len(t) for t in tokens) / len(tokens)
        if avg_len < 2.0 and len(tokens) >= 4:
            return True, (
                f"Gibberish: avg token length {avg_len:.1f} "
                f"across {len(tokens)} tokens"
            )

        # 5. Alternating / mock caps pattern
        # e.g. "jusT dO tHiS pRoperlY" — tokens that mix upper and lower
        # case in an alternating or random-internal pattern (not whole-word
        # ALL-CAPS, not normal Title Case).  A token is "mock-capped" when
        # it is ≥ 3 chars long, is neither all-upper nor title-case (*only*
        # first char upper), and contains at least one upper-case character
        # after the first position.  If ≥ 40 % of long alpha tokens show
        # this pattern the output is flagged.
        alpha_tokens = [t for t in tokens if len(t) >= 3 and t.isalpha()]
        if len(alpha_tokens) >= 4:
            def _is_mock_capped(t: str) -> bool:
                if t.isupper():          # ALL CAPS — valid stylistic shout
                    return False
                if t[0].isupper() and t[1:].islower():  # Normal Title Case
                    return False
                # Has upper-case anywhere after the first character
                return any(c.isupper() for c in t[1:])

            mock_count = sum(1 for t in alpha_tokens if _is_mock_capped(t))
            mock_ratio = mock_count / len(alpha_tokens)
            if mock_ratio >= 0.40:
                return True, (
                    f"Gibberish: alternating/mock-caps in {mock_ratio:.0%} "
                    f"of alpha tokens ({mock_count}/{len(alpha_tokens)})"
                )

        # 6. CamelCase hashtag-word detector
        # e.g. "DamnImInnaFuckedUpStateNow" — a single run-together token
        # with ≥ 2 internal uppercase letters (beyond the first char) and
        # length ≥ 12.  One such token is enough to reject the output.
        # Normal long English words ("approximately", "congratulations") are
        # all-lowercase after the first char, so false-positive risk is low.
        for tok in tokens:
            if len(tok) >= 12 and tok.isalpha():
                internal_upper = sum(1 for c in tok[1:] if c.isupper())
                if internal_upper >= 2:
                    return True, (
                        f"Gibberish: CamelCase hashtag-word '{tok}' "
                        f"({internal_upper} internal uppercase letters)"
                    )

        return False, ""

    @staticmethod
    def _is_leetspeak(text: str) -> tuple[bool, str]:
        """Detect leet-speak / number-substitution style output.

        Flags text where digits are used as letter replacements in a way
        that is characteristic of generated leet-speak, e.g.::

            cr4p y0ur damn1ng b1tch 5 h3ll g0t ur cr4p

        Heuristics
        ----------
        1. Count *leet tokens* — whitespace-separated tokens that contain
           at least one digit used as a common letter substitute
           (``0``→o, ``1``→i/l, ``2``→to/too, ``3``→e, ``4``→a, ``5``→s,
           ``8``→b, ``@``→a).  A token counts if it contains a mix of alpha
           chars AND one of these substitution digits (pure numbers like
           ``42`` are excluded).
        2. If **≥ 1 leet token** found → flag as leet-speak.
        """
        import re as _re
        # Digit/symbol characters that stand for letters in leet-speak
        # '2' added to catch '2myself', '2u', '2late' etc. ("2" → to/too)
        LEET_PATTERN = _re.compile(r'(?<=[a-zA-Z])[0123458@]|[0123458@](?=[a-zA-Z])')
        tokens = text.split()
        leet_tokens = [
            tok for tok in tokens
            if _re.search(r'[a-zA-Z]', tok) and LEET_PATTERN.search(tok)
        ]
        if len(leet_tokens) >= 1:
            return True, (
                f"Leet-speak detected: {len(leet_tokens)} leet tokens "
                f"({', '.join(leet_tokens[:4])})"
            )
        return False, ""

    def _validate(
        self, original: str, generated: str, max_extra_chars: int = 15,
    ) -> tuple[bool, str]:
        """Validate the generated toxic message.

        Validation rules:
        reject LLM-style prefixes, multi-line outputs, length violations,
        and gibberish / non-natural-language output.
        """
        if not generated:
            return False, "Empty output"

        # Check for LLM-style response prefixes
        lo = generated.lower().strip()
        bad_starts = [
            "here's", "here is", "rewritten:", "rewritten message:",
            "the rewritten", "i've rewritten", "i have rewritten",
            "sure", "certainly", "of course", "output",
        ]
        for pfx in bad_starts:
            if lo.startswith(pfx):
                return False, f"LLM-style response detected (starts with '{pfx}')"

        # Check for meta-commentary / artifact keywords anywhere in text
        meta_keywords = [
            "warning", "input", "output", "user", "rewrite", "rewritten",
            "length", "characters", "chars", "response", "intensity",
            "censored",
        ]
        for keyword in meta_keywords:
            if keyword in lo:
                return False, f"Meta-commentary keyword detected: '{keyword}'"
        
        # Check for specific problematic patterns
        if "okay," in lo:
            return False, "Artifact: 'Okay,' detected (LLM-style response)"
        
        # Check for parentheses (often indicate meta-commentary or char counts)
        if "(" in generated or ")" in generated:
            return False, "Artifact: parentheses detected (likely meta-commentary)"
        
        # Check for square brackets (often indicate stage directions or meta-commentary)
        if "[" in generated or "]" in generated:
            return False, "Artifact: square brackets detected (likely meta-commentary)"
        
        # Check for asterisks (often indicate stage directions or emphasis markup)
        if "*" in generated:
            return False, "Artifact: asterisks detected (likely stage directions or markup)"
        
        # Check for numbered list patterns (e.g., "1.", "2.", etc.)
        import re as _re
        if _re.search(r'\d+\.', generated):
            return False, "Artifact: numbered list pattern detected (e.g., '1.', '2.')"
        
        # Check for colons (often indicate explanatory text or format artifacts)
        if ":" in generated:
            return False, "Artifact: colon detected (likely format artifact or explanation)"

        # ── Artifact / style checks ────────────────────────────────────
        # 1a. Starts with '@' (e.g. "@user ...") — model addressing artefact
        if generated.lstrip().startswith("@"):
            return False, "Artifact: output starts with '@'"

        # 1b. Starts with markdown bold '**' (e.g. "**Output**")
        if generated.lstrip().startswith("**"):
            return False, "Artifact: output starts with markdown bold '**'"

        # 2. Leet-speak / number-for-letter substitution
        import re as _re
        is_leet, leet_reason = self._is_leetspeak(generated)
        if is_leet:
            return False, leet_reason

        # 3. Check for multi-line responses (likely includes explanation)
        if '\n' in generated.strip():
            return False, "Multi-line output detected (should be single message)"

        # Check for gibberish / random characters
        is_gib, gib_reason = self._is_gibberish(generated)
        if is_gib:
            return False, gib_reason

        # Check length constraint
        orig_len = len(original)
        gen_len = len(generated)
        max_allowed = orig_len + max_extra_chars
        min_allowed = orig_len - 15

        if gen_len > max_allowed:
            return False, f"Too long: {gen_len} chars (max: {max_allowed}, original: {orig_len})"
        if gen_len < min_allowed:
            return False, f"Too short: {gen_len} chars (min: {min_allowed}, original: {orig_len})"

        return True, "Valid"

    # ── Batched GPU generation ─────────────────────────────────────────

    def rewrite_many(
        self,
        requests: List[tuple],
        max_retries: int = 10,
        batch_size: int = 8,
    ) -> List[Optional[str]]:
        """Batch-rewrite a list of request tuples.

        Groups requests by escalation level so all prompts in a sub-batch
        share the same steering alpha.  Failed validations are retried in
        subsequent rounds up to *max_retries* times, progressively shrinking
        to individual retries.

        Each request may be either ``(message, escalation_level, message_type)``
        or ``(message, escalation_level, message_type, context_window)``.

        Returns a list of ``Optional[str]`` in the same order as *requests*.
        """
        from collections import defaultdict

        results: List[Optional[str]] = [None] * len(requests)
        # Only queue requests whose escalation_level is supported
        pending: List[int] = [
            i for i, request in enumerate(requests)
            if self.esc_to_level.get(request[1]) is not None
        ]

        for attempt in range(max_retries):
            if not pending:
                break

            # Group by escalation level (→ same alpha)
            groups: Dict[int, List[int]] = defaultdict(list)
            for idx in pending:
                esc = requests[idx][1]
                groups[esc].append(idx)

            still_pending: List[int] = []

            for esc, group_indices in groups.items():
                level = self.esc_to_level[esc]
                alpha = self.level_to_alpha[level]
                features = self.toxicity_features.get(esc, self._fallback_features)
                feat_str = "\n".join(f"- {f}" for f in features)

                for batch_start in range(0, len(group_indices), batch_size):
                    sub = group_indices[batch_start : batch_start + batch_size]

                    prompts = []
                    for idx in sub:
                        request = requests[idx]
                        msg = request[0]
                        msg_type = request[2] if len(request) > 2 else "escalation"
                        context_window = (
                            request[3] if len(request) > 3 else None
                        )
                        prompt_text = self.rewrite_prompt.format(
                            msg_type=msg_type,
                            esc_level=esc,
                            orig_len=len(msg),
                            context_window=(
                                context_window
                                or "[no surrounding context available]"
                            ),
                            message=msg,
                            features=feat_str,
                            len_lo=max(1, len(msg) - 10),
                            len_hi=len(msg) + 10,
                        )
                        prompts.append(self._format_prompt(prompt_text))

                    inputs = self.tokenizer(
                        prompts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=1024,
                    ).to(self._input_device)

                    with torch.no_grad(), self._steering_hook(alpha):
                        output_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=self.max_new_tokens,
                            do_sample=True,
                            temperature=self.temperature,
                            pad_token_id=self.tokenizer.eos_token_id,
                        )

                    prompt_len = inputs.input_ids.shape[1]

                    for j, idx in enumerate(sub):
                        orig_msg = requests[idx][0]
                        new_ids = output_ids[j, prompt_len:]
                        text = self.tokenizer.decode(
                            new_ids, skip_special_tokens=True
                        ).strip()

                        for stop_seq in ["\n\n", "Original:", "MESSAGE:"]:
                            if stop_seq in text:
                                text = text[: text.index(stop_seq)]
                        text = text.strip().strip('"').strip("'").strip()

                        ok, reason = self._validate(orig_msg, text)
                        if ok:
                            results[idx] = text
                            if attempt > 0:
                                print(f"      ✓ [{idx}] valid after {attempt + 1} attempts")
                        else:
                            still_pending.append(idx)
                            if attempt < max_retries - 1:
                                print(f"      ⚠ [{idx}] attempt {attempt + 1} failed: {reason}")

            pending = still_pending

        for idx in pending:
            msg = requests[idx][0]
            print(f"      ✗ [{idx}] all {max_retries} attempts failed: {msg[:50]!r}")

        return results

    # ── Batch convenience ──────────────────────────────────────────────

    def rewrite_batch(
        self,
        messages: List[Dict],
        escalation_level: int,
        max_retries: int = 10,
    ) -> List[Dict]:
        """
        Rewrite a list of message dicts in-place style.

        Each dict must have at least ``content`` and ``label`` keys.
        Returns a new list of dicts with ``original_content``,
        ``content`` (rewritten), and ``toxicity_injected`` fields.
        """
        results = []
        for msg in messages:
            rewritten = self.rewrite(
                message=msg["content"],
                escalation_level=escalation_level,
                message_type=msg.get("label", "escalation"),
                max_retries=max_retries,
            )
            out = dict(msg)
            if rewritten is not None:
                out["original_content"] = msg["content"]
                out["content"] = rewritten
                out["toxicity_injected"] = True
            else:
                out["toxicity_injected"] = False
                out["injection_failed"] = True
            results.append(out)
        return results
