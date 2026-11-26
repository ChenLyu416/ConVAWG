#!/usr/bin/env python3
"""
Deterministic text-quality detectors shared by stage 7 conversion
(convert_dialogues_to_minimal.py) and the post-hoc fix pass
(fix_converted_dialogues.py).

Detects the pattern-detectable defect classes observed in judge results:
- leaked LLM refusal preambles (unspeakered "I can't write a chat
  transcript ..." lines at the top of a dialogue),
- CJK style-label leakage appended to message text (e.g. "u still
  gone?典型被动攻击式回应"),
- meta-annotations ("<32 chars", "[email]", "{placeholder}"),
- spaced-out letter sequences and other mechanical gibberish signatures.

Semantic gibberish (garbled-but-ASCII tokens such as "swarzin base rdy 2
drop") is NOT reliably regex-detectable; those lines are only *flagged* for
review when they co-occur with a mechanical signature, and are otherwise
left to the LLM/agent review pass.
"""

import re
from typing import Any, Dict, List, Tuple

# "Speaker: message" — every legitimate dialogue line has this prefix.
SPEAKER_PREFIX_RE = re.compile(r"^\s*[^:\n]{1,32}:\s")

# Refusal phrasing at the start of a line (curly and straight apostrophes).
REFUSAL_START_RE = re.compile(
    r"^\s*(?:"
    r"I\s*(?:can[’']?t|cannot|won[’']?t|am\s+not\s+able\s+to|"
    r"[’']?m\s+not\s+able\s+to|[’']?m\s+unable\s+to)\s+"
    r"(?:write|create|generate|produce|provide|continue|assist|help|complete|fulfil?l)\b"
    r"|As\s+an\s+AI\b"
    r"|I[’']?m\s+sorry,?\s+but\s+I\s+(?:can[’']?t|cannot|won[’']?t)\b"
    r")",
    re.IGNORECASE,
)

# Meta-objects that mark a refusal as assistant-voice even inside a
# speakered line ("Wei: I can't write this dialogue ...").
REFUSAL_META_OBJECT_RE = re.compile(
    r"\b(?:transcript|dialogue|chat\s+log|roleplay|role-play|scenario|"
    r"this\s+request|content\s+polic\w+|AI\s+(?:assistant|model)|language\s+model)\b",
    re.IGNORECASE,
)

# CJK ranges (Han + kana + CJK punctuation + fullwidth forms).
CJK_CHAR_RE = re.compile(r"[　-〿぀-ヿ一-鿿＀-￯]")

# Characters allowed in a *trailing* CJK run (CJK + whitespace + any
# non-word punctuation, so "典型…，暗示失望" strips cleanly).
_TRAILING_CJK_ALLOWED_RE = re.compile(
    r"^[　-〿぀-ヿ一-鿿＀-￯\s\W]*$"
)

META_ANNOTATION_RE = re.compile(
    r"<\s*\d+\s*chars?"
    r"|\[(?:email|phone|name|address|link|url|redacted)\]"
    r"|\{[a-z_]{3,}\}"
    r"|<<[^>\n]{1,40}>>",
    re.IGNORECASE,
)

# "s e s s 1 c r e"-style spaced-out letters.
SPACED_LETTERS_RE = re.compile(r"(?:\b[A-Za-z]\s+){4,}[A-Za-z]\b")

# "*who's-your-flippant-idol-viagra*"-style asterisk-wrapped chains.
ASTERISK_CHAIN_RE = re.compile(r"\*[^*\s]{12,}\*")

# "nn@siren"-style @-fragments that are not real e-mail addresses.
AT_FRAGMENT_RE = re.compile(r"\b\w+@[A-Za-z]+\b(?!\.[A-Za-z])")


def split_speaker(line: str) -> Tuple[str, str]:
    """Split "Speaker: message" into (speaker, message); speaker is "" if absent."""
    m = SPEAKER_PREFIX_RE.match(line)
    if not m:
        return "", line
    head = m.group(0)
    return head.rstrip().rstrip(":").strip(), line[len(head):]


def is_leaked_refusal_line(line: str) -> bool:
    """True for unspeakered assistant-refusal lines (safe to auto-remove).

    Speakered lines never match here: a victim texting "I can't write to
    you anymore" is legitimate dialogue.
    """
    if not line.strip():
        return False
    if SPEAKER_PREFIX_RE.match(line):
        return False
    return bool(REFUSAL_START_RE.match(line.strip()))


def is_suspect_speakered_refusal(line: str) -> bool:
    """True for speakered lines that read as assistant refusals (flag only)."""
    speaker, body = split_speaker(line)
    if not speaker:
        return False
    return bool(REFUSAL_START_RE.match(body.strip()) and REFUSAL_META_OBJECT_RE.search(body))


def strip_trailing_cjk(line: str) -> Tuple[str, bool]:
    """Strip a CJK run that extends to the end of the line.

    "u still gone?典型被动攻击式回应，暗示失望" -> ("u still gone?", True).
    Embedded CJK ("D拖ag-you-back") is left untouched (returns changed=False)
    because mechanical removal would corrupt the surrounding word.
    """
    m = CJK_CHAR_RE.search(line)
    if not m:
        return line, False
    if not _TRAILING_CJK_ALLOWED_RE.match(line[m.start():]):
        return line, False
    cleaned = line[: m.start()].rstrip().rstrip(",;、").rstrip()
    if not cleaned or not CJK_CHAR_RE.search(line[m.start():]):
        return line, False
    return cleaned, True


def detect_line_issues(line: str) -> List[str]:
    """Return flag tags for issues on this line (after any auto-cleaning)."""
    tags: List[str] = []
    if CJK_CHAR_RE.search(line):
        tags.append("embedded_cjk")
    if META_ANNOTATION_RE.search(line):
        tags.append("meta_annotation")
    if SPACED_LETTERS_RE.search(line):
        tags.append("spaced_letters")
    if ASTERISK_CHAIN_RE.search(line):
        tags.append("asterisk_chain")
    if AT_FRAGMENT_RE.search(line):
        tags.append("at_fragment")
    if is_suspect_speakered_refusal(line):
        tags.append("speakered_refusal")
    return tags


def clean_dialogue_text(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Auto-clean one dialogue transcript and flag residual suspect lines.

    Auto-fixes (applied): remove unspeakered refusal lines; strip trailing
    CJK runs. Everything else is only recorded as a flag.

    Returns (cleaned_text, records) where each record is
    {"line_index", "line", "action" | "flags", ["cleaned_line"]}.
    line_index refers to the ORIGINAL text's line numbering.
    """
    records: List[Dict[str, Any]] = []
    kept: List[str] = []
    for idx, line in enumerate(text.split("\n")):
        if is_leaked_refusal_line(line):
            records.append({"line_index": idx, "line": line, "action": "removed_refusal_line"})
            continue
        cleaned, changed = strip_trailing_cjk(line)
        if changed:
            records.append({
                "line_index": idx,
                "line": line,
                "action": "stripped_trailing_cjk",
                "cleaned_line": cleaned,
            })
            line = cleaned
        flags = detect_line_issues(line)
        if flags:
            records.append({"line_index": idx, "line": line, "flags": flags})
        kept.append(line)
    # Drop leading blank lines left behind by a removed preamble.
    while kept and not kept[0].strip() and any(r.get("action") == "removed_refusal_line" for r in records):
        kept.pop(0)
    return "\n".join(kept), records
