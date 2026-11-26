#!/usr/bin/env python3
"""
Post-hoc quality-fix pass over stage-7 converted dialogue files.

Targets the two defect classes that dominate judge criticism of ConVAWG's
global metrics:

1. Character-name collisions — the scenario's established cast
   (victim / perpetrator / third_party_actors with roles like "GP",
   "police officer") gets reused inside message text for unrelated
   incidental characters (a child being put to bed, a bar co-worker),
   which judges read as continuity errors.
2. Residual generation artifacts — leaked LLM refusal preambles, CJK
   style-label leakage, garbled/gibberish lines.

Deciding whether a specific name mention conflicts with the canonical role
is a judgment call, so this is a two-phase harness — a reviewer (a human or
an LLM agent) makes the call between the phases:

1. scan  — deterministically build, per input file, a review packet
           (<stem>_scan.json in --workdir): the canonical cast with roles,
           every cast-name mention in message text with its line and
           dialogue context, cast names used as speakers, mechanically
           detected suspect lines, the auto-fixes that `apply` will make,
           and (optionally) judge explanations for the file.
2.         (the reviewer reads each scan packet and writes a matching
           <stem>_fix_plan.json with three optional lists: "renames"
           (old_name, new_name, dialogue_indices), "line_replacements"
           (with new_line) and "line_removals"; see the apply step below
           in this file for the exact fields)
3. apply — mechanically execute auto-fixes + fix plans: case-preserving
           whole-word renames scoped to chosen dialogues, exact-line
           replacements/removals, refusal-preamble removal, trailing-CJK
           stripping. Writes fixed files in place (with backups in the
           workdir) or to a fresh output dir, plus fix_report.json.

Usage examples
--------------
# 1. Scan a converted dir, enriching packets with judge explanations
python fix_converted_dialogues.py scan \\
    --input-dir data/ConVAWG/dialogues \\
    --workdir /path/to/scratch/fix_work \\
    --judge-results Generation_Results/8_judge_results/<judge_run>

# 2. (reviewer writes <stem>_fix_plan.json files into the workdir)

# 3. Apply into a new directory (drop-in replacement for re-judging)
python fix_converted_dialogues.py apply \\
    --workdir /path/to/scratch/fix_work \\
    --output-dir Generation_Results/converted_dialogues_fixed

# Or fix in place (originals backed up under <workdir>/backup/)
python fix_converted_dialogues.py apply \\
    --workdir /path/to/scratch/fix_work --in-place

# Auto-fixes only (no fix plans needed): refusal preambles + CJK leakage
python fix_converted_dialogues.py apply \\
    --workdir /path/to/scratch/fix_work --in-place --auto-only
"""

import argparse
import json
import logging
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from dialogue_text_filters import (
    clean_dialogue_text,
    split_speaker,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Dialogue text variants, in the priority order used by the judge.
TEXT_FIELDS = ("caa_rewritten_dialogue", "toxic_injected_dialogue", "dialogue", "enforced_dialogue")

JUDGE_NOTE_RE = re.compile(
    r"nam(?:e|ing)|collision|conflict|garbl|gibberish|refus|non.?sequitur|glitch|corrupt|inconsisten",
    re.IGNORECASE,
)


def _match_case(occurrence: str, new_name: str) -> str:
    """Render new_name in the same case pattern as the matched occurrence."""
    if occurrence.islower():
        return new_name.lower()
    if occurrence.isupper():
        return new_name.upper()
    if occurrence[:1].isupper():
        return new_name[:1].upper() + new_name[1:]
    return new_name


def _rename_in_text(text: str, old: str, new: str, include_speaker_lines: bool = True) -> Tuple[str, int]:
    """Whole-word, case-insensitive, case-preserving rename. Returns (text, n)."""
    pattern = re.compile(rf"\b{re.escape(old)}\b", re.IGNORECASE)
    count = 0

    def repl(m: "re.Match[str]") -> str:
        nonlocal count
        count += 1
        return _match_case(m.group(0), new)

    if include_speaker_lines:
        return pattern.sub(repl, text), count

    lines_out: List[str] = []
    for line in text.split("\n"):
        speaker, body = split_speaker(line)
        if speaker:
            prefix = line[: len(line) - len(body)]
            lines_out.append(prefix + pattern.sub(repl, body))
        else:
            lines_out.append(pattern.sub(repl, line))
    return "\n".join(lines_out), count


def _rename_in_obj(obj: Any, old: str, new: str) -> Tuple[Any, int]:
    """Rename inside every string of a nested dict/list (for contexts/participants)."""
    if isinstance(obj, str):
        return _rename_in_text(obj, old, new)
    if isinstance(obj, dict):
        total = 0
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            out[k], n = _rename_in_obj(v, old, new)
            total += n
        return out, total
    if isinstance(obj, list):
        total = 0
        out_l: List[Any] = []
        for v in obj:
            fixed, n = _rename_in_obj(v, old, new)
            out_l.append(fixed)
            total += n
        return out_l, total
    return obj, 0


class DialogueFixer:
    """Two-phase (scan/apply) post-hoc fixer for converted dialogue files."""

    def __init__(self, workdir: Path):
        self.workdir = workdir
        self.workdir.mkdir(parents=True, exist_ok=True)

    # ── scan phase ───────────────────────────────────────────────────

    def scan(self, input_dir: Path, judge_results_dir: Optional[Path] = None) -> Dict[str, Any]:
        files = [
            f for f in sorted(input_dir.glob("*_dialogues.json"))
            if self._is_dialogue_file(f)
        ]
        if not files:
            logger.error("No *_dialogues.json files found in %s", input_dir)
            sys.exit(1)

        summary: Dict[str, Any] = {
            "input_dir": str(input_dir),
            "files_scanned": 0,
            "files_needing_review": 0,
            "total_name_mentions": 0,
            "total_auto_fixes": 0,
            "total_suspect_lines": 0,
            "files": [],
        }

        for f in tqdm(files, desc="Scanning", unit="file"):
            packet = self._scan_file(f, judge_results_dir)
            out = self.workdir / f"{f.stem}_scan.json"
            with open(out, "w", encoding="utf-8") as fh:
                json.dump(packet, fh, indent=2, ensure_ascii=False)

            n_mentions = len(packet["review"]["name_mentions"]) + len(packet["review"]["speaker_usages"])
            n_suspects = len(packet["review"]["suspect_lines"])
            n_auto = len(packet["auto_fixes"])
            summary["files_scanned"] += 1
            summary["total_name_mentions"] += n_mentions
            summary["total_auto_fixes"] += n_auto
            summary["total_suspect_lines"] += n_suspects
            if not packet["clean"]:
                summary["files_needing_review"] += 1
            summary["files"].append({
                "file": f.name,
                "scan": out.name,
                "clean": packet["clean"],
                "name_mentions": n_mentions,
                "auto_fixes": n_auto,
                "suspect_lines": n_suspects,
                "judge_notes": len(packet["review"]["judge_notes"]),
            })

        summary_path = self.workdir / "scan_summary.json"
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False)

        logger.info(
            "Scanned %d files: %d need review, %d cast-name mentions, %d auto-fixes, %d suspect lines",
            summary["files_scanned"], summary["files_needing_review"],
            summary["total_name_mentions"], summary["total_auto_fixes"],
            summary["total_suspect_lines"],
        )
        logger.info("Scan packets in %s (summary: %s)", self.workdir, summary_path.name)
        return summary

    @staticmethod
    def _is_dialogue_file(path: Path) -> bool:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return isinstance(data, dict) and isinstance(data.get("dialogues"), list)
        except Exception:
            return False

    def _scan_file(self, path: Path, judge_results_dir: Optional[Path]) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        cast = self._build_cast(data)
        cast_names = {c["name"].lower(): c for c in cast if c.get("name")}

        auto_fixes: List[Dict[str, Any]] = []
        suspect_lines: List[Dict[str, Any]] = []
        name_mentions: List[Dict[str, Any]] = []
        speaker_usages: List[Dict[str, Any]] = []
        dialogue_meta: List[Dict[str, Any]] = []

        for idx, dlg in enumerate(data.get("dialogues", [])):
            participants = self._participant_names(dlg)
            context = dlg.get("conversation_context") or {}
            dialogue_meta.append({
                "dialogue_index": idx,
                "event_id": dlg.get("event_id"),
                "timestamp": dlg.get("timestamp"),
                "participants": participants,
                "context_action": context.get("action"),
                "context_description": (context.get("description") or "")[:300],
            })
            participants_lower = {p.lower() for p in participants}
            primary_field = next(
                (f for f in TEXT_FIELDS if isinstance(dlg.get(f), str) and dlg.get(f)), None
            )

            for field in TEXT_FIELDS:
                text = dlg.get(field)
                if not text or not isinstance(text, str):
                    continue
                _, records = clean_dialogue_text(text)
                for rec in records:
                    entry = {"dialogue_index": idx, "field": field, **rec}
                    if "action" in rec:
                        auto_fixes.append(entry)
                    elif field == primary_field:
                        suspect_lines.append(entry)

                if field != primary_field:
                    continue  # review items only from the field the judge consumes

                for li, line in enumerate(text.split("\n")):
                    speaker, body = split_speaker(line)
                    if speaker and speaker.lower() in cast_names:
                        member = cast_names[speaker.lower()]
                        if member["kind"] == "third_party":
                            speaker_usages.append({
                                "dialogue_index": idx,
                                "field": field,
                                "name": member["name"],
                                "canonical_role": member.get("role"),
                                "line_index": li,
                                "line": line,
                            })
                    for lname, member in cast_names.items():
                        if member["kind"] != "third_party" and lname in participants_lower:
                            continue  # victim/perp addressed by their partner: normal
                        if speaker and speaker.lower() == lname:
                            continue  # already recorded as speaker usage
                        if re.search(rf"\b{re.escape(member['name'])}\b", body, re.IGNORECASE):
                            name_mentions.append({
                                "dialogue_index": idx,
                                "field": field,
                                "name": member["name"],
                                "kind": member["kind"],
                                "canonical_role": member.get("role"),
                                "name_is_participant": lname in participants_lower,
                                "line_index": li,
                                "line": line,
                            })

        judge_notes = self._collect_judge_notes(path, judge_results_dir, cast_names)

        clean = not (auto_fixes or suspect_lines or name_mentions or speaker_usages or judge_notes)
        return {
            "source_file": str(path),
            "scenario_id": data.get("scenario_id"),
            "n_dialogues": len(data.get("dialogues", [])),
            "clean": clean,
            "cast": cast,
            "dialogues": dialogue_meta,
            "auto_fixes": auto_fixes,
            "review": {
                "name_mentions": name_mentions,
                "speaker_usages": speaker_usages,
                "suspect_lines": suspect_lines,
                "judge_notes": judge_notes,
            },
        }

    @staticmethod
    def _build_cast(data: Dict[str, Any]) -> List[Dict[str, Any]]:
        vawg = data.get("vawg_scenarios") or {}
        cast: List[Dict[str, Any]] = []
        victim = data.get("victim_name") or (vawg.get("victim_profile") or {}).get("name")
        perp = data.get("perpetrator_name") or (vawg.get("perpetrator_profile") or {}).get("name")
        if victim:
            cast.append({"name": victim, "kind": "victim", "role": "victim"})
        if perp:
            cast.append({"name": perp, "kind": "perpetrator", "role": "perpetrator"})
        seen = {c["name"].lower() for c in cast}
        for actor in vawg.get("third_party_actors") or []:
            name = (actor or {}).get("name")
            if not name or name.lower() in seen:
                continue
            seen.add(name.lower())
            role_bits = [actor.get("role"), actor.get("relationship_to_victim")]
            cast.append({
                "name": name,
                "kind": "third_party",
                "role": " / ".join(b for b in role_bits if b) or "third party",
            })
        for name in data.get("third_party_names") or []:
            if name and name.lower() not in seen:
                seen.add(name.lower())
                cast.append({"name": name, "kind": "third_party", "role": "third party (role unspecified)"})
        return [c for c in cast if len(c["name"]) >= 3]

    @staticmethod
    def _participant_names(dlg: Dict[str, Any]) -> List[str]:
        names: List[str] = []
        participants = dlg.get("participants") or {}
        for side in ("sender", "receiver"):
            name = ((participants.get(side) or {}).get("name") or "").strip()
            if name:
                names.append(name)
        return names

    @staticmethod
    def _collect_judge_notes(
        path: Path, judge_results_dir: Optional[Path], cast_names: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        if not judge_results_dir:
            return []
        jr_path = judge_results_dir / f"{path.stem}_judge_results.json"
        if not jr_path.exists():
            return []
        try:
            with open(jr_path, "r", encoding="utf-8") as fh:
                jr = json.load(fh)
        except Exception:
            return []

        notes: List[Dict[str, Any]] = []
        for metric, rating in (jr.get("global_ratings") or {}).items():
            expl = (rating or {}).get("explanation") or ""
            if expl:
                notes.append({
                    "scope": "global",
                    "metric": metric,
                    "rating": rating.get("rating"),
                    "explanation": expl,
                })
        name_re = None
        if cast_names:
            name_re = re.compile(
                r"\b(?:" + "|".join(re.escape(c["name"]) for c in cast_names.values()) + r")\b",
                re.IGNORECASE,
            )
        for pd in jr.get("per_dialogue_ratings") or []:
            for metric, rating in (pd.get("ratings") or {}).items():
                if not isinstance(rating, dict):
                    continue
                expl = rating.get("explanation") or ""
                if not expl:
                    continue
                if JUDGE_NOTE_RE.search(expl) or (name_re and name_re.search(expl)):
                    notes.append({
                        "scope": "per_dialogue",
                        "dialogue_index": pd.get("dialogue_index"),
                        "event_id": pd.get("event_id"),
                        "metric": metric,
                        "rating": rating.get("rating"),
                        "explanation": expl,
                    })
        return notes

    # ── apply phase ──────────────────────────────────────────────────

    def apply(
        self,
        output_dir: Optional[Path] = None,
        in_place: bool = False,
        auto_only: bool = False,
        no_auto: bool = False,
        regen_pdf: bool = False,
    ) -> Dict[str, Any]:
        scans = sorted(self.workdir.glob("*_scan.json"))
        if not scans:
            logger.error("No *_scan.json packets in %s — run the scan phase first", self.workdir)
            sys.exit(1)

        backup_dir = self.workdir / "backup"
        report: Dict[str, Any] = {"mode": "in_place" if in_place else "output_dir", "files": []}

        for scan_path in tqdm(scans, desc="Applying", unit="file"):
            with open(scan_path, "r", encoding="utf-8") as fh:
                packet = json.load(fh)
            src = Path(packet["source_file"])
            if not src.exists():
                logger.warning("Source file missing, skipping: %s", src)
                continue
            with open(src, "r", encoding="utf-8") as fh:
                data = json.load(fh)

            file_report: Dict[str, Any] = {
                "file": src.name,
                "auto_fixes_applied": [],
                "plan_ops_applied": [],
                "plan_ops_failed": [],
            }

            if not no_auto:
                self._apply_auto_fixes(data, file_report)

            plan_path = self.workdir / f"{src.stem}_fix_plan.json"
            if not auto_only and plan_path.exists():
                with open(plan_path, "r", encoding="utf-8") as fh:
                    plan = json.load(fh)
                self._apply_plan(data, plan, packet, file_report)

            changed = bool(file_report["auto_fixes_applied"] or file_report["plan_ops_applied"])
            file_report["changed"] = changed
            report["files"].append(file_report)

            if in_place:
                dest = src
                if changed:
                    backup_dir.mkdir(parents=True, exist_ok=True)
                    backup_path = backup_dir / src.name
                    if not backup_path.exists():  # keep the true original on reruns
                        shutil.copy2(src, backup_path)
            else:
                output_dir.mkdir(parents=True, exist_ok=True)
                dest = output_dir / src.name

            if changed or not in_place:
                with open(dest, "w", encoding="utf-8") as fh:
                    json.dump(data, fh, indent=2, ensure_ascii=False)

            if regen_pdf and (changed or not in_place):
                self._regen_pdf(data, dest)

        n_changed = sum(1 for f in report["files"] if f["changed"])
        n_failed = sum(len(f["plan_ops_failed"]) for f in report["files"])
        report["files_changed"] = n_changed
        report["plan_ops_failed_total"] = n_failed
        report_path = self.workdir / "fix_report.json"
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
        logger.info("Applied fixes: %d/%d files changed, %d plan ops failed (report: %s)",
                    n_changed, len(report["files"]), n_failed, report_path)
        if n_failed:
            logger.warning("Some plan ops failed to match — inspect 'plan_ops_failed' in %s", report_path)
        return report

    @staticmethod
    def _apply_auto_fixes(data: Dict[str, Any], file_report: Dict[str, Any]) -> None:
        for idx, dlg in enumerate(data.get("dialogues", [])):
            for field in TEXT_FIELDS:
                text = dlg.get(field)
                if not text or not isinstance(text, str):
                    continue
                cleaned, records = clean_dialogue_text(text)
                actions = [r for r in records if "action" in r]
                if actions:
                    dlg[field] = cleaned
                    for rec in actions:
                        file_report["auto_fixes_applied"].append(
                            {"dialogue_index": idx, "field": field, **rec}
                        )

    def _apply_plan(
        self,
        data: Dict[str, Any],
        plan: Dict[str, Any],
        packet: Dict[str, Any],
        file_report: Dict[str, Any],
    ) -> None:
        dialogues = data.get("dialogues", [])
        existing_names = {c["name"].lower() for c in packet.get("cast", [])}
        for dlg in dialogues:
            for field in TEXT_FIELDS:
                for line in (dlg.get(field) or "").split("\n"):
                    speaker, _ = split_speaker(line)
                    if speaker:
                        existing_names.add(speaker.lower())

        for op in plan.get("line_removals", []) or []:
            self._line_op(dialogues, op, file_report, replacement=None)

        for op in plan.get("line_replacements", []) or []:
            new_line = op.get("new_line")
            if new_line is None:
                file_report["plan_ops_failed"].append({**op, "error": "missing new_line"})
                continue
            self._line_op(dialogues, op, file_report, replacement=new_line)

        renamed_pairs: Dict[str, str] = {}  # new_name (lower) -> old_name (lower)
        for op in plan.get("renames", []) or []:
            old = (op.get("old_name") or "").strip()
            new = (op.get("new_name") or "").strip()
            if not old or not new:
                file_report["plan_ops_failed"].append({**op, "error": "missing old_name/new_name"})
                continue
            # A new name may be reused across ops only for the same old name
            # (same entity fixed in several dialogues); anything else collides.
            if new.lower() in existing_names or renamed_pairs.get(new.lower(), old.lower()) != old.lower():
                file_report["plan_ops_failed"].append(
                    {**op, "error": f"new_name '{new}' collides with an existing name in this file"}
                )
                continue
            indices = op.get("dialogue_indices", "all")
            if indices == "all":
                indices = list(range(len(dialogues)))
            include_speakers = op.get("include_speaker_lines", True)
            include_context = op.get("include_context", True)
            total = 0
            for i in indices:
                if not (0 <= i < len(dialogues)):
                    continue
                dlg = dialogues[i]
                for field in TEXT_FIELDS:
                    text = dlg.get(field)
                    if not text or not isinstance(text, str):
                        continue
                    dlg[field], n = _rename_in_text(text, old, new, include_speakers)
                    total += n
                if include_speakers and dlg.get("participants"):
                    dlg["participants"], n = _rename_in_obj(dlg["participants"], old, new)
                    total += n
                if include_context and dlg.get("conversation_context"):
                    dlg["conversation_context"], n = _rename_in_obj(dlg["conversation_context"], old, new)
                    total += n
            if total:
                renamed_pairs[new.lower()] = old.lower()
                file_report["plan_ops_applied"].append({**op, "occurrences_renamed": total})
            else:
                file_report["plan_ops_failed"].append({**op, "error": "no occurrences matched"})

    @staticmethod
    def _line_op(
        dialogues: List[Dict[str, Any]],
        op: Dict[str, Any],
        file_report: Dict[str, Any],
        replacement: Optional[str],
    ) -> None:
        idx = op.get("dialogue_index")
        target = op.get("old_line", op.get("line"))
        if idx is None or target is None or not (0 <= idx < len(dialogues)):
            file_report["plan_ops_failed"].append({**op, "error": "bad dialogue_index or missing line"})
            return
        dlg = dialogues[idx]
        matched_fields: List[str] = []
        for field in TEXT_FIELDS:
            text = dlg.get(field)
            if not text or not isinstance(text, str):
                continue
            lines = text.split("\n")
            for li, line in enumerate(lines):
                if line == target or line.strip() == target.strip():
                    if replacement is None:
                        del lines[li]
                    else:
                        lines[li] = replacement
                    dlg[field] = "\n".join(lines)
                    matched_fields.append(field)
                    break
        if matched_fields:
            file_report["plan_ops_applied"].append({**op, "fields": matched_fields})
        else:
            file_report["plan_ops_failed"].append({**op, "error": "line not found"})

    @staticmethod
    def _regen_pdf(data: Dict[str, Any], json_path: Path) -> None:
        try:
            from convert_dialogues_to_minimal import REPORTLAB_AVAILABLE, create_pdf
        except ImportError:
            logger.warning("convert_dialogues_to_minimal not importable — skipping PDF")
            return
        if not REPORTLAB_AVAILABLE:
            return
        observe_toxic = any(
            ("caa_rewritten_dialogue" in d or "toxic_injected_dialogue" in d) and "enforced_dialogue" in d
            for d in data.get("dialogues", [])
        )
        create_pdf(data, json_path.with_suffix(".pdf"), observe_toxic)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-hoc name-collision and artifact fix pass over converted dialogue files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage examples\n--------------\n", 1)[-1],
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_scan = sub.add_parser("scan", help="Build per-file review packets into the workdir")
    p_scan.add_argument("--input-dir", required=True, help="Converted dialogues dir (stage-7 output)")
    p_scan.add_argument("--workdir", required=True, help="Scratch dir for scan packets and fix plans")
    p_scan.add_argument("--judge-results", default=None,
                        help="Optional judge results dir; explanations are added to the packets")

    p_apply = sub.add_parser("apply", help="Execute auto-fixes and agent fix plans")
    p_apply.add_argument("--workdir", required=True, help="Workdir containing *_scan.json (+ *_fix_plan.json)")
    group = p_apply.add_mutually_exclusive_group(required=True)
    group.add_argument("--in-place", action="store_true",
                       help="Overwrite source files (originals backed up under <workdir>/backup/)")
    group.add_argument("--output-dir", default=None,
                       help="Write all files (fixed or not) into this dir as a drop-in replacement")
    p_apply.add_argument("--auto-only", action="store_true",
                         help="Apply only deterministic auto-fixes; ignore *_fix_plan.json")
    p_apply.add_argument("--no-auto", action="store_true",
                         help="Apply only agent fix plans; skip deterministic auto-fixes")
    p_apply.add_argument("--regen-pdf", action="store_true",
                         help="Regenerate the PDF next to each written JSON (needs reportlab)")

    args = parser.parse_args()
    fixer = DialogueFixer(Path(args.workdir))

    if args.command == "scan":
        fixer.scan(
            Path(args.input_dir),
            Path(args.judge_results) if args.judge_results else None,
        )
    else:
        fixer.apply(
            output_dir=Path(args.output_dir) if args.output_dir else None,
            in_place=args.in_place,
            auto_only=args.auto_only,
            no_auto=args.no_auto,
            regen_pdf=args.regen_pdf,
        )


if __name__ == "__main__":
    main()
