#!/usr/bin/env python3
"""
Script to convert dialogue JSON files to a minimal format.
Extracts only specific fields: scenario_id, victim_name, perpetrator_name,
third_party_names, vawg_scenarios, num_events, and filtered dialogue data.

During conversion, dialogue text is passed through deterministic cleaning
(dialogue_text_filters.py): leaked LLM refusal preambles are removed,
trailing CJK style-label leakage is stripped, and residual suspect lines
(embedded CJK, meta-annotations, spaced-letter gibberish) are flagged into
conversion_clean_report.json in the output folder. Disable with --no-clean.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

from dialogue_text_filters import clean_dialogue_text

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
    from reportlab.lib.enums import TA_LEFT, TA_CENTER
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


def create_pdf(data: Dict[str, Any], output_path: Path, observe_toxic: bool = False) -> bool:
    """
    Create a reader-friendly PDF from the minimal JSON data.
    
    Args:
        data: Minimal JSON data dictionary
        output_path: Path to output PDF file
        observe_toxic: If True, show both enforced and toxic dialogues
        
    Returns:
        True if successful, False otherwise
    """
    if not REPORTLAB_AVAILABLE:
        return False
        
    try:
        doc = SimpleDocTemplate(str(output_path), pagesize=letter,
                               topMargin=0.5*inch, bottomMargin=0.5*inch,
                               leftMargin=0.75*inch, rightMargin=0.75*inch)
        
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            textColor='#1a1a1a',
            spaceAfter=12,
            alignment=TA_CENTER
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=12,
            textColor='#2c3e50',
            spaceAfter=8,
            spaceBefore=12
        )
        
        subheading_style = ParagraphStyle(
            'CustomSubHeading',
            parent=styles['Heading3'],
            fontSize=10,
            textColor='#34495e',
            spaceAfter=6,
            spaceBefore=8
        )
        
        victim_style = ParagraphStyle(
            'VictimStyle',
            parent=styles['Normal'],
            fontSize=9,
            textColor='#16a085',
            leftIndent=20,
            spaceAfter=4
        )
        
        perpetrator_style = ParagraphStyle(
            'PerpetratorStyle',
            parent=styles['Normal'],
            fontSize=9,
            textColor='#c0392b',
            leftIndent=20,
            spaceAfter=4
        )
        
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=9,
            spaceAfter=4
        )
        
        dialogue_style = ParagraphStyle(
            'DialogueStyle',
            parent=styles['Normal'],
            fontSize=8,
            leftIndent=30,
            rightIndent=10,
            spaceAfter=2,
            fontName='Courier'
        )
        
        # Title
        story.append(Paragraph(f"<b>Dialogue Scenario {data.get('scenario_id', 'Unknown')}</b>", title_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Basic Information
        story.append(Paragraph("<b>Scenario Information</b>", heading_style))
        story.append(Paragraph(f"<b>Victim:</b> {data.get('victim_name', 'N/A')}", normal_style))
        story.append(Paragraph(f"<b>Perpetrator:</b> {data.get('perpetrator_name', 'N/A')}", normal_style))
        
        if data.get('third_party_names'):
            third_parties = ', '.join(data.get('third_party_names', []))
            story.append(Paragraph(f"<b>Third Parties:</b> {third_parties}", normal_style))
        
        story.append(Paragraph(f"<b>Number of Events:</b> {data.get('num_events', 'N/A')}", normal_style))
        story.append(Spacer(1, 0.2*inch))
        
        # VAWG Scenarios - Full details
        if data.get('vawg_scenarios'):
            story.append(Paragraph("<b>VAWG Scenario Context</b>", heading_style))
            vawg = data['vawg_scenarios']
            
            if vawg.get('persona_id'):
                story.append(Paragraph(f"<b>Persona ID:</b> {vawg['persona_id']}", normal_style))
            
            if vawg.get('persona_text'):
                story.append(Paragraph(f"<b>Persona:</b> {vawg['persona_text']}", normal_style))
                story.append(Spacer(1, 0.1*inch))
            
            # Victim Profile
            if vawg.get('victim_profile'):
                story.append(Paragraph("<b>Victim Profile</b>", subheading_style))
                victim = vawg['victim_profile']
                
                if victim.get('name'):
                    story.append(Paragraph(f"<b>Name:</b> {victim['name']}", normal_style))
                if victim.get('age'):
                    story.append(Paragraph(f"<b>Age:</b> {victim['age']}", normal_style))
                if victim.get('gender'):
                    story.append(Paragraph(f"<b>Gender:</b> {victim['gender']}", normal_style))
                if victim.get('occupation'):
                    story.append(Paragraph(f"<b>Occupation:</b> {victim['occupation']}", normal_style))
                if victim.get('ethnicity'):
                    story.append(Paragraph(f"<b>Ethnicity:</b> {victim['ethnicity']}", normal_style))
                
                if victim.get('demographics'):
                    story.append(Paragraph(f"<b>Demographics:</b> {str(victim['demographics'])}", normal_style))
                
                if victim.get('vulnerabilities'):
                    story.append(Paragraph("<b>Vulnerabilities:</b>", normal_style))
                    for vuln in victim['vulnerabilities']:
                        vuln_escaped = vuln.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                        story.append(Paragraph(f"• {vuln_escaped}", dialogue_style))
                
                if victim.get('support_seeking'):
                    story.append(Paragraph("<b>Support Seeking:</b>", normal_style))
                    for support in victim['support_seeking']:
                        support_escaped = support.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                        story.append(Paragraph(f"• {support_escaped}", dialogue_style))
                
                story.append(Spacer(1, 0.1*inch))
            
            # Perpetrator Profile
            if vawg.get('perpetrator_profile'):
                story.append(Paragraph("<b>Perpetrator Profile</b>", subheading_style))
                perp = vawg['perpetrator_profile']
                
                if perp.get('name'):
                    story.append(Paragraph(f"<b>Name:</b> {perp['name']}", normal_style))
                if perp.get('age'):
                    story.append(Paragraph(f"<b>Age:</b> {perp['age']}", normal_style))
                if perp.get('gender'):
                    story.append(Paragraph(f"<b>Gender:</b> {perp['gender']}", normal_style))
                if perp.get('occupation'):
                    story.append(Paragraph(f"<b>Occupation:</b> {perp['occupation']}", normal_style))
                if perp.get('ethnicity'):
                    story.append(Paragraph(f"<b>Ethnicity:</b> {perp['ethnicity']}", normal_style))
                if perp.get('relationship_to_victim'):
                    story.append(Paragraph(f"<b>Relationship to Victim:</b> {perp['relationship_to_victim']}", normal_style))
                
                if perp.get('demographics'):
                    story.append(Paragraph(f"<b>Demographics:</b> {str(perp['demographics'])}", normal_style))
                
                if perp.get('motivations'):
                    story.append(Paragraph("<b>Motivations:</b>", normal_style))
                    for motiv in perp['motivations']:
                        motiv_escaped = motiv.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                        story.append(Paragraph(f"• {motiv_escaped}", dialogue_style))
                
                story.append(Spacer(1, 0.1*inch))
            
            # Event Graph
            if vawg.get('event_graph'):
                story.append(Paragraph("<b>Event Graph</b>", subheading_style))
                graph = vawg['event_graph']
                
                if graph.get('crime_type'):
                    story.append(Paragraph(f"<b>Crime Type:</b> {graph['crime_type']}", normal_style))
                if graph.get('crime_subtype'):
                    story.append(Paragraph(f"<b>Crime Subtype:</b> {graph['crime_subtype']}", normal_style))
                
                if graph.get('events'):
                    story.append(Paragraph(f"<b>Number of Graph Events:</b> {len(graph['events'])}", normal_style))
                
                story.append(Spacer(1, 0.1*inch))
        
        # Dialogues
        dialogues = data.get('dialogues', [])
        if dialogues:
            story.append(PageBreak())
            story.append(Paragraph("<b>Dialogue Events</b>", heading_style))
            story.append(Spacer(1, 0.1*inch))
            
            for idx, dialogue in enumerate(dialogues, 1):
                # Event header with prominent timestamp
                event_id = dialogue.get('event_id', 'Unknown')
                timestamp = dialogue.get('timestamp', 'Unknown')
                story.append(Paragraph(f"<b>Event {idx}: {event_id}</b> — <i>{timestamp}</i>", subheading_style))
                story.append(Spacer(1, 0.05*inch))
                
                # Context - Full details
                context = dialogue.get('conversation_context', {})
                if context:
                    story.append(Paragraph("<b>Conversation Context:</b>", normal_style))
                    if context.get('action'):
                        story.append(Paragraph(f"<b>Action:</b> {context['action']}", dialogue_style))
                    if context.get('description'):
                        desc_escaped = context['description'].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                        story.append(Paragraph(f"<b>Description:</b> {desc_escaped}", dialogue_style))
                    if context.get('purpose'):
                        purpose_escaped = context['purpose'].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                        story.append(Paragraph(f"<b>Purpose:</b> {purpose_escaped}", dialogue_style))
                    if context.get('crime_type'):
                        story.append(Paragraph(f"<b>Crime Type:</b> {context['crime_type']}", dialogue_style))
                    if context.get('tone'):
                        story.append(Paragraph(f"<b>Tone:</b> {context['tone']}", dialogue_style))
                    if context.get('emotional_state'):
                        story.append(Paragraph(f"<b>Emotional State:</b> {context['emotional_state']}", dialogue_style))
                    if context.get('escalation_level') is not None:
                        story.append(Paragraph(f"<b>Escalation Level:</b> {context['escalation_level']}", dialogue_style))
                
                # Participants - Full details
                participants = dialogue.get('participants', {})
                if participants:
                    story.append(Paragraph("<b>Participants:</b>", normal_style))
                    if participants.get('sender'):
                        sender = participants['sender']
                        story.append(Paragraph(f"<b>Sender:</b> {sender.get('name', 'N/A')} (Role: {sender.get('role', 'N/A')})", dialogue_style))
                    if participants.get('receiver'):
                        receiver = participants['receiver']
                        story.append(Paragraph(f"<b>Receiver:</b> {receiver.get('name', 'N/A')} (Role: {receiver.get('role', 'N/A')})", dialogue_style))
                
                story.append(Spacer(1, 0.05*inch))
                
                # Dialogue content
                if observe_toxic:
                    # Show both enforced and toxic/rewritten versions
                    if dialogue.get('enforced_dialogue'):
                        story.append(Paragraph(f"<b>Original Enforced Dialogue</b> <i>({timestamp})</i>", normal_style))
                        for line in dialogue['enforced_dialogue'].split('\n'):
                            if line.strip():
                                line_escaped = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                                story.append(Paragraph(line_escaped, dialogue_style))
                        story.append(Spacer(1, 0.1*inch))
                    
                    # Prefer CAA-rewritten, fall back to toxic_injected
                    rewritten = dialogue.get('caa_rewritten_dialogue') or dialogue.get('toxic_injected_dialogue')
                    rewritten_label = 'CAA-Rewritten Dialogue' if dialogue.get('caa_rewritten_dialogue') else 'Toxic Injected Dialogue'
                    if rewritten:
                        story.append(Paragraph(f"<b>{rewritten_label}</b> <i>({timestamp})</i>", normal_style))
                        for line in rewritten.split('\n'):
                            if line.strip():
                                line_escaped = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                                story.append(Paragraph(line_escaped, dialogue_style))
                else:
                    # Show single dialogue version
                    dialogue_text = dialogue.get('dialogue')
                    if dialogue_text:
                        story.append(Paragraph(f"<b>Dialogue</b> <i>({timestamp})</i>", normal_style))
                        for line in dialogue_text.split('\n'):
                            if line.strip():
                                line_escaped = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                                story.append(Paragraph(line_escaped, dialogue_style))
                
                story.append(Spacer(1, 0.15*inch))
        
        # Build PDF
        doc.build(story)
        return True
        
    except Exception as e:
        print(f"Error creating PDF: {e}", file=sys.stderr)
        return False


def extract_dialogue_fields(dialogue: Dict[str, Any], observe_toxic: bool = False) -> Dict[str, Any]:
    """
    Extract only required fields from a single dialogue entry.
    
    Args:
        dialogue: Original dialogue dictionary
        observe_toxic: If True, include both enforced_dialogue and toxic_injected_dialogue
        
    Returns:
        Filtered dialogue dictionary with only required fields
    """
    filtered = {
        "event_id": dialogue.get("event_id"),
        "timestamp": dialogue.get("timestamp"),
        "participants": dialogue.get("participants"),
        "conversation_context": dialogue.get("conversation_context"),
    }
    
    if observe_toxic:
        # Include both enforced and rewritten/toxic dialogues for comparison
        filtered["enforced_dialogue"] = dialogue.get("enforced_dialogue")
        # Use original key name from the source
        if dialogue.get("caa_rewritten_dialogue"):
            filtered["caa_rewritten_dialogue"] = dialogue["caa_rewritten_dialogue"]
        elif dialogue.get("toxic_injected_dialogue"):
            filtered["toxic_injected_dialogue"] = dialogue["toxic_injected_dialogue"]
    else:
        # Priority: caa_rewritten_dialogue > toxic_injected_dialogue > enforced_dialogue
        if dialogue.get("caa_rewritten_dialogue"):
            filtered["dialogue"] = dialogue["caa_rewritten_dialogue"]
        elif dialogue.get("toxic_injected_dialogue"):
            filtered["dialogue"] = dialogue["toxic_injected_dialogue"]
        elif dialogue.get("enforced_dialogue"):
            filtered["dialogue"] = dialogue["enforced_dialogue"]
        else:
            filtered["dialogue"] = None
    
    return filtered


def clean_minimal_dialogues(minimal_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Run deterministic text cleaning over every dialogue text field in place.

    Auto-fixes (leaked refusal preambles removed, trailing CJK stripped) are
    applied directly to the text; residual suspect lines are only flagged.

    Returns a list of records, each {dialogue_index, field, line_index, line,
    action|flags, [cleaned_line]} (see dialogue_text_filters.clean_dialogue_text).
    """
    records: List[Dict[str, Any]] = []
    text_fields = ("dialogue", "caa_rewritten_dialogue", "toxic_injected_dialogue", "enforced_dialogue")
    for idx, dlg in enumerate(minimal_data.get("dialogues", [])):
        for field in text_fields:
            text = dlg.get(field)
            if not text or not isinstance(text, str):
                continue
            cleaned, recs = clean_dialogue_text(text)
            if any("action" in r for r in recs):
                dlg[field] = cleaned
            for rec in recs:
                records.append({"dialogue_index": idx, "field": field, **rec})
    return records


def convert_json_file(input_path: Path, output_path: Path, observe_toxic: bool = False,
                      generate_pdf: bool = True, clean: bool = True) -> "tuple[bool, List[Dict[str, Any]]]":
    """
    Convert a single JSON file to minimal format.

    Args:
        input_path: Path to input JSON file
        output_path: Path to output JSON file
        observe_toxic: If True, include both enforced_dialogue and toxic_injected_dialogue
        generate_pdf: If True, also generate a PDF version (default: True)
        clean: If True, apply deterministic text cleaning and collect flags (default: True)

    Returns:
        Tuple of (success, clean_records) where clean_records lists the
        auto-fixes applied and suspect lines flagged for this file.
    """
    clean_records: List[Dict[str, Any]] = []
    try:
        # Read input file
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extract required top-level fields
        minimal_data = {
            "scenario_id": data.get("scenario_id"),
            "victim_name": data.get("victim_name"),
            "perpetrator_name": data.get("perpetrator_name"),
            "third_party_names": data.get("third_party_names"),
            "vawg_scenarios": data.get("vawg_scenarios"),
            "num_events": data.get("num_events"),
        }

        # Process dialogues array
        if "dialogues" in data and isinstance(data["dialogues"], list):
            minimal_data["dialogues"] = [
                extract_dialogue_fields(dialogue, observe_toxic)
                for dialogue in data["dialogues"]
            ]
        else:
            minimal_data["dialogues"] = []

        if clean:
            clean_records = clean_minimal_dialogues(minimal_data)

        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write output file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(minimal_data, f, indent=2, ensure_ascii=False)
        
        # Generate PDF if requested
        if generate_pdf:
            pdf_path = output_path.with_suffix('.pdf')
            if not create_pdf(minimal_data, pdf_path, observe_toxic):
                if not REPORTLAB_AVAILABLE:
                    print(f"\nWarning: reportlab not installed. PDF generation skipped.", file=sys.stderr)
                    print(f"Install with: pip install reportlab", file=sys.stderr)

        return True, clean_records

    except Exception as e:
        print(f"Error processing {input_path}: {e}", file=sys.stderr)
        return False, clean_records


def convert_folder(input_folder: str, output_folder: str, recursive: bool = False, observe_toxic: bool = False,
                   generate_pdf: bool = True, clean: bool = True) -> None:
    """
    Convert all JSON files in a folder to minimal format.

    Args:
        input_folder: Path to folder containing input JSON files
        output_folder: Path to folder for output JSON files
        recursive: Whether to process subdirectories recursively
        observe_toxic: If True, include both enforced_dialogue and toxic_injected_dialogue
        generate_pdf: If True, also generate PDF versions (default: True)
        clean: If True, apply deterministic text cleaning and write
            conversion_clean_report.json into the output folder (default: True)
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    if not input_path.exists():
        print(f"Error: Input folder '{input_folder}' does not exist.", file=sys.stderr)
        sys.exit(1)
    
    if not input_path.is_dir():
        print(f"Error: '{input_folder}' is not a directory.", file=sys.stderr)
        sys.exit(1)
    
    # Find all JSON files
    pattern = "**/*.json" if recursive else "*.json"
    json_files = list(input_path.glob(pattern))
    
    if not json_files:
        print(f"No JSON files found in '{input_folder}'.", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(json_files)} JSON file(s) to process...")

    success_count = 0
    fail_count = 0
    clean_report: Dict[str, Any] = {"auto_fixes": 0, "suspect_lines": 0, "files": []}

    for json_file in json_files:
        # Calculate relative path to maintain directory structure
        relative_path = json_file.relative_to(input_path)
        output_file = output_path / relative_path

        print(f"Processing: {relative_path}...", end=" ")

        ok, records = convert_json_file(json_file, output_file, observe_toxic, generate_pdf, clean)
        if ok:
            print("✓")
            success_count += 1
        else:
            print("✗")
            fail_count += 1
        if records:
            n_fixes = sum(1 for r in records if "action" in r)
            n_flags = len(records) - n_fixes
            clean_report["auto_fixes"] += n_fixes
            clean_report["suspect_lines"] += n_flags
            clean_report["files"].append({"file": str(relative_path), "records": records})

    print(f"\nConversion complete!")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {fail_count}")
    if clean:
        report_path = output_path / "conversion_clean_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(clean_report, f, indent=2, ensure_ascii=False)
        print(f"  Text cleaning: {clean_report['auto_fixes']} auto-fix(es), "
              f"{clean_report['suspect_lines']} suspect line(s) flagged")
        print(f"  Cleaning report: {report_path}")
    print(f"\nOutput files saved to: {output_folder}")


# ── GPT Gibberish Reviewer ─────────────────────────────────────────────


GIBBERISH_REVIEW_PROMPT = """You are a quality-control reviewer for synthetic dialogue datasets.

I will give you a dialogue (a sequence of "Speaker: message" lines).
Your job is to identify ANY lines that contain gibberish, garbled text,
random characters, or nonsensical sequences that do NOT read like natural
language — even if the rest of the dialogue is fine.

Examples of gibberish:
- "nn@siren 0h 710 s e s s 1 c r e g u o"
- "*who's-your-flippant-idol-viagra*"
- "asdkjf lkjsdf kjsdf"
- "You have a fucking mental breakdown? <32 chars"
- Spaced-out letter sequences like "s e s s 1 c r e"
- Text with meta-annotations like "<32 chars" or "[email]"

If ALL lines are natural language (even if rude, toxic, or offensive), respond with exactly:
PASS

If you find gibberish, respond with:
FAIL
Then list each gibberish line on its own line, prefixed with "- ".

Do NOT flag lines just for being rude, vulgar, or offensive — only flag
lines that are NOT coherent natural language."""


def review_dialogues_for_gibberish(
    output_folder: str,
    gpt_model: str = "gpt-5.2",
    gpt_api_base: str | None = None,
    recursive: bool = False,
) -> dict:
    """Scan all converted dialogue files for gibberish using GPT.

    Returns a dict with keys:
        files_checked (int), files_with_issues (int), total_issues (int),
        issues (list of dicts with file, scenario_id, event_id, details)
    """
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai package not installed. Cannot run gibberish review.", file=sys.stderr)
        return {"files_checked": 0, "files_with_issues": 0, "total_issues": 0, "issues": []}

    client_kwargs = {}
    if gpt_api_base:
        client_kwargs["base_url"] = gpt_api_base
    client = OpenAI(**client_kwargs)

    out_path = Path(output_folder)
    pattern = "**/*.json" if recursive else "*.json"
    json_files = sorted(out_path.glob(pattern))

    issues: list[dict] = []
    files_checked = 0
    files_with_issues = 0

    print(f"\n{'=' * 60}")
    print(f"GIBBERISH REVIEW — scanning {len(json_files)} file(s) with {gpt_model}")
    print(f"{'=' * 60}")

    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        scenario_id = data.get("scenario_id", jf.stem)
        dialogues = data.get("dialogues", [])
        file_has_issue = False
        files_checked += 1

        for dlg in dialogues:
            event_id = dlg.get("event_id", "?")

            # Pick whichever dialogue field is present
            text = (
                dlg.get("caa_rewritten_dialogue")
                or dlg.get("toxic_injected_dialogue")
                or dlg.get("dialogue")
                or dlg.get("enforced_dialogue")
            )
            if not text or not text.strip():
                continue

            # Ask GPT to review
            try:
                request_kwargs = {
                    "model": gpt_model,
                    "temperature": 0.0,
                    "messages": [
                        {"role": "system", "content": GIBBERISH_REVIEW_PROMPT},
                        {"role": "user", "content": text},
                    ],
                }

                # GPT-5-style APIs use max_completion_tokens.
                # Fallback to max_tokens for older-compatible endpoints.
                try:
                    resp = client.chat.completions.create(
                        **request_kwargs,
                        max_completion_tokens=500,
                    )
                except Exception as e:
                    if "max_completion_tokens" in str(e):
                        resp = client.chat.completions.create(
                            **request_kwargs,
                            max_tokens=500,
                        )
                    else:
                        raise
                answer = resp.choices[0].message.content.strip()
            except Exception as e:
                print(f"  ⚠ GPT error on {scenario_id}/{event_id}: {e}")
                continue

            if answer.startswith("FAIL"):
                file_has_issue = True
                detail_lines = answer.split("\n", 1)
                details = detail_lines[1].strip() if len(detail_lines) > 1 else "(no details)"
                issues.append({
                    "file": str(jf.relative_to(out_path)),
                    "scenario_id": scenario_id,
                    "event_id": event_id,
                    "details": details,
                })
                print(f"  ✗ {scenario_id} / {event_id} — gibberish found:")
                for line in details.split("\n"):
                    print(f"      {line}")

        if file_has_issue:
            files_with_issues += 1
        else:
            print(f"  ✓ {scenario_id} — all dialogues OK")

    # ── Summary ──────────────────────────────────────────────────
    total_issues = len(issues)
    print(f"\n{'=' * 60}")
    print(f"GIBBERISH REVIEW COMPLETE")
    print(f"  Files checked:      {files_checked}")
    print(f"  Files with issues:  {files_with_issues}")
    print(f"  Total issues:       {total_issues}")
    print(f"{'=' * 60}")

    # Write report JSON next to the output folder
    report = {
        "files_checked": files_checked,
        "files_with_issues": files_with_issues,
        "total_issues": total_issues,
        "issues": issues,
    }
    report_path = out_path / "gibberish_review_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  Report saved to: {report_path}\n")

    return report


def main():
    """Main entry point for the script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert dialogue JSON files to minimal format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all JSON files in a folder (generates PDFs by default)
  python convert_dialogues_to_minimal.py -i ./input_folder -o ./output_folder
  
  # Convert recursively (include subdirectories)
  python convert_dialogues_to_minimal.py -i ./input_folder -o ./output_folder -r
  
  # Include both enforced and toxic dialogues for comparison
  python convert_dialogues_to_minimal.py -i ./input_folder -o ./output_folder --observe-toxic
  
  # Disable PDF generation
  python convert_dialogues_to_minimal.py -i ./input_folder -o ./output_folder --no-pdf
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Input folder containing JSON files'
    )
    
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output folder for converted JSON files'
    )
    
    parser.add_argument(
        '-r', '--recursive',
        action='store_true',
        help='Process subdirectories recursively'
    )
    
    parser.add_argument(
        '--observe-toxic',
        action='store_true',
        help='Include both enforced_dialogue and toxic_injected_dialogue for comparison'
    )
    
    parser.add_argument(
        '--no-pdf',
        action='store_true',
        help='Disable PDF generation (enabled by default)'
    )

    parser.add_argument(
        '--no-clean',
        action='store_true',
        help='Disable deterministic text cleaning (refusal-preamble removal, '
             'CJK-leakage stripping, suspect-line flagging; enabled by default)'
    )
    
    parser.add_argument(
        '--review-gibberish',
        action='store_true',
        help='After conversion, use GPT to scan all dialogues for gibberish / garbled text'
    )
    
    parser.add_argument(
        '--gpt-model',
        default='gpt-5.2',
        help='GPT model for gibberish review (default: gpt-5.2)'
    )
    
    parser.add_argument(
        '--gpt-api-base',
        default=None,
        help='Custom OpenAI API base URL for gibberish review'
    )
    
    args = parser.parse_args()

    convert_folder(args.input, args.output, args.recursive, args.observe_toxic,
                   not args.no_pdf, not args.no_clean)

    # ── Post-conversion gibberish review ────────────────────────────
    if args.review_gibberish:
        review_dialogues_for_gibberish(
            output_folder=args.output,
            gpt_model=args.gpt_model,
            gpt_api_base=args.gpt_api_base,
            recursive=args.recursive,
        )


if __name__ == "__main__":
    main()
