#!/usr/bin/env python3
"""
Extract key sections (Introduction, Background, Facts, Circumstances, etc.)
from downloaded DHR PDFs and save to extracted_key_texts/ folder.

Uses multiple extraction strategies in order of preference:
1. PDF TOC/Outline
2. Markdown heading detection via pymupdf4llm
3. Font-size/bold heuristics
4. Regex pattern matching (fallback)
"""

import os
import re
import json
import logging
import statistics
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional

try:
    import fitz  # PyMuPDF
except ImportError:
    print("Installing PyMuPDF...")
    os.system("pip install pymupdf")
    import fitz

try:
    import pymupdf4llm
except ImportError:
    print("Installing pymupdf4llm...")
    os.system("pip install pymupdf4llm")
    import pymupdf4llm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Target section names (case-insensitive, partial match)
TARGETS = (
    "introduction",
    "background",
    "fact",
    "circumstance",
    "summary",
    "overview",
    "the incident",
)

# Maximum content length per section (characters)
MAX_SECTION_LENGTH = 30000


def _norm_text(s: str) -> str:
    """Normalize text for fuzzy matching."""
    return re.sub(r"[^a-z0-9]+", " ", s.casefold()).strip()


def _title_matches(title: str, targets: Tuple[str, ...] = TARGETS) -> bool:
    """Check if title matches any target section names."""
    nt = _norm_text(title)
    for t in targets:
        if _norm_text(t) in nt:
            return True
    return False


def _is_toc_content(text: str) -> bool:
    """Detect if content is likely a table of contents (lots of page numbers, short lines)."""
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if not lines or len(lines) < 5:
        return False
    
    # Count lines that are just numbers or end with numbers
    number_lines = sum(1 for l in lines if re.match(r'^\d+$', l) or re.search(r'\d{1,3}$', l))
    # Count very short lines (likely TOC entries)
    short_lines = sum(1 for l in lines if len(l) < 60)
    
    # If >50% are number-heavy or very short, it's likely a TOC
    return (number_lines / len(lines) > 0.3) or (short_lines / len(lines) > 0.7)


def _clean_content(content: str, max_length: int = MAX_SECTION_LENGTH) -> str:
    """
    Clean and trim section content.
    - Remove TOC artifacts
    - Stop at appendix/glossary
    - Limit length
    """
    # Stop at common end markers
    for stop_marker in ["appendix", "annex", "glossary", "definitions", "bibliography", "references"]:
        stop_pattern = re.compile(rf'^\s*(?:\d+[\.\)]?\s*)?{stop_marker}', re.I | re.M)
        match = stop_pattern.search(content)
        if match:
            content = content[:match.start()].strip()
            break
    
    # Remove excessive whitespace
    content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
    
    # Trim to max length at sentence boundary
    if len(content) > max_length:
        truncated = content[:max_length]
        # Try to end at a sentence
        last_period = truncated.rfind('. ')
        if last_period > max_length * 0.8:  # Only if we're close to the end
            truncated = truncated[:last_period + 1]
        content = truncated + "\n\n[Content truncated due to length]"
    
    return content.strip()


# ---------- Strategy A: TOC/Outline ----------

def extract_sections_by_toc(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Use the PDF's outline/TOC to locate page ranges for target sections.
    """
    out = []
    try:
        with fitz.open(pdf_path) as doc:
            toc = doc.get_toc()
            if not toc:
                return out

            for i, (level, title, page) in enumerate(toc):
                if not _title_matches(title):
                    continue
                start = max(0, page - 1)
                end_page = len(doc)
                # Find next entry of same or higher level
                for j in range(i + 1, len(toc)):
                    l2, _, p2 = toc[j]
                    if l2 <= level:
                        end_page = p2 - 1
                        break
                
                text = "".join(doc[p].get_text("text") for p in range(start, end_page))
                
                # Skip if it looks like TOC content
                if _is_toc_content(text):
                    continue
                    
                cleaned = _clean_content(text)
                if cleaned and len(cleaned) > 200:  # Minimum viable content
                    out.append({
                        "heading": title.strip(),
                        "level": level,
                        "pages": (start + 1, end_page),
                        "content": cleaned,
                        "method": "toc",
                    })
    except Exception as e:
        logger.debug(f"TOC extraction failed for {pdf_path}: {e}")
    return out


# ---------- Strategy B: Markdown headings ----------

def extract_sections_from_markdown(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Convert PDF -> Markdown with PyMuPDF4LLM, then slice by # headings.
    """
    sections = []
    try:
        with fitz.open(pdf_path) as doc:
            md = pymupdf4llm.to_markdown(doc)

        h_re = re.compile(r"^(#{1,6})\s*(.+?)\s*$", flags=re.M)
        matches = list(h_re.finditer(md))

        for i, m in enumerate(matches):
            level = len(m.group(1))
            title = m.group(2)
            if not _title_matches(title):
                continue
                
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(md)
            content = md[start:end].strip()
            
            if _is_toc_content(content):
                continue
                
            cleaned = _clean_content(content)
            if cleaned and len(cleaned) > 200:
                sections.append({
                    "heading": title.strip(),
                    "level": level,
                    "content": cleaned,
                    "method": "markdown",
                })
    except Exception as e:
        logger.debug(f"Markdown extraction failed for {pdf_path}: {e}")
    return sections


# ---------- Strategy C: Font-size / bold heuristics ----------

def _page_median_font_size(page: fitz.Page) -> float:
    """Get median font size for a page."""
    sizes = []
    try:
        rd = page.get_text("rawdict")
        for b in rd.get("blocks", []):
            for l in b.get("lines", []):
                for s in l.get("spans", []):
                    if s.get("text", "").strip():
                        sizes.append(s.get("size", 0))
    except:
        pass
    return statistics.median(sizes) if sizes else 10.0


def extract_sections_by_font(pdf_path: str, size_multiplier: float = 1.25) -> List[Dict[str, Any]]:
    """
    Heuristic: treat lines with larger/bold fonts as headings.
    Improved version using per-page scanning with cleaner logic.
    """
    results = []
    try:
        with fitz.open(pdf_path) as doc:
            medians = [_page_median_font_size(p) for p in doc]
            
            for pno, page in enumerate(doc):
                rd = page.get_text("rawdict")
                median = medians[pno]

                # Build lines as (text, maxsize, is_boldish)
                lines = []
                for b in rd.get("blocks", []):
                    for l in b.get("lines", []):
                        txt_parts = []
                        maxsize = 0.0
                        boldish = False
                        for s in l.get("spans", []):
                            t = s.get("text", "")
                            if t:
                                txt_parts.append(t)
                            size = s.get("size", 0.0)
                            maxsize = max(maxsize, size)
                            fontname = s.get("font", "")
                            if any(k in fontname for k in ("Bold", "Black", "Semibold", "Demi")):
                                boldish = True
                        line_text = "".join(txt_parts).strip()
                        if line_text:
                            lines.append((line_text, maxsize, boldish))

                # Scan for target headings on this page
                i = 0
                while i < len(lines):
                    line, maxsize, boldish = lines[i]
                    is_heading = (maxsize >= size_multiplier * median) or boldish
                    if is_heading and _title_matches(line):
                        # Collect until next heading-like line
                        buf = []
                        j = i + 1
                        while j < len(lines):
                            ltxt, lsize, lbold = lines[j]
                            # Stop at next heading
                            if (lsize >= size_multiplier * median) or lbold:
                                break
                            buf.append(ltxt)
                            j += 1
                        content = "\n".join(buf).strip()
                        
                        if _is_toc_content(content):
                            i = j
                            continue
                            
                        cleaned = _clean_content(content)
                        if cleaned and len(cleaned) > 200:
                            results.append({
                                "heading": line.strip(),
                                "page": pno + 1,
                                "content": cleaned,
                                "method": "font",
                            })
                        i = j
                    else:
                        i += 1
    except Exception as e:
        logger.debug(f"Font extraction failed for {pdf_path}: {e}")
    return results


# ---------- Strategy D: Regex pattern matching on full text ----------

def extract_sections_by_regex(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Fall back to regex pattern matching on full document text.
    Look for section headers and extract narrative content.
    """
    results = []
    try:
        with fitz.open(pdf_path) as doc:
            full_text = ""
            for page in doc:
                full_text += page.get_text("text") + "\n"

        # More precise pattern - looking for actual section headers
        # Match: "1. Introduction", "INTRODUCTION", "1.1 Introduction", etc.
        section_pattern = re.compile(
            r'^[\s]*(?:(?:\d+(?:\.\d+)*|[IVX]+)[\.\)]?\s+)?(INTRODUCTION|BACKGROUND|'
            r'FACTS?\s+OF\s+THE\s+CASE|CIRCUMSTANCES?|SUMMARY\s+OF\s+(?:THE\s+)?CASE|'
            r'CASE\s+SUMMARY|OVERVIEW|THE\s+INCIDENT)[\s:]*$',
            re.IGNORECASE | re.MULTILINE
        )

        matches = list(section_pattern.finditer(full_text))
        
        for i, m in enumerate(matches):
            heading = m.group(0).strip()
            start = m.end()
            
            # Find end: next section match or reasonable stopping point
            end = len(full_text)
            if i + 1 < len(matches):
                end = matches[i + 1].start()
            
            content = full_text[start:end].strip()
            
            # Skip if looks like TOC
            if _is_toc_content(content):
                continue
            
            cleaned = _clean_content(content)
            
            if cleaned and len(cleaned) > 200:
                results.append({
                    "heading": heading,
                    "content": cleaned,
                    "method": "regex",
                })
    except Exception as e:
        logger.debug(f"Regex extraction failed for {pdf_path}: {e}")
    return results


# ---------- Orchestrator ----------

def extract_key_sections(pdf_path: str) -> Dict[str, Any]:
    """
    Try multiple strategies to extract key sections.
    Prioritize quality: TOC > Markdown > Font > Regex.
    Only use fallbacks if previous methods yield no results.
    """
    all_sections = []
    methods_used = []

    # Strategy A: TOC (most reliable when available)
    toc_sections = extract_sections_by_toc(pdf_path)
    if toc_sections:
        all_sections.extend(toc_sections)
        methods_used.append("toc")
        logger.debug(f"TOC extracted {len(toc_sections)} sections")

    # Strategy B: Markdown (good structure detection)
    if not all_sections:
        md_sections = extract_sections_from_markdown(pdf_path)
        if md_sections:
            all_sections.extend(md_sections)
            methods_used.append("markdown")
            logger.debug(f"Markdown extracted {len(md_sections)} sections")

    # Strategy C: Font heuristics (works when structure is visual)
    if not all_sections:
        font_sections = extract_sections_by_font(pdf_path)
        if font_sections:
            all_sections.extend(font_sections)
            methods_used.append("font")
            logger.debug(f"Font extracted {len(font_sections)} sections")

    # Strategy D: Regex (last resort)
    if not all_sections:
        regex_sections = extract_sections_by_regex(pdf_path)
        if regex_sections:
            all_sections.extend(regex_sections)
            methods_used.append("regex")
            logger.debug(f"Regex extracted {len(regex_sections)} sections")

    # Deduplicate by heading similarity
    seen_headings = set()
    unique_sections = []
    for sec in all_sections:
        norm_heading = _norm_text(sec["heading"])[:50]
        if norm_heading not in seen_headings:
            seen_headings.add(norm_heading)
            unique_sections.append(sec)

    return {
        "methods_used": methods_used or ["none"],
        "sections": unique_sections,
    }


def process_all_pdfs(
    input_dir: str = "downloaded_pdfs",
    output_dir: str = "extracted_key_texts",
) -> Dict[str, Any]:
    """
    Process all PDFs in input_dir and save extracted sections to output_dir.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pdf_files = list(input_path.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDFs to process")

    stats = {
        "total": len(pdf_files),
        "success": 0,
        "no_sections": 0,
        "errors": 0,
        "files_processed": [],
    }

    for pdf_file in pdf_files:
        try:
            logger.info(f"Processing: {pdf_file.name}")
            result = extract_key_sections(str(pdf_file))
            
            if result["sections"]:
                # Combine all section contents
                combined_text = ""
                for sec in result["sections"]:
                    combined_text += f"\n{'='*80}\n"
                    combined_text += f"SECTION: {sec['heading']}\n"
                    combined_text += f"Method: {sec['method']}\n"
                    if "pages" in sec:
                        combined_text += f"Pages: {sec['pages']}\n"
                    combined_text += f"{'='*80}\n\n"
                    combined_text += sec["content"]
                    combined_text += "\n\n"

                # Save as .txt file
                output_file = output_path / f"{pdf_file.stem}.txt"
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(f"Source: {pdf_file.name}\n")
                    f.write(f"Extraction methods: {', '.join(result['methods_used'])}\n")
                    f.write(f"Sections found: {len(result['sections'])}\n")
                    f.write(combined_text)

                # Also save structured JSON
                json_file = output_path / f"{pdf_file.stem}.json"
                with open(json_file, "w", encoding="utf-8") as f:
                    json.dump({
                        "source": pdf_file.name,
                        "methods_used": result["methods_used"],
                        "sections": result["sections"],
                    }, f, indent=2)

                stats["success"] += 1
                stats["files_processed"].append({
                    "file": pdf_file.name,
                    "sections": len(result["sections"]),
                    "methods": result["methods_used"],
                })
                logger.info(f"  ✓ Extracted {len(result['sections'])} sections")
            else:
                stats["no_sections"] += 1
                logger.warning(f"  ✗ No key sections found")

        except Exception as e:
            stats["errors"] += 1
            logger.error(f"  ✗ Error processing {pdf_file.name}: {e}")

    # Save summary stats
    stats_file = output_path / "_extraction_stats.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"EXTRACTION COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total PDFs: {stats['total']}")
    logger.info(f"Success: {stats['success']}")
    logger.info(f"No sections found: {stats['no_sections']}")
    logger.info(f"Errors: {stats['errors']}")
    logger.info(f"Output directory: {output_path}")

    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract key sections from DHR PDFs"
    )
    parser.add_argument(
        "--input-dir",
        default="downloaded_pdfs",
        help="Directory containing PDFs (default: downloaded_pdfs)",
    )
    parser.add_argument(
        "--output-dir",
        default="extracted_key_texts",
        help="Output directory for extracted texts (default: extracted_key_texts)",
    )
    parser.add_argument(
        "--single",
        help="Process a single PDF file instead of directory",
    )
    
    args = parser.parse_args()
    
    if args.single:
        # Process single file
        result = extract_key_sections(args.single)
        print(f"Methods used: {result['methods_used']}")
        for sec in result["sections"]:
            print("=" * 80)
            print(f"Heading: {sec['heading']}")
            print(f"Method: {sec['method']}")
            print(sec["content"][:500], "..." if len(sec["content"]) > 500 else "")
    else:
        # Process all PDFs
        process_all_pdfs(args.input_dir, args.output_dir)
