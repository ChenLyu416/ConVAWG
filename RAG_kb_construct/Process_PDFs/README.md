# PDF Processing for Homicide Reviews

This directory contains scripts to scrape and process PDF documents from the UK Home Office Homicide Review database.

## What It Does

1. **Scrapes PDF links** from the Homicide Review website (filtered for female victims)
2. **Downloads PDFs** to `downloaded_pdfs/` directory
3. **Extracts text** from each PDF
4. **Truncates at "Appendix"** section
5. **Saves cleaned text** to `extracted_texts/` directory

## Quick Start

### Option 1: Simple Run
```bash
bash run.sh
```

### Option 2: Manual Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run script
python3 scrape_and_extract.py
```

## Output Structure

```
Process_PDFs/
├── downloaded_pdfs/          # Downloaded PDF files
│   ├── Case_Name_001.pdf
│   ├── Case_Name_002.pdf
│   └── ...
├── extracted_texts/          # Extracted text files
│   ├── Case_Name_001.txt
│   ├── Case_Name_002.txt
│   └── ...
├── pdf_links.json           # Metadata of found PDFs
└── pdf_processing.log       # Processing log
```

## Features

### Intelligent Appendix Detection

The script detects appendix sections using multiple patterns:
- "Appendix"
- "Appendices"
- "Annex"
- "Supplementary materials"

Only content **before** the appendix is saved.

### Robust PDF Extraction

Uses two extraction methods with automatic fallback:
1. **pdfplumber** (primary) - Better text quality
2. **PyPDF2** (fallback) - Wider compatibility

### Rate Limiting

- Polite delays between requests (0.5-1 second)
- Session-based requests with proper headers
- Error handling and retry logic

### Progress Tracking

- Progress bars for downloads and extraction
- Detailed logging to `pdf_processing.log`
- Summary statistics at completion

## Configuration

Edit `scrape_and_extract.py` to customize:

```python
# Number of pages to scrape (100 PDFs per page)
pdf_links = scraper.scrape_pdf_links(max_pages=20)

# Output directories
pdf_dir = Path("downloaded_pdfs")
text_dir = Path("extracted_texts")
```

## Search Filters

The script searches for:
- **Sex of victim**: Female
- **Sorted by**: Upload date (descending)
- **Page size**: 100 results per page

To modify filters, edit the `build_search_url()` method in the script.

## Troubleshooting

### "No PDFs found"
- Website may be down or changed structure
- Check internet connection
- Verify URL in browser

### "Failed to extract text"
- PDF may be scanned image (OCR needed)
- PDF may be corrupted
- Check `pdf_processing.log` for details

### "Permission denied"
- Ensure write permissions in directory
- Check disk space

## Advanced Usage

### Process Specific PDFs Only

```python
from pathlib import Path
from scrape_and_extract import PDFTextExtractor

extractor = PDFTextExtractor(output_dir="my_texts")
pdf_path = Path("my_document.pdf")
text_path = extractor.process_pdf(pdf_path)
```

### Skip Download (Process Existing PDFs)

```python
from pathlib import Path
from scrape_and_extract import PDFTextExtractor

# Process all PDFs in a directory
extractor = PDFTextExtractor(output_dir="extracted_texts")
pdf_dir = Path("downloaded_pdfs")
pdfs = list(pdf_dir.glob("*.pdf"))
extractor.process_all(pdfs)
```

### Custom Appendix Patterns

```python
import re
from scrape_and_extract import PDFTextExtractor

extractor = PDFTextExtractor()
# Add custom pattern
extractor.appendix_patterns.append(
    re.compile(r'\bsupporting\s+documents?\b', re.I)
)
```

## Integration with Main Pipeline

After extracting texts, you can process them with the main extraction pipeline:

```bash
# Copy extracted texts to data directory
cp extracted_texts/*.txt ../data/cases/

# Run extraction pipeline
cd ..
bash hpc/submit_job.sh
```

## Dependencies

- **requests** - HTTP requests
- **beautifulsoup4** - HTML parsing
- **PyPDF2** - PDF text extraction (fallback)
- **pdfplumber** - PDF text extraction (primary)
- **tqdm** - Progress bars
- **lxml** - XML/HTML parsing

## Performance

- **Scraping**: ~1 second per page
- **Download**: ~2-5 seconds per PDF (depends on size)
- **Extraction**: ~1-3 seconds per PDF

For 100 PDFs:
- Total time: ~10-15 minutes
- Disk space: ~50-200 MB (PDFs + texts)

## Logs

Check `pdf_processing.log` for:
- Download status
- Extraction success/failure
- Appendix detection
- Error messages

## Notes

- PDFs are **not** deleted after extraction (kept in `downloaded_pdfs/`)
- Existing files are skipped (resume-friendly)
- Text encoding: UTF-8
- Preserves original PDF naming in output files

## Example Output

```
==========================================
Homicide Review PDF Scraper and Text Extractor
==========================================

[1] Scraping PDF links...
Found 145 PDFs

[2] Downloading PDFs...
Downloading PDFs: 100%|████████| 145/145 [12:34<00:00,  5.2s/it]
Downloaded 145 PDFs to: downloaded_pdfs

[3] Extracting text from PDFs...
Extracting text: 100%|████████| 145/145 [04:23<00:00,  1.8s/it]
Extracted text from 142 PDFs to: extracted_texts

==========================================
Summary
==========================================
PDFs found:       145
PDFs downloaded:  145
Texts extracted:  142

PDF directory:    /path/to/Process_PDFs/downloaded_pdfs
Text directory:   /path/to/Process_PDFs/extracted_texts
==========================================
```
