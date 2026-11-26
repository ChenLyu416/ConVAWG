#!/usr/bin/env python3
"""
PDF Scraper and Text Extractor for Homicide Review Cases

This script:
1. Scrapes PDF links from the Homicide Review website
2. Downloads all PDFs with female victims
3. Extracts text from each PDF up to "Appendix" section
4. Saves extracted text with proper naming
"""

import os
import re
import sys
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from urllib.parse import urljoin, urlparse, parse_qs, urlencode

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

try:
    import PyPDF2
except ImportError:
    print("PyPDF2 not found. Installing...")
    os.system("pip install PyPDF2")
    import PyPDF2

try:
    import pdfplumber
except ImportError:
    print("pdfplumber not found. Installing...")
    os.system("pip install pdfplumber")
    import pdfplumber


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pdf_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class HomicideReviewScraper:
    """Scraper for Homicide Review PDFs."""
    
    BASE_URL = "https://homicide-review.homeoffice.gov.uk"
    
    def __init__(self, output_dir: str = "downloaded_pdfs"):
        """Initialize scraper."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def build_search_url(self, page: int = 0, page_size: int = 100) -> str:
        """Build search URL for female victims (page is 0-indexed)."""
        params = {
            'Keywords[]': '',
            'Victim-Specific Information.Victims.Sex of victim[]': 'Female',
            'pagination[pageSize]': str(page_size),
            'pagination[pageNumber]': str(page)
        }
        
        query_string = urlencode(params, safe='[]')
        return f"{self.BASE_URL}/?{query_string}"
    
    def scrape_pdf_links(self, max_pages: int = 5) -> List[Dict[str, str]]:
        """
        Scrape PDF links from search results (pages are 0-indexed).
        
        Returns:
            List of dicts with 'url', 'title', 'filename' keys
        """
        pdf_links = []
        seen_urls = set()  # Track unique download URLs
        
        for page in range(max_pages):
            logger.info(f"Scraping page {page + 1}/{max_pages}...")
            url = self.build_search_url(page=page)
            
            try:
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find download links (the site uses /download/ endpoints, not direct .pdf links)
                links = soup.find_all('a', href=re.compile(r'^/download/', re.I))
                
                if not links:
                    logger.info(f"No download links found on page {page}. Stopping.")
                    break
                
                page_count = 0
                for link in links:
                    href = link.get('href')
                    
                    if href and href not in seen_urls:
                        seen_urls.add(href)
                        
                        # Use the 'download' attribute for title/filename
                        title = link.get('download') or link.get('aria-label', '').replace('Download ', '') or 'Unknown'
                        full_url = urljoin(self.BASE_URL, href)
                        filename = self._sanitize_filename(title) + '.pdf'
                        
                        pdf_links.append({
                            'url': full_url,
                            'title': title,
                            'filename': filename
                        })
                        page_count += 1
                
                logger.info(f"Found {page_count} unique PDFs on page {page} (total: {len(pdf_links)})")
                
                # If we got fewer than expected, we might be at the last page
                if page_count == 0:
                    logger.info(f"No new PDFs on page {page}. Stopping.")
                    break
                    
                time.sleep(1)  # Be nice to the server
                
            except Exception as e:
                logger.error(f"Error scraping page {page}: {e}")
                break
        
        logger.info(f"Total unique PDFs found: {len(pdf_links)}")
        return pdf_links
    
    def _sanitize_filename(self, title: str) -> str:
        """Sanitize title for use as filename."""
        # Remove special characters
        sanitized = re.sub(r'[^\w\s-]', '', title)
        # Replace spaces with underscores
        sanitized = re.sub(r'\s+', '_', sanitized)
        # Truncate to reasonable length
        sanitized = sanitized[:200]
        return sanitized.strip('_') or 'untitled'
    
    def download_pdf(self, pdf_info: Dict[str, str]) -> Optional[Path]:
        """
        Download a single PDF.
        
        Args:
            pdf_info: Dict with 'url' and 'filename'
            
        Returns:
            Path to downloaded file or None
        """
        url = pdf_info['url']
        filename = pdf_info['filename']
        output_path = self.output_dir / filename
        
        # Skip if already downloaded
        if output_path.exists():
            logger.info(f"Already exists: {filename}")
            return output_path
        
        try:
            logger.info(f"Downloading: {filename}")
            response = self.session.get(url, timeout=60, stream=True)
            response.raise_for_status()
            
            # Save file
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Downloaded: {filename}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")
            return None
    
    def download_all(self, pdf_links: List[Dict[str, str]]) -> List[Path]:
        """Download all PDFs with progress bar."""
        downloaded = []
        
        for pdf_info in tqdm(pdf_links, desc="Downloading PDFs"):
            path = self.download_pdf(pdf_info)
            if path:
                downloaded.append(path)
            time.sleep(0.5)  # Rate limiting
        
        return downloaded


class PDFTextExtractor:
    """Extract text from PDFs up to Appendix section."""
    
    def __init__(self, output_dir: str = "extracted_texts"):
        """Initialize extractor."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Patterns to detect appendix section
        self.appendix_patterns = [
            re.compile(r'\bappendix\b', re.I),
            re.compile(r'\bappendices\b', re.I),
            re.compile(r'\bannex\b', re.I),
            re.compile(r'\bsupplementary\s+materials?\b', re.I),
        ]
    
    def extract_with_pdfplumber(self, pdf_path: Path) -> str:
        """Extract text using pdfplumber (better quality)."""
        text = ""
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
        except Exception as e:
            logger.warning(f"pdfplumber failed for {pdf_path.name}: {e}")
            return ""
        
        return text
    
    def extract_with_pypdf2(self, pdf_path: Path) -> str:
        """Extract text using PyPDF2 (fallback)."""
        text = ""
        
        try:
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
        except Exception as e:
            logger.warning(f"PyPDF2 failed for {pdf_path.name}: {e}")
            return ""
        
        return text
    
    def extract_text(self, pdf_path: Path) -> str:
        """
        Extract text from PDF.
        
        Tries pdfplumber first, falls back to PyPDF2.
        """
        logger.info(f"Extracting text from: {pdf_path.name}")
        
        # Try pdfplumber first (better quality)
        text = self.extract_with_pdfplumber(pdf_path)
        
        # Fallback to PyPDF2
        if not text.strip():
            logger.info(f"Trying PyPDF2 for {pdf_path.name}")
            text = self.extract_with_pypdf2(pdf_path)
        
        if not text.strip():
            logger.error(f"No text extracted from {pdf_path.name}")
            return ""
        
        return text
    
    def find_appendix_position(self, text: str) -> Optional[int]:
        """
        Find position where appendix section starts.
        
        Returns:
            Character position or None
        """
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Check if line matches appendix pattern
            for pattern in self.appendix_patterns:
                if pattern.search(line_stripped):
                    # Check if this is likely a header (short line, possibly numbered)
                    if len(line_stripped) < 100:
                        # Calculate character position
                        char_pos = sum(len(l) + 1 for l in lines[:i])
                        logger.info(f"Found appendix at line {i}: '{line_stripped}'")
                        return char_pos
        
        return None
    
    def truncate_at_appendix(self, text: str) -> str:
        """Remove appendix section from text."""
        appendix_pos = self.find_appendix_position(text)
        
        if appendix_pos:
            logger.info(f"Truncating at position {appendix_pos}")
            return text[:appendix_pos].strip()
        else:
            logger.info("No appendix found, keeping full text")
            return text.strip()
    
    def save_text(self, text: str, pdf_path: Path) -> Path:
        """Save extracted text to file."""
        # Create output filename (same as PDF but .txt)
        output_filename = pdf_path.stem + '.txt'
        output_path = self.output_dir / output_filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        logger.info(f"Saved text to: {output_path}")
        return output_path
    
    def process_pdf(self, pdf_path: Path) -> Optional[Path]:
        """
        Process single PDF: extract text and truncate at appendix.
        
        Returns:
            Path to saved text file or None
        """
        # Check if already processed
        output_filename = pdf_path.stem + '.txt'
        output_path = self.output_dir / output_filename
        
        if output_path.exists():
            logger.info(f"Already processed: {pdf_path.name}")
            return output_path
        
        try:
            # Extract text
            text = self.extract_text(pdf_path)
            
            if not text:
                return None
            
            # Truncate at appendix
            truncated_text = self.truncate_at_appendix(text)
            
            # Save
            return self.save_text(truncated_text, pdf_path)
            
        except Exception as e:
            logger.error(f"Failed to process {pdf_path.name}: {e}")
            return None
    
    def process_all(self, pdf_paths: List[Path]) -> List[Path]:
        """Process all PDFs with progress bar."""
        processed = []
        
        for pdf_path in tqdm(pdf_paths, desc="Extracting text"):
            result = self.process_pdf(pdf_path)
            if result:
                processed.append(result)
        
        return processed


def main():
    """Main execution function."""
    print("=" * 80)
    print("Homicide Review PDF Scraper and Text Extractor")
    print("=" * 80)
    print()
    
    # Configuration
    pdf_dir = Path("downloaded_pdfs")
    text_dir = Path("extracted_texts")
    
    # Step 1: Scrape and download PDFs
    print("\n[1] Scraping PDF links...")
    scraper = HomicideReviewScraper(output_dir=str(pdf_dir))
    
    pdf_links = scraper.scrape_pdf_links(max_pages=10)  # Should cover all ~500 documents
    
    if not pdf_links:
        print("No PDFs found. Exiting.")
        return
    
    print(f"\nFound {len(pdf_links)} PDFs")
    
    # Save links to JSON for reference
    links_file = Path("pdf_links.json")
    with open(links_file, 'w') as f:
        json.dump(pdf_links, f, indent=2)
    print(f"Saved PDF links to: {links_file}")
    
    # Download PDFs
    print("\n[2] Downloading PDFs...")
    downloaded_pdfs = scraper.download_all(pdf_links)
    print(f"\nDownloaded {len(downloaded_pdfs)} PDFs to: {pdf_dir}")
    
    # Step 2: Extract text
    print("\n[3] Extracting text from PDFs...")
    extractor = PDFTextExtractor(output_dir=str(text_dir))
    
    extracted_texts = extractor.process_all(downloaded_pdfs)
    print(f"\nExtracted text from {len(extracted_texts)} PDFs to: {text_dir}")
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"PDFs found:       {len(pdf_links)}")
    print(f"PDFs downloaded:  {len(downloaded_pdfs)}")
    print(f"Texts extracted:  {len(extracted_texts)}")
    print(f"\nPDF directory:    {pdf_dir.absolute()}")
    print(f"Text directory:   {text_dir.absolute()}")
    print(f"Log file:         pdf_processing.log")
    print("=" * 80)


if __name__ == "__main__":
    main()
