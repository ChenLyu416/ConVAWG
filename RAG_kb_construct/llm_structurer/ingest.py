"""Document ingestion and preprocessing module."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import re


logger = logging.getLogger(__name__)


class DocumentIngestor:
    """Handles ingestion of raw case documents."""

    def __init__(self):
        """Initialize the document ingestor."""
        self.logger = logging.getLogger(__name__)

    def ingest_file(self, file_path: str) -> Dict[str, Any]:
        """
        Ingest a single file.

        Args:
            file_path: Path to the file to ingest

        Returns:
            Dictionary with case_id and raw_text
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if path.suffix == '.json':
            return self._ingest_json(path)
        elif path.suffix in ['.txt', '.text']:
            return self._ingest_text(path)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")

    def _ingest_json(self, path: Path) -> Dict[str, Any]:
        """Ingest a JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            if 'case_id' not in data:
                data['case_id'] = path.stem
            if 'raw_text' not in data and 'text' in data:
                data['raw_text'] = data['text']
            if 'raw_text' not in data:
                raise ValueError(f"JSON file must contain 'raw_text' or 'text' field: {path}")
            return data
        else:
            raise ValueError(f"JSON file must contain a dictionary: {path}")

    def _ingest_text(self, path: Path) -> Dict[str, Any]:
        """Ingest a plain text file."""
        with open(path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
        
        return {
            'case_id': path.stem,
            'raw_text': raw_text,
            'source': str(path)
        }

    def ingest_directory(self, directory: str, pattern: str = "*") -> List[Dict[str, Any]]:
        """
        Ingest all matching files from a directory.

        Args:
            directory: Directory path
            pattern: Glob pattern for file matching

        Returns:
            List of ingested documents
        """
        dir_path = Path(directory)
        
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        documents = []
        supported_extensions = ['.json', '.txt', '.text']
        
        for file_path in dir_path.glob(pattern):
            if file_path.is_file() and file_path.suffix in supported_extensions:
                try:
                    doc = self.ingest_file(str(file_path))
                    documents.append(doc)
                    self.logger.info(f"Ingested: {file_path}")
                except Exception as e:
                    self.logger.error(f"Failed to ingest {file_path}: {e}")
        
        self.logger.info(f"Ingested {len(documents)} documents from {directory}")
        return documents

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess raw text.

        Args:
            text: Raw text to preprocess

        Returns:
            Preprocessed text
        """
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text

    def save_ingested(self, documents: List[Dict[str, Any]], output_path: str):
        """
        Save ingested documents to JSONL file.

        Args:
            documents: List of ingested documents
            output_path: Output file path
        """
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output, 'w', encoding='utf-8') as f:
            for doc in documents:
                f.write(json.dumps(doc) + '\n')
        
        self.logger.info(f"Saved {len(documents)} documents to {output_path}")

    def load_ingested(self, input_path: str) -> List[Dict[str, Any]]:
        """
        Load previously ingested documents from JSONL file.

        Args:
            input_path: Input JSONL file path

        Returns:
            List of documents
        """
        documents = []
        
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    documents.append(json.loads(line))
        
        self.logger.info(f"Loaded {len(documents)} documents from {input_path}")
        return documents
