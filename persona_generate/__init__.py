"""Fix imports for VAWG generation modules."""
# This file ensures proper imports work

import sys
from pathlib import Path

# Add parent directories to Python path
parent_dir = Path(__file__).parent.parent
rag_dir = parent_dir / "RAG_kb_construct"

if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

if str(rag_dir) not in sys.path:
    sys.path.insert(0, str(rag_dir))
