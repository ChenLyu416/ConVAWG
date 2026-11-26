"""Simple script to build persona index from KB_homicide JSON files."""
import json
import logging
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_persona_docs(jsonl_file):
    """Load persona documents from JSONL file."""
    personas = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            personas.append(data)
    return personas


def build_index(jsonl_file, output_dir, model_name="all-MiniLM-L6-v2"):
    """Build FAISS index from persona documents."""
    
    logger.info(f"Loading persona documents from {jsonl_file}...")
    personas = load_persona_docs(jsonl_file)
    logger.info(f"Loaded {len(personas)} persona documents")
    
    # Initialize embedding model
    logger.info(f"Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)
    embedding_dim = model.get_sentence_embedding_dimension()
    logger.info(f"Embedding dimension: {embedding_dim}")
    
    # Create texts for embedding.
    # The index is used with case-level documents (KB_homicide) in retrieval.py,
    # but we keep compatibility with persona-level docs that may include
    # `summary_text` / `context_text`.
    texts = []
    for doc in personas:
        text = (
            doc.get("summary_text", "")
            or doc.get("context_text", "")
            or doc.get("summary", "")
        )

        if not text:
            raw_text = doc.get("raw_text", "")
            if raw_text:
                # Prefer the facts section when present; otherwise truncate.
                marker = "**2. The Facts**"
                if marker in raw_text:
                    text = raw_text.split(marker, 1)[1][:1500]
                else:
                    text = raw_text[:1500]

        if not text:
            role = doc.get("role", "")
            case_id = doc.get("case_id", "")
            text = f"{role} in case {case_id}".strip()

        texts.append(text)
    
    # Generate embeddings
    logger.info("Generating embeddings (this may take a few minutes)...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    logger.info(f"Generated embeddings shape: {embeddings.shape}")
    
    # Build FAISS index
    logger.info("Building FAISS index...")
    index = faiss.IndexFlatIP(embedding_dim)  # Inner product (cosine similarity)
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Add to index
    index.add(embeddings.astype(np.float32))
    logger.info(f"Index built with {index.ntotal} vectors")
    
    # Save index and documents
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    index_file = output_dir / "index.faiss"
    docs_file = output_dir / "persona_docs.jsonl"
    
    logger.info(f"Saving index to {index_file}...")
    faiss.write_index(index, str(index_file))
    
    logger.info(f"Copying persona docs to {docs_file}...")
    import shutil
    shutil.copy(jsonl_file, docs_file)
    
    # Save metadata
    metadata = {
        # Keep the old key for backward compatibility with existing logs/scripts.
        "num_personas": len(personas),
        "num_documents": len(personas),
        "embedding_model": model_name,
        "embedding_dim": embedding_dim,
        "index_type": "IndexFlatIP"
    }
    metadata_file = output_dir / "index_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"✓ Index built successfully!")
    logger.info(f"  - Index file: {index_file}")
    logger.info(f"  - Persona docs: {docs_file}")
    logger.info(f"  - Metadata: {metadata_file}")
    logger.info(f"  - Total personas: {len(personas)}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python build_persona_index_simple.py <jsonl_file> <output_dir>")
        print("Example: python build_persona_index_simple.py outputs/persona_docs_full.jsonl outputs/persona_index_full")
        sys.exit(1)
    
    jsonl_file = sys.argv[1]
    output_dir = sys.argv[2]
    
    build_index(jsonl_file, output_dir)
