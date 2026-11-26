#!/bin/bash
# Build persona index from all KB_homicide cases

set -e

echo "=========================================="
echo "Building Persona Index from KB_homicide"
echo "=========================================="
echo

cd "$(dirname "${BASH_SOURCE[0]}")"

echo "Using Python: $(which python)"
python --version

echo "Step 1: Converting KB_homicide JSON files to JSONL..."
python convert_json_to_jsonl.py ../RAG_kb_construct/KB_homicide outputs/persona_docs_full.jsonl

echo
echo "Step 2: Building FAISS index from persona documents..."
python build_persona_index_simple.py outputs/persona_docs_full.jsonl outputs/persona_index_full

echo
echo "=========================================="
echo "✓ Persona index built successfully!"
echo "=========================================="
echo
echo "Index location: outputs/persona_index_full/"
echo "Total personas indexed: $(wc -l < outputs/persona_docs_full.jsonl)"
