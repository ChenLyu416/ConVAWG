#!/bin/bash
# Structure every extracted DHR text into a JSON case record (one file per case).
# Run from RAG_kb_construct/ after Process_PDFs/scrape_and_extract.py and
# Process_PDFs/extract_key_sections.py. Needs a running vLLM server for the
# extractor model named in CONFIG (see the top-level README).
#
# Usage: bash process_all_dhrs.sh [INPUT_DIR] [OUTPUT_DIR]

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

CONFIG="configs/qwen_14b_gguf.yaml"     # extractor used to build the paper's knowledge base
INPUT_DIR="${1:-Process_PDFs/extracted_key_texts}"
OUTPUT_DIR="${2:-KB_homicide}"

mkdir -p "$OUTPUT_DIR"
echo "Config: $CONFIG"
echo "Input:  $INPUT_DIR ($(find "$INPUT_DIR" -name '*.txt' -type f | wc -l) files)"
echo "Output: $OUTPUT_DIR"

python -m llm_structurer.cli extract \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --config "$CONFIG"

echo "Done: $(find "$OUTPUT_DIR" -name '*.json' -type f | wc -l) case records in $OUTPUT_DIR"
