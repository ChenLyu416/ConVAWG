#!/usr/bin/env python3
"""Convert directory of JSON files to single JSONL file for persona index."""

import json
import glob
import sys
from pathlib import Path

def convert_json_dir_to_jsonl(input_dir: str, output_file: str):
    """Convert directory of JSON files to JSONL."""
    input_path = Path(input_dir)
    json_files = sorted(input_path.glob("*.json"))
    
    # Filter out processing.log if it exists
    json_files = [f for f in json_files if f.name != "processing.log"]
    
    print(f"Found {len(json_files)} JSON files in {input_dir}")
    
    with open(output_file, 'w', encoding='utf-8') as out:
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Add case_id from filename if not present
                if 'case_id' not in data:
                    data['case_id'] = json_file.stem
                
                # Write as single line
                out.write(json.dumps(data, ensure_ascii=False) + '\n')
                
            except Exception as e:
                print(f"Error processing {json_file}: {e}", file=sys.stderr)
    
    print(f"Converted {len(json_files)} files to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_json_to_jsonl.py <input_dir> <output_file>")
        print("Example: python convert_json_to_jsonl.py ../RAG_kb_construct/KB_homicide outputs/persona_docs_full.jsonl")
        sys.exit(1)
    
    convert_json_dir_to_jsonl(sys.argv[1], sys.argv[2])
