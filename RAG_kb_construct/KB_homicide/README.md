# Structured DHR knowledge base (not distributed)

Stage 1 retrieves from a knowledge base of structured records extracted from
publicly available UK Domestic Homicide Reviews (DHRs). The reviews describe
real victims, so neither the source documents nor the extracted records are
redistributed with this repository. Rebuild the knowledge base locally:

```bash
# 1. Download the reviews and extract text (URL list: Process_PDFs/pdf_links.json)
cd RAG_kb_construct/Process_PDFs
python scrape_and_extract.py
python extract_key_sections.py

# 2. Structure each review into a JSON record (one file per case); place the
#    resulting records in this directory, or pass their location as kb_homicide_dir
cd ..
bash process_all_dhrs.sh          # writes the records to KB_homicide/

# 3. Build the retrieval index used by Stage 1
cd ../persona_generate
bash build_full_index.sh
```

The record schema is defined in `RAG_kb_construct/llm_structurer/schema.py`,
the extractor prompts in `RAG_kb_construct/llm_structurer/prompts/`, the
extractor configuration in `RAG_kb_construct/configs/qwen_14b_gguf.yaml`, and
the indexing and retrieval logic in
`persona_generate/build_persona_index_simple.py` and
`persona_generate/retrieval.py`.

`Process_PDFs/pdf_links.json` lists the 489 review links collected from the
public library; the paper's knowledge base holds 410 structured records.

To run the pipeline without the knowledge base, set
`stage_1_persona_generation.zero_shot: true` (Stage 1 then skips retrieval).
