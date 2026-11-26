# ConVAWG

Code for the EACL 2027 paper **"ConVAWG: A Retrieval-Grounded Framework for
Controlled Synthetic Dialogue Generation in Violence Against Women and Girls"**.

ConVAWG generates synthetic multi-turn chat dialogues that depict Violence
Against Women and Girls (VAWG) scenarios as relational, temporally unfolding
interactions. Scenarios are built from persona seeds, UK Crown Prosecution
Service (CPS) crime definitions and retrieved Domestic Homicide Review (DHR)
cases; they are converted into hierarchical event timelines, played out as
multi-scene role-play dialogues, and finally receive targeted,
activation-steered toxicity on selected perpetrator messages.

> **Content warning.** This project concerns domestic abuse, coercive control,
> stalking and sexual offences. The dataset and any output of this code contain
> synthetic abusive and distressing language. All dialogues are synthetic; no
> real person's messages are included.

## What is released

| Component | Where | Availability |
|---|---|---|
| Generation pipeline, evaluation, baselines, downstream tasks | this repository | public, [MIT](LICENSE) |
| ConVAWG dataset (6,171 dialogue events, 200 scenarios, 18 CPS-aligned crime types), with scenario and event-planning metadata | [Hugging Face](https://huggingface.co/datasets/ChenL49/ConVAWG) | gated; access on accepting the [data-use agreement](DATA_USE_AGREEMENT.md) |
| Calibrated steering vectors and Stage 4 toxification prompts | — | not public; justified request only ([details](toxic_steering/README.md)) |
| Source DHR documents and the structured case knowledge base | — | not redistributed; [rebuild instructions](RAG_kb_construct/KB_homicide/README.md) |

## Repository layout

The paper describes four generation stages; the code runs them as eight
pipeline stages.

| Paper stage | Pipeline stage | Module |
|---|---|---|
| 1. Real-case-guided persona and scenario construction | 1 | `RAG_kb_construct/`, `persona_generate/` |
| 2. Hierarchical scenario-to-chat event planning | 2 – event graph extraction | `event_generation/event_graph_from_summary/` |
| | 3 – event graph expansion | `event_generation/graph_expansion/` |
| | 4 – chat script generation | `event_generation/graph_to_chat/` |
| 3. Role-play online dialogue generation | 5 | `roleplay_dialogue/` |
| 4. Targeted toxicity injection | 6 | `toxic_steering/levelled_rewrite/` |
| Formatting and cleaning | 7 | `convert_dialogues_to_minimal.py`, `dialogue_text_filters.py` |
| LLM-as-Judge evaluation | 8 | `LLM_judge/` |

```
ConVAWG/
├── run_batch_pipeline.py        # end-to-end orchestrator
├── pipeline_config.yaml         # all stage settings
├── RAG_kb_construct/            # DHR scraping, structuring (schema + extractor prompts), LLM backends
├── persona_generate/            # Stage 1: retrieval + scenario generation (prompts, output schema)
├── event_generation/            # Stage 2: event graph → expansion → chat scripts
├── roleplay_dialogue/           # Stage 3: role-play generation and prompt templates
├── toxic_steering/              # Stage 4: orchestration of the CAA-steered rewrite
├── convert_dialogues_to_minimal.py, dialogue_text_filters.py, fix_converted_dialogues.py
├── LLM_judge/                   # six-metric LLM judge + Detoxify, perplexity and diversity scorers
├── Experiments/
│   ├── Baselines/               # direct-prompt baselines and DiaSynth / SynDG / SPASM re-implementations
│   └── Downstream_tasks/        # behaviour and relationship classification, escalation forecasting, CGA and MentalManip transfer
├── analysis/                    # results tables, significance tests, scale comparison, agreement
├── docs/                        # CPS VAWG definitions and crime-type reference
└── data/                        # placeholder for the access-controlled dataset
```

Prompts referred to in the paper's appendix:

| Prompt / schema | File |
|---|---|
| DHR record schema and extractor prompts | `RAG_kb_construct/llm_structurer/schema.py`, `RAG_kb_construct/llm_structurer/prompts/` |
| DHR extractor configuration | `RAG_kb_construct/configs/qwen_14b_gguf.yaml`, `RAG_kb_construct/process_all_dhrs.sh` |
| Indexing and retrieval | `persona_generate/build_persona_index_simple.py`, `persona_generate/retrieval.py` |
| Stage 1 scenario prompts and output schema | `persona_generate/prompt_templates.py`, `persona_generate/schemas/vawg_scenario.py` |
| Stage 1 consistency refinement | `persona_generate/postprocess_scenarios.py` |
| Stage 2 event planning prompts | `event_generation/*/*.py` |
| Stage 3 role-play prompts and style notes | `roleplay_dialogue/roleplay_prompt_template.yaml`, `roleplay_dialogue/generate_roleplay_dialogues.py` |
| LLM-as-Judge rubrics | `LLM_judge/prompts.py` |
| Downstream-task LLM prompts | `Experiments/Downstream_tasks/*/run_llm_baseline.py`, `persona_relation/llm_baseline_clf.py` |

## Installation

Python 3.10 or newer.

```bash
git clone https://github.com/ChenLyu416/ConVAWG.git
cd ConVAWG
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # then fill in the API keys you need and export them
```

API backends (OpenAI, DeepSeek, Gemini) read keys from the environment.
Open-weight models are served by a separately launched vLLM server
(OpenAI-compatible; default `http://localhost:8000/v1`, override with
`VLLM_BASE_URL` or the `base_url` field of a backend config):

```bash
python -m vllm.entrypoints.openai.api_server --model <model> --port 8000
```

Backend configs live in `RAG_kb_construct/configs/` (one YAML per model);
judge configs in `LLM_judge/configs/`.

## Running the pipeline

```bash
python run_batch_pipeline.py                         # uses pipeline_config.yaml
python run_batch_pipeline.py --config my_config.yaml
```

Each run creates `Generation_Results/run_<timestamp>_<pid>/` with a copy of the
config, a log, and one sub-directory per stage. A stage is skipped when its
output directory already contains JSON files; to continue an interrupted run,
point the stage directories in the config at the existing run directory
(absolute paths). Delete a stage's output directory to regenerate it.

Two things are needed before the default configuration runs end to end:

1. **Knowledge base for Stage 1.** Rebuild it from the public DHR library
   ([instructions](RAG_kb_construct/KB_homicide/README.md)), or set
   `stage_1_persona_generation.zero_shot: true` to skip retrieval.
2. **Steering assets for stage 6.** These are withheld (see below), so stage 6
   is disabled by default and stage 7 reads the stage 5 output. With the assets,
   set `global.enable_stage_6: true`, fill in `stage_6_toxic_injection`, and
   point `stage_7_convert_dialogues.input_dir` at the stage 6 output.

The 200 persona seeds used in the paper are in
`persona_generate/VAWG_sampled_200_personas.csv`.

### Ablations

| Condition | How to run |
|---|---|
| w/o Style Notes | `stage_5_roleplay_dialogue.bare_prompt: true` |
| w/o Steering | `global.enable_stage_6: false` |
| Backbone | set the `model` key of stages 1–5 to `deepseek_v4_flash`, `gemini_flash_lite`, `qwen35_35b` or `llama3_8b` (configs in `RAG_kb_construct/configs/`) |

## Evaluation

```bash
# Six-metric LLM judge (coherence, humanlikeness, persona consistency, toxicity realism;
# crime fidelity, scenario realism), 1–5 scale
python LLM_judge/run_judge.py --input data/ConVAWG/dialogues --config LLM_judge/configs/judge_config.yaml
python LLM_judge/run_judge.py --input <file.json> --dry-run          # inspect prompts only
python LLM_judge/run_multi_judge_parallel.py --input data/ConVAWG/dialogues

# Automatic scorers
python LLM_judge/run_detoxify.py --help
```

See [`LLM_judge/README.md`](LLM_judge/README.md) for configuration and output
format, and [`analysis/README.md`](analysis/README.md) for the scripts behind
the results tables, significance tests and agreement statistics.

### Baselines

```bash
cd Experiments/Baselines
python baseline_generate.py --model gpt5.2 --persona-ids $(python -c "import json;print(*json.load(open('matched_50_personas.json'))['all_50'])")
python frameworks/prepare_shared_inputs.py --help     # shared scenario inputs for the framework baselines
python frameworks/run_diasynth.py --help              # likewise run_syndg.py, run_spasm.py
```

`matched_50_personas.json` lists the 50 personas on which ConVAWG and all
baselines are compared in the paper.

### Downstream tasks

Each directory under `Experiments/Downstream_tasks/` has a `build_dataset.py`,
training / evaluation scripts and a run script (`run_experiments.sh`, or
`run_all_baselines.sh` for `persona_relation/`):
`toxic_classification/` (toxic behaviour classification),
`persona_relation/` (perpetrator–victim relationship classification),
`escalation_forecasting/`, `cga_transfer/` (zero-shot transfer to
Conversations Gone Awry) and `synthetic_transfer/` (MentalManip transfer and
augmentation). They read the dataset from `data/ConVAWG/` by default
([layout](data/README.md)).

## Dataset

The dataset is released for non-commercial research under a data-use agreement
that prohibits harassment, profiling of real individuals, training systems
primarily intended to generate abusive content, and application to real-world
cases. Read the [agreement](DATA_USE_AGREEMENT.md) before accepting it.

**Access:** complete the agreement form on the dataset page,
https://huggingface.co/datasets/ChenL49/ConVAWG. Submitting the form
constitutes acceptance, and access is granted immediately. Then:

```bash
huggingface-cli login
huggingface-cli download ChenL49/ConVAWG --repo-type dataset --local-dir data/ConVAWG
```

The dialogue files are then in `data/ConVAWG/dialogues/`. The release also
contains the plans behind each dialogue, in the formats the pipeline stages
exchange: `scenarios/`, `event_graphs/`, `expanded_event_graphs/` and
`chat_scripts/` ([details](data/README.md)).

## Steering vectors and toxification prompts

The calibrated steering configuration (vectors, steered model, layer and
coefficients) and the Stage 4 prompts are not included, because together they
make an open-weight model a controllable generator of abusive text.
`toxic_steering/` contains the orchestration code and documents the expected
asset formats. Requests that state the research purpose and the safeguards in
place can be sent to the authors.

## Citation

```bibtex
@misc{lyu2026convawg,
  title         = {{ConVAWG}: A Retrieval-Grounded Framework for Controlled Synthetic Dialogue Generation in Violence Against Women and Girls},
  author        = {Lyu, Chen and Tan, Xingwei and Cullen, Simon and Wilson, Shelley and Arthurs, Lois and Jhumka, Arshad and Pergola, Gabriele},
  year          = {2026},
  eprint        = {2608.11200},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2608.11200},
  note          = {To appear at EACL 2027}
}
```

## Contact

Chen Lyu (`chen.lyu@warwick.ac.uk`), University of Warwick.

## Licence

Code: [MIT](LICENSE). Dataset: [ConVAWG Data Use Agreement](DATA_USE_AGREEMENT.md).
The CPS definitions in `docs/` summarise public Crown Prosecution Service
guidance; Domestic Homicide Reviews are published by the UK Home Office and
are not redistributed here.
