# LLM-as-Judge — VAWG Dialogue Evaluator

Evaluates auto-generated VAWG (Violence Against Women and Girls) dialogues using
an LLM judge. Each dialogue file (`*_dialogues.json`) is scored on **six metrics**:
four per-dialogue and two global.

---

## Metrics

### Per-Dialogue (one rating per dialogue)

| Metric | Description |
|---|---|
| **Coherence** | Turn-level logical consistency and conversational flow |
| **Humanlikeness** | Naturalness and conversational authenticity |
| **Persona Consistency** | Stability of character traits and speaking style |
| **Toxicity Realism** | Realism of toxic/abusive language at appropriate escalation points |

### Global (single rating over the full dialogue sequence)

| Metric | Description |
|---|---|
| **Crime Fidelity** | Domain-level realism of VAWG dynamics |
| **Scenario Realism** | Faithful and plausible alignment with the given scenario |

All ratings are on a **1–5 scale** (Toxicity Realism also allows **N/A** when no
toxic language is present).

---

## Setup

Install the repository requirements (`pip install -r requirements.txt`).
The Qwen3.5 judge needs a separately launched vLLM server.

---

## Configuration

One config per judge used in the paper, all at temperature 0.1:

| Judge | Config | Backend |
|---|---|---|
| GPT-5.2 | `configs/judge_config.yaml` | `openai` (`OPENAI_API_KEY`) |
| DeepSeek-V4-Flash | `configs/judge_config_deepseek.yaml` | `deepseek` (`DEEPSEEK_API_KEY`) |
| Gemini-3.1-Flash-Lite | `configs/judge_config_gemini.yaml` | `gemini` (`GEMINI_API_KEY`) |
| Qwen3.5-35B-A3B | `configs/judge_config_qwen_vllm.yaml` | `vllm` server at `http://localhost:8003/v1` |

Each config also toggles the automatic scorers that run alongside the judge:
Detoxify (the toxicity score reported in the paper), GPT-2 perplexity and
lexical diversity.

---

## Usage

### Evaluate a single file

```bash
# From inside LLM_judge/
python run_judge.py --input ../data/ConVAWG/dialogues/18015_dialogues.json

# From the repository root
python LLM_judge/run_judge.py --input data/ConVAWG/dialogues/18015_dialogues.json
```

### Batch evaluate a directory

```bash
# From the project root
python LLM_judge/run_judge.py --input data/ConVAWG/dialogues/
```

### Use another judge

```bash
python run_judge.py --input path/to/file.json --config configs/judge_config_deepseek.yaml
```

### Dry-run (inspect prompts without calling the LLM)

```bash
python run_judge.py --input path/to/file.json --dry-run
```

### Full options

```
--input   / -i   File, directory, or glob pattern
--config  / -c   Config YAML path (default: configs/judge_config.yaml)
--output-dir / -o   Output directory (default: from config)
--backend        Backend override (openai, deepseek, gemini, vllm)
--base-url       Server URL override (vllm backend)
--model   / -m   LLM model override
--api-key        API key override
--keep-llm-errors  Keep files whose ratings contain LLM errors instead of failing
--dry-run        Print prompts, do not call LLM
--verbose / -v   DEBUG logging
```

---

## Output format

Results are saved to `results/<stem>_judge_results.json`:

```json
{
  "scenario_id": "18015",
  "victim_name": "Poppy",
  "perpetrator_name": "Florence",
  "num_dialogues_evaluated": 27,

  "per_dialogue_ratings": [
    {
      "event_id": "E1.1.pre",
      "dialogue_index": 0,
      "timestamp": "2025-12-06T19:18:44",
      "escalation_level": 1,
      "crime_type": null,
      "dialogue_source": "enforced_dialogue",
      "ratings": {
        "coherence":           {"rating": "5",      "explanation": "…"},
        "humanlikeness":       {"rating": "4",      "explanation": "…"},
        "persona_consistency": {"rating": "3",      "explanation": "…"},
        "toxicity_realism":    {"rating": "N/A",    "explanation": ""}
      }
    }
  ],

  "global_ratings": {
    "crime_fidelity":   {"rating": "4", "explanation": "…"},
    "scenario_realism": {"rating": "3", "explanation": "…"}
  },

  "summary": {
    "coherence":           {"modal_rating": "5", "average_score": 4.41, "average_label": "4", "counts": {"5": 20, "4": 7}},
    "humanlikeness":       {"modal_rating": "5", "average_score": 4.52, "average_label": "5", "counts": {"5": 22, "4": 5}},
    "persona_consistency": {"modal_rating": "3", "average_score": 3.37, "average_label": "3", "counts": {"4": 10, "3": 17}},
    "toxicity_realism":    {"modal_rating": "4", "average_score": 4.10, "average_label": "4", "counts": {"5": 15, "3": 5, "N/A": 7}},
    "crime_fidelity":      {"rating": "4", "score": 4},
    "scenario_realism":    {"rating": "3", "score": 3}
  }
}
```

---

## Module structure

```
LLM_judge/
├── configs/                     # one config per judge (GPT-5.2, DeepSeek, Gemini, Qwen3.5)
├── llm_backend.py               # OpenAI / DeepSeek / Gemini / vLLM-server backends
├── prompts.py                   # judge system prompt and the two rubrics
├── judge.py                     # core DialogueJudge class
├── run_judge.py                 # CLI entry point
├── run_multi_judge_parallel.py  # run the API judges side by side
├── detoxify_scorer.py, run_detoxify.py   # Detoxify toxicity
├── perplexity_scorer.py, diversity_scorer.py
└── README.md
```

---

## Programmatic usage

```python
import yaml
from LLM_judge.judge import DialogueJudge

cfg = yaml.safe_load(open("LLM_judge/configs/judge_config.yaml"))
judge = DialogueJudge(cfg)

result = judge.evaluate_file(
    "data/ConVAWG/dialogues/18015_dialogues.json"
)

# Per-dialogue ratings
for d in result["per_dialogue_ratings"]:
    print(d["event_id"], d["ratings"]["coherence"]["rating"])

# Global ratings
print(result["global_ratings"]["crime_fidelity"]["rating"])
```
