# Escalation Forecasting

Conversation-derailment-style forecasting on ConVAWG: given the chronological
history of a scenario's online conversations, predict how the **next**
perpetrator–victim conversation will escalate — *before seeing it*.

ConVAWG scenarios are multi-session and follow the abuse-cycle rubric
(Appendix B.4 of the paper), so trajectories are non-monotonic — models must
distinguish genuine de-escalation from the reconciliation ("honeymoon") phase.

## Task definitions

One example per labelled perpetrator–victim event with ≥2 labelled
predecessors (dynamic prefix evaluation, as in the CGA/CRAFT protocol). The
input is the raw text of **all** preceding conversations (including
third-party/support conversations); history events carry **no** escalation
annotations.

| Task | Label | Type |
|------|-------|------|
| `level` | escalation level 0–4 of the next conversation | ordinal 5-class |
| `jump`  | next level > current level (imminent escalation) | binary (~18% positive) |

Escalation rubric: 0 normal, 1 emerging control, 2 manipulation/monitoring,
3 coercive intimidation, 4 severe threats or violent escalation.

Dataset (scenario-level 70/15/15 split, seed 42, from
`data/ConVAWG/dialogues`):
train 3,329 / dev 673 / test 716 examples over 140/30/30 scenarios.

## Systems

**Fine-tuned encoders** (`train_forecaster.py`) — tokenizer truncates from
the *left* so the most recent conversations are always kept:

- `microsoft/deberta-v3-base` — short-context encoder (512 tokens)
- `allenai/longformer-base-4096` — sparse-attention long-context (4,096 tokens)
- `answerdotai/ModernBERT-base` — modern long-context encoder (4,096 tokens here)

**LLM baselines** (`run_llm_baseline.py`), following 2025 forecasting work:

- `--method direct` — zero-shot next-level prediction from the history
- `--method generative` — generation-based forecasting (Forecasting
  Conversation Derailments Through Generation, INLG 2025): sample N=5
  plausible next conversations at T=0.9, rate each at T=0, take the median
  rating as consensus

**Trivial baselines** (computed by `evaluate_forecasts.py`, always report
these): *persistence* (copy current level), *majority*, *markov1* (argmax of
the train transition matrix). Persistence reaches ~0.70 accuracy / 0.68 QWK —
that is the bar to beat.

## Metrics

- `level`: accuracy, macro-F1, MAE, quadratic-weighted kappa (QWK)
- `jump`: F1 / precision / recall (+ AUPRC/AUROC for encoders); for
  level-task systems, jump is derived as `pred > cur_level`
- **Early-warning analysis** (per test scenario): detection rate of the first
  severe (level-4) event, mean lead time in events and in days, false-alert
  rate on scenarios that never reach level 4

## Running

```bash
python build_dataset.py                       # → data/

# encoders
python train_forecaster.py --task level --gpu 0
python train_forecaster.py --task jump  --gpu 0

# LLM baselines (Qwen3.5 needs a running vLLM server)
python run_llm_baseline.py --config configs/llm_baseline_qwen_vllm.yaml --method direct
python run_llm_baseline.py --config configs/llm_baseline_qwen_vllm.yaml --method generative
python run_llm_baseline.py --config configs/llm_baseline_gpt52.yaml --method direct
python run_llm_baseline.py --config configs/llm_baseline_deepseek.yaml --method direct

# aggregate everything + baselines + early-warning table
python evaluate_forecasts.py --pred results/*/predictions.jsonl \
    results_llm_baseline/*/*/predictions.jsonl
```

`bash run_experiments.sh` runs the full encoder grid sequentially.

Note on cost: `direct` sends the full rendered history (~10k tokens/example ×
716 test examples) per API model; `generative` multiplies that by ~6 calls.
Run API configs deliberately.

## Note on label leakage

Escalation labels conditioned the generation prompts, so *classifying* a
visible dialogue's level partly measures prompt-following. Forecasting does
not have this shortcut: the model never sees the target conversation, and the
persistence / markov1 baselines quantify how much of the trajectory is
predictable from label dynamics alone.
