# Targeted toxicity injection (paper Stage 4, pipeline stage 6)

This directory contains the orchestration code for the final generation stage:
a sampled subset of perpetrator *escalation-act* messages is rewritten by a
chat model steered with Contrastive Activation Addition (CAA), at a strength
matched to the escalation level of the source event.

```
levelled_rewrite/
├── run_levelled_rewrite.py   # CLI: label → sample → rewrite → save
├── message_labeler.py        # LLM labeling of perpetrator messages (12 act labels)
├── steered_rewriter.py       # forward-hook CAA steering + batched rewriting
├── config.yaml               # sampling ratios, labeling backend, asset paths
└── prompts.example.yaml      # schema of the (withheld) rewriting prompts
```

## What is and is not released

As stated in the paper's Ethical Considerations, the **calibrated steering
configuration** and the **toxification prompts** are not part of the public
release, because together they turn an open-weight chat model into a
controllable generator of abusive text. Concretely, this repository does not
contain:

- the steering vectors;
- the calibration (steered model, steering layer, per-level coefficients) and
  the pair-construction, vector-extraction and calibration procedures;
- the Stage 4 system prompt, rewrite template and per-level style guidelines.

These may be provided to researchers upon justified request; contact the
authors (see the top-level README).

Everything else in the pipeline runs without them. With stage 6 disabled
(`global.enable_stage_6: false` in `pipeline_config.yaml`) the pipeline
produces the non-steered dialogues, i.e. the "w/o Steering" condition of the
paper's ablation.

## Method in brief

At inference time a single forward hook on transformer layer `l*` adds a fixed
direction to the hidden state of every token position:

```
h' = h + alpha * v_hat          (||v_hat||_2 = 1, alpha >= 0)
```

`v_hat` is an L2-normalised steering vector estimated from toxic/clean
contrastive pairs; `alpha` is the only magnitude control and is calibrated per
model into three operating points. Source escalation levels 2, 3 and 4 map to
steering levels 1, 2 and 3. Only messages labelled as escalation acts are
eligible, and only a per-level fraction of them (`injection_ratio`) is
rewritten, so toxicity appears where the scenario makes it plausible. Victim
messages are never modified.

## Expected asset formats

| Asset | Format |
|---|---|
| Steering vector | `<vector_dir>/layer_{N}.pt` — a 1-D `torch` tensor of the model's hidden size, L2-normalised |
| Calibration | JSON: `{"layer": N, "levels": {"1": a1, "2": a2, "3": a3}}` with `a1 <= a2 <= a3` |
| Prompts | YAML following `levelled_rewrite/prompts.example.yaml` |

## Running

Through the pipeline: set the four asset paths under `stage_6_toxic_injection`
in `pipeline_config.yaml` and enable stage 6. Standalone:

```bash
cd toxic_steering
python -m levelled_rewrite.run_levelled_rewrite \
    --input  ../Generation_Results/roleplay_dialogues \
    --output ../Generation_Results/caa_rewritten_dialogues \
    --model <hf-model-id> \
    --vector-dir <dir-with-layer_N.pt> \
    --calibration <calibration.json> \
    --prompts <prompts.yaml>
```

Message labeling uses an OpenAI-compatible endpoint (`OPENAI_API_KEY`, or a
local vLLM server with `--label-backend vllm`). The steered model itself is
loaded in-process with `transformers`, because steering needs direct access to
hidden states.
