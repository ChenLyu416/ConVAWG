# CGA Cross-Corpus Forecasting Transfer (ConVAWG → Conversations Gone Awry)

Does a forecaster trained on ConVAWG's *synthetic* escalation trajectories
transfer zero-shot to derailment forecasting on *real* public conversations?
The test set is entirely external — the Conversations Gone Awry (CGA) corpora
of real Wikipedia talk-page and Reddit ChangeMyView discussions — and no CGA
data is used for training the transferred model.

## Task mapping

Train: ConVAWG **`level`** forecasting (`../escalation_forecasting`, five-class —
"what is the escalation level of the next conversation?").
Test (zero-shot): CGA **derailment** forecasting (binary — "will this
conversation end in a personal attack?"). The derailment score is
**P(next level ≥ 3)**, i.e. the summed probability of levels 3 and 4
(`--score_classes 3 4`, the default), and the reported metric is AUROC.

| | ConVAWG `level` | CGA derailment |
|---|---|---|
| unit | multi-session scenario (many conversations) | single comment thread |
| input | history of whole conversations | history of comments |
| positive | next conversation reaches coercive intimidation or above | conversation ends in a personal attack (wiki) / moderator-removed comment (cmv) |

**Semantics caveat:** these are related but distinct constructs: imminent
worsening of an ongoing abusive dynamic versus a first overtly antisocial
comment. Both corpora are label-balanced, so chance AUROC is 0.5.

## Data

Built by `build_dataset.py` directly from the ConvoKit file server (no
`convokit` install — the corpora's `conversations.json` + `utterances.jsonl`
are parsed with the standard library, keeping the pinned environment intact).
Official corpus splits are kept (val → dev):

| corpus | source | train | dev | test | test balance (0/1) | mean history comments |
|---|---|---|---|---|---|---|
| `wiki` | Wikipedia talk pages (Zhang et al., ACL 2018) | 2,503 | 838 | 837 | 417/420 | 5.1 |
| `cmv` | Reddit ChangeMyView (Chang & Danescu-Niculescu-Mizil, EMNLP 2019) | 3,658 | 1,232 | 1,220 | 598/622 | 5.7 |

Protocol (standard CGA/CRAFT forecasting setup):

- The attack / removed comment is **never** in the input. Wiki: history is cut
  strictly before the first comment flagged `comment_has_personal_attack`
  (in 2,088/2,094 positives that is the final comment). CMV: no per-comment
  flag exists; the removed comment is the final one and is dropped.
- The final comment is also dropped from civil conversations in both corpora,
  so history length carries no label signal.
- Wikipedia section headers are excluded; conversations with <2 remaining
  history comments are dropped (wiki: 0 dropped; cmv: 732 dropped).
- Comments are rendered as the per-comment analogue of ConVAWG's history
  blocks — `[comment k | ts | speaker]` + text, blocks joined by a blank
  line, tokenizer truncating from the **left** (recency window), exactly as
  in `train_forecaster.py`.

The JSONL schema is field-compatible with `escalation_forecasting`
(`history`, `jump`, placeholder `cur_level`/`next_level`), so the existing
training and evaluation code paths work unchanged on this data.

## Running

```bash
python build_dataset.py                       # → data/wiki, data/cmv (downloads ~100 MB once)

# Zero-shot transfer: ConVAWG-trained level forecaster on the CGA test sets
# (checkpoints appear under ../escalation_forecasting/results/<model>_level/checkpoint-*)
python evaluate_cga.py \
    --checkpoint ../escalation_forecasting/results/microsoft_deberta-v3-base_level/checkpoint-<N> \
    --data_dir data/wiki
python evaluate_cga.py --checkpoint <same> --data_dir data/cmv

# Smoke test (CPU, seconds):
python evaluate_cga.py --checkpoint <ckpt> --data_dir data/wiki --limit 8 --cpu
```

`bash run_experiments.sh` runs the full zero-shot grid (DeBERTa-v3-base,
Longformer-4096, ModernBERT-base on both corpora) once the level checkpoints
exist.

## Metrics

Positive class = derailment. The paper reports **AUROC** (threshold-free).
`results.json` also contains accuracy, F1, precision and recall at the 0.5
threshold and at a CGA-dev-calibrated threshold, and AUPRC. Predictions go to
`results/<run>/predictions.jsonl`.
