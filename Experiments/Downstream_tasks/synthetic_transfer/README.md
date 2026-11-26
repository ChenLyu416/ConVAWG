# Synthetic-to-Real Transfer (ConVAWG ↔ MentalManip)

Does training on ConVAWG's *synthetic* dialogues produce detectors that work
on *real-derived* data? The protocol follows the cross-dataset evaluation of
Song et al. (2025).

## Shared task

Dialogue-level binary detection of **manipulative / abusive conversation**.

- **ConVAWG** (synthetic): one example per ConVAWG dialogue.
  Positive = escalation ≥ 2 (manipulation-and-monitoring onwards);
  negative = escalation ≤ 1 plus third-party support conversations (hard
  negatives that *discuss* abuse without enacting it).
  Scenario-level 70/15/15 split (seed 42):
  train 4,340 / dev 903 / test 916 (75% positive).
- **MentalManip** (real-derived; Wang et al., ACL 2024): 2,915
  consensus-labelled movie dialogues (`audreyeleven/MentalManip`,
  config `mentalmanip_con`), stratified 70/15/15:
  train 2,040 / dev 436 / test 439 (69% positive).

Format alignment: speakers in **both** datasets are anonymised to
`Person1/Person2/…` so the transfer measures content generalisation, not
formatting artifacts. Label priors are naturally close (75% vs 69%).

## Experiment matrix (`train_transfer.py --matrix`)

Every run trains one `microsoft/deberta-v3-base` and evaluates on **both**
test sets (macro-F1 and AUROC):

| Spec | Paper row |
|------|-----------|
| `mentalmanip` | MentalManip (MM), full real training data |
| `convawg+mentalmanip` | ConVAWG → MM (sequential training) |
| `mentalmanip:500` | MM₅₀₀, low-resource real-only control |
| `convawg+mentalmanip:500` | ConVAWG → MM₅₀₀ |

The per-escalation-level positive rates of the MentalManip-only detector on
ConVAWG are computed from its `predictions_*.jsonl`.

## Running

```bash
python build_dataset.py                 # → data/convawg, data/mentalmanip
python train_transfer.py --matrix --gpu 0
python train_transfer.py --aggregate_only            # summary table
```

`bash run_experiments.sh` runs the same grid.

## Interpretation caveats

- MentalManip's positives are *mental manipulation* (gaslighting,
  guilt-tripping…); ConVAWG's positives include overt threats/abuse that
  MentalManip may treat differently. Perfect transfer is not expected — report
  the gap honestly and analyse per-class errors (`predictions_*.jsonl`).
- Both directions matter: real→synthetic transfer tells you whether ConVAWG
  looks distributionally like real abuse data to a trained detector.
