# Analysis scripts

Scripts behind the paper's baseline comparison and agreement statistics. They
read LLM-judge result files (`<persona_id>_dialogues_judge_results.json`, as
written by `LLM_judge/run_judge.py`) and write CSV / LaTeX / JSON reports to
`analysis/outputs/` (git-ignored).

| Script | Produces |
|---|---|
| `compute_main_results_table.py` | per-metric cells of Table 1 and of the per-judge appendix table: persona-level macro-averages over the four judges, with Avg-D and Avg-G |
| `compute_significance_tests.py` | the six-metric Avg and the significance markers of Table 1; Friedman omnibus test, Holm-corrected paired Wilcoxon tests, bootstrap CIs, effect sizes, per-judge robustness |
| `compute_judge_robustness_and_distributions.py` | leave-one-judge-out comparison and the rating-distribution figure |
| `compute_scale_comparison_table.py` | scale comparison (dialogues per scenario, turns and words per dialogue) |
| `human_annotation/gwet_ac2.py` | Gwet's AC2 for inter-annotator agreement and human–LLM alignment |

## Expected inputs

The unit of analysis is the persona; the comparison uses the 50 matched
personas in `Experiments/Baselines/matched_50_personas.json`.

```
Generation_Results/8_judge_results/
    convawg_gpt52/  convawg_deepseek/  convawg_gemini/  convawg_qwen/     # ConVAWG, one directory per judge
Experiments/Baselines/results/<setting>/
    judge_results/  judge_results_deepseek/  judge_results_gemini/  judge_results_qwen/
```

`<setting>` is one of the baseline directory names listed in `SYSTEMS` at the
top of `compute_main_results_table.py` (for example `gpt5.2` or
`diasynth_qwen3.5-35b`). Adjust the path constants at the top of that file if
your results are laid out differently.

`gwet_ac2.py` expects per-annotator rating directories under
`analysis/human_annotation/annotation_results/` and LLM-judge files under
`analysis/human_annotation/llm_result/` (one judge at a time: pass each
judge's directory with `--llm-dir`). The human ratings are not distributed
with this repository.

```bash
python analysis/compute_main_results_table.py
python analysis/compute_significance_tests.py
python analysis/compute_judge_robustness_and_distributions.py
python analysis/compute_scale_comparison_table.py
python analysis/human_annotation/gwet_ac2.py --help
```
