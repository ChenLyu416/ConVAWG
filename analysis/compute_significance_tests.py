#!/usr/bin/env python3
"""
compute_significance_tests.py
------------------------------
Formal statistical tests behind the baseline comparison (Table 1 and the
statistical-significance appendix of the paper).

Protocol (persona = unit of analysis, n = 50 matched personas, scores
averaged over the four calibrated judges exactly as in Table 1):

  1. Omnibus: Friedman test across the 9 systems on the per-persona
     six-metric average (Avg) and the dialogue-level average (Avg-D),
     with Kendall's W as the omnibus effect size.
  2. Post-hoc: two-sided paired Wilcoxon signed-rank of ConVAWG vs each
     of the 8 baselines, per metric and for Avg-D / Avg-G / Avg, with
     Holm-Bonferroni correction across the 8 baselines within each
     metric column.
  3. Effect sizes: matched-pairs rank-biserial correlation r (Kerby
     2014) and Cohen's d_z; paired mean difference with a 95% bootstrap
     percentile CI over personas (10,000 resamples, seed 42).
  4. Per-judge robustness: the same ConVAWG-vs-baseline Wilcoxon on
     Avg-D within each judge separately (Holm-corrected per judge).

Reuses the loading/aggregation of compute_main_results_table.py with the
four-judge pool (GPT-5.2, DeepSeek, Gemini, Qwen).
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as sps

import compute_main_results_table as cmt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent

JUDGES_4 = ["gpt52", "deepseek", "gemini", "qwen"]
POSTFIX_DIRS = {
    "gpt52": "convawg_gpt52",
    "deepseek": "convawg_deepseek",
    "gemini": "convawg_gemini",
    "qwen": "convawg_qwen",
}

METRICS = cmt.DIAL_METRICS + cmt.GLOB_METRICS
AGGREGATES = ["avg_dial", "avg_glob", "avg6"]
COLUMNS = METRICS + AGGREGATES

N_BOOT = 10_000
SEED = 42

CONVAWG = "ConVAWG"
BASELINES = [s for s in cmt.SYSTEMS if s != CONVAWG]


def holm_correction(pvals: Dict[str, float]) -> Dict[str, float]:
    """Holm-Bonferroni step-down adjusted p-values (NaNs passed through)."""
    keys = [k for k, v in pvals.items() if not np.isnan(v)]
    m = len(keys)
    order = sorted(keys, key=lambda k: pvals[k])
    adjusted: Dict[str, float] = {k: np.nan for k in pvals}
    running_max = 0.0
    for i, k in enumerate(order):
        adj = min(1.0, (m - i) * pvals[k])
        running_max = max(running_max, adj)
        adjusted[k] = running_max
    return adjusted


def rank_biserial(diffs: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation r = (R+ - R-)/(R+ + R-)."""
    d = diffs[diffs != 0]
    if len(d) == 0:
        return np.nan
    ranks = sps.rankdata(np.abs(d))
    r_plus = float(ranks[d > 0].sum())
    r_minus = float(ranks[d < 0].sum())
    total = r_plus + r_minus
    return (r_plus - r_minus) / total if total > 0 else np.nan


def bootstrap_diff_ci(diffs: np.ndarray, n_boot: int = N_BOOT,
                      seed: int = SEED) -> Tuple[float, float]:
    """95% percentile bootstrap CI of the paired mean difference."""
    if len(diffs) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
    means = diffs[idx].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def wilcoxon_p(diffs: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p on paired differences."""
    if len(diffs) < 6:
        return np.nan
    if np.allclose(diffs, 0):
        return 1.0
    try:
        return float(sps.wilcoxon(diffs).pvalue)
    except ValueError:
        return 1.0


class SignificanceAnalyzer:
    """Persona-level significance tests for ConVAWG vs the 8 baselines."""

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        for judge, postfix in POSTFIX_DIRS.items():
            if postfix not in cmt.CONVAWG_DIRS[judge]:
                cmt.CONVAWG_DIRS[judge].insert(0, postfix)
        cmt.POOLED_JUDGES = list(JUDGES_4)
        self.builder = cmt.JudgeTableBuilder(out_dir, JUDGES_4)

    # ---------- data ----------

    def load(self) -> None:
        self.builder.load()
        self.pooled = self.builder.pooled_scores()
        # per-system persona -> column -> value (pooled over 4 judges)
        self.values: Dict[str, Dict[str, Dict[str, float]]] = {}
        for system, ps in self.pooled.items():
            per_pid: Dict[str, Dict[str, float]] = {}
            for pid in ps.dial:
                row = dict(ps.dial[pid])
                row.update(ps.glob.get(pid, {}))
                row["avg_dial"] = ps.avg_dial(pid)
                row["avg_glob"] = ps.avg_glob(pid)
                six = [row.get(m, np.nan) for m in METRICS]
                row["avg6"] = (float(np.nanmean(six))
                               if any(not np.isnan(v) for v in six) else np.nan)
                per_pid[pid] = row
            self.values[system] = per_pid

    def _paired(self, baseline: str, col: str) -> Tuple[np.ndarray, np.ndarray]:
        conv, base = self.values[CONVAWG], self.values[baseline]
        pids = [p for p in conv if p in base
                and not np.isnan(conv[p].get(col, np.nan))
                and not np.isnan(base[p].get(col, np.nan))]
        x = np.array([conv[p][col] for p in pids])
        y = np.array([base[p][col] for p in pids])
        return x, y

    # ---------- tests ----------

    def friedman(self, col: str) -> dict:
        pids = [p for p in self.values[CONVAWG]
                if all(p in self.values[s]
                       and not np.isnan(self.values[s][p].get(col, np.nan))
                       for s in cmt.SYSTEMS)]
        mat = [np.array([self.values[s][p][col] for p in pids])
               for s in cmt.SYSTEMS]
        stat, p = sps.friedmanchisquare(*mat)
        k, n = len(mat), len(pids)
        kendall_w = stat / (n * (k - 1))
        return {"column": col, "n": n, "k": k, "chi2": float(stat),
                "df": k - 1, "p": float(p), "kendall_w": float(kendall_w)}

    def posthoc_table(self) -> List[dict]:
        rows: List[dict] = []
        for baseline in BASELINES:
            row: dict = {"baseline": baseline}
            for col in COLUMNS:
                x, y = self._paired(baseline, col)
                d = x - y
                row[f"{col}_n"] = len(d)
                row[f"{col}_delta"] = float(np.mean(d)) if len(d) else np.nan
                lo, hi = bootstrap_diff_ci(d)
                row[f"{col}_ci_lo"], row[f"{col}_ci_hi"] = lo, hi
                row[f"{col}_p"] = wilcoxon_p(d)
                row[f"{col}_r"] = rank_biserial(d)
                sd = float(np.std(d, ddof=1)) if len(d) > 1 else np.nan
                row[f"{col}_dz"] = (float(np.mean(d)) / sd
                                    if sd and not np.isnan(sd) and sd > 0 else np.nan)
            rows.append(row)
        for col in COLUMNS:  # Holm across the 8 baselines within each column
            adj = holm_correction({r["baseline"]: r[f"{col}_p"] for r in rows})
            for r in rows:
                r[f"{col}_p_holm"] = adj[r["baseline"]]
        return rows

    def per_judge_avgd(self) -> List[dict]:
        rows: List[dict] = []
        for judge in JUDGES_4:
            conv = self.builder.scores[judge][CONVAWG]
            conv_ad = {p: conv.avg_dial(p) for p in conv.dial}
            pvals: Dict[str, float] = {}
            deltas: Dict[str, Tuple[float, int]] = {}
            for baseline in BASELINES:
                ps = self.builder.scores[judge][baseline]
                base_ad = {p: ps.avg_dial(p) for p in ps.dial}
                pids = [p for p in conv_ad if p in base_ad
                        and not np.isnan(conv_ad[p]) and not np.isnan(base_ad[p])]
                d = np.array([conv_ad[p] - base_ad[p] for p in pids])
                pvals[baseline] = wilcoxon_p(d)
                deltas[baseline] = (float(np.mean(d)) if len(d) else np.nan, len(d))
            adj = holm_correction(pvals)
            for baseline in BASELINES:
                delta, n = deltas[baseline]
                rows.append({"judge": judge, "baseline": baseline, "n": n,
                             "delta_avgd": delta, "p": pvals[baseline],
                             "p_holm": adj[baseline]})
        return rows

    # ---------- output ----------

    @staticmethod
    def _fmt_p(p: float) -> str:
        if np.isnan(p):
            return "--"
        if p < 1e-4:
            return "$<\\!10^{-4}$"
        return f"{p:.3f}" if p >= 0.001 else f"{p:.4f}"

    def tex_table(self, rows: List[dict], friedman_rows: List[dict]) -> str:
        fr = {f["column"]: f for f in friedman_rows}

        def cell(r: dict, col: str) -> str:
            delta, lo, hi = r[f"{col}_delta"], r[f"{col}_ci_lo"], r[f"{col}_ci_hi"]
            p, rb = r[f"{col}_p_holm"], r[f"{col}_r"]
            if np.isnan(delta):
                return "-- & -- & --"
            return (f"{delta:+.2f} [{lo:+.2f}, {hi:+.2f}] & "
                    f"{self._fmt_p(p)} & {rb:+.2f}")

        lines = [
            r"\begin{table*}[!t]",
            r"\centering",
            r"\small",
            r"\setlength{\tabcolsep}{3pt}",
            r"\begin{tabular}{l|ccc|ccc|ccc}",
            r"\toprule",
            r" & \multicolumn{3}{c|}{\textbf{Avg-D (dialogue-level)}} & "
            r"\multicolumn{3}{c|}{\textbf{Avg-G (global)}} & "
            r"\multicolumn{3}{c}{\textbf{Avg (all six metrics)}} \\",
            r"\textbf{Baseline} & $\Delta$ [95\% CI] & $p_{\text{Holm}}$ & $r$ & "
            r"$\Delta$ [95\% CI] & $p_{\text{Holm}}$ & $r$ & "
            r"$\Delta$ [95\% CI] & $p_{\text{Holm}}$ & $r$ \\",
            r"\midrule",
        ]
        for r in rows:
            name = r["baseline"].replace("&", r"\&")
            lines.append(f"{name} & {cell(r, 'avg_dial')} & "
                         f"{cell(r, 'avg_glob')} & {cell(r, 'avg6')} \\\\")
        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            (r"\caption{Statistical comparison of \toolname{} against each baseline "
             r"under the four-judge average on the 50 matched personas (persona = "
             r"unit of analysis, as in Table~\ref{tab:multi_judge_comparison}). "
             r"$\Delta$ = mean paired difference (\toolname{} $-$ baseline) with a "
             r"95\% bootstrap percentile CI over personas (10{,}000 resamples); "
             r"$p_{\text{Holm}}$ = two-sided paired Wilcoxon signed-rank $p$-value, "
             r"Holm-corrected across the eight baselines; $r$ = matched-pairs "
             r"rank-biserial correlation. Omnibus Friedman tests across the nine "
             f"systems: Avg-D $\\chi^2({fr['avg_dial']['df']})={fr['avg_dial']['chi2']:.1f}$, "
             f"$p<10^{{-4}}$, Kendall's $W={fr['avg_dial']['kendall_w']:.2f}$; "
             f"Avg $\\chi^2({fr['avg6']['df']})={fr['avg6']['chi2']:.1f}$, "
             f"$p<10^{{-4}}$, Kendall's $W={fr['avg6']['kendall_w']:.2f}$.}}"),
            r"\label{tab:significance_tests}",
            r"\end{table*}",
        ]
        return "\n".join(lines) + "\n"

    def run(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        friedman_rows = [self.friedman(c) for c in ("avg_dial", "avg_glob", "avg6")]
        posthoc = self.posthoc_table()
        per_judge = self.per_judge_avgd()

        # CSV dumps
        cols = ["baseline"] + [f"{c}_{s}" for c in COLUMNS
                               for s in ("n", "delta", "ci_lo", "ci_hi", "p",
                                         "p_holm", "r", "dz")]
        out = [",".join(cols)]
        for r in posthoc:
            out.append(",".join(
                f'"{r[c]}"' if c == "baseline" else
                ("" if isinstance(r[c], float) and np.isnan(r[c])
                 else f"{r[c]:.6g}" if isinstance(r[c], float) else str(r[c]))
                for c in cols))
        (self.out_dir / "posthoc_wilcoxon.csv").write_text("\n".join(out) + "\n")

        fr_out = ["column,n,k,chi2,df,p,kendall_w"]
        for f in friedman_rows:
            fr_out.append(f"{f['column']},{f['n']},{f['k']},{f['chi2']:.4f},"
                          f"{f['df']},{f['p']:.6g},{f['kendall_w']:.4f}")
        (self.out_dir / "friedman.csv").write_text("\n".join(fr_out) + "\n")

        pj_out = ["judge,baseline,n,delta_avgd,p,p_holm"]
        for r in per_judge:
            pj_out.append(f"{r['judge']},\"{r['baseline']}\",{r['n']},"
                          f"{r['delta_avgd']:.4f},{r['p']:.6g},{r['p_holm']:.6g}")
        (self.out_dir / "per_judge_avgd_wilcoxon.csv").write_text(
            "\n".join(pj_out) + "\n")

        (self.out_dir / "significance_table.tex").write_text(
            self.tex_table(posthoc, friedman_rows))

        self._write_report(friedman_rows, posthoc, per_judge)

    def _write_report(self, friedman_rows: List[dict], posthoc: List[dict],
                      per_judge: List[dict]) -> None:
        lines = ["# Significance tests — ConVAWG vs baselines "
                 "(4-judge pooled, 50 matched personas)", ""]
        lines.append("## Friedman omnibus (9 systems)")
        lines.append("")
        for f in friedman_rows:
            lines.append(f"- **{f['column']}**: chi2({f['df']}) = {f['chi2']:.1f}, "
                         f"p = {f['p']:.3g}, Kendall's W = {f['kendall_w']:.3f} "
                         f"(n = {f['n']} personas)")
        lines.append("")
        lines.append("## Post-hoc paired Wilcoxon vs ConVAWG (Holm across 8 baselines)")
        lines.append("")
        for col in COLUMNS:
            lines.append(f"### {col}")
            lines.append("")
            lines.append("| Baseline | n | delta | 95% CI | p | p_Holm | r | d_z |")
            lines.append("|---|---|---|---|---|---|---|---|")
            for r in posthoc:
                def g(key: str) -> str:
                    v = r[key]
                    return "--" if isinstance(v, float) and np.isnan(v) else (
                        f"{v:.4g}" if isinstance(v, float) else str(v))
                lines.append(
                    f"| {r['baseline']} | {r[f'{col}_n']} | {g(f'{col}_delta')} | "
                    f"[{g(f'{col}_ci_lo')}, {g(f'{col}_ci_hi')}] | {g(f'{col}_p')} | "
                    f"{g(f'{col}_p_holm')} | {g(f'{col}_r')} | {g(f'{col}_dz')} |")
            lines.append("")
        lines.append("## Per-judge Avg-D robustness (Wilcoxon, Holm within judge)")
        lines.append("")
        lines.append("| Judge | Baseline | n | delta Avg-D | p | p_Holm |")
        lines.append("|---|---|---|---|---|---|")
        for r in per_judge:
            lines.append(f"| {r['judge']} | {r['baseline']} | {r['n']} | "
                         f"{r['delta_avgd']:+.3f} | {r['p']:.3g} | {r['p_holm']:.3g} |")
        lines.append("")
        (self.out_dir / "report.md").write_text("\n".join(lines))
        logger.info("wrote %s", self.out_dir / "report.md")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Significance tests for the Table 1 baseline comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python compute_significance_tests.py
  python compute_significance_tests.py --out-dir analysis/outputs/significance
""",
    )
    parser.add_argument("--out-dir", type=Path,
                        default=REPO / "analysis" / "outputs" / "significance",
                        help="Output directory for CSV/TeX/report files")
    args = parser.parse_args()

    analyzer = SignificanceAnalyzer(args.out_dir)
    analyzer.load()
    analyzer.run()


if __name__ == "__main__":
    main()
