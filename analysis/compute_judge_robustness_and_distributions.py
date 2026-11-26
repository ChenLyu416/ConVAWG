#!/usr/bin/env python3
"""
compute_judge_robustness_and_distributions.py
---------------------------------------------
Two robustness analyses of the baseline comparison (judge self-preference /
circularity, and ceiling-compressed ratings):

  1. Leave-one-judge-out (LOO) robustness: recompute the Table 1
     aggregation (persona = unit of analysis, n = 50 matched personas)
     with each judge held out in turn — in particular excluding the
     GPT-5.2 judge, which shares the generation backbone — and rerun
     the ConVAWG-vs-baseline Holm-corrected Wilcoxon tests on the
     3-judge pool.
  2. Score distributions: raw 1-5 rating histograms per system and
     metric, pooled over the four calibrated judges, quantifying the
     ceiling concentration (share of ratings at 5 / at >=4), plus a
     stacked-share figure for the appendix.

Reuses the loading/aggregation of compute_main_results_table.py with the
four-judge pool (GPT-5.2, DeepSeek, Gemini, Qwen).
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

CONVAWG = "ConVAWG"
BASELINES = [s for s in cmt.SYSTEMS if s != CONVAWG]
METRICS = cmt.DIAL_METRICS + cmt.GLOB_METRICS
RATINGS = [1, 2, 3, 4, 5]

METRIC_TITLES = {
    "coherence": "Coherence",
    "humanlikeness": "Humanlikeness",
    "persona_consistency": "Persona Consistency",
    "toxicity_realism": "Toxicity Realism",
    "crime_fidelity": "Crime Fidelity",
    "scenario_realism": "Scenario Realism",
}
SYSTEM_SHORT = {
    "ConVAWG": "ConVAWG",
    "GPT-5.2": "GPT-5.2",
    "DeepSeek-V4-Flash": "DeepSeek",
    "Gemini-3.1-Flash-Lite": "Gemini",
    "Qwen3.5-35B-A3B": "Qwen3.5",
    "LLaMA-3.1-70B-Instruct": "Llama-70B",
    "DiaSynth (Qwen3.5)": "DiaSynth",
    "SynDG (Qwen3.5)": "SynDG",
    "SPASM (Qwen3.5)": "SPASM",
}
# single-hue sequential ramp, light (rating 1) -> dark (rating 5)
RATING_COLORS = ["#dbe9f6", "#a8cbe8", "#6da6d4", "#3d7fbb", "#175a9d"]


def holm_correction(pvals: Dict[str, float]) -> Dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, running_max = {}, 0.0
    for i, (key, p) in enumerate(items):
        running_max = max(running_max, (m - i) * p)
        adj[key] = min(1.0, running_max)
    return adj


def rank_biserial(diffs: np.ndarray) -> float:
    nz = diffs[diffs != 0]
    if len(nz) == 0:
        return 0.0
    ranks = sps.rankdata(np.abs(nz))
    pos = ranks[nz > 0].sum()
    neg = ranks[nz < 0].sum()
    return float((pos - neg) / ranks.sum())


def wilcoxon_p(diffs: np.ndarray) -> float:
    nz = diffs[diffs != 0]
    if len(nz) < 6:
        return 1.0
    try:
        return float(sps.wilcoxon(diffs).pvalue)
    except ValueError:
        return 1.0


class RobustnessAnalyzer:
    """LOO judge robustness + raw rating distributions for Table 1 systems."""

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        for judge, postfix in POSTFIX_DIRS.items():
            if postfix not in cmt.CONVAWG_DIRS[judge]:
                cmt.CONVAWG_DIRS[judge].insert(0, postfix)
        cmt.POOLED_JUDGES = list(JUDGES_4)
        self.builder = cmt.JudgeTableBuilder(out_dir, JUDGES_4)
        # counts[system][metric] -> np.array(5,) of rating counts (pooled judges)
        self.counts: Dict[str, Dict[str, np.ndarray]] = {}

    def load(self) -> None:
        self.builder.load()
        self._load_raw_counts()

    # ---------- raw rating histograms ----------

    def _system_dirs(self, judge: str, system: str) -> List[Path]:
        setting = cmt.SYSTEMS[system]
        if setting is None:
            return [cmt.CONVAWG_JUDGE / d for d in cmt.CONVAWG_DIRS[judge]]
        return [cmt.JULY / setting / cmt.JUDGE_SUBDIR[judge]]

    def _load_raw_counts(self) -> None:
        for system in cmt.SYSTEMS:
            per_metric = {m: np.zeros(5, dtype=int) for m in METRICS}
            for judge in JUDGES_4:
                dirs = self._system_dirs(judge, system)
                for pid in self.builder.p50:
                    f = self.builder._result_file(dirs, pid)
                    if f is None:
                        continue
                    try:
                        data = json.load(open(f))
                    except (json.JSONDecodeError, OSError):
                        continue
                    for item in data.get("per_dialogue_ratings", []):
                        ratings = item.get("ratings", {})
                        for m in cmt.DIAL_METRICS:
                            v = cmt._parse_rating(ratings.get(m))
                            if not np.isnan(v) and float(v).is_integer():
                                per_metric[m][int(v) - 1] += 1
                    g = data.get("global_ratings", {})
                    for m in cmt.GLOB_METRICS:
                        v = cmt._parse_rating(g.get(m))
                        if not np.isnan(v) and float(v).is_integer():
                            per_metric[m][int(v) - 1] += 1
            self.counts[system] = per_metric
            total = int(sum(c.sum() for c in per_metric.values()))
            logger.info("raw ratings %-22s : %d cells", system, total)

    # ---------- LOO pooling ----------

    def _pooled_values(self, judges: List[str]) -> Dict[str, Dict[str, Dict[str, float]]]:
        """system -> pid -> column -> value, pooled over the given judges."""
        values: Dict[str, Dict[str, Dict[str, float]]] = {}
        for system in cmt.SYSTEMS:
            per_pid: Dict[str, Dict[str, float]] = {}
            for pid in self.builder.p50:
                row: Dict[str, float] = {}
                for m in cmt.DIAL_METRICS:
                    vs = [self.builder.scores[j][system].dial[pid][m]
                          for j in judges
                          if pid in self.builder.scores[j][system].dial
                          and not np.isnan(self.builder.scores[j][system].dial[pid][m])]
                    row[m] = float(np.mean(vs)) if vs else np.nan
                for m in cmt.GLOB_METRICS:
                    vs = [self.builder.scores[j][system].glob[pid][m]
                          for j in judges
                          if pid in self.builder.scores[j][system].glob
                          and not np.isnan(self.builder.scores[j][system].glob[pid][m])]
                    row[m] = float(np.mean(vs)) if vs else np.nan
                dial = [row[m] for m in cmt.DIAL_METRICS if not np.isnan(row[m])]
                glob_ = [row[m] for m in cmt.GLOB_METRICS if not np.isnan(row[m])]
                if not dial:
                    continue
                row["avg_dial"] = float(np.mean(dial))
                row["avg_glob"] = float(np.mean(glob_)) if glob_ else np.nan
                six = [row[m] for m in METRICS]
                row["avg6"] = (float(np.nanmean(six))
                               if any(not np.isnan(v) for v in six) else np.nan)
                per_pid[pid] = row
            values[system] = per_pid
        return values

    def loo_tables(self) -> List[dict]:
        rows: List[dict] = []
        for held_out in JUDGES_4:
            pool = [j for j in JUDGES_4 if j != held_out]
            values = self._pooled_values(pool)
            means = {}
            for system, per_pid in values.items():
                for col in ("avg_dial", "avg6"):
                    vals = np.array([r[col] for r in per_pid.values()])
                    means.setdefault(system, {})[col] = float(np.nanmean(vals))
            ranks = {col: sorted(cmt.SYSTEMS, key=lambda s: -means[s][col])
                     for col in ("avg_dial", "avg6")}
            pvals: Dict[str, Dict[str, float]] = {"avg_dial": {}, "avg6": {}}
            stats: Dict[str, dict] = {}
            for baseline in BASELINES:
                st: dict = {}
                for col in ("avg_dial", "avg6"):
                    conv, base = values[CONVAWG], values[baseline]
                    pids = [p for p in conv if p in base
                            and not np.isnan(conv[p].get(col, np.nan))
                            and not np.isnan(base[p].get(col, np.nan))]
                    d = np.array([conv[p][col] - base[p][col] for p in pids])
                    st[f"{col}_n"] = len(d)
                    st[f"{col}_delta"] = float(np.mean(d)) if len(d) else np.nan
                    st[f"{col}_r"] = rank_biserial(d)
                    pvals[col][baseline] = wilcoxon_p(d)
                stats[baseline] = st
            adj = {col: holm_correction(pvals[col]) for col in pvals}
            for system in cmt.SYSTEMS:
                row = {"held_out_judge": held_out, "system": system,
                       "avg_dial": means[system]["avg_dial"],
                       "avg6": means[system]["avg6"],
                       "rank_avg_dial": ranks["avg_dial"].index(system) + 1,
                       "rank_avg6": ranks["avg6"].index(system) + 1}
                if system != CONVAWG:
                    st = stats[system]
                    for col in ("avg_dial", "avg6"):
                        row[f"{col}_delta"] = st[f"{col}_delta"]
                        row[f"{col}_r"] = st[f"{col}_r"]
                        row[f"{col}_p"] = pvals[col][system]
                        row[f"{col}_p_holm"] = adj[col][system]
                rows.append(row)
        return rows

    # ---------- distribution outputs ----------

    def distribution_rows(self) -> List[dict]:
        rows: List[dict] = []
        for system in cmt.SYSTEMS:
            for m in METRICS:
                c = self.counts[system][m]
                n = int(c.sum())
                row = {"system": system, "metric": m, "n": n}
                for r in RATINGS:
                    row[f"n_{r}"] = int(c[r - 1])
                    row[f"share_{r}"] = float(c[r - 1] / n) if n else np.nan
                row["share_ge4"] = float((c[3] + c[4]) / n) if n else np.nan
                rows.append(row)
        return rows

    def plot_distributions(self, fig_path: Path) -> None:
        fig, axes = plt.subplots(2, 3, figsize=(11.5, 5.2), sharey=True)
        systems = list(cmt.SYSTEMS)
        x = np.arange(len(systems))
        for ax, m in zip(axes.flat, METRICS):
            bottom = np.zeros(len(systems))
            for r in RATINGS:
                shares = np.array([
                    self.counts[s][m][r - 1] / self.counts[s][m].sum()
                    if self.counts[s][m].sum() else 0.0
                    for s in systems])
                ax.bar(x, shares, bottom=bottom, width=0.72,
                       color=RATING_COLORS[r - 1], edgecolor="white",
                       linewidth=0.6, label=str(r) if m == METRICS[0] else None)
                bottom += shares
            ax.set_title(METRIC_TITLES[m], fontsize=10)
            ax.set_xticks(x)
            labels = [SYSTEM_SHORT[s] for s in systems]
            ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7.5)
            for tick, s in zip(ax.get_xticklabels(), systems):
                if s == CONVAWG:
                    tick.set_fontweight("bold")
            ax.set_ylim(0, 1)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(axis="y", labelsize=8)
        for ax in axes[:, 0]:
            ax.set_ylabel("Share of ratings", fontsize=9)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, title="Rating", loc="upper center",
                   ncol=5, fontsize=8, title_fontsize=9, frameon=False,
                   bbox_to_anchor=(0.5, 1.06))
        fig.tight_layout()
        for suffix in (".pdf", ".png"):
            fig.savefig(fig_path.with_suffix(suffix), dpi=300,
                        bbox_inches="tight")
        plt.close(fig)
        logger.info("figure written to %s.{pdf,png}", fig_path)

    # ---------- report ----------

    def run(self, fig_path: Path) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        loo_rows = self.loo_tables()
        dist_rows = self.distribution_rows()

        loo_cols = ["held_out_judge", "system", "avg_dial", "avg6",
                    "rank_avg_dial", "rank_avg6", "avg_dial_delta", "avg_dial_r",
                    "avg_dial_p", "avg_dial_p_holm", "avg6_delta", "avg6_r",
                    "avg6_p", "avg6_p_holm"]
        lines = [",".join(loo_cols)]
        for r in loo_rows:
            lines.append(",".join(
                f"{r[c]:.4f}" if isinstance(r.get(c), float) else str(r.get(c, ""))
                for c in loo_cols))
        (self.out_dir / "loo_judge_table.csv").write_text("\n".join(lines) + "\n")

        dist_cols = (["system", "metric", "n"]
                     + [f"n_{r}" for r in RATINGS]
                     + [f"share_{r}" for r in RATINGS] + ["share_ge4"])
        lines = [",".join(dist_cols)]
        for r in dist_rows:
            lines.append(",".join(
                f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c])
                for c in dist_cols))
        (self.out_dir / "rating_distributions.csv").write_text("\n".join(lines) + "\n")

        self.plot_distributions(fig_path)
        self._write_report(loo_rows, dist_rows)

    def _write_report(self, loo_rows: List[dict], dist_rows: List[dict]) -> None:
        text = ["# LOO judge robustness + rating distributions",
                "(post-fix ConVAWG, 4 calibrated judges, 50 matched personas)", ""]
        for held_out in JUDGES_4:
            rows = [r for r in loo_rows if r["held_out_judge"] == held_out]
            rows.sort(key=lambda r: r["rank_avg6"])
            text.append(f"## Held-out judge: {held_out} (3-judge pool)")
            text.append("| system | Avg-D | rank | Avg | rank | dAvg | Holm-p(Avg) | r |")
            text.append("|---|---|---|---|---|---|---|---|")
            for r in rows:
                if r["system"] == CONVAWG:
                    text.append(f"| **{r['system']}** | {r['avg_dial']:.3f} | "
                                f"{r['rank_avg_dial']} | {r['avg6']:.3f} | "
                                f"{r['rank_avg6']} | -- | -- | -- |")
                else:
                    text.append(f"| {r['system']} | {r['avg_dial']:.3f} | "
                                f"{r['rank_avg_dial']} | {r['avg6']:.3f} | "
                                f"{r['rank_avg6']} | {r['avg6_delta']:+.3f} | "
                                f"{r['avg6_p_holm']:.2e} | {r['avg6_r']:+.2f} |")
            text.append("")
        text.append("## Ceiling concentration (pooled over 4 judges)")
        text.append("| system | metric | n | share@5 | share>=4 |")
        text.append("|---|---|---|---|---|")
        for r in dist_rows:
            text.append(f"| {r['system']} | {r['metric']} | {r['n']} | "
                        f"{r['share_5']:.3f} | {r['share_ge4']:.3f} |")
        all_counts = np.zeros(5, dtype=int)
        conv_counts = np.zeros(5, dtype=int)
        for system in cmt.SYSTEMS:
            for m in METRICS:
                all_counts += self.counts[system][m]
                if system == CONVAWG:
                    conv_counts += self.counts[system][m]
        text.append("")
        text.append(f"All systems pooled: n={all_counts.sum()}, "
                    f"share@5={all_counts[4]/all_counts.sum():.3f}, "
                    f"share>=4={(all_counts[3]+all_counts[4])/all_counts.sum():.3f}")
        text.append(f"ConVAWG pooled:     n={conv_counts.sum()}, "
                    f"share@5={conv_counts[4]/conv_counts.sum():.3f}, "
                    f"share>=4={(conv_counts[3]+conv_counts[4])/conv_counts.sum():.3f}")
        (self.out_dir / "report.md").write_text("\n".join(text) + "\n")
        logger.info("report written to %s", self.out_dir / "report.md")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Example:
  python compute_judge_robustness_and_distributions.py \\
      --out-dir analysis/outputs/judge_robustness \\
      --fig analysis/outputs/figures/score_distributions
""")
    parser.add_argument("--out-dir", type=Path,
                        default=REPO / "analysis" / "outputs" / "judge_robustness")
    parser.add_argument("--fig", type=Path,
                        default=REPO / "analysis" / "outputs" / "figures" / "score_distributions",
                        help="figure path without extension (.pdf and .png written)")
    args = parser.parse_args()

    analyzer = RobustnessAnalyzer(args.out_dir)
    analyzer.load()
    analyzer.run(args.fig)


if __name__ == "__main__":
    main()
