#!/usr/bin/env python3
"""
compute_main_results_table.py
------------------------------
Build the main baseline-comparison results tables (per-judge + pooled) from
LLM-judge outputs, using the persona as the unit of analysis.

Methodology:
  * Per-dialogue metrics (coherence, humanlikeness, persona_consistency,
    toxicity_realism) are averaged WITHIN each persona first, then
    macro-averaged across the 50 matched personas — so every cell is a
    "mean over n personas", matching the granularity of the global metrics
    (crime_fidelity, scenario_realism: one rating per persona).
  * No single flat average across the two metric families. Instead two
    columns: Avg (dial.) and Avg (glob.).
  * Uncertainty: 95% bootstrap CI, resampling personas (dialogues are
    nested within personas), 10,000 resamples, seed 42.
  * Significance: paired Wilcoxon signed-rank of ConVAWG vs each baseline
    on matched personas, per judge and pooled.
  * Pooled main table = per-persona scores averaged across the complete
    judges first, then macro-averaged; inter-judge agreement reported as
    average pairwise Spearman over per-(system, persona) scores.

Ratings are read from ``per_dialogue_ratings[i].ratings.<metric>.rating``
(strings "1".."5"; PARSE_ERROR cells are dropped and counted) and
``global_ratings.<metric>.rating``.
"""

import argparse
import itertools
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as sps
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent
JULY = REPO / "Experiments/Baselines/results"  # <setting>/<judge subdir>/
CONVAWG_JUDGE = REPO / "Generation_Results/8_judge_results"

DIAL_METRICS = ["coherence", "humanlikeness", "persona_consistency", "toxicity_realism"]
GLOB_METRICS = ["crime_fidelity", "scenario_realism"]

JUDGE_SUBDIR = {
    "gpt52": "judge_results",
    "deepseek": "judge_results_deepseek",
    "gemini": "judge_results_gemini",
    "qwen": "judge_results_qwen",
}

# ConVAWG per-judge result dirs (fallback dirs searched in order for missing personas)
CONVAWG_DIRS = {
    "gpt52": ["convawg_gpt52"],
    "deepseek": ["convawg_deepseek"],
    "gemini": ["convawg_gemini"],
    "qwen": ["convawg_qwen"],
}

# display name -> baseline setting dir under Experiments/Baselines/results (None = ConVAWG)
SYSTEMS = {
    "ConVAWG": None,
    "GPT-5.2": "gpt5.2",
    "DeepSeek-V4-Flash": "deepseek-v4-flash",
    "Gemini-3.1-Flash-Lite": "gemini-flash-lite",
    "Qwen3.5-35B-A3B": "qwen3.5-35b",
    "LLaMA-3.1-70B-Instruct": "llama3.1-70b-awq",
    "DiaSynth (Qwen3.5)": "diasynth_qwen3.5-35b",
    "SynDG (Qwen3.5)": "syndg_qwen3.5-35b",
    "SPASM (Qwen3.5)": "spasm_qwen3.5-35b",
}

POOLED_JUDGES: List[str] = ["gpt52", "deepseek", "gemini", "qwen"]
N_BOOT = 10_000
SEED = 42


def _parse_rating(cell: Optional[dict]) -> float:
    """Return rating as float in [1, 5], else NaN (PARSE_ERROR etc.)."""
    if not isinstance(cell, dict):
        return np.nan
    try:
        v = float(str(cell.get("rating", "")).strip())
    except ValueError:
        return np.nan
    return v if 1.0 <= v <= 5.0 else np.nan


class PersonaScores:
    """Persona-level scores for one (system, judge) cell."""

    def __init__(self) -> None:
        self.dial: Dict[str, Dict[str, float]] = {}   # pid -> metric -> persona mean
        self.glob: Dict[str, Dict[str, float]] = {}   # pid -> metric -> rating
        self.n_cells = 0
        self.n_parse_err = 0

    def add_file(self, pid: str, data: dict) -> None:
        dial_vals: Dict[str, List[float]] = {m: [] for m in DIAL_METRICS}
        for item in data.get("per_dialogue_ratings", []):
            ratings = item.get("ratings", {})
            for m in DIAL_METRICS:
                v = _parse_rating(ratings.get(m))
                self.n_cells += 1
                if np.isnan(v):
                    self.n_parse_err += 1
                else:
                    dial_vals[m].append(v)
        self.dial[pid] = {m: (float(np.mean(vs)) if vs else np.nan)
                          for m, vs in dial_vals.items()}
        g = data.get("global_ratings", {})
        gvals = {}
        for m in GLOB_METRICS:
            v = _parse_rating(g.get(m))
            self.n_cells += 1
            if np.isnan(v):
                self.n_parse_err += 1
            gvals[m] = v
        self.glob[pid] = gvals

    def avg_dial(self, pid: str) -> float:
        vals = [v for v in self.dial.get(pid, {}).values() if not np.isnan(v)]
        return float(np.mean(vals)) if vals else np.nan

    def avg_glob(self, pid: str) -> float:
        vals = [v for v in self.glob.get(pid, {}).values() if not np.isnan(v)]
        return float(np.mean(vals)) if vals else np.nan


def bootstrap_ci(values: np.ndarray, n_boot: int = N_BOOT,
                 seed: int = SEED) -> Tuple[float, float]:
    vals = values[~np.isnan(values)]
    if len(vals) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), size=(n_boot, len(vals)))
    means = vals[idx].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def paired_wilcoxon(a: Dict[str, float], b: Dict[str, float]) -> Tuple[float, int]:
    """Two-sided Wilcoxon signed-rank on personas present+valid in both."""
    pids = [p for p in a if p in b and not np.isnan(a[p]) and not np.isnan(b[p])]
    if len(pids) < 6:
        return (np.nan, len(pids))
    x = np.array([a[p] for p in pids])
    y = np.array([b[p] for p in pids])
    if np.allclose(x, y):
        return (1.0, len(pids))
    try:
        return (float(sps.wilcoxon(x, y).pvalue), len(pids))
    except ValueError:
        return (1.0, len(pids))


class JudgeTableBuilder:
    def __init__(self, out_dir: Path, judges: List[str]) -> None:
        self.out_dir = out_dir
        self.judges = judges
        self.p50: List[str] = json.load(
            open(REPO / "Experiments/Baselines/matched_50_personas.json"))["all_50"]
        # scores[judge][system] -> PersonaScores
        self.scores: Dict[str, Dict[str, PersonaScores]] = {}

    # ---------- loading ----------

    def _result_file(self, dirs: List[Path], pid: str) -> Optional[Path]:
        for d in dirs:
            for pat in (f"{pid}_dialogues_judge_results.json",
                        f"{pid}_dialogue_judge_results.json"):
                f = d / pat
                if f.exists():
                    return f
        return None

    def load(self) -> None:
        for judge in self.judges:
            self.scores[judge] = {}
            for system, setting in SYSTEMS.items():
                if setting is None:
                    dirs = [CONVAWG_JUDGE / d for d in CONVAWG_DIRS[judge]]
                else:
                    dirs = [JULY / setting / JUDGE_SUBDIR[judge]]
                ps = PersonaScores()
                for pid in self.p50:
                    f = self._result_file(dirs, pid)
                    if f is None:
                        continue
                    try:
                        ps.add_file(pid, json.load(open(f)))
                    except (json.JSONDecodeError, OSError) as e:
                        logger.warning("unreadable %s: %s", f, e)
                self.scores[judge][system] = ps
                logger.info("loaded %-8s x %-22s : %d personas",
                            judge, system, len(ps.dial))

    # ---------- aggregation ----------

    def cell_stats(self, judge: str, system: str) -> dict:
        ps = self.scores[judge][system]
        row: dict = {"system": system}
        for m in DIAL_METRICS:
            vals = np.array([ps.dial[p][m] for p in ps.dial])
            row[m] = float(np.nanmean(vals)) if len(vals) else np.nan
        for m in GLOB_METRICS:
            vals = np.array([ps.glob[p][m] for p in ps.glob])
            row[m] = float(np.nanmean(vals)) if len(vals) else np.nan
        ad = np.array([ps.avg_dial(p) for p in ps.dial])
        ag = np.array([ps.avg_glob(p) for p in ps.glob])
        row["avg_dial"] = float(np.nanmean(ad)) if len(ad) else np.nan
        row["avg_glob"] = float(np.nanmean(ag)) if len(ag) else np.nan
        row["avg_dial_ci"] = bootstrap_ci(ad)
        row["avg_glob_ci"] = bootstrap_ci(ag)
        row["n_personas"] = int(np.sum(~np.isnan(ad)))
        row["n_personas_glob"] = int(np.sum(~np.isnan(ag)))
        row["parse_err_pct"] = (100.0 * ps.n_parse_err / ps.n_cells
                                if ps.n_cells else np.nan)
        return row

    def judge_table(self, judge: str) -> List[dict]:
        rows = [self.cell_stats(judge, s) for s in SYSTEMS]
        conv = self.scores[judge]["ConVAWG"]
        conv_ad = {p: conv.avg_dial(p) for p in conv.dial}
        conv_ag = {p: conv.avg_glob(p) for p in conv.glob}
        for row in rows:
            if row["system"] == "ConVAWG":
                row["p_dial"], row["p_glob"] = np.nan, np.nan
                continue
            ps = self.scores[judge][row["system"]]
            row["p_dial"], _ = paired_wilcoxon(
                conv_ad, {p: ps.avg_dial(p) for p in ps.dial})
            row["p_glob"], _ = paired_wilcoxon(
                conv_ag, {p: ps.avg_glob(p) for p in ps.glob})
        return rows

    def pooled_scores(self) -> Dict[str, PersonaScores]:
        """Per persona, average each metric across the pooled judges."""
        pooled: Dict[str, PersonaScores] = {}
        for system in SYSTEMS:
            ps = PersonaScores()
            for pid in self.p50:
                dial, glob_ = {}, {}
                for m in DIAL_METRICS:
                    vs = [self.scores[j][system].dial[pid][m]
                          for j in POOLED_JUDGES
                          if pid in self.scores[j][system].dial
                          and not np.isnan(self.scores[j][system].dial[pid][m])]
                    dial[m] = float(np.mean(vs)) if vs else np.nan
                for m in GLOB_METRICS:
                    vs = [self.scores[j][system].glob[pid][m]
                          for j in POOLED_JUDGES
                          if pid in self.scores[j][system].glob
                          and not np.isnan(self.scores[j][system].glob[pid][m])]
                    glob_[m] = float(np.mean(vs)) if vs else np.nan
                if any(not np.isnan(v) for v in dial.values()):
                    ps.dial[pid] = dial
                    ps.glob[pid] = glob_
            pooled[system] = ps
        return pooled

    def judge_agreement(self) -> Tuple[List[Tuple[str, str, float, int]], float]:
        """Average pairwise Spearman over per-(system, persona) overall scores."""
        per_judge: Dict[str, Dict[Tuple[str, str], float]] = {}
        for j in POOLED_JUDGES:
            d = {}
            for system in SYSTEMS:
                ps = self.scores[j][system]
                for pid in ps.dial:
                    vals = ([v for v in ps.dial[pid].values() if not np.isnan(v)]
                            + [v for v in ps.glob.get(pid, {}).values()
                               if not np.isnan(v)])
                    if vals:
                        d[(system, pid)] = float(np.mean(vals))
            per_judge[j] = d
        pairs = []
        for a, b in itertools.combinations(POOLED_JUDGES, 2):
            common = sorted(set(per_judge[a]) & set(per_judge[b]))
            if len(common) < 10:
                continue
            rho = sps.spearmanr([per_judge[a][k] for k in common],
                                [per_judge[b][k] for k in common]).statistic
            pairs.append((a, b, float(rho), len(common)))
        mean_rho = float(np.mean([r for _, _, r, _ in pairs])) if pairs else np.nan
        return pairs, mean_rho

    # ---------- output ----------

    @staticmethod
    def _csv_rows(rows: List[dict]) -> str:
        cols = ["system"] + DIAL_METRICS + ["avg_dial", "avg_dial_ci_lo", "avg_dial_ci_hi"] \
            + GLOB_METRICS + ["avg_glob", "avg_glob_ci_lo", "avg_glob_ci_hi",
                              "n_personas", "n_personas_glob", "parse_err_pct",
                              "p_dial", "p_glob"]
        out = [",".join(cols)]
        for r in rows:
            vals = [r["system"]]
            for c in cols[1:]:
                if c == "avg_dial_ci_lo":
                    v = r["avg_dial_ci"][0]
                elif c == "avg_dial_ci_hi":
                    v = r["avg_dial_ci"][1]
                elif c == "avg_glob_ci_lo":
                    v = r["avg_glob_ci"][0]
                elif c == "avg_glob_ci_hi":
                    v = r["avg_glob_ci"][1]
                else:
                    v = r.get(c, np.nan)
                vals.append("" if (isinstance(v, float) and np.isnan(v))
                            else (f"{v:.4f}" if isinstance(v, float) else str(v)))
            out.append(",".join(f'"{v}"' if "," in v else v for v in vals))
        return "\n".join(out) + "\n"

    def _tex_table(self, rows: List[dict], caption: str, label: str) -> str:
        def rank_marks(col: str) -> Dict[str, str]:
            vals = {r["system"]: r[col] for r in rows if not np.isnan(r[col])}
            order = sorted(vals, key=vals.get, reverse=True)
            marks = {}
            if order:
                marks[order[0]] = "best"
            if len(order) > 1:
                marks[order[1]] = "second"
            return marks

        marks = {c: rank_marks(c) for c in DIAL_METRICS + GLOB_METRICS
                 + ["avg_dial", "avg_glob"]}

        def fmt(r: dict, col: str, sig_col: Optional[str] = None) -> str:
            v = r[col]
            if np.isnan(v):
                return "--"
            s = f"{v:.2f}"
            m = marks[col].get(r["system"])
            if m == "best":
                s = rf"\textbf{{{s}}}"
            elif m == "second":
                s = rf"\underline{{{s}}}"
            if sig_col and not np.isnan(r.get(sig_col, np.nan)) \
                    and r[sig_col] < 0.05 and r["system"] != "ConVAWG":
                s += r"$^{*}$"
            return s

        def fmt_ci(r: dict, col: str, ci_col: str, sig_col: str) -> str:
            v = r[col]
            if np.isnan(v):
                return "--"
            lo, hi = r[ci_col]
            s = fmt(r, col, sig_col)
            if not np.isnan(lo):
                s += rf" {{\scriptsize [{lo:.2f}, {hi:.2f}]}}"
            return s

        lines = [
            r"\begin{table*}[t]",
            r"\centering",
            r"\small",
            r"\setlength{\tabcolsep}{4pt}",
            r"\begin{tabular}{l cccc c cc c}",
            r"\toprule",
            r" & \multicolumn{5}{c}{\textit{Per-dialogue (mean over "
            r"$n{=}50$ personas)}} & \multicolumn{3}{c}{\textit{Global "
            r"(one rating per persona)}} \\",
            r"\cmidrule(lr){2-6} \cmidrule(lr){7-9}",
            r"System & Coh. & Human. & Persona & Tox.\ real. & "
            r"Avg (dial.) & Crime fid. & Scen.\ real. & Avg (glob.) \\",
            r"\midrule",
        ]
        for r in rows:
            cells = [r["system"].replace("&", r"\&")]
            for m in DIAL_METRICS:
                cells.append(fmt(r, m))
            cells.append(fmt_ci(r, "avg_dial", "avg_dial_ci", "p_dial"))
            for m in GLOB_METRICS:
                cells.append(fmt(r, m))
            cells.append(fmt_ci(r, "avg_glob", "avg_glob_ci", "p_glob"))
            lines.append(" & ".join(cells) + r" \\")
        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\end{table*}",
        ]
        return "\n".join(lines) + "\n"

    def run(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        all_tables: Dict[str, List[dict]] = {}
        for judge in tqdm(self.judges, desc="judges"):
            rows = self.judge_table(judge)
            all_tables[judge] = rows
            (self.out_dir / f"per_judge_{judge}.csv").write_text(
                self._csv_rows(rows))

        # pooled
        pooled = self.pooled_scores()
        saved_scores = self.scores
        self.scores = {"pooled": pooled}
        conv = pooled["ConVAWG"]
        conv_ad = {p: conv.avg_dial(p) for p in conv.dial}
        conv_ag = {p: conv.avg_glob(p) for p in conv.glob}
        pooled_rows = []
        for system in SYSTEMS:
            row = self.cell_stats("pooled", system)
            ps = pooled[system]
            if system == "ConVAWG":
                row["p_dial"], row["p_glob"] = np.nan, np.nan
            else:
                row["p_dial"], _ = paired_wilcoxon(
                    conv_ad, {p: ps.avg_dial(p) for p in ps.dial})
                row["p_glob"], _ = paired_wilcoxon(
                    conv_ag, {p: ps.avg_glob(p) for p in ps.glob})
            pooled_rows.append(row)
        self.scores = saved_scores
        all_tables["pooled"] = pooled_rows
        (self.out_dir / "pooled_main.csv").write_text(self._csv_rows(pooled_rows))

        pairs, mean_rho = self.judge_agreement()
        agree_csv = "judge_a,judge_b,spearman_rho,n_items\n" + "\n".join(
            f"{a},{b},{r:.4f},{n}" for a, b, r, n in pairs) \
            + f"\nAVERAGE,,{mean_rho:.4f},\n"
        (self.out_dir / "judge_agreement.csv").write_text(agree_csv)

        judge_full = {"gpt52": "GPT-5.2", "deepseek": "DeepSeek-V4-Flash",
                      "gemini": "Gemini-3.1-Flash-Lite", "qwen": "Qwen3.5-35B-A3B"}
        pooled_names = ", ".join(judge_full[j] for j in POOLED_JUDGES)
        caption = (
            f"Judge scores on the 50 matched personas, pooled over "
            f"{len(POOLED_JUDGES)} judges ({pooled_names}). "
            "Per-dialogue metrics are averaged within each persona "
            "first, then macro-averaged over personas; global metrics are one "
            "1--5 rating per persona, so single-context baselines approach the "
            "ceiling on internal-consistency-driven criteria. Avg columns show "
            "mean [95\\% bootstrap CI over personas]. Bold = best, underline = "
            "second best; $^{*}$ = differs from ConVAWG at $p<0.05$ "
            "(paired Wilcoxon signed-rank on matched personas). "
            f"Inter-judge agreement: mean pairwise Spearman $\\rho = {mean_rho:.2f}$."
        )
        (self.out_dir / "pooled_main_table.tex").write_text(
            self._tex_table(pooled_rows, caption, "tab:main-results"))

        self._write_report(all_tables, pairs, mean_rho)

    def _write_report(self, tables: Dict[str, List[dict]],
                      pairs: List[Tuple[str, str, float, int]],
                      mean_rho: float) -> None:
        judge_names = {"gpt52": "GPT-5.2", "deepseek": "DeepSeek-V4-Flash",
                       "gemini": "Gemini-3.1-Flash-Lite", "qwen": "Qwen3.5-35B-A3B",
                       "pooled": f"POOLED ({len(POOLED_JUDGES)} judges: "
                                 f"{', '.join(POOLED_JUDGES)})"}
        lines = ["# Main results — persona-level macro-averages (matched 50 personas)",
                 "",
                 "Per-dialogue metrics: persona mean first, then macro-average. "
                 "Global metrics: one rating per persona. "
                 "CI: 95% bootstrap over personas. "
                 "p: paired Wilcoxon vs ConVAWG on matched personas.",
                 ""]
        for judge in ["pooled"] + self.judges:
            rows = tables.get(judge)
            if not rows:
                continue
            lines.append(f"## {judge_names.get(judge, judge)}")
            lines.append("")
            lines.append("| System | Coh. | Hum. | Pers. | Tox. | **Avg dial** "
                         "[95% CI] | Crime | Scen. | **Avg glob** [95% CI] | "
                         "n | p(dial) | p(glob) |")
            lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
            for r in rows:
                def f(v: float) -> str:
                    return "--" if (isinstance(v, float) and np.isnan(v)) else f"{v:.2f}"
                ad_lo, ad_hi = r["avg_dial_ci"]
                ag_lo, ag_hi = r["avg_glob_ci"]
                ad_ci = f" [{ad_lo:.2f}, {ad_hi:.2f}]" if not np.isnan(ad_lo) else ""
                ag_ci = f" [{ag_lo:.2f}, {ag_hi:.2f}]" if not np.isnan(ag_lo) else ""
                pd_, pg = r.get("p_dial", np.nan), r.get("p_glob", np.nan)
                pd_s = "--" if np.isnan(pd_) else (f"{pd_:.4f}" if pd_ >= 0.0001 else "<1e-4")
                pg_s = "--" if np.isnan(pg) else (f"{pg:.4f}" if pg >= 0.0001 else "<1e-4")
                lines.append(
                    f"| {r['system']} | {f(r['coherence'])} | {f(r['humanlikeness'])} | "
                    f"{f(r['persona_consistency'])} | {f(r['toxicity_realism'])} | "
                    f"**{f(r['avg_dial'])}**{ad_ci} | {f(r['crime_fidelity'])} | "
                    f"{f(r['scenario_realism'])} | **{f(r['avg_glob'])}**{ag_ci} | "
                    f"{r['n_personas']} | {pd_s} | {pg_s} |")
            lines.append("")
        lines.append("## Inter-judge agreement (Spearman over per-(system,persona) scores)")
        lines.append("")
        for a, b, r, n in pairs:
            lines.append(f"- {judge_names[a]} vs {judge_names[b]}: rho = {r:.3f} (n={n})")
        lines.append(f"- **Average pairwise rho = {mean_rho:.3f}**")
        lines.append("")
        (self.out_dir / "report.md").write_text("\n".join(lines))
        logger.info("wrote %s", self.out_dir / "report.md")


def main() -> None:
    global POOLED_JUDGES
    parser = argparse.ArgumentParser(
        description="Persona-level macro-averaged baseline results tables",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python compute_main_results_table.py
  python compute_main_results_table.py --out-dir analysis/outputs/full_results
  python compute_main_results_table.py --judges gpt52 qwen
""",
    )
    parser.add_argument("--out-dir", type=Path,
                        default=REPO / "analysis" / "outputs" / "full_results",
                        help="Output directory for CSV/TeX/report files")
    parser.add_argument("--judges", nargs="+",
                        default=["gpt52", "deepseek", "gemini", "qwen"],
                        choices=list(JUDGE_SUBDIR),
                        help="Judges to include")
    parser.add_argument("--pooled-judges", nargs="+", default=POOLED_JUDGES,
                        choices=list(JUDGE_SUBDIR),
                        help="Judges pooled into the main table and agreement stats")
    args = parser.parse_args()

    POOLED_JUDGES = args.pooled_judges

    builder = JudgeTableBuilder(args.out_dir, args.judges)
    builder.load()
    builder.run()


if __name__ == "__main__":
    main()
