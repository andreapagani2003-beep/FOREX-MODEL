"""Phase 2 report: comparison table, per-spec figures, markdown summary."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from usdjpy_mr.config import Config
from usdjpy_mr.stats.runner import SpecSampleResult, recommend, summary_table
from usdjpy_mr.utils.logging import get_logger

log = get_logger(__name__)

# Categorical palette (fixed slot order) and chrome, from the dataviz reference palette.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


def _style():
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "text.color": INK,
            "axes.labelcolor": INK2,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "lines.linewidth": 1.2,
            "font.size": 9,
            "legend.frameon": False,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
        }
    )


def _shade(ax, cfg: Config, color="#f0efec"):
    for a in cfg.stats.anchors:
        ax.axvspan(pd.Timestamp(a.start), pd.Timestamp(a.end), color=color, zorder=0)


def spec_figure(
    res: SpecSampleResult, rolling_eg: pd.DataFrame, df: pd.DataFrame, cfg: Config, out: Path
) -> Path:
    """Four single-axis panels for one (spec, full sample): spot vs static fair value + residual,
    beta paths, rolling Engle-Granger p-value, residual z-score with anchor windows."""
    import matplotlib.pyplot as plt

    _style()
    S = res.series
    T = res.tables
    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
    y = df["usdjpy"].loc[S["resid_static"].index]
    fv = y - S["resid_static"]

    ax = axes[0]
    ax.plot(y.index, y, color=SERIES[0], label="USD/JPY")
    ax.plot(
        fv.index,
        fv,
        color=SERIES[1],
        label=f"static fair value (beta {T['static_ols']['beta']:.1f})",
    )
    ax.plot(
        S["resid_kalman"].index,
        y - S["resid_kalman"],
        color=SERIES[2],
        lw=0.9,
        label="Kalman fair value",
    )
    _shade(ax, cfg)
    ax.set_title(
        f"{res.spec}: spot vs spread-implied fair value  (EG p={T['engle_granger']['y_on_x']['pvalue']:.2f}, "
        f"Johansen trace {T['johansen']['trace_r0']:.1f} vs crit95 {T['johansen']['trace_r0_crit95']:.1f})"
    )
    ax.set_ylabel("USD/JPY")
    ax.legend(loc="upper left", ncol=3)

    ax = axes[1]
    for i, (name, s) in enumerate(S["rolling_beta"].items()):
        ax.plot(s.index, s, color=SERIES[i + 1], lw=0.9, label=name.replace("ols_", "rolling OLS "))
    ax.plot(S["kalman_beta"].index, S["kalman_beta"], color=SERIES[0], label="Kalman")
    ax.axhline(T["static_ols"]["beta"], color=INK2, lw=0.8, ls="--", label="static")
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_title("beta paths (USD/JPY per 1pp of spread)")
    ax.set_ylabel("beta")
    ax.set_ylim(-40, 60)
    ax.legend(loc="upper left", ncol=5)

    ax = axes[2]
    r = rolling_eg
    ax.plot(
        r.index,
        r["pvalue"],
        color=SERIES[0],
        label=f"EG p-value, {cfg.stats.rolling_coint.window}-day window",
    )
    ax.axhline(
        cfg.stats.significance,
        color=SERIES[7 % len(SERIES)] if len(SERIES) > 7 else "#e34948",
        lw=0.8,
        ls="--",
        label="5% level",
    )
    ax.fill_between(
        r.index,
        0,
        1,
        where=r["reject_5pct"],
        color="#cde2fb",
        zorder=0,
        label="reject unit root in residual",
    )
    share = r["reject_5pct"].mean()
    ax.set_title(f"rolling Engle-Granger: {share:.0%} of windows reject at 5%")
    ax.set_ylabel("p-value")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", ncol=3)

    ax = axes[3]
    z = S["z_static"]
    ax.plot(
        z.index,
        z,
        color=SERIES[0],
        lw=0.9,
        label=f"z-score of static residual ({cfg.stats.zscore_window}-day)",
    )
    for lvl in (2, -2):
        ax.axhline(lvl, color=INK2, lw=0.6, ls=":")
    ax.axhline(0, color=INK, lw=0.6)
    _shade(ax, cfg)
    hl = T["ou_static"]
    ax.set_title(
        f"residual z-score; OU half-life {hl['half_life']:.0f} days "
        f"(95% CI {hl['hl_ci_low']:.0f}..{'inf' if not np.isfinite(hl['hl_ci_high']) else f'{hl["hl_ci_high"]:.0f}'}); "
        f"shaded = handoff anchor windows"
    )
    ax.set_ylabel("z")
    ax.legend(loc="upper left")

    fig.align_ylabels(axes)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def _md_table(df: pd.DataFrame, cols: list[str], fmt: dict[str, str]) -> str:
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float | np.floating):
                cells.append(
                    "inf"
                    if np.isinf(v)
                    else ("" if np.isnan(v) else fmt.get(c, "{:.3g}").format(v))
                )
            elif isinstance(v, bool | np.bool_):
                cells.append("yes" if v else "no")
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep, *rows])


def write_report(
    results: list[SpecSampleResult],
    rolling_eg: dict[str, pd.DataFrame],
    df: pd.DataFrame,
    cfg: Config,
    figures: dict[str, Path],
) -> Path:
    scfg = cfg.stats
    s = summary_table(results)
    rec = recommend(s)
    lo, hi = scfg.half_life_bounds
    any_pass = bool(s["pass"].any())

    lines = [
        "# Phase 2 — Statistical confirmation",
        "",
        f"Data: `daily.parquet`, {df.index[0].date()} → {df.index[-1].date()}, {len(df)} rows, "
        f"convention `{cfg.alignment.convention}`. Significance {scfg.significance}. "
        f"Half-life band {lo:g}–{hi:g} bars.",
        "",
        "## Verdict",
        "",
        f"**Acceptance {'MET' if any_pass else 'NOT MET'}.** Criterion (handoff §5): at least one "
        "specification cointegrated at 5% on post-2016 AND half-life in band AND no unexplained "
        "break in the last 24 months.",
        "",
    ]
    if rec["primary"]:
        lines += [
            f"Primary: `{rec['primary']}`; secondary: `{rec['secondary']}` ({rec['reason']})."
        ]
    else:
        lines += [f"No primary/secondary recommendation: {rec['reason']}."]
    lines += [
        "",
        "## Comparison across specifications",
        "",
        "EG p = Engle-Granger p-value (USD/JPY on spread); Johansen trace r=0 vs 95% critical "
        "value; beta in USD/JPY per 1pp; half-life in trading days from the OU fit of the "
        "static-OLS residual with bootstrap 95% CI; breaks = sequential sup-F at 5% "
        "(sieve-bootstrap p-values).",
        "",
    ]
    cols = [
        "spec",
        "sample",
        "nobs",
        "usdjpy_order",
        "spread_order",
        "eg_p_y_on_x",
        "eg_p_x_on_y",
        "johansen_trace_r0",
        "johansen_crit95",
        "johansen_rank",
        "beta_static",
        "beta_se",
        "r2",
        "beta_kalman_last",
        "beta_roll500_range",
        "half_life",
        "hl_ci_low",
        "hl_ci_high",
        "hl_kalman",
        "breaks_found",
        "breaks_unexplained_recent",
        "anchors_pass",
        "cointegrated_5pct",
        "hl_in_band",
        "pass",
    ]
    fmt = {
        "eg_p_y_on_x": "{:.3f}",
        "eg_p_x_on_y": "{:.3f}",
        "johansen_trace_r0": "{:.1f}",
        "johansen_crit95": "{:.1f}",
        "beta_static": "{:.2f}",
        "beta_se": "{:.2f}",
        "r2": "{:.3f}",
        "beta_kalman_last": "{:.1f}",
        "half_life": "{:.0f}",
        "hl_ci_low": "{:.0f}",
        "hl_ci_high": "{:.0f}",
        "hl_kalman": "{:.0f}",
    }
    lines += [_md_table(s, cols, fmt), ""]

    lines += ["## Rolling cointegration (500-day Engle-Granger, full sample)", ""]
    for spec, r in rolling_eg.items():
        share = r["reject_5pct"].mean()
        yr = r.groupby(r.index.year)["pvalue"].median()
        lines += [
            f"- `{spec}`: {share:.1%} of {len(r)} windows reject at 5%. Median p by window-end year: "
            + ", ".join(f"{int(k)}: {v:.2f}" for k, v in yr.items())
        ]
    lines += ["", "## Structural breaks", ""]
    for res in results:
        T = res.tables
        if T["sup_f"]:
            for b in T["sup_f"]:
                lines += [
                    f"- `{res.spec}` / {res.sample}: break {b['date']} (sup-F {b['sup_f']:.0f}, p={b['pvalue']:.3f}), "
                    f"beta {b['beta_before']:.1f} → {b['beta_after']:.1f}; nearest known: {b['nearest_known']} "
                    f"{b['nearest_label']} ({b['distance_days']}d) → "
                    f"{'explained' if b['explained'] else 'UNEXPLAINED'}{', recent' if b['recent'] else ''}"
                ]
    full10 = next(r for r in results if r.spec == scfg.specs[0].name and r.sample == "full")
    lines += [
        "",
        f"Chow tests at the known policy dates (`{full10.spec}`, full sample). Classical p assumes iid "
        "errors and is meaningless here (the residual is near-integrated); the sieve-bootstrap p "
        "is the one to read.",
        "",
        "| date | event | F | p classical | p bootstrap | beta before → after | resid var ratio |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in full10.tables["chow"]:
        lines += [
            f"| {c['date']} | {c['label']} | {c['f_stat']:.0f} | {c['pvalue']:.1e} | {c['pvalue_boot']:.3f} | "
            f"{c['beta_before']:.1f} → {c['beta_after']:.1f} | {c['var_ratio']:.2f} |"
        ]

    lines += [
        "",
        "## Sanity anchors (handoff §5.7)",
        "",
        "Sign of the static-OLS residual in the anchor windows (full sample):",
        "",
    ]
    for a in full10.tables["anchors_static"]:
        if a["in_sample"]:
            lines += [
                f"- {a['window']}: expected {a['expected']}, observed {a['observed']} "
                f"(mean residual {a['resid_mean']:+.1f}, z-score mean {a['z_mean']:+.2f}) → "
                f"{'pass' if a['pass'] else 'FAIL'}"
            ]
    lines += ["", "## Figures", ""]
    for spec, path in figures.items():
        lines += [f"- `{spec}`: `{path.relative_to(cfg.root)}`"]
    lines += [
        "",
        "## Notes on method",
        "",
        "- Stationarity: ADF (H0 unit root) and KPSS (H0 stationary) on levels and first differences.",
        "- Cointegration: Engle-Granger in both orderings (MacKinnon p-values); Johansen trace and "
        "max-eigenvalue with det_order 0, 1 lag difference.",
        "- Beta: static OLS; rolling OLS 250/500/750; Kalman filter with random-walk (alpha, beta), "
        f"delta {scfg.kalman.delta:g}, observation variance from the first {scfg.kalman.init_from_first_n} obs.",
        "- OU: AR(1) on the residual, theta = -ln(1 + slope), half-life ln2/theta; 95% CI from a "
        f"{scfg.ou.n_boot}-draw residual bootstrap.",
        "- Breaks: Chow at known dates with classical and sieve-bootstrap p; unknown breaks by "
        f"sequential sup-F (trim {scfg.breaks.trim}, up to {scfg.breaks.max_breaks}, "
        f"{scfg.breaks.n_boot} sieve-bootstrap draws). A found break within "
        f"{scfg.breaks.tolerance_days} days of a known policy date counts as explained.",
        "- The `10y_real` spec uses US TIPS real yield minus *nominal* JGB yield: there is no JGB "
        "real-yield series in the pipeline, so it is a partial real spread.",
    ]
    out = cfg.reports_dir / "phase2_summary.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("wrote %s", out)
    return out


def dump_results(
    results: list[SpecSampleResult], rolling_eg: dict[str, pd.DataFrame], cfg: Config
) -> Path:
    out_dir = cfg.reports_dir / "phase2"
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = [
        {
            "spec": r.spec,
            "sample": r.sample,
            "start": r.start,
            "end": r.end,
            "nobs": r.nobs,
            **r.tables,
        }
        for r in results
    ]
    (out_dir / "results.json").write_text(
        json.dumps(tables, indent=1, default=str), encoding="utf-8"
    )
    for spec, r in rolling_eg.items():
        r.to_csv(out_dir / f"rolling_eg_{spec}.csv")
    for r in results:
        if r.sample == "full":
            pd.DataFrame({k: v for k, v in r.series.items() if isinstance(v, pd.Series)}).to_csv(
                out_dir / f"series_{r.spec}_full.csv"
            )
    return out_dir
