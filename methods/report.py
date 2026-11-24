"""One self-contained HTML report: what each method is, how it did out-of-sample, and the scenario forecasts."""
import base64
import html
import io
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402
import pandas as pd              # noqa: E402

from .evaluate import METHOD_LABELS, METRICS  # noqa: E402
import re  # noqa: E402

INK, MUTED, GRID = "#1f2933", "#6b7785", "#e4e7eb"
COLORS = {"bma": "#1b4f8a", "bma_asym": "#4a7fc1", "spline": "#5b8c5a", "hybrid": "#7a5c99",
          "direct": "#b5179e", "direct_class": "#e88b4a",
          "combination": "#2a9d8f", "class": "#c08a1e", "ar1": "#8e6c8a",
          "drift": "#9aa5b1", "random_walk": "#c94f4f"}
SCEN_COLORS = {"baseline": "#1b4f8a", "upside": "#2a9d8f", "downside": "#c08a1e", "severe": "#c94f4f"}

SOURCES = [
    ("Marcellino, Stock & Watson (2006), A comparison of direct and iterated multistep AR methods for forecasting "
     "macroeconomic time series, Journal of Econometrics 135", "https://doi.org/10.1016/j.jeconom.2005.07.020"),
    ("Chevillon (2007), Direct multi-step estimation and forecasting, Journal of Economic Surveys 21(4)",
     "https://doi.org/10.1111/j.1467-6419.2007.00518.x"),
    ("Jorda (2005), Estimation and inference of impulse responses by local projections, American Economic Review 95(1)",
     "https://doi.org/10.1257/0002828053828518"),
    ("Barnichon & Brownlees (2019), Impulse response estimation by smooth local projections, "
     "Review of Economics and Statistics 101(3)", "https://doi.org/10.1162/rest_a_00778"),
    ("Li, Plagborg-Moller & Wolf (2024), Local projections vs. VARs: lessons from thousands of DGPs, "
     "Journal of Econometrics 244(2)", "https://arxiv.org/abs/2104.00655"),
    ("Goulet Coulombe, Leroux, Stevanovic & Surprenant (2022), How is machine learning useful for macroeconomic "
     "forecasting?, Journal of Applied Econometrics 37(5)", "https://doi.org/10.1002/jae.2910"),
    ("Barbieri & Berger (2004), Optimal predictive model selection, Annals of Statistics 32(3)",
     "https://doi.org/10.1214/009053604000000238"),
    ("Hyndman, Ahmed, Athanasopoulos & Shang (2011), Optimal combination forecasts for hierarchical time series, "
     "Computational Statistics & Data Analysis 55(9)", "https://doi.org/10.1016/j.csda.2011.03.006"),
    ("IMF Global Financial Stability Report, Oct 2020, Online Annex 4.1 (Global Solvency Stress Test)",
     "https://www.imf.org/-/media/Files/Publications/GFSR/2020/October/English/onlineannex41.ashx"),
    ("Guerrieri & Welch (2012), Can Macro Variables Used in Stress Testing Forecast the Performance of Banks? "
     "Fed FEDS 2012-49", "https://www.federalreserve.gov/pubs/feds/2012/201249/index.html"),
    ("Hirtle, Kovner, Vickery & Bhanot (2016), The CLASS model, NY Fed Staff Report 663",
     "https://www.newyorkfed.org/research/staff_reports/sr663.html"),
    ("Federal Reserve, Supervisory Stress Test Model Documentation: PPNR models",
     "https://www.federalreserve.gov/supervisionreg/files/pre-provision-net-revenue-models.pdf"),
    ("Federal Reserve, 2025 Supervisory Stress Test Methodology: Descriptions of Supervisory Models",
     "https://www.federalreserve.gov/publications/2025-june-supervisory-stress-test-methodology-descriptions-supervisory-models.htm"),
]

METHOD_TEXT = [
    ("bma", "IMF: Bayesian model averaging with sign constraints",
     "At every origin the candidate drivers (series with an expected sign in the target's sign file) are screened "
     "on their partial correlation with the target; the best 10 with the right sign are kept (at most 2 per theme, "
     "no near-duplicates), each at its best lag 0-2. Every combination of 0 to 3 of them is estimated as "
     "y<sub>i,t</sub> = a<sub>i</sub> + &rho; y<sub>i,t-1</sub> + &Sigma; b<sub>k</sub> x<sub>k,t-lag</sub> with bank fixed effects. "
     "Equations whose long-run multipliers b<sub>k</sub>/(1&minus;&rho;) have the wrong sign are discarded. "
     "The forecast is the average of the remaining equations weighted by exp(&minus;BIC/2), which is Bayesian "
     "model averaging with equal prior odds. Nothing is selected: every admissible equation contributes."),
    ("combination", "Guerrieri &amp; Welch (Fed): forecast combination",
     "One equation per macro variable: y<sub>t</sub> on y<sub>t-1</sub> and lags 1-4 of that variable (bank effects in a "
     "panel). The 10 variables most correlated with the target are used (signs are not imposed, as in the paper). "
     "The forecast is the equal-weighted average of the equations."),
    ("class", "NY Fed CLASS-style: one fixed, simple equation",
     "One small equation per target with drivers chosen on economic grounds (set in config.yaml under "
     "class_drivers), estimated on the panel, and kept identical at every origin. A transparent challenger."),
    ("rxb", "Federal Reserve: income = rate &times; balance",
     "For an income target, each method forecasts the parts (rates and balances) and the income is rebuilt from them "
     "with the same formula, instead of forecasting the dollar amount itself. Shown as &ldquo;method (rate x balance)&rdquo;."),
    ("ident", "Accounting identities",
     "When targets are linked by an identity (total loans = mortgages + commercial loans, say) and all of them are "
     "forecast in the run, their forecasts are made to add up: either every target moves a little, the least certain "
     "forecasts the most (weighted least squares reconciliation, Hyndman et al.), or one target takes the whole gap. "
     "Shown as &ldquo;method (identities)&rdquo;, next to the unadjusted forecasts. With only some of the targets in "
     "the run, or <code>identities_on: false</code>, the identity is ignored."),
    ("asym", "Asymmetric effects (IMF BMA with rises and falls)",
     "The same Bayesian model averaging, but each driver may also enter with separate slopes for its rises and its "
     "falls (max(x,0) and min(x,0)), for example unemployment hurting more when it rises than it helps when it falls. "
     "The asymmetric form costs one more parameter in the BIC, so it only takes weight when the data support it; "
     "both slopes must have the expected sign. Stays linear in each regime, so it extrapolates into severe scenarios."),
    ("spline", "Monotone piecewise-linear effects",
     "The core drivers of the BMA (inclusion probability of at least 0.5, Barbieri &amp; Berger's median probability "
     "model) enter as piecewise-linear curves with knots at the terciles of their history. The curve may bend, but "
     "every piece must slope in the expected direction (estimated by bounded least squares), and the end pieces "
     "continue in a straight line beyond the data, so severe scenarios still have an effect."),
    ("hybrid", "Linear model + monotone boosted trees",
     "The BMA's weighted-average linear equation, plus shallow gradient-boosted trees fitted to its residuals and "
     "constrained to be monotone in each driver's expected direction. Trees are flat outside the range of their "
     "training data, so beyond history only the linear part moves the forecast: the trees add curvature where there "
     "is evidence, the linear part extrapolates. Nonlinearity helps most in uncertain and stressed periods "
     "(Goulet Coulombe et al. 2022), but with about 100 quarters it must earn its place out-of-sample."),
    ("direct", "Direct forecasts (smooth local projections)",
     "Instead of one quarterly equation iterated forward, one equation per horizon h for the cumulative change over "
     "h quarters, on the current change and each driver's cumulative change over the same h quarters of the scenario "
     "(a local projection, Jorda 2005). All horizons are estimated together with a penalty on the second differences "
     "of the coefficients across h (a P-spline with a knot at every horizon), so the paths are smooth, as in the smooth "
     "local projections of Barnichon &amp; Brownlees (2019); coefficients are "
     "bounded to their expected sign. Iterated forecasts are more efficient when the model is right; direct ones are "
     "more robust when it is misspecified (Marcellino, Stock &amp; Watson 2006; Chevillon 2007), at the cost of more "
     "variance at long horizons (Li, Plagborg-Moller &amp; Wolf 2024). <i>direct</i> uses the BMA's core drivers, "
     "<i>direct_class</i> the CLASS-style drivers."),
    ("bench", "Benchmarks", "AR(1) with bank effects; drift (the bank's average change); random walk (no change)."),
    ("conf", "Conformal bands",
     "The 80% and 95% bands come from the out-of-sample errors of the same method at the same horizon "
     "(finite-sample conformal quantiles of log errors for balances, of differences for rates). The coverage columns "
     "check them honestly: at each origin the band is built only from errors already known at that origin, "
     "then compared with what happened."),
]


def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _ax(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def _color(m):
    base = m.split(" (")[0]
    return COLORS.get(base, "#555")


def _label(m):
    base = m.split(" (")[0]
    lab = METHOD_LABELS.get(base, base)
    return lab + (" &mdash; rate &times; balance" if "rate x balance" in m else "") + \
        (" &mdash; with identities" if "(identities)" in m else "")


def _plain(m):
    return html.unescape(_label(m).replace("&mdash;", "-").replace("&times;", "x"))


def chart_horizon(s):
    byh = s["byh"]
    show = list(s["lb"].index[:6])
    if "random_walk" in byh.index and "random_walk" not in show:
        show.append("random_walk")
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for m in show:
        ax.plot(byh.columns, byh.loc[m].values, color=_color(m), lw=2.2 if m == show[0] else 1.4,
                ls="--" if "(" in m else "-", label=_plain(m))
    ax.set_xlabel("Quarters ahead", color=MUTED, fontsize=8)
    ax.set_ylabel(f"{METRICS[s['rank']['metric']]} ({s['target'].error_units()})", color=MUTED, fontsize=8)
    _ax(ax)
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper left")
    return _png(fig)


def chart_scenarios(s, macro, best, H, levels):
    t = s["target"]
    tot = s["scen_total"][best]
    hist = _hist_total(t)
    hist = hist[hist.index > macro.hist_end - 24]
    fut = [macro.hist_end + h for h in range(0, H + 1)]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.plot([q.to_timestamp() for q in hist.index], hist.values, color=INK, lw=1.8, label="History")
    x = [q.to_timestamp() for q in fut]
    last = hist.iloc[-1]
    for sc, by in tot.items():
        p = np.r_[last, by["Total"]]
        c = SCEN_COLORS.get(sc, "#555")
        ax.plot(x, p, color=c, lw=1.8, label=sc)
        if sc in ("baseline", "severe"):
            for lv, a in zip(sorted(levels), (0.22, 0.10)):
                lo, hi = [], []
                for h in range(1, H + 1):
                    ql, qh = s["qt"].get((best, h), {}).get(lv, (np.nan, np.nan))
                    v = by["Total"][h - 1]
                    lo.append(v * np.exp(ql) if t.form == "dlog" else v + ql)
                    hi.append(v * np.exp(qh) if t.form == "dlog" else v + qh)
                ax.fill_between(x[1:], lo, hi, color=c, alpha=a, lw=0)
    ax.axvline(x[0], color=GRID, lw=1)
    ax.set_ylabel(t.units, color=MUTED, fontsize=8)
    _ax(ax)
    ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper left")
    return _png(fig)


def chart_methods_severe(s, macro, H, scen="severe"):
    t = s["target"]
    hist = _hist_total(t)
    hist = hist[hist.index > macro.hist_end - 16]
    x = [q.to_timestamp() for q in [macro.hist_end + h for h in range(0, H + 1)]]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.plot([q.to_timestamp() for q in hist.index], hist.values, color=INK, lw=1.8, label="History")
    best = s["lb"].index[0]
    for m, by in s["scen_total"].items():
        if scen not in by or ("(identities)" in m and m != best):
            continue
        ax.plot(x, np.r_[hist.iloc[-1], by[scen]["Total"]], color=_color(m),
                ls="--" if "(" in m else "-", lw=1.4, label=_plain(m))
    ax.set_ylabel(t.units, color=MUTED, fontsize=8)
    _ax(ax)
    ax.legend(fontsize=6.5, frameon=False, ncol=2, loc="upper left")
    return _png(fig)


def chart_pip(s, macro):
    if "bma" not in s["fits"]:
        return None
    pip = s["fits"]["bma"].info["pip"]
    items = sorted(pip.items(), key=lambda kv: kv[1])
    if not items:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(items) + 0.6))
    labels = [f"{macro.describe(v)[:46]} (lag {l})" for (v, l), _ in items]
    ax.barh(labels, [p for _, p in items], color=COLORS["bma"], height=0.6)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Inclusion probability (share of BMA weight)", color=MUTED, fontsize=8)
    _ax(ax)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.grid(axis="y", visible=False)
    return _png(fig)


def _hist_total(t):
    df = pd.DataFrame(t.levels)
    return df.sum(axis=1, min_count=1) if t.form == "dlog" else df.mean(axis=1)


def _fmt(v, d=3):
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:,.{d}f}"


REF_LABELS = {"random_walk": "random walk", "ar1": "AR(1)", "drift": "drift", "class": "CLASS-style",
              "combination": "combination", "bma": "IMF BMA"}


def _ref(r):
    base = r.split(" (")[0]
    return REF_LABELS.get(base, base) + (" (rate x balance)" if "rate x balance" in r else "")


def _slug(n):
    return re.sub(r"[^A-Za-z0-9]+", "-", n).strip("-")


def lb_table(s):
    lb, rk = s["lb"], s["rank"]
    met = rk["metric"]
    refs = rk.get("refs") or []
    cols = [("avg_err", "Average error"), ("rmse", "RMSE"), ("p90", "90th pct"),
            ("cvar", f"Tail: worst {rk['tail_share'] * 100:.0f}%"), ("worst", "Largest"),
            ("combined", f"Average + {rk['penalty']:g} &times; tail excess")]
    key = "avg_err" if met == "mean" else met
    head = ("<tr><th>#</th><th>Method</th>" + "".join(
        f"<th class='{'rk' if k == key else ''}'>{lab}{' &#9650;' if k == key else ''}</th>" for k, lab in cols)
        + "".join(f"<th>vs {_ref(r)}</th>" for r in refs)
        + (f"<th>Beats {_ref(refs[0])} (share of origins)</th>" if refs else "")
        + "<th>80% band coverage</th><th>95% band coverage</th></tr>")
    rows = []
    beats = f"beats_{refs[0]}_share" if refs else None
    for i, (m, r) in enumerate(lb.iterrows(), 1):
        cells = "".join(f"<td class='{'rk' if k == key else ''}'>{'<b>' if k == key else ''}{_fmt(r.get(k))}"
                        f"{'</b>' if k == key else ''}</td>" for k, _ in cols)
        vs = "".join(f"<td>{_fmt(r.get(f'vs_{ref}_pct'), 0)}%</td>" for ref in refs)
        bt = (f"<td>{_fmt(r.get(beats) * 100 if pd.notna(r.get(beats)) else np.nan, 0)}%</td>" if beats else "")
        rows.append(
            f"<tr class='{'best' if i == 1 else ''}'><td>{i}</td><td><span class='dot' style='background:{_color(m)}'></span>"
            f"{_label(m)}</td>{cells}{vs}{bt}"
            f"<td>{_fmt(r.get('coverage_80', np.nan) * 100, 0)}%</td><td>{_fmt(r.get('coverage_95', np.nan) * 100, 0)}%</td></tr>")
    return f"<table class='lb'>{head}{''.join(rows)}</table>"


def settings_html(s, macro):
    st = s["settings"]
    est, rs = st["estimation"], st["restrictions"]
    bits = [f"Estimation: <b>{'panel (banks pooled, bank fixed effects)' if est['mode'] == 'panel' else 'each bank on its own'}</b>"]
    if rs.get("signs", True):
        bits.append("sign restrictions on: " + ", ".join(sorted(rs.get("apply_to", ["bma"]))))
    else:
        bits.append("sign restrictions <b>off</b>")
    for k, lab in (("force_in", "always in"), ("exclude", "excluded"), ("free_sign", "free sign")):
        if rs.get(k):
            bits.append(f"{lab}: " + ", ".join(html.escape(str(v)) for v in rs[k]))
    if rs.get("not_together"):
        bits.append("not together: " + "; ".join("{" + ", ".join(map(str, g)) + "}" for g in rs["not_together"]))
    if rs.get("include_unsigned"):
        bits.append("unsigned series allowed (free sign)")
    w = s["fits"].get("class")
    if w is not None and w.info.get("wrong_sign"):
        bits.append("<b>CLASS-style equation has a wrong sign:</b> " + ", ".join(w.info["wrong_sign"]))
    c = s["fits"].get("combination")
    if c is not None and c.info.get("dropped_for_sign"):
        bits.append(f"combination: {len(c.info['dropped_for_sign'])} equation(s) dropped for a wrong sign")
    return "<p class='note'>" + " &middot; ".join(bits) + "</p>"


def _kind(f):
    k = f[2] if len(f) > 2 else None
    return "" if k is None else ", rises only" if k == "+" else ", falls only" if k == "-" else ""


def bma_table(s, macro):
    top = s["fits"]["bma"].info["top"][:5]
    rows = []
    for e in top:
        d = "<br>".join(f"{html.escape(macro.describe(f[0]))} (lag {f[1]}{_kind(f)}): {b:+.4f}"
                        for f, b in zip(e["drivers"], e["beta"])) \
            or "<i>no macro driver (AR only)</i>"
        rows.append(f"<tr><td>{e['weight'] * 100:.1f}%</td><td>{e['rho']:.2f}</td><td>{d}</td></tr>")
    return ("<table class='small'><tr><th>BMA weight</th><th>&rho;</th><th>Drivers and coefficients</th></tr>"
            + "".join(rows) + "</table>")


def identity_block(s):
    ids = s.get("identities") or []
    if not ids:
        return ""
    out = []
    lb = s["lb"]
    for d in ids:
        if not d["active"]:
            out.append(f"<p class='note'>Identity <b>{html.escape(d['name'])}</b> ({html.escape(d['equation'])}): "
                       f"{html.escape(d['reason'])}.</p>")
            continue
        how = ("every target adjusted, weighted by " + {"errors": "out-of-sample error variance", "size": "size",
                                                         "equal": "equal weights"}[d["weights"]]
               if d["method"] == "reconcile" else f"{html.escape(str(d['residual']))} takes the whole gap")
        rows = []
        for m in [i for i in lb.index if "(" not in i]:
            k = m + " (identities)"
            if k in lb.index:
                a_, b_ = lb.loc[m, "score"], lb.loc[k, "score"]
                rows.append(f"<tr><td>{_label(m)}</td><td>{_fmt(a_)}</td><td>{_fmt(b_)}</td>"
                            f"<td>{_fmt((1 - b_ / a_) * 100, 1)}%</td></tr>")
        out.append(f"<h3>Identity: {html.escape(d['name'])}</h3><p class='note'>{html.escape(d['equation'])}; {how}. "
                   f"Largest gap in history: {_fmt(d['history_gap_pct'], 3)}% of the largest term; the scenario "
                   f"forecasts moved {_fmt(d['adjust_pct'], 3)}% on average to add up.</p>"
                   "<table class='small'><tr><th>Method</th><th>Score, own forecast</th><th>Score, with identity</th>"
                   "<th>Improvement</th></tr>" + "".join(rows) + "</table>")
    return "".join(out)


def bma_block(s, macro):
    if "bma" not in s["fits"]:
        return ""
    i = s["fits"]["bma"].info
    warn = f"<p class='note'><b>Warning:</b> {html.escape(i['warning'])}</p>" if i.get("warning") else ""
    return (f"<h3>IMF BMA: the five heaviest equations (all data)</h3><p class='note'>{i['n_models']} of {i['n_tried']} "
            f"equations passed the restrictions and share the weight.</p>" + warn + bma_table(s, macro))


def extra_block(s, macro):
    """Details of the asymmetric, spline, hybrid and direct fits on all the data."""
    f = s["fits"]
    out = []
    a = f.get("bma_asym")
    if a is not None and a.info.get("asym"):
        items = [(d, p) for d, p in a.info["asym"].items() if a.info["pip"].get(d, 0) >= 0.1]
        if items:
            out.append("<b>Asymmetric effects:</b> share of the BMA weight using separate rise/fall slopes: " + "; ".join(
                f"{html.escape(macro.describe(v))} (lag {l}) {p * 100:.0f}% of {a.info['pip'][(v, l)] * 100:.0f}%"
                for (v, l), p in items))
    sp = f.get("spline")
    if sp is not None:
        parts = []
        for k, sl in sp.info["slopes"].items():
            v = k.split("@")[0]
            kn = ", ".join(f"{x:.4g}" for x in sp.info["knots"][k])
            parts.append(f"{html.escape(macro.describe(v))}: slopes {', '.join(f'{x:+.3f}' for x in sl)} (knots {kn})")
        out.append("<b>Monotone piecewise-linear effects</b> (slope below, between and above the knots): " +
                   "; ".join(parts))
    hy = f.get("hybrid")
    if hy is not None:
        out.append(f"<b>Hybrid:</b> the trees explain {hy.info['tree_share'] * 100:.1f}% of the variance of the target "
                   f"on top of the linear equation; outside the range of history they add nothing.")
    for m in ("direct", "direct_class"):
        d = f.get(m)
        if d is not None:
            co = "; ".join(f"{html.escape(macro.describe(k.split('@')[0]))}: {v[0]:+.3f} at 1q, "
                           f"{v[min(3, len(v) - 1)]:+.3f} at 4q, {v[-1]:+.3f} at {len(v)}q"
                           for k, v in d.info["coef_h"].items())
            out.append(f"<b>{_label(m)}:</b> effect of each driver's cumulative change: {co or 'no drivers'}.")
    return "".join(f"<p class='note'>{x}</p>" for x in out)


def class_line(s, macro):
    f = s["fits"].get("class")
    if f is None:
        return ""
    parts = [f"{html.escape(macro.describe(k.split('@')[0]))} (lag {k.split('@')[1]}): {v:+.4f}" for k, v in f.info["coef"].items()]
    return f"<p class='note'><b>CLASS-style equation:</b> &rho; = {f.info['rho']:.2f}; " + "; ".join(parts) + "</p>"


def range_html(s, macro):
    rc = s["range"]
    if rc is None or not len(rc):
        return "<p class='note'>No BMA driver leaves its historical range in any scenario.</p>"
    rows = "".join(f"<tr><td>{html.escape(macro.describe(r.driver))}</td><td>{r.scenario}</td>"
                   f"<td>{r.quarters_below_history}</td><td>{r.quarters_above_history}</td>"
                   f"<td>{r.scenario_min:.4f} / {r.history_min:.4f}</td><td>{r.scenario_max:.4f} / {r.history_max:.4f}</td></tr>"
                   for r in rc.itertuples())
    return ("<p class='note'>Scenario values outside the range seen in history (the equations extrapolate there):</p>"
            "<table class='small'><tr><th>Driver</th><th>Scenario</th><th>Quarters below</th><th>Quarters above</th>"
            "<th>Min: scenario / history</th><th>Max: scenario / history</th></tr>" + rows + "</table>")


def write(summary, macro, cfg, out, quick=False):
    H = int(cfg["horizon"])
    levels = cfg["intervals"]["levels"]
    nav, body = [], []

    # overview
    ov = []
    for n, s in summary.items():
        lb = s["lb"]
        best = lb.index[0]
        rank_bma = list(lb.index).index("bma") + 1 if "bma" in lb.index else None
        ref = (s["rank"].get("refs") or [None])[0]
        ov.append(f"<tr><td><a href='#{_slug(n)}'>{html.escape(s['target'].label)}</a></td><td>{_label(best)}</td>"
                  f"<td>{METRICS[s['rank']['metric']]}</td>"
                  f"<td>{_fmt(lb.loc[best, 'score'])} {html.escape(s['target'].error_units())}</td>"
                  f"<td>{_fmt(lb.loc[best, 'avg_err'])}</td>"
                  f"<td>{(_fmt(lb.loc[best].get(f'vs_{ref}_pct'), 0) + '% vs ' + _ref(ref)) if ref else ''}</td><td>{rank_bma or ''}</td>"
                  f"<td>{_fmt(lb.loc[best].get('coverage_80', np.nan) * 100, 0)}%</td></tr>")
    first_o = cfg["oos"]["start"]
    body.append(f"""
<section id='overview'><h1>Public stress-testing methods, compared out-of-sample</h1>
<p class='lead'>Each target is forecast with methods published by the IMF, the Federal Reserve and the New York Fed,
and with three benchmarks. Every method is re-estimated at every origin from {first_o} on (every
{'quarter' if cfg['oos'].get('step', 1) == 1 else str(cfg['oos'].get('step')) + ' quarters'}), using only data up to that origin,
and forecasts 1&ndash;{H} quarters ahead with the macro values that actually happened. Errors are on the level:
percent of the actual level for balances and income, units (percentage points) for rates.
{'<b>Quick run: origins every 4th quarter.</b>' if quick else ''}</p>
<p class='lead'>Methods are ranked by the metric chosen in <code>ranking.metric</code>. The tail metrics (RMSE, 90th
percentile, the average of the worst errors, the largest error) punish a method that is right on average but
sometimes deviates a lot; every metric is computed at each horizon and averaged over horizons 1&ndash;{H}.</p>
<table class='lb'><tr><th>Target</th><th>Best method</th><th>Ranked by</th><th>Its score</th><th>Its average error</th>
<th>vs benchmark</th><th>Rank of IMF BMA</th><th>80% band coverage</th></tr>{''.join(ov)}</table>
</section>""")
    nav.append("<a href='#overview'>Overview</a>")

    mt = "".join(f"<h3>{t}</h3><p>{d}</p>" for _, t, d in METHOD_TEXT)
    src = "".join(f"<li><a href='{u}'>{html.escape(t)}</a></li>" for t, u in SOURCES)
    body.append(f"<section id='methods'><h2>The methods</h2>{mt}<h3>Sources</h3><ul>{src}</ul></section>")
    nav.append("<a href='#methods'>The methods</a>")

    for n, s in summary.items():
        t = s["target"]
        best = s["lb"].index[0]
        rxb = ""
        if any("rate x balance" in m for m in s["lb"].index):
            lb = s["lb"]
            pairs = []
            for m in [i for i in lb.index if "(" not in i]:
                k = f"{m} (rate x balance)"  # noqa
                if k in lb.index:
                    d, c = lb.loc[m, "score"], lb.loc[k, "score"]
                    pairs.append(f"<tr><td>{_label(m)}</td><td>{_fmt(d)}</td><td>{_fmt(c)}</td>"
                                 f"<td>{_fmt((1 - c / d) * 100, 0)}%</td></tr>")
            rxb = ("<h3>Federal Reserve approach: forecast the parts, then multiply</h3>"
                   "<table class='small'><tr><th>Method</th><th>Direct: forecast income itself</th>"
                   "<th>Rate &times; balance</th><th>Improvement</th></tr>" + "".join(pairs) + "</table>")
        pip_img = chart_pip(s, macro)
        body.append(f"""
<section id='{_slug(n)}'><h2>{html.escape(t.label)}</h2>
<p class='note'>{len(t.banks)} bank{'s' if len(t.banks) > 1 else ''} ({html.escape(', '.join(t.banks))}); modelled as {'log growth' if t.form == 'dlog' else 'quarterly change'};
errors in {html.escape(t.error_units())}; {len(s['signs'])} candidate drivers, {sum(1 for v in s['signs'].values() if v)} with a sign restriction.
Ranked by: <b>{METRICS[s['rank']['metric']]}</b>.</p>
{settings_html(s, macro)}
{lb_table(s)}
{rxb}
{identity_block(s)}
<div class='grid'>
<figure><img src='{chart_horizon(s)}'><figcaption>{METRICS[s['rank']['metric']]} by horizon (out-of-sample).</figcaption></figure>
<figure><img src='{chart_scenarios(s, macro, best, H, levels)}'><figcaption>Scenarios with the most accurate method
({_label(best)}){', total of the banks' if len(t.banks) > 1 and t.form == 'dlog' else ', average of the banks' if len(t.banks) > 1 else ''}.
Shaded: 80% and 95% conformal bands for baseline and severe.</figcaption></figure>
<figure><img src='{chart_methods_severe(s, macro, H)}'><figcaption>Severe scenario: every method.</figcaption></figure>
{f"<figure><img src='{pip_img}'><figcaption>IMF BMA on all the data: how much of the weight each screened driver carries.</figcaption></figure>" if pip_img else ''}
</div>
{bma_block(s, macro)}
{class_line(s, macro)}
{extra_block(s, macro)}
<h3>Scenario range check</h3>
{range_html(s, macro)}
</section>""")
        nav.append(f"<a href='#{_slug(n)}'>{html.escape(t.label)}</a>")

    body.append("""
<section id='notes'><h2>Reading the results</h2>
<ul>
<li>The test is conditional on the macro path that happened. That is the right test for scenario models, but real
forecasts also carry the error of the macro forecast itself.</li>
<li>The coverage columns should be close to 80% and 95%. They are computed in real time: early bands are built from the
errors of the calm years only, and then meet the shocks of 2020 and 2022, so coverage below the nominal level is the usual
result and an honest warning. The scenario bands are wider, because they use every out-of-sample error, shocks included.</li>
<li>The example data are simulated. On them the true drivers are known (see the data notes of the main project), so
the BMA inclusion probabilities can be checked against the truth; on real data they are evidence, not proof.</li>
<li>All equations are in changes. A target that is pulled back towards a level (a rate that tracks a bond yield,
say) is better served by an error-correction term, which none of these published methods uses by default.</li>
</ul></section>""")
    nav.append("<a href='#notes'>Reading the results</a>")

    page = f"""<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>Public stress-testing methods compared</title><style>
body{{margin:0;font:14px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:{INK};background:#fafbfc}}
nav{{position:fixed;top:0;left:0;width:230px;height:100vh;overflow:auto;background:#fff;border-right:1px solid {GRID};padding:20px 14px;box-sizing:border-box}}
nav a{{display:block;color:{MUTED};text-decoration:none;padding:5px 8px;border-radius:6px;font-size:13px}}
nav a:hover{{background:#eef2f7;color:{INK}}}
main{{margin-left:230px;padding:28px 44px;max-width:1180px}}
section{{background:#fff;border:1px solid {GRID};border-radius:10px;padding:22px 26px;margin-bottom:22px}}
h1{{font-size:24px;margin:0 0 8px}} h2{{font-size:19px;margin:0 0 6px}} h3{{font-size:15px;margin:18px 0 6px}}
.lead{{color:#3e4c59;max-width:900px}} .note{{color:{MUTED};font-size:13px}}
table{{border-collapse:collapse;width:100%;margin:8px 0;font-size:12.5px}}
th{{text-align:left;color:{MUTED};font-weight:600;border-bottom:1px solid {GRID};padding:6px 8px}}
td{{border-bottom:1px solid #f0f2f5;padding:6px 8px;font-variant-numeric:tabular-nums}}
tr.best td{{background:#f2f7fc}} th.rk{{color:#1b4f8a}} code{{background:#f0f2f5;padding:1px 4px;border-radius:4px}}
.dot{{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:7px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:12px}}
figure{{margin:0}} figure img{{width:100%}} figcaption{{color:{MUTED};font-size:12px}}
a{{color:#1b4f8a}}
@media(max-width:900px){{nav{{display:none}} main{{margin-left:0;padding:14px}} .grid{{grid-template-columns:1fr}}}}
</style></head><body><nav><b style='display:block;margin:0 8px 12px'>Contents</b>{''.join(nav)}</nav>
<main>{''.join(body)}</main></body></html>"""
    path = os.path.join(out, "report.html")
    with open(path, "w") as f:
        f.write(page)
    return path
