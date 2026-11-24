"""Fit every method at every origin, forecast, score, and forecast the scenarios.

The out-of-sample test is rolling-origin: at each origin every method is re-estimated (the BMA and the
combination also re-screen their drivers) on data up to the origin only, then forecasts 1..H quarters ahead
using the macro values that actually happened. Errors are measured on the level of the target.
"""
import numpy as np
import pandas as pd

from . import models as M


METHOD_LABELS = {
    "bma": "IMF: Bayesian model averaging, sign-constrained",
    "combination": "Guerrieri-Welch: forecast combination",
    "class": "NY Fed CLASS-style fixed equation",
    "bma_asym": "IMF BMA with asymmetric effects",
    "spline": "Monotone piecewise-linear effects",
    "hybrid": "Linear BMA + monotone boosted trees",
    "direct": "Direct (smooth local projection), BMA drivers",
    "direct_class": "Direct (smooth local projection), CLASS drivers",
    "ar1": "AR(1) benchmark",
    "drift": "Drift (average growth) benchmark",
    "random_walk": "Random walk (no change) benchmark",
}


def parse_drivers(lst):
    out = []
    for s in lst or []:
        v, _, l = str(s).partition("@")
        out.append((v.strip(), int(l or 0)))
    return out


def first_row(macro, cfg):
    lags = max(cfg["candidates"].get("max_lag", 2), cfg["methods"]["combination"].get("lags", 4))
    x = macro.history()
    return x.index[0] + lags + 1          # x starts with a NaN (the first change), then needs `lags` lags


ALL_METHODS = ["bma", "bma_asym", "spline", "hybrid", "direct", "direct_class", "combination", "class",
               "ar1", "drift", "random_walk"]
NEEDS_BMA = {"spline", "hybrid", "direct"}


def methods_to_run(cfg):
    """methods.run (default: all), plus every benchmark named in ranking.compare_against."""
    run = list(cfg["methods"].get("run") or ALL_METHODS)
    for b in (cfg.get("ranking", {}) or {}).get("compare_against") or []:
        base = b.split(" (")[0]
        if base not in run:
            run.append(base)
    bad = [m for m in run if m not in ALL_METHODS]
    if bad:
        raise ValueError(f"methods.run: unknown {bad}; choose from {ALL_METHODS}")
    return run


def fit_all(target, macro, signs, themes, origin, cfg, first):
    """Every chosen method on data up to `origin`: {method: ModelSet}."""
    rows = M.Rows(target, macro.history(), origin, first)
    restr = M.Restrictions(cfg)
    run = methods_to_run(cfg)
    out = {}
    H = int(cfg["horizon"])
    bma_set = None
    if "bma" in run or "bma_asym" in run or NEEDS_BMA & set(run):
        screened = M.screen(rows, signs, themes, cfg, signed=restr.uses_signs("bma"), restr=restr)
        bma_set = M.bma(rows, screened, signs, cfg, restr)
        bma_set.info["screened"] = screened
        if "bma" in run:
            out["bma"] = bma_set
        if "bma_asym" in run:
            out["bma_asym"] = M.bma(rows, screened, signs, cfg, restr, asym=True, name="bma_asym")
            out["bma_asym"].info["screened"] = screened
    if "combination" in run:
        cs = M.screen(rows, signs, themes, cfg, signed=restr.uses_signs("combination"),
                      n_keep=cfg["methods"]["combination"].get("n_vars", 10), restr=restr)
        out["combination"] = M.combination(rows, cs, cfg, signs, restr)
    class_drivers = parse_drivers(target.cfg.get("class_drivers"))
    if "class" in run and cfg["methods"].get("class", {}).get("enabled", True) and class_drivers:
        out["class"] = M.fixed(rows, class_drivers, "class", signs, check_signs=restr.uses_signs("class"))
    if bma_set is not None:
        core = M.median_model(bma_set)
        if "spline" in run:
            out["spline"] = M.spline(rows, core, signs, cfg, restr)
        if "hybrid" in run:
            out["hybrid"] = M.hybrid(rows, bma_set, signs, cfg, restr)
        if "direct" in run:
            out["direct"] = M.direct(target, macro.history(), origin, first, core, signs, cfg, H, restr, "direct")
    if "direct_class" in run and class_drivers:
        out["direct_class"] = M.direct(target, macro.history(), origin, first, class_drivers, signs, cfg, H, restr,
                                       "direct_class")
    for b in ("random_walk", "drift", "ar1"):
        if b in run:
            out[b] = getattr(M, b)(rows)
    return out


def forecast_bank(ms, target, b, origin, xpath, H):
    """Level path (H,) of bank b from origin, plus the change paths of each equation."""
    i = target.banks.index(b)
    y_last = target.y[b].get(origin, np.nan)
    if np.isnan(y_last):
        y_last = 0.0
    yp, P = ms.forecast(i, y_last, xpath, origin, H)
    L0 = float(target.levels[b].loc[origin])
    return target.rebuild(L0, yp), P


def actual_path(target, b, origin, H):
    """Organic actual levels after the origin (NaN where not yet known)."""
    y = target.y[b]
    L0 = float(target.levels[b].loc[origin])
    ys = np.array([y.get(origin + h, np.nan) for h in range(1, H + 1)])
    lv = target.rebuild(L0, np.nan_to_num(ys))
    gap = np.where(np.isnan(ys))[0]
    if len(gap):
        lv[gap[0]:] = np.nan          # after the first unknown quarter the level is unknown
    return lv


def origins(target, macro, cfg):
    last = max(s.index[-1] for s in target.levels.values())
    o = pd.Period(str(cfg["oos"]["start"]), freq="Q")
    step = int(cfg["oos"].get("step", 1))
    out = []
    while o < last:
        out.append(o)
        o += step
    return out


def run_target(target, macro, signs, themes, cfg, log=print):
    """Out-of-sample forecasts of every method.

    Returns {"fc": {method: {origin: {bank: level path}}}, "act": {origin: {bank: actual path}}, "fits": last fits}
    """
    H = int(cfg["horizon"])
    first = first_row(macro, cfg)
    xh = macro.history()
    fc, act = {}, {}
    ors = origins(target, macro, cfg)
    for k, o in enumerate(ors):
        fits = fit_all(target, macro, signs, themes, o, cfg, first)
        act[o] = {}
        for b in target.banks:
            if o not in target.levels[b].index:
                continue
            act[o][b] = actual_path(target, b, o, H)
            for m, ms in fits.items():
                lv, _ = forecast_bank(ms, target, b, o, xh, H)
                fc.setdefault(m, {}).setdefault(o, {})[b] = lv
        if k % 10 == 0 or k == len(ors) - 1:
            log(f"  {target.name}: origin {o} ({k + 1}/{len(ors)}), BMA {fits['bma'].info['n_models']} admissible "
                f"equations")
    return {"fc": fc, "act": act}


def compose(spec_target, parts, results):
    """Fed approach: rebuild the target from the parts' forecasts with the target's expression.

    parts: names of part targets; results: {name: run_target output}. Returns forecasts keyed by method.
    """
    expr = spec_target.cfg["expression"]
    fc = {}
    methods = set.intersection(*[set(results[p]["fc"]) for p in parts]) - {"random_walk"}   # same as direct
    for m in methods:
        for o, banks in results[parts[0]]["fc"][m].items():
            for b in banks:
                env = {p: results[p]["fc"][m].get(o, {}).get(b) for p in parts}
                if any(v is None for v in env.values()):
                    continue
                val = _eval(expr, env)
                fc.setdefault(m, {}).setdefault(o, {})[b] = np.asarray(val, float)
    return fc


def _eval(expr, env):
    return eval(expr, {"__builtins__": {}}, env)


# --------------------------------------------------------------------------------------------- scoring
def errors(target, fc, act, H):
    """Long table of errors: method, origin, bank, h, abs error (reported units), signed error (for bands)."""
    rec = []
    for m, byo in fc.items():
        for o, banks in byo.items():
            for b, f in banks.items():
                a = act.get(o, {}).get(b)
                if a is None:
                    continue
                for h in range(H):
                    if np.isnan(a[h]) or np.isnan(f[h]) or (target.form == "dlog" and f[h] <= 0):
                        continue
                    rec.append((m, o, b, h + 1, float(target.abs_error(f[h], a[h])), float(target.error(f[h], a[h]))))
    return pd.DataFrame(rec, columns=["method", "origin", "bank", "h", "abs_err", "err"])


METRICS = {
    "mean": "Average error",
    "rmse": "Root mean squared error",
    "p90": "90th percentile of errors",
    "cvar": "Average of the worst errors (tail)",
    "worst": "Largest single error",
    "combined": "Average + penalty x tail excess",
}


def _metrics(a, tail_share, penalty):
    """Error metrics of one set of absolute errors."""
    a = np.sort(np.asarray(a, float))
    n = len(a)
    if not n:
        return {k: np.nan for k in METRICS}
    k = max(int(np.ceil(n * tail_share)), 1)
    mean = a.mean()
    cvar = a[-k:].mean()
    return {"mean": mean, "rmse": float(np.sqrt((a ** 2).mean())), "p90": float(np.quantile(a, 0.9)),
            "cvar": float(cvar), "worst": float(a[-1]), "combined": float(mean + penalty * (cvar - mean))}


def leaderboard(E, H, ranking=None):
    """One row per method, ordered by the ranking metric.

    Every metric is computed horizon by horizon over all origins and banks, then averaged over horizons 1..H,
    so long horizons (whose errors are naturally larger) do not dominate the tail metrics.

        mean      average absolute error (for balances: % of the level, i.e. a MAPE)
        rmse      root mean squared error: squares the errors, so large deviations weigh more
        p90       the error exceeded in 10% of cases
        cvar      average of the worst tail_share of errors (expected shortfall): how bad the bad cases are
        worst     the single largest error, at any horizon (not averaged)
        combined  mean + penalty * (cvar - mean): the average, plus a charge for the excess of the bad cases
    """
    ranking = ranking or {}
    metric = ranking.get("metric", "mean")
    if metric not in METRICS:
        raise ValueError(f"ranking.metric must be one of {list(METRICS)}")
    tail, pen = float(ranking.get("tail_share", 0.10)), float(ranking.get("penalty", 0.5))
    per = {}
    for (m, h), g in E.groupby(["method", "h"]):
        per[(m, h)] = _metrics(g["abs_err"].values, tail, pen)
    P = pd.DataFrame(per).T
    P.index.names = ["method", "h"]
    out = P.groupby(level="method").mean()
    out["worst"] = E.groupby("method")["abs_err"].max()            # the single largest error, any horizon
    out.insert(0, "score", out[metric])
    byh = P[metric].unstack("h")
    means = P["mean"].unstack("h")
    for h in (1, 4, 8, H):
        if h in means.columns:
            out[f"h{h}"] = means[h]
    refs = ranking.get("compare_against") or ["random_walk", "ar1"]
    for ref in refs:
        if ref in out.index:
            out[f"vs_{ref}_pct"] = (1 - out["score"] / out.loc[ref, "score"]) * 100
    # how often the method beats the first benchmark, origin by origin (bank- and horizon-averaged)
    po = E.groupby(["method", "origin"])["abs_err"].mean().unstack("method")
    ref = next((r for r in refs if r in po.columns), None)
    if ref:
        out[f"beats_{ref}_share"] = [(po[m] < po[ref]).mean() if m in po else np.nan for m in out.index]
    out = out.rename(columns={"mean": "avg_err"})
    out.attrs.update(metric=metric, refs=[r for r in refs if r in out.index], tail_share=tail, penalty=pen)
    return out.sort_values("score"), byh


def conformal_quantiles(E, levels):
    """Signed-error quantiles by method and horizon for each band level: {(method, h): {level: (lo, hi)}}."""
    q = {}
    for (m, h), g in E.groupby(["method", "h"]):
        e = np.sort(g["err"].values)
        n = len(e)
        q[(m, h)] = {lv: _band(e, lv) for lv in levels} if n else {}
    return q


def _band(e, lv):
    n = len(e)
    a = (1 - lv) / 2
    # finite-sample conformal ranks: the k-th smallest with k = ceil((n+1)*p)
    lo_k = max(int(np.floor((n + 1) * a)), 1)
    hi_k = min(int(np.ceil((n + 1) * (1 - a))), n)
    return float(e[lo_k - 1]), float(e[hi_k - 1])


def coverage(E, levels, min_cal):
    """Honest coverage: the band at origin o uses only errors already known at o (origin + h <= o)."""
    out = {}
    for m, g in E.groupby("method"):
        hits = {lv: [] for lv in levels}
        for h, gh in g.groupby("h"):
            gh = gh.sort_values("origin")
            ors = gh["origin"].values
            errs = gh["err"].values
            for o in np.unique(ors):
                cal = errs[ors + h <= o]
                if len(cal) < min_cal:
                    continue
                cal = np.sort(cal)
                now = errs[ors == o]
                for lv in levels:
                    lo, hi = _band(cal, lv)
                    hits[lv] += list((now >= lo) & (now <= hi))
        out[m] = {lv: (float(np.mean(v)) if v else np.nan) for lv, v in hits.items()}
    return out


# ------------------------------------------------------------------------------------------- scenarios
def scenarios(target, macro, signs, themes, cfg, fits=None):
    """Forecast every scenario from the last history quarter with models fitted on all the data."""
    H = int(cfg["horizon"])
    origin = macro.hist_end
    fits = fits or fit_all(target, macro, signs, themes, origin, cfg, first_row(macro, cfg))
    out = {}
    for s in macro.scenarios:
        xp = macro.x[s]
        for m, ms in fits.items():
            for b in target.banks:
                if origin not in target.levels[b].index:
                    continue
                lv, P = forecast_bank(ms, target, b, origin, xp, H)
                out.setdefault(m, {}).setdefault(s, {})[b] = lv
                if m == "bma":
                    out.setdefault("_bma_models", {}).setdefault(s, {})[b] = np.array(
                        [target.rebuild(float(target.levels[b].loc[origin]), p) for p in P])
    return out, fits


def compose_scenarios(spec_target, parts, scen):
    expr = spec_target.cfg["expression"]
    out = {}
    methods = set.intersection(*[set(k for k in scen[p] if not k.startswith("_")) for p in parts]) - {"random_walk"}
    for m in methods:
        for s, banks in scen[parts[0]][m].items():
            for b in banks:
                env = {p: scen[p][m][s].get(b) for p in parts}
                if any(v is None for v in env.values()):
                    continue
                out.setdefault(m, {}).setdefault(s, {})[b] = np.asarray(_eval(expr, env), float)
    return out


def range_check(fits, macro, H):
    """Drivers of the final BMA (inclusion probability >= 0.1) that leave their historical range in a scenario."""
    ms = fits["bma"]
    rows = []
    hist = macro.history()
    for (v, l), p in ms.info["pip"].items():
        if p < 0.1:
            continue
        lo, hi = hist[v].min(), hist[v].max()
        for s in macro.scenarios:
            path = macro.x[s][v].loc[macro.hist_end + 1: macro.hist_end + H]
            out_lo, out_hi = int((path < lo).sum()), int((path > hi).sum())
            if out_lo or out_hi:
                rows.append({"driver": v, "lag": l, "inclusion": p, "scenario": s,
                             "quarters_below_history": out_lo, "quarters_above_history": out_hi,
                             "scenario_min": float(path.min()), "history_min": float(lo),
                             "scenario_max": float(path.max()), "history_max": float(hi)})
    return pd.DataFrame(rows)
