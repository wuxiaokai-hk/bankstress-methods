"""Run every public method on every target, test them out-of-sample, forecast the scenarios, write the report.

    python run.py                      # config.yaml, full test
    python run.py --quick              # origins every 4th quarter (a fast first look)
    python run.py --targets loans_panel nii
    python run.py --config my.yaml
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from methods import data as D          # noqa: E402
from methods import evaluate as V      # noqa: E402
from methods import report as R        # noqa: E402
from methods import identities as I    # noqa: E402


def total(fc_or_act, form):
    """Sum of banks (balances, dlog) or average (rates, diff) for each method/origin or scenario."""
    out = {}
    for k, banks in fc_or_act.items():
        paths = [p for p in banks.values() if p is not None]
        if not paths:
            continue
        A = np.vstack(paths)
        out[k] = {"Total": A.sum(axis=0) if form == "dlog" else A.mean(axis=0)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--quick", action="store_true", help="origins every 4th quarter")
    ap.add_argument("--targets", nargs="*", help="only these targets (parts of a rate x balance target run as needed)")
    ap.add_argument("--mode", choices=["panel", "separate"], help="override estimation.mode for every target")
    ap.add_argument("--banks", nargs="*", help="override estimation.banks for every target (names, or: all)")
    ap.add_argument("--metric", choices=["mean", "rmse", "p90", "cvar", "worst", "combined"],
                    help="override ranking.metric")
    ap.add_argument("--identities", choices=["on", "off"], help="override identities_on")
    ap.add_argument("--no-report", action="store_true")
    a = ap.parse_args(argv)

    t0 = time.time()
    cfg = yaml.safe_load(open(a.config))
    base = os.path.dirname(os.path.abspath(a.config))
    if a.quick:
        cfg["oos"]["step"] = 4
    if a.mode or a.banks:
        cfg["estimation"] = dict(cfg.get("estimation") or {})
        if a.mode:
            cfg["estimation"]["mode"] = a.mode
        if a.banks:
            cfg["estimation"]["banks"] = "all" if a.banks == ["all"] else a.banks
        for t in cfg["targets"]:
            t.pop("estimation", None)
    if a.metric:
        cfg["ranking"] = dict(cfg.get("ranking") or {}, metric=a.metric)
    out = os.path.join(base, cfg.get("output_dir", "outputs"))
    os.makedirs(out, exist_ok=True)
    H = int(cfg["horizon"])
    levels = cfg["intervals"]["levels"]

    macro = D.Macro(cfg, base)
    themes = D.themes(macro)
    print(f"Macro: {macro.levels[macro.scenarios[0]].shape[1]} series, history to {macro.hist_end}, "
          f"scenarios {', '.join(macro.scenarios)}")

    tcs = {t["name"]: t for t in cfg["targets"]}
    want = a.targets or list(tcs)
    unknown = [n for n in want if n not in tcs]
    if unknown:
        raise SystemExit(f"Unknown targets {unknown}; the config has {list(tcs)}")
    order = [n for n in tcs if n in want]

    cache = {}
    done = {}            # (target, mode, banks) -> {unit: (target, results, scenarios, fits, signs)}
    summary = {}

    def estimate(tc, tcfg, est, banks):
        """Run one target in one estimation mode for some banks; parts of rate x balance first."""
        key = (tc["name"], est["mode"], tuple(banks))
        if key in done:
            return done[key]
        t = D.Target(tc, base, cache).subset(banks)
        signs = D.load_signs(os.path.join(base, tc["signs"]), macro, tcfg)
        units = {}
        if est["mode"] == "panel":
            units[tc["name"]] = t
        else:
            for b in banks:
                units[f"{tc['name']} | {b}"] = t.subset([b], name=f"{tc['name']} | {b}", label=f"{t.label} - {b}")
        res = {}
        for u, tu in units.items():
            print(f"\n{tu.label}: {len(tu.banks)} bank(s), {est['mode']}, form {tu.form}, {len(signs)} candidate drivers "
                  f"({sum(1 for v in signs.values() if v)} with a sign restriction)")
            r = V.run_target(tu, macro, signs, themes, tcfg)
            sc, fits = V.scenarios(tu, macro, signs, themes, tcfg)
            res[u] = (tu, r, sc, fits, signs)
        rxb = tc.get("rate_x_balance")
        if rxb:
            parts = rxb["parts"]
            pres = {p: estimate(tcs[p], settings(cfg, tcs[p], est), est, banks) for p in parts}
            for u, (tu, r, sc, fits, signs) in res.items():
                suffix = u[len(tc["name"]):]                      # '' (panel) or ' | Bank A'
                R_ = {p: pres[p][p + suffix][1] for p in parts}
                S_ = {p: pres[p][p + suffix][2] for p in parts}
                for m, f in V.compose(tu, parts, R_).items():
                    r["fc"][f"{m} (rate x balance)"] = f
                for m, f in V.compose_scenarios(tu, parts, S_).items():
                    sc[f"{m} (rate x balance)"] = f
        done[key] = res
        return res

    # ---- 1. estimate every target (and the parts of rate x balance targets)
    units = {}
    for n in order:
        tc = tcs[n]
        tcfg = settings(cfg, tc)
        est = tcfg["estimation"]
        allbanks = D.Target(tc, base, cache).banks
        banks = allbanks if est.get("banks", "all") in ("all", None) else [str(b) for b in est["banks"]]
        for u, (t, r, sc, fits, signs) in estimate(tc, tcfg, est, banks).items():
            units[u] = {"base": n, "suffix": u[len(n):], "target": t, "r": r, "sc": sc, "fits": fits,
                        "signs": signs, "settings": tcfg}

    # ---- 2. accounting identities between targets
    idents = [I.Identity(x) for x in cfg.get("identities") or []]
    E0 = {u: V.errors(x["target"], x["r"]["fc"], x["r"]["act"], H) for u, x in units.items()} if idents else {}
    switch = cfg.get("identities_on", True) if a.identities is None else a.identities == "on"
    ident_report = I.apply(idents, units, E0, H, switch=switch)

    # ---- 3. scores, bands, files
    for u, x in units.items():
        t, r, sc, fits, signs, tcfg = x["target"], x["r"], x["sc"], x["fits"], x["signs"], x["settings"]
        if True:
            # ---- scores
            E = V.errors(t, r["fc"], r["act"], H)
            lb, byh = V.leaderboard(E, H, tcfg.get("ranking"))
            cov = V.coverage(E, levels, cfg["intervals"]["min_calibration"])
            for lv in levels:
                lb[f"coverage_{int(lv * 100)}"] = [cov.get(m, {}).get(lv, np.nan) for m in lb.index]
            q = V.conformal_quantiles(E, levels)

            # totals (sum of banks / average rate) for charts and their own bands
            fct = {m: total_by_origin(byo, t.form) for m, byo in r["fc"].items()}
            actt = total_by_origin(r["act"], t.form)
            Et = V.errors(t, fct, actt, H)
            qt = V.conformal_quantiles(Et, levels)
            sct = {m: total(byscen, t.form) for m, byscen in sc.items() if not m.startswith("_")}

            rc = V.range_check(fits, macro, H) if "bma" in fits else pd.DataFrame()
            summary[u] = {"target": t, "lb": lb, "byh": byh, "E": E, "q": q, "qt": qt, "scen": sc, "scen_total": sct,
                          "fits": fits, "range": rc, "signs": signs, "settings": tcfg,
                          "identities": [d for d in ident_report if x["base"] in I.parse(d["equation"])],
                          "rank": {"metric": lb.attrs.get("metric"), "refs": lb.attrs.get("refs"),
                                   "tail_share": lb.attrs.get("tail_share"), "penalty": lb.attrs.get("penalty")}}
            print(f"\n{t.label}")
            print(lb.drop(columns=[c for c in lb.columns if c.startswith("h")]).round(3).to_string())

            # ---- files
            safe = u.replace(" | ", "__").replace(" ", "_")
            E.assign(target=u, origin=E["origin"].astype(str)).to_csv(os.path.join(out, f"errors_{safe}.csv"), index=False)
            _scenario_rows(t, sc, q, sct, qt, macro, levels, H).to_csv(os.path.join(out, f"scenarios_{safe}.csv"), index=False)
    if ident_report:
        pd.DataFrame([{k: (", ".join(v) if isinstance(v, list) else v) for k, v in d.items()} for d in ident_report]
                     ).to_csv(os.path.join(out, "identities.csv"), index=False)

    lbs = pd.concat({n: s["lb"] for n, s in summary.items()}, names=["target", "method"]).reset_index()
    lbs.to_csv(os.path.join(out, "leaderboards.csv"), index=False)
    pip = []
    for n, s in summary.items():
        if "bma" not in s["fits"]:
            continue
        bm = s["fits"]["bma"]
        for (v, l), p in bm.info["pip"].items():
            pip.append({"target": n, "driver": v, "lag": l, "inclusion_probability": p,
                        "description": macro.describe(v), "expected_sign": s["signs"].get(v)})
    pd.DataFrame(pip).to_csv(os.path.join(out, "bma_inclusion.csv"), index=False)
    rcs = [s["range"].assign(target=n) for n, s in summary.items() if len(s["range"])]
    if rcs:
        pd.concat(rcs).to_csv(os.path.join(out, "scenario_range_check.csv"), index=False)
    with open(os.path.join(out, "run.json"), "w") as f:
        json.dump({"config": os.path.abspath(a.config), "quick": a.quick, "targets": order,
                   "seconds": round(time.time() - t0, 1)}, f, indent=2)
    if not a.no_report:
        path = R.write(summary, macro, cfg, out, quick=a.quick)
        print(f"\nReport: {path}")
    print(f"Done in {time.time() - t0:.0f} s. Outputs in {out}")
    return summary


DEFAULT_SETTINGS = {
    "estimation": {"mode": "panel", "banks": "all"},
    "restrictions": {"signs": True, "apply_to": ["bma"], "free_sign": [], "force_in": [], "exclude": [],
                     "not_together": [], "include_unsigned": False},
    "ranking": {"metric": "mean", "tail_share": 0.10, "penalty": 0.5, "compare_against": ["random_walk", "ar1"]},
}


def _merge(a, b):
    out = dict(a)
    for k, v in (b or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def settings(cfg, tc, est=None):
    """The config seen by one target: defaults < global settings < the target's own settings.
    est forces the estimation mode and banks (the parts of a rate x balance target follow the target)."""
    c = _merge(cfg, {})
    for k, d in DEFAULT_SETTINGS.items():
        c[k] = _merge(_merge(d, cfg.get(k)), tc.get(k))
    if tc.get("methods"):
        c["methods"] = _merge(cfg["methods"], tc["methods"])
    if est is not None:
        c["estimation"] = dict(est)
    if c["estimation"]["mode"] not in ("panel", "separate"):
        raise SystemExit(f"{tc['name']}: estimation.mode must be panel or separate")
    return c


def total_by_origin(byo, form):
    """{origin: {bank: path}} -> {origin: {'Total': path}}, only where every bank active at the origin is known."""
    out = {}
    for o, banks in byo.items():
        paths = list(banks.values())
        A = np.vstack(paths)
        out[o] = {"Total": A.sum(axis=0) if form == "dlog" else A.mean(axis=0)}
    return out


def _scenario_rows(t, sc, q, sct, qt, macro, levels, H):
    dates = [str(macro.hist_end + h) for h in range(1, H + 1)]
    rows = []

    def add(m, s, b, path, qq):
        for h in range(H):
            r = {"method": m, "scenario": s, "bank": b, "date": dates[h], "forecast": path[h]}
            for lv in levels:
                lo, hi = qq.get((m, h + 1), {}).get(lv, (np.nan, np.nan))
                tag = int(lv * 100)
                if t.form == "dlog":
                    r[f"lo{tag}"], r[f"hi{tag}"] = path[h] * np.exp(lo), path[h] * np.exp(hi)
                else:
                    r[f"lo{tag}"], r[f"hi{tag}"] = path[h] + lo, path[h] + hi
            rows.append(r)

    for m, byscen in sc.items():
        if m.startswith("_"):
            continue
        for s, banks in byscen.items():
            for b, p in banks.items():
                add(m, s, b, p, q)
            if len(t.banks) > 1:
                add(m, s, "Total", sct[m][s]["Total"], qt)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
