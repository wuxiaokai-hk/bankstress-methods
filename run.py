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
    ap.add_argument("--targets", nargs="*", help="only these targets (parts of a rate x balance target are added)")
    ap.add_argument("--no-report", action="store_true")
    a = ap.parse_args(argv)

    t0 = time.time()
    cfg = yaml.safe_load(open(a.config))
    base = os.path.dirname(os.path.abspath(a.config))
    if a.quick:
        cfg["oos"]["step"] = 4
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
    for n in list(want):
        for p in (tcs[n].get("rate_x_balance") or {}).get("parts", []):
            if p not in want:
                want.insert(0, p)
    order = [n for n in tcs if n in want and not tcs[n].get("rate_x_balance")] + \
            [n for n in tcs if n in want and tcs[n].get("rate_x_balance")]

    cache = {}
    T, RES, SCEN, FITS = {}, {}, {}, {}
    summary = {}
    for n in order:
        tc = tcs[n]
        t = D.Target(tc, base, cache)
        signs = D.load_signs(os.path.join(base, tc["signs"]), macro, cfg)
        print(f"\n{t.label}: {len(t.banks)} bank(s), form {t.form}, {len(signs)} signed candidate drivers")
        r = V.run_target(t, macro, signs, themes, cfg)
        sc, fits = V.scenarios(t, macro, signs, themes, cfg)
        rxb = tc.get("rate_x_balance")
        if rxb:
            parts = rxb["parts"]
            cf = V.compose(t, parts, RES)
            cs = V.compose_scenarios(t, parts, SCEN)
            for m in cf:
                r["fc"][f"{m} (rate x balance)"] = cf[m]
            for m in cs:
                sc[f"{m} (rate x balance)"] = cs[m]
        T[n], RES[n], SCEN[n], FITS[n] = t, r, sc, fits

        # ---- scores
        E = V.errors(t, r["fc"], r["act"], H)
        lb, byh = V.leaderboard(E, H)
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

        rc = V.range_check(fits, macro, H)
        summary[n] = {"target": t, "lb": lb, "byh": byh, "E": E, "q": q, "qt": qt, "scen": sc, "scen_total": sct,
                      "fits": fits, "range": rc, "signs": signs}
        print(lb.round(3).to_string())

        # ---- files
        E.assign(target=n, origin=E["origin"].astype(str)).to_csv(os.path.join(out, f"errors_{n}.csv"), index=False)
        _scenario_rows(t, sc, q, sct, qt, macro, levels, H).to_csv(os.path.join(out, f"scenarios_{n}.csv"), index=False)

    lbs = pd.concat({n: s["lb"] for n, s in summary.items()}, names=["target", "method"]).reset_index()
    lbs.to_csv(os.path.join(out, "leaderboards.csv"), index=False)
    pip = []
    for n, s in summary.items():
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
