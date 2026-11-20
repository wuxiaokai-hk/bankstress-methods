"""Reading the inputs: macro scenarios, sign files and bank targets.

Everything is quarterly and indexed by pandas Period('Q').

Target forms
    dlog   y_t = ln L_t - ln L_t-1        balances, income: anything that is always positive
    diff   y_t = L_t - L_t-1              rates, spreads, ratios
An acquisition or divestiture recorded in the events file is taken out of the change in its quarter
(organic change), the same adjustment the Federal Reserve makes for mergers.
"""
import os

import numpy as np
import pandas as pd


def _q(s):
    return pd.PeriodIndex(pd.Series(s).astype(str).str.replace(" ", ""), freq="Q")


# ----------------------------------------------------------------------------------------------- macro
class Macro:
    """Macro levels and transformed changes for every scenario.

    levels[scen]  DataFrame (quarters x mnemonics), history + scenario path
    x[scen]       the same, transformed to quarterly change (dlog or diff, from the metadata)
    meta          metadata indexed by mnemonic
    hist_end      last quarter that is history (identical across scenarios)
    """

    def __init__(self, cfg, base):
        mc = cfg["macro"]
        self.meta = pd.read_csv(os.path.join(base, mc["metadata"])).set_index("mnemonic")
        dc = mc.get("date_column", "date")
        self.levels, self.x = {}, {}
        for name, f in mc["scenarios"].items():
            df = pd.read_csv(os.path.join(base, f))
            df.index = _q(df.pop(dc))
            df = df.apply(pd.to_numeric, errors="coerce")
            self.levels[name] = df
            self.x[name] = self.transform(df)
        self.scenarios = list(self.levels)
        self.hist_end = self._hist_end()

    def transform(self, df):
        out = {}
        for c in df.columns:
            t = self.meta["transform"].get(c, "dlog") if c in self.meta.index else "dlog"
            s = df[c]
            if t == "dlog" and (s.dropna() > 0).all():
                out[c] = np.log(s).diff()
            else:
                out[c] = s.diff()
        return pd.DataFrame(out, index=df.index)

    def _hist_end(self):
        first = self.levels[self.scenarios[0]]
        same = pd.Series(True, index=first.index)
        for s in self.scenarios[1:]:
            d = self.levels[s].reindex(first.index)
            same &= np.isclose(d.values, first.values, equal_nan=True).all(axis=1)
        diff_at = np.where(~same.values)[0]
        return first.index[diff_at[0] - 1] if len(diff_at) else first.index[-1]

    def history(self):
        """Transformed history (the scenarios agree on it)."""
        return self.x[self.scenarios[0]].loc[: self.hist_end]

    def describe(self, code):
        return str(self.meta["description"].get(code, code)) if code in self.meta.index else code


def load_signs(path, macro, cfg):
    """{mnemonic: +1/-1} for the candidate drivers of one target."""
    cc = cfg["candidates"]
    m = pd.read_csv(path).set_index("mnemonic")
    m = m[m["expected_sign"].fillna(0).astype(int) != 0]
    if cc.get("geographies"):
        m = m[m["geography"].isin(cc["geographies"])]
    if cc.get("exclude_themes"):
        m = m[~m["theme"].isin(cc["exclude_themes"])]
    m = m[m.index.isin(macro.levels[macro.scenarios[0]].columns)]
    return {k: int(np.sign(v)) for k, v in m["expected_sign"].items()}


def themes(macro):
    return macro.meta["theme"].to_dict()


# ---------------------------------------------------------------------------------------------- targets
class Target:
    """One target: a level series per bank and its (organic) change.

    levels  {bank: Series of levels}
    y       {bank: Series of changes (dlog or diff), organic}
    """

    def __init__(self, tc, base, frame_cache=None):
        self.cfg = tc
        self.name = tc["name"]
        self.label = tc.get("label", self.name)
        self.form = tc.get("form", "dlog")
        self.units = tc.get("units", "")
        frame = _read(os.path.join(base, tc["file"]), frame_cache)
        bc = tc.get("bank_column")
        if bc is None:
            frame = frame.assign(_bank="Bank")
            bc = "_bank"
        self.levels = {}
        for b, g in frame.groupby(bc, sort=True):
            g = g.set_index("_q").sort_index()
            if tc.get("expression"):
                s = g.eval(tc["expression"])
            else:
                s = g[tc["column"]]
            s = pd.to_numeric(s, errors="coerce").dropna()
            self.levels[str(b)] = s[~s.index.duplicated(keep="last")]
        if self.form == "dlog":
            bad = [b for b, s in self.levels.items() if (s <= 0).any()]
            if bad:
                raise ValueError(f"{self.name}: dlog needs positive values (not true for {bad}); use form: diff")
        events = _events(tc, base, bc != "_bank")
        self.y = {b: self._organic(s, events.get(b, {})) for b, s in self.levels.items()}
        self.banks = sorted(self.levels)

    def _organic(self, s, ev):
        y = np.log(s).diff() if self.form == "dlog" else s.diff()
        prev = s.shift(1)
        for q, a in ev.items():
            if q in s.index and q != s.index[0]:
                y.loc[q] = np.log((s.loc[q] - a) / prev.loc[q]) if self.form == "dlog" else s.loc[q] - a - prev.loc[q]
        return y

    # level after a path of changes, starting from level L0
    def rebuild(self, L0, ypath):
        ypath = np.asarray(ypath, float)
        return L0 * np.exp(np.cumsum(ypath)) if self.form == "dlog" else L0 + np.cumsum(ypath)

    def error(self, forecast, actual):
        """Signed error used for intervals: log ratio (dlog) or difference (diff)."""
        return np.log(actual / forecast) if self.form == "dlog" else actual - forecast

    def abs_error(self, forecast, actual):
        """Error reported in tables: percent of the actual level (dlog) or units (diff)."""
        return np.abs(forecast / actual - 1) * 100 if self.form == "dlog" else np.abs(forecast - actual)

    def error_units(self):
        return "% of the actual level" if self.form == "dlog" else self.units


def _read(path, cache):
    if cache is not None and path in cache:
        return cache[path]
    df = pd.read_csv(path)
    df["_q"] = _q(df["date"])
    if cache is not None:
        cache[path] = df
    return df


def _events(tc, base, panel):
    """{bank: {quarter: amount}} for the target's series; amounts of acquisitions are positive."""
    f = tc.get("events")
    if not f:
        return {}
    ev = pd.read_csv(os.path.join(base, f))
    col = tc.get("column")
    if "series" in ev.columns and col:
        ev = ev[ev["series"] == col]
    out = {}
    for _, r in ev.iterrows():
        b = str(r[tc["bank_column"]]) if panel else "Bank"
        out.setdefault(b, {})[pd.Period(str(r["date"]), freq="Q")] = float(r["amount"])
    return out
