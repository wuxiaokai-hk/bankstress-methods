"""Accounting identities between targets, e.g. total loans = mortgages + commercial loans.

Each target is forecast on its own, so the forecasts need not add up. When an identity is switched on and every
target in it is forecast in the run, the forecasts are made to add up in one of two ways:

    reconcile  every target moves a little; the adjustment is shared in proportion to how uncertain each forecast
               is (its out-of-sample error variance at that horizon), so precise forecasts move least. This is the
               weighted least squares projection of forecast reconciliation (Hyndman et al. 2011; Wickramasuriya,
               Athanasopoulos & Hyndman 2019, "MinT" with a diagonal covariance).
    residual   one named target takes the whole gap (e.g. total loans = the sum of the forecast parts).

If the identity does not hold exactly in history (other items not modelled), the gap at the forecast origin is
carried forward, so the identity holds with that gap.

The reconciled forecasts are added as methods called "<method> (identities)", next to the unadjusted ones, so the
leaderboard shows whether imposing the identity helps.
"""
import re

import numpy as np

SUFFIX = " (identities)"


def parse(expr):
    """'total = a + b - 0.5*c' -> {'total': 1, 'a': -1, 'b': -1, 'c': 0.5} (all terms moved to one side)."""
    if "=" not in expr:
        raise ValueError(f"identity '{expr}' needs an '='")
    lhs, rhs = expr.split("=", 1)
    out = {}
    for side, sgn in ((lhs, 1.0), (rhs, -1.0)):
        s = side.replace(" ", "")
        if not s:
            raise ValueError(f"identity '{expr}': empty side")
        if s[0] not in "+-":
            s = "+" + s
        for sign, term in re.findall(r"([+-])([^+-]+)", s):
            m = re.fullmatch(r"(?:([0-9.]+)\*)?([A-Za-z_][A-Za-z0-9_]*)(?:\*([0-9.]+))?", term)
            if not m:
                raise ValueError(f"identity '{expr}': cannot read term '{term}' (use name, number*name)")
            c = float(m.group(1) or m.group(3) or 1.0) * (1 if sign == "+" else -1) * sgn
            out[m.group(2)] = out.get(m.group(2), 0.0) + c
    return out


class Identity:
    def __init__(self, spec):
        if isinstance(spec, str):
            spec = {"equation": spec}
        self.equation = spec["equation"]
        self.name = spec.get("name", self.equation)
        self.coefs = parse(self.equation)
        self.method = spec.get("method", "reconcile")
        self.residual = spec.get("residual")
        self.enforce = spec.get("enforce", "auto")          # auto | true | false
        self.weights = spec.get("weights", "errors")         # errors | size | equal
        if self.method not in ("reconcile", "residual"):
            raise ValueError(f"identity {self.name}: method must be reconcile or residual")
        if self.method == "residual" and self.residual not in self.coefs:
            raise ValueError(f"identity {self.name}: residual must be one of {list(self.coefs)}")
        if self.weights not in ("errors", "size", "equal"):
            raise ValueError(f"identity {self.name}: weights must be errors, size or equal")

    @property
    def targets(self):
        return list(self.coefs)

    def status(self, available, switch=True):
        """(active, reason)."""
        if not switch:
            return False, "identities switched off (identities_on: false)"
        if self.enforce in (False, "false", "never"):
            return False, "switched off for this identity (enforce: false)"
        missing = [t for t in self.targets if t not in available]
        if missing:
            if self.enforce in (True, "true", "always"):
                raise SystemExit(f"identity {self.name}: enforce is true but {missing} are not in this run; "
                                 f"add them with --targets, or set enforce: auto")
            return False, f"not imposed: {', '.join(missing)} not forecast in this run"
        return True, "imposed"

    def adjust(self, F, W, gap):
        """Make sum_i a_i F_i = gap at every horizon.

        F: (k, H) forecasts in target order; W: (k, H) variances (weights); gap: (k-independent) scalar or (H,).
        """
        a = np.array([self.coefs[t] for t in self.targets])[:, None]
        miss = (a * F).sum(axis=0) - gap
        if self.method == "residual":
            r = self.targets.index(self.residual)
            G = F.copy()
            G[r] = F[r] - miss / a[r, 0]
            return G
        denom = (a ** 2 * W).sum(axis=0)
        denom = np.where(denom > 0, denom, 1.0)
        return F - W * a * (miss / denom)


def history_gap(ident, levels):
    """Gap sum_i a_i L_i in history, by bank: {bank: Series}; levels {target: {bank: Series}}."""
    banks = set.intersection(*[set(levels[t]) for t in ident.targets])
    out = {}
    for b in banks:
        s = None
        for t in ident.targets:
            x = ident.coefs[t] * levels[t][b]
            s = x if s is None else s.add(x, fill_value=np.nan)
        out[b] = s.dropna()
    return out


# ------------------------------------------------------------------------------------------ applying them
class _ErrVar:
    """Squared out-of-sample errors of one target, by method and horizon, to weight a reconciliation."""

    def __init__(self, E):
        self.d = {}
        for (m, h), g in E.groupby(["method", "h"]):
            o = np.array([p.ordinal for p in g["origin"]])
            self.d[(m, h)] = (o, g["err"].values ** 2)

    def var(self, m, h, before=None, min_n=8):
        if (m, h) not in self.d:
            return np.nan
        o, e2 = self.d[(m, h)]
        if before is not None:
            e2 = e2[o + h <= before]
        return float(e2.mean()) if len(e2) >= min_n else np.nan


def _weights(ident, units, F, m, H, ev, before):
    """(k, H) variances of the forecasts F (levels) in target order."""
    k = len(ident.targets)
    if ident.weights == "equal":
        return np.ones((k, H))
    W = np.zeros((k, H))
    for i, t in enumerate(ident.targets):
        tu = units[t]["target"]
        for h in range(H):
            scale = F[i, h] ** 2 if tu.form == "dlog" else 1.0
            v = ev[t].var(m, h + 1, before) if ident.weights == "errors" else np.nan
            if np.isnan(v):                                  # too few errors yet: proportional to size
                v = 1.0 if tu.form == "dlog" else np.nan
            W[i, h] = v * scale if not np.isnan(v) else F[i, h] ** 2
    return W


def apply(idents, units, E, H, switch=True, log=print):
    """Impose the active identities on the forecasts of every unit.

    units: {unit name: {"base": target name, "suffix": '' or ' | Bank A', "target", "r", "sc"}}
    E: {unit name: unadjusted error table}. Adds methods '<m> (identities)' to r['fc'] and sc in place.
    Returns a list of dicts describing each identity and what it did.
    """
    report = []
    by_base = {}
    for u, x in units.items():
        by_base.setdefault(x["base"], {})[x["suffix"]] = u
    for ident in idents:
        active, why = ident.status(set(by_base), switch)
        rec = {"name": ident.name, "equation": ident.equation, "method": ident.method,
               "residual": ident.residual, "weights": ident.weights, "active": active, "reason": why,
               "units": [], "history_gap_pct": np.nan, "adjust_pct": np.nan}
        if not active:
            log(f"Identity {ident.name}: {why}")
            report.append(rec)
            continue
        suffixes = set.intersection(*[set(by_base[t]) for t in ident.targets])
        if not suffixes:
            rec.update(active=False, reason="not imposed: its targets use different estimation modes or banks")
            log(f"Identity {ident.name}: {rec['reason']}")
            report.append(rec)
            continue
        gaps, adj = [], []
        for sfx in sorted(suffixes):
            U = {t: units[by_base[t][sfx]] for t in ident.targets}
            ev = {t: _ErrVar(E[by_base[t][sfx]]) for t in ident.targets}
            methods = set.intersection(*[set(U[t]["r"]["fc"]) for t in ident.targets])
            methods = sorted(m for m in methods if SUFFIX not in m and "(" not in m and m != "random_walk")
            # (a random walk is already coherent: every level stays where it is)
            levels = {t: U[t]["target"].levels for t in ident.targets}
            hg = history_gap(ident, levels)
            for b, g in hg.items():
                size = max(np.abs(ident.coefs[t] * levels[t][b]).max() for t in ident.targets)
                gaps.append(float(np.abs(g).max() / size * 100) if len(g) else np.nan)
            # out-of-sample forecasts
            for m in methods:
                fcs = [U[t]["r"]["fc"][m] for t in ident.targets]
                for o in fcs[0]:
                    banks = set.intersection(*[set(f.get(o, {})) for f in fcs])
                    for b in banks:
                        F = np.vstack([f[o][b] for f in fcs])
                        gap = hg.get(b)
                        g0 = float(gap.get(o, 0.0)) if gap is not None else 0.0
                        W = _weights(ident, U, F, m, H, ev, o.ordinal)
                        G = ident.adjust(F, W, g0)
                        for i, t in enumerate(ident.targets):
                            U[t]["r"]["fc"].setdefault(m + SUFFIX, {}).setdefault(o, {})[b] = G[i]
            # scenario forecasts (weights from every error)
            for m in methods:
                scs = [U[t]["sc"].get(m, {}) for t in ident.targets]
                for s in scs[0]:
                    banks = set.intersection(*[set(x.get(s, {})) for x in scs])
                    for b in banks:
                        F = np.vstack([x[s][b] for x in scs])
                        gap = hg.get(b)
                        g0 = float(gap.iloc[-1]) if gap is not None and len(gap) else 0.0
                        W = _weights(ident, U, F, m, H, ev, None)
                        G = ident.adjust(F, W, g0)
                        adj.append(float(np.nanmean(np.abs(G - F) / np.abs(F)) * 100))
                        for i, t in enumerate(ident.targets):
                            U[t]["sc"].setdefault(m + SUFFIX, {}).setdefault(s, {})[b] = G[i]
            rec["units"].append(sfx.strip(" |") or "panel")
        rec["history_gap_pct"] = float(np.nanmax(gaps)) if gaps else np.nan
        rec["adjust_pct"] = float(np.nanmean(adj)) if adj else np.nan
        log(f"Identity {ident.name}: imposed ({ident.method}); largest gap in history {rec['history_gap_pct']:.3g}% "
            f"of the largest term; scenario forecasts moved {rec['adjust_pct']:.3g}% on average")
        report.append(rec)
    return report
