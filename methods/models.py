"""The equations and the methods that build them.

Every method produces a ModelSet: one or more linear equations of the same form,

    y_i,t = a_i + rho * y_i,t-1 + sum_k b_k * x_k,t-lag_k + e_i,t

(a_i is a bank fixed effect; with one bank it is the constant), and a weight for each equation.
The forecast of a ModelSet is the weighted average of its equations' forecasts.

    bma           IMF Global Solvency Stress Test: all subsets of the screened drivers (0..max_vars), estimated
                  with bank fixed effects; equations whose long-run multipliers b_k / (1 - rho) have the wrong
                  sign are dropped; the rest are averaged with weights exp(-BIC / 2) (Bayesian model averaging
                  with equal prior model probabilities and the BIC approximation to the marginal likelihood).
    combination   Guerrieri & Welch (2012, Fed FEDS 2012-49): one equation per macro variable with lags 1..4
                  of that variable, equal weights. No sign restrictions.
    class         NY Fed CLASS model (Hirtle et al. 2016): one small fixed equation per target, chosen on
                  economic grounds and kept the same at every origin.
    ar1           y_t on y_t-1 and the bank effect only.
    drift         the bank's average change over the estimation sample.
    random_walk   no change (y = 0): the level stays where it is.
"""
from itertools import combinations

import numpy as np


class Rows:
    """The estimation rows of one target up to an origin, with any set of macro features.

    Rows are bank-quarters t <= origin with y_t and y_t-1 known and t >= first (the same rows for every
    equation, so their BIC values are comparable).
    """

    def __init__(self, target, xhist, origin, first):
        B, T, Y, YL = [], [], [], []
        for i, b in enumerate(target.banks):
            y = target.y[b]
            y = y[(y.index >= first) & (y.index <= origin)]
            yl = target.y[b].shift(1).reindex(y.index)
            ok = y.notna() & yl.notna()
            B += [i] * int(ok.sum())
            T += list(y.index[ok])
            Y += list(y[ok].values)
            YL += list(yl[ok].values)
        self.nb = len(target.banks)
        self.B = np.array(B, int)
        self.T = T
        self.Y = np.array(Y, float)
        self.YL = np.array(YL, float)
        self.n = len(self.Y)
        self.xhist = xhist
        self.counts = np.bincount(self.B, minlength=self.nb).astype(float)
        self.present = self.counts > 0

    def col(self, var, lag):
        s = self.xhist[var].shift(lag)
        return s.reindex(self.T).values.astype(float)

    def X(self, feats):
        return np.column_stack([self.col(v, l) for v, l in feats]) if feats else np.zeros((self.n, 0))

    def means(self, A):
        """Bank means of the columns of A: (nb, k)."""
        A = A.reshape(self.n, -1)
        S = np.zeros((self.nb, A.shape[1]))
        np.add.at(S, self.B, A)
        c = np.where(self.counts > 0, self.counts, 1)[:, None]
        return S / c

    def demean(self, A):
        A = A.reshape(self.n, -1)
        return A - self.means(A)[self.B]


class ModelSet:
    """Equations over a common feature list.

    feats  [(var, lag)]          rho (M,)    beta (M, K)    alpha (M, nb)    w (M,)
    """

    def __init__(self, name, feats, rho, beta, alpha, w, info=None):
        self.name, self.feats = name, list(feats)
        self.rho, self.beta, self.alpha = np.asarray(rho, float), np.asarray(beta, float), np.asarray(alpha, float)
        self.w = np.asarray(w, float) / np.sum(w)
        self.info = info or {}

    def paths(self, bank_index, y_last, xpath, start, H):
        """Change paths of every equation for one bank: (M, H).

        xpath: DataFrame of transformed macro values (history + scenario) covering start-max_lag .. start+H.
        start: the origin quarter (the forecast is for start+1 .. start+H).
        """
        M = len(self.rho)
        K = len(self.feats)
        Xf = np.zeros((H, K))
        for k, (v, l) in enumerate(self.feats):
            s = xpath[v]
            Xf[:, k] = [s.get(start + h - l, np.nan) for h in range(1, H + 1)]
        Xf = np.nan_to_num(Xf)
        out = np.zeros((M, H))
        prev = np.full(M, y_last)
        a = self.alpha[:, bank_index]
        for h in range(H):
            prev = a + self.rho * prev + self.beta @ Xf[h]
            out[:, h] = prev
        return out

    def forecast(self, bank_index, y_last, xpath, start, H):
        P = self.paths(bank_index, y_last, xpath, start, H)
        return self.w @ P, P


def _ols(Yd, Zd):
    coef, *_ = np.linalg.lstsq(Zd, Yd, rcond=None)
    r = Yd - Zd @ coef
    return coef, float(r @ r)


def _alpha(rows, rho, beta, Xbar):
    """Bank effects of M equations: (M, nb)."""
    Ybar = rows.means(rows.Y)[:, 0]
    YLbar = rows.means(rows.YL)[:, 0]
    a = Ybar[None, :] - rho[:, None] * YLbar[None, :] - beta @ Xbar.T
    a[:, ~rows.present] = 0.0
    return a


def _bic(ssr, n, k):
    return n * np.log(max(ssr, 1e-300) / n) + k * np.log(n)


# ---------------------------------------------------------------------------------------------- restrictions
class Restrictions:
    """How drivers may enter (config.yaml, `restrictions`).

    signs        impose the expected signs at all (False: every driver has a free sign)
    apply_to     which methods impose them: bma, combination, class
    force_in     drivers in every BMA equation (and added to every combination equation): 'VAR' or 'VAR@lag'
    not_together groups of drivers of which an equation may hold at most one
    max_corr     two drivers more correlated than this are not both screened in (None: no limit)
    """

    def __init__(self, cfg):
        rs = cfg.get("restrictions", {}) or {}
        self.signs = bool(rs.get("signs", True))
        self.apply_to = set(rs.get("apply_to", ["bma"]))
        self.force = []
        for x in rs.get("force_in") or []:
            v, _, l = str(x).partition("@")
            self.force.append((v.strip(), int(l) if l.strip() else None))
        self.groups = [set(str(v).split("@")[0].strip() for v in g) for g in (rs.get("not_together") or [])]
        mc = rs.get("max_corr", cfg["candidates"].get("max_corr", 0.85))
        self.max_corr = None if mc in (None, "none", "None") else float(mc)

    def uses_signs(self, method):
        return self.signs and method in self.apply_to

    def allowed(self, names):
        """False if the drivers break a not_together group."""
        names = set(names)
        return all(len(names & g) <= 1 for g in self.groups)


def _sign_ok(b, sgn):
    """Every restricted driver (sgn != 0) has the expected sign; b and sgn are arrays over the same drivers."""
    b, sgn = np.asarray(b, float), np.asarray(sgn, int)
    on = sgn != 0
    return not np.any(np.sign(b[on]) != sgn[on])


# ------------------------------------------------------------------------------------------------ screening
def screen(rows, signs, themes, cfg, signed=True, n_keep=None, restr=None):
    """Rank candidate drivers by their partial correlation with y (after y_t-1 and bank effects).

    For each variable the lag 0..max_lag with the best correlation is used. With signed=True a driver with an
    expected sign (+1/-1) is kept only if its correlation has that sign; drivers with sign 0 are ranked by the
    size of the correlation. Forced drivers (restrictions.force_in) are always kept, first, on top of n_keep.
    Returns [(var, lag, score)].
    """
    cc = cfg["candidates"]
    restr = restr or Restrictions(cfg)
    Yd = rows.demean(rows.Y)[:, 0]
    YLd = rows.demean(rows.YL)[:, 0]
    bl = (YLd @ Yd) / max(YLd @ YLd, 1e-12)
    r = Yd - bl * YLd
    rn = np.sqrt(r @ r)
    forced = {v: l for v, l in restr.force}

    def scored(v, sgn, lags):
        top = None
        for lag in lags:
            x = rows.col(v, lag)
            if np.isnan(x).any():
                continue
            xd = rows.demean(x)[:, 0]
            xd = xd - ((YLd @ xd) / max(YLd @ YLd, 1e-12)) * YLd
            nx = np.sqrt(xd @ xd)
            if nx < 1e-12 or rn < 1e-12:
                continue
            c = (xd @ r) / (nx * rn)
            score = sgn * c if (signed and sgn) else abs(c)
            if top is None or score > top[2]:
                top = (v, lag, score, xd / nx)
        return top

    all_lags = range(cc.get("max_lag", 2) + 1)
    keep = []
    for v, l in restr.force:
        t = scored(v, signs.get(v, 0), [l] if l is not None else all_lags)
        if t is None:
            raise ValueError(f"restrictions.force_in: {v} has missing values in the estimation sample")
        keep.append(t)
    best = []
    for v, sgn in signs.items():
        if v in forced:
            continue
        top = scored(v, sgn, all_lags)
        if top and top[2] > 0:
            best.append(top)
    best.sort(key=lambda t: -t[2])
    per_theme = {}
    n_keep = n_keep or cc.get("n_screen", 10)
    n_forced = len(keep)
    for v, lag, score, z in best:
        th = themes.get(v, "")
        if per_theme.get(th, 0) >= cc.get("max_per_theme", 2):
            continue
        if restr.max_corr is not None and any(abs(z @ z2) > restr.max_corr for *_, z2 in keep):
            continue
        keep.append((v, lag, score, z))
        per_theme[th] = per_theme.get(th, 0) + 1
        if len(keep) - n_forced >= n_keep:
            break
    return [(v, lag, s) for v, lag, s, _ in keep]


# ------------------------------------------------------------------------------------------------- methods
def bma(rows, screened, signs, cfg, restr=None, _relax=False):
    restr = restr or Restrictions(cfg)
    feats = [(v, l) for v, l, _ in screened]
    K = len(feats)
    X = rows.X(feats)
    Yd = rows.demean(rows.Y)[:, 0]
    Zall = np.column_stack([rows.demean(rows.YL), rows.demean(X)]) if K else rows.demean(rows.YL)
    Xbar = rows.means(X) if K else np.zeros((rows.nb, 0))
    use = restr.uses_signs("bma")
    forced_names = {v for v, _ in restr.force}
    sgn = np.array([signs.get(v, 0) if use and not (_relax and v in forced_names) else 0 for v, _ in feats], int)
    F = [k for k, (v, _) in enumerate(feats) if v in forced_names]
    free = [k for k in range(K) if k not in F]
    max_vars = max(cfg["methods"]["bma"].get("max_vars", 3), len(F))
    rhos, betas, bics, subsets = [], [], [], []
    tried = 0
    for m in range(0, min(max_vars - len(F), len(free)) + 1):
        for extra in combinations(free, m):
            S = tuple(sorted(F + list(extra)))
            if not restr.allowed(feats[k][0] for k in S):
                continue
            tried += 1
            cols = [0] + [1 + j for j in S]
            coef, ssr = _ols(Yd, Zall[:, cols])
            rho = coef[0]
            b = np.zeros(K)
            b[list(S)] = coef[1:]
            if not (-1 < rho < 1):
                continue
            if S and not _sign_ok(b[list(S)], sgn[list(S)]):    # long-run multiplier b/(1-rho), 1-rho > 0
                continue
            rhos.append(rho)
            betas.append(b)
            bics.append(_bic(ssr, rows.n, 1 + len(S) + int(rows.present.sum())))
            subsets.append(S)
    if not rhos:
        if F and not _relax:
            # every equation fails because a forced driver has the wrong sign: keep the forced drivers,
            # drop their sign check, and say so in the report
            ms = bma(rows, screened, signs, cfg, restr, _relax=True)
            ms.info["warning"] = ("No equation passed with the forced drivers' expected signs; their sign check was "
                                  "dropped (the data disagree with the expected sign).")
            return ms
        raise ValueError("bma: no equation passes the restrictions (check force_in, signs and not_together)")
    rho, beta, bic = np.array(rhos), np.array(betas).reshape(len(rhos), K), np.array(bics)
    w = np.exp(-0.5 * (bic - bic.min()))
    w /= w.sum()
    alpha = _alpha(rows, rho, beta, Xbar)
    pip = {feats[k]: float(w[[k in S for S in subsets]].sum()) for k in range(K)}
    order = np.argsort(-w)[:10]
    top = [{"drivers": [feats[k] for k in subsets[i]], "weight": float(w[i]), "rho": float(rho[i]),
            "beta": [float(beta[i, k]) for k in subsets[i]]} for i in order]
    return ModelSet("bma", feats, rho, beta, alpha, w,
                    {"pip": pip, "top": top, "n_models": len(w), "n_tried": tried, "signs_used": use})


def combination(rows, screened, cfg, signs=None, restr=None):
    """One equation per screened variable (lags 1..L), plus any forced drivers in every equation; equal weights.
    With 'combination' in restrictions.apply_to, equations whose summed lag coefficients have the wrong sign are
    dropped."""
    restr = restr or Restrictions(cfg)
    signs = signs or {}
    L = cfg["methods"]["combination"].get("lags", 4)
    forced = [v for v, _ in restr.force]
    vars_ = [v for v, _, _ in screened if v not in forced]
    allv = forced + vars_
    feats = [(v, l) for v in allv for l in range(1, L + 1)]
    K = len(feats)
    X = rows.X(feats)
    Yd = rows.demean(rows.Y)[:, 0]
    Xd = rows.demean(X)
    YLd = rows.demean(rows.YL)
    Xbar = rows.means(X)
    use = restr.uses_signs("combination")
    fcols = list(range(len(forced) * L))
    rho, beta, used, dropped = [], [], [], []
    for i, v in enumerate(vars_ or ([None] if forced else [])):
        own = [] if v is None else list(range((len(forced) + i) * L, (len(forced) + i + 1) * L))
        cols = fcols + own
        coef, _ = _ols(Yd, np.column_stack([YLd, Xd[:, cols]]))
        b = np.zeros(K)
        b[cols] = coef[1:]
        if use:
            sums = [b[allv.index(u) * L:(allv.index(u) + 1) * L].sum() for u in forced + ([v] if v else [])]
            sg = [signs.get(u, 0) for u in forced + ([v] if v else [])]
            if not _sign_ok(sums, sg):
                dropped.append(v)
                continue
        rho.append(coef[0])
        beta.append(b)
        used.append(v)
    if not rho:
        return ar1(rows)
    rho, beta = np.array(rho), np.array(beta)
    return ModelSet("combination", feats, rho, beta, _alpha(rows, rho, beta, Xbar), np.ones(len(rho)),
                    {"variables": used, "dropped_for_sign": dropped})


def fixed(rows, feats, name="class", signs=None, check_signs=False):
    X = rows.X(feats)
    ok = ~np.isnan(X).any(axis=1) if len(feats) else np.ones(rows.n, bool)
    if not ok.all():
        raise ValueError(f"{name}: missing macro values for {feats}")
    Yd = rows.demean(rows.Y)[:, 0]
    Z = np.column_stack([rows.demean(rows.YL), rows.demean(X)]) if feats else rows.demean(rows.YL)
    coef, ssr = _ols(Yd, Z)
    rho, beta = np.array([coef[0]]), np.array([coef[1:]])
    wrong = []
    if check_signs and signs:
        wrong = [f"{v}@{l}" for (v, l), b in zip(feats, coef[1:]) if signs.get(v, 0) and np.sign(b) != signs[v]]
    return ModelSet(name, feats, rho, beta, _alpha(rows, rho, beta, rows.means(X) if feats else np.zeros((rows.nb, 0))),
                    [1.0], {"coef": dict(zip([f"{v}@{l}" for v, l in feats], map(float, coef[1:]))), "rho": float(coef[0]),
                            "wrong_sign": wrong})


def ar1(rows):
    return fixed(rows, [], "ar1")


def drift(rows):
    mu = rows.means(rows.Y)[:, 0]
    mu[~rows.present] = 0.0
    return ModelSet("drift", [], [0.0], np.zeros((1, 0)), mu[None, :], [1.0])


def random_walk(rows):
    return ModelSet("random_walk", [], [0.0], np.zeros((1, 0)), np.zeros((1, rows.nb)), [1.0])
