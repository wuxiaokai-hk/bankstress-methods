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
from itertools import combinations, product

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

    def col(self, var, lag, kind=None):
        s = self.xhist[var].shift(lag)
        return transform(s.reindex(self.T).values.astype(float), kind)

    def X(self, feats):
        return np.column_stack([self.col(*f) for f in feats]) if feats else np.zeros((self.n, 0))

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


def transform(x, kind):
    """A feature is (var, lag) or (var, lag, kind); kind changes how the macro change x enters:
        None              x itself
        '+'               max(x, 0): only increases (asymmetric effects)
        '-'               min(x, 0): only decreases
        ('seg', lo, hi)   the part of x inside [lo, hi], measured from lo (one piece of a piecewise-linear curve;
                          lo = -inf or hi = +inf for the open end pieces, which extrapolate linearly)
    """
    if kind is None:
        return x
    if kind == "+":
        return np.maximum(x, 0.0)
    if kind == "-":
        return np.minimum(x, 0.0)
    _, lo, hi = kind
    if np.isinf(lo):
        return np.minimum(x, hi) - hi
    if np.isinf(hi):
        return np.maximum(x, lo) - lo
    return np.clip(x, lo, hi) - lo


def feat_name(f):
    v, l = f[0], f[1]
    k = f[2] if len(f) > 2 else None
    tag = "" if k is None else (" rising" if k == "+" else " falling" if k == "-" else
                                f" in [{k[1]:.3g}, {k[2]:.3g}]")
    return f"{v}@{l}{tag}"


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
        for k, f in enumerate(self.feats):
            v, l = f[0], f[1]
            s = xpath[v]
            Xf[:, k] = transform(np.array([s.get(start + h - l, np.nan) for h in range(1, H + 1)], float),
                                 f[2] if len(f) > 2 else None)
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

    FAMILY = {"bma_asym": "bma", "spline": "bma", "hybrid": "bma", "direct": "bma", "direct_class": "class"}

    def uses_signs(self, method):
        """Methods built on the BMA's drivers (bma_asym, spline, hybrid, direct) follow 'bma';
        direct_class follows 'class'; each can also be named on its own in apply_to."""
        return self.signs and (method in self.apply_to or self.FAMILY.get(method) in self.apply_to)

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
def bma(rows, screened, signs, cfg, restr=None, _relax=False, asym=False, name="bma"):
    """Bayesian model averaging over every admissible equation (IMF GST).

    With asym=True (method bma_asym) each driver may also enter asymmetrically: separate slopes for its rises and
    its falls, max(x, 0) and min(x, 0). An equation then chooses, for each of its drivers, the symmetric or the
    asymmetric form; the asymmetric form costs one more parameter in the BIC, so it only gets weight when the data
    support it. Both slopes must have the expected sign (the effect is monotone, only its strength differs).
    """
    restr = restr or Restrictions(cfg)
    drivers = [(v, l) for v, l, _ in screened]
    feats, alts = [], []
    for v, l in drivers:
        k = len(feats)
        feats.append((v, l))
        a_ = [[k]]
        if asym:
            feats += [(v, l, "+"), (v, l, "-")]
            a_.append([k + 1, k + 2])
        alts.append(a_)
    K = len(feats)
    X = rows.X(feats)
    Yd = rows.demean(rows.Y)[:, 0]
    Zall = np.column_stack([rows.demean(rows.YL), rows.demean(X)]) if K else rows.demean(rows.YL)
    Xbar = rows.means(X) if K else np.zeros((rows.nb, 0))
    use = restr.uses_signs("bma")
    forced_names = {v for v, _ in restr.force}
    sgn = np.array([signs.get(f[0], 0) if use and not (_relax and f[0] in forced_names) else 0 for f in feats], int)
    D = len(drivers)
    F = [d for d, (v, _) in enumerate(drivers) if v in forced_names]
    free = [d for d in range(D) if d not in F]
    max_vars = max(cfg["methods"]["bma"].get("max_vars", 3), len(F))
    rhos, betas, bics, subsets, chosen = [], [], [], [], []
    tried = 0
    for m in range(0, min(max_vars - len(F), len(free)) + 1):
        for extra in combinations(free, m):
            ds = tuple(sorted(F + list(extra)))
            if not restr.allowed(drivers[d][0] for d in ds):
                continue
            for pick in product(*[range(len(alts[d])) for d in ds]):
                S = [c for d, p_ in zip(ds, pick) for c in alts[d][p_]]
                tried += 1
                cols = [0] + [1 + j for j in S]
                coef, ssr = _ols(Yd, Zall[:, cols])
                rho = coef[0]
                b = np.zeros(K)
                b[S] = coef[1:]
                if not (-1 < rho < 1):
                    continue
                if S and not _sign_ok(b[S], sgn[S]):    # long-run multiplier b/(1-rho), 1-rho > 0
                    continue
                rhos.append(rho)
                betas.append(b)
                bics.append(_bic(ssr, rows.n, 1 + len(S) + int(rows.present.sum())))
                subsets.append(tuple(S))
                chosen.append(dict(zip(ds, pick)))
    if not rhos:
        if F and not _relax:
            # every equation fails because a forced driver has the wrong sign: keep the forced drivers,
            # drop their sign check, and say so in the report
            ms = bma(rows, screened, signs, cfg, restr, _relax=True, asym=asym, name=name)
            ms.info["warning"] = ("No equation passed with the forced drivers' expected signs; their sign check was "
                                  "dropped (the data disagree with the expected sign).")
            return ms
        raise ValueError("bma: no equation passes the restrictions (check force_in, signs and not_together)")
    rho, beta, bic = np.array(rhos), np.array(betas).reshape(len(rhos), K), np.array(bics)
    w = np.exp(-0.5 * (bic - bic.min()))
    w /= w.sum()
    alpha = _alpha(rows, rho, beta, Xbar)
    pip = {drivers[d]: float(sum(wi for wi, c in zip(w, chosen) if d in c)) for d in range(D)}
    asym_share = {drivers[d]: float(sum(wi for wi, c in zip(w, chosen) if c.get(d) == 1)) for d in range(D)} if asym else {}
    order = np.argsort(-w)[:10]
    top = [{"drivers": [feats[k] for k in subsets[i]], "weight": float(w[i]), "rho": float(rho[i]),
            "beta": [float(beta[i, k]) for k in subsets[i]]} for i in order]
    return ModelSet(name, feats, rho, beta, alpha, w,
                    {"pip": pip, "asym": asym_share, "top": top, "n_models": len(w), "n_tried": tried,
                     "signs_used": use})


def median_model(ms):
    """Drivers with inclusion probability >= 0.5 (Barbieri & Berger's median probability model); at least the
    single heaviest driver if none reaches 0.5."""
    pip = ms.info.get("pip", {})
    keep = [d for d, p in pip.items() if p >= 0.5]
    if not keep and pip:
        keep = [max(pip, key=pip.get)]
    return keep


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


# ------------------------------------------------------------------------------------- nonlinear: monotone curves
def spline(rows, drivers, signs, cfg, restr=None):
    """Shape-constrained piecewise-linear effects (a monotone linear spline).

    Each driver's effect is a piecewise-linear curve with knots at quantiles of its history (methods.spline.knots,
    default the terciles). The curve may bend at the knots but, when the driver has an expected sign, every piece
    must slope that way, so the effect is monotone. Beyond the range of the data the end pieces continue in a
    straight line, so a severe scenario still moves the forecast. Estimated by bounded least squares with bank
    fixed effects and y_t-1, like the other equations.
    """
    from scipy.optimize import lsq_linear

    restr = restr or Restrictions(cfg)
    qs = cfg["methods"].get("spline", {}).get("knots", [1 / 3, 2 / 3])
    use = restr.uses_signs("spline")
    feats, lb, ub = [], [-0.999], [0.999]
    knots = {}
    for v, l in drivers:
        x = rows.col(v, l)
        ks = sorted(set(np.round(np.quantile(x, qs), 12)))
        knots[(v, l)] = ks
        edges = [-np.inf] + ks + [np.inf]
        sg = signs.get(v, 0) if use else 0
        for lo, hi in zip(edges[:-1], edges[1:]):
            feats.append((v, l, ("seg", lo, hi)))
            lb.append(0.0 if sg > 0 else -np.inf)
            ub.append(0.0 if sg < 0 else np.inf)
    X = rows.X(feats)
    Z = np.column_stack([rows.demean(rows.YL), rows.demean(X)]) if feats else rows.demean(rows.YL)
    Yd = rows.demean(rows.Y)[:, 0]
    res = lsq_linear(Z, Yd, bounds=(np.array(lb), np.array(ub)), method="bvls")
    coef = res.x
    rho, beta = np.array([coef[0]]), np.array([coef[1:]])
    Xbar = rows.means(X) if feats else np.zeros((rows.nb, 0))
    slopes = {f"{v}@{l}": [float(b) for f, b in zip(feats, coef[1:]) if f[:2] == (v, l)] for v, l in drivers}
    return ModelSet("spline", feats, rho, beta, _alpha(rows, rho, beta, Xbar), [1.0],
                    {"drivers": list(drivers), "knots": {f"{v}@{l}": k for (v, l), k in knots.items()},
                     "slopes": slopes, "rho": float(coef[0])})


# ------------------------------------------------------------------- nonlinear: linear model + boosted trees
class HybridSet:
    """Linear BMA equation + monotone gradient-boosted trees on its residuals.

    y_t = a_i + rho y_t-1 + b'x_t  +  g(x_t, y_t-1)
    The linear part is the BMA's weighted-average equation; g is fitted to its residuals with shallow trees,
    constrained to be monotone in each driver in the direction of its expected sign. Trees are flat outside the
    range of their training data, so beyond history only the linear part moves the forecast: the trees add
    curvature inside the range seen, the linear part extrapolates.
    """

    def __init__(self, base, gbm, xfeats, info):
        self.name, self.base, self.gbm, self.xfeats, self.info = "hybrid", base, gbm, xfeats, info

    def forecast(self, bank_index, y_last, xpath, start, H):
        b = self.base
        K = len(b.feats)
        Xf = np.zeros((H, K))
        for k, f in enumerate(b.feats):
            s = xpath[f[0]]
            Xf[:, k] = transform(np.array([s.get(start + h - f[1], np.nan) for h in range(1, H + 1)], float),
                                 f[2] if len(f) > 2 else None)
        Xf = np.nan_to_num(Xf)
        a = b.alpha[0, bank_index]
        out = np.zeros(H)
        prev = y_last
        for h in range(H):
            lin = a + b.rho[0] * prev + b.beta[0] @ Xf[h]
            g = self.gbm.predict(np.r_[Xf[h, self.xfeats], prev][None, :])[0] if self.gbm is not None else 0.0
            prev = lin + g
            out[h] = prev
        return out, out[None, :]


def hybrid(rows, bma_set, signs, cfg, restr=None):
    from sklearn.ensemble import HistGradientBoostingRegressor

    restr = restr or Restrictions(cfg)
    hc = cfg["methods"].get("hybrid", {})
    w = bma_set.w
    rho = np.array([w @ bma_set.rho])
    beta = (w @ bma_set.beta)[None, :]
    alpha = (w @ bma_set.alpha)[None, :]
    base = ModelSet("hybrid_linear", bma_set.feats, rho, beta, alpha, [1.0])
    X = rows.X(base.feats)
    fitted = alpha[0, rows.B] + rho[0] * rows.YL + (X @ beta[0] if X.shape[1] else 0.0)
    resid = rows.Y - fitted
    xfeats = [k for k, f in enumerate(base.feats) if len(f) == 2]        # the plain drivers
    use = restr.uses_signs("hybrid")
    mono = [int(signs.get(base.feats[k][0], 0)) if use else 0 for k in xfeats] + [0]
    G = np.column_stack([X[:, xfeats], rows.YL]) if xfeats else rows.YL[:, None]
    gbm = HistGradientBoostingRegressor(
        max_depth=hc.get("max_depth", 2), learning_rate=hc.get("learning_rate", 0.05),
        max_iter=hc.get("trees", 100), min_samples_leaf=max(hc.get("min_leaf", 10), rows.n // 20),
        l2_regularization=hc.get("l2", 1.0), monotonic_cst=mono, early_stopping=False, random_state=0)
    gbm.fit(G, resid)
    share = float(np.var(gbm.predict(G)) / max(np.var(rows.Y), 1e-18))
    rng = {feat_name(base.feats[k]): (float(np.min(X[:, k])), float(np.max(X[:, k]))) for k in xfeats}
    return HybridSet(base, gbm, xfeats, {"tree_share": share, "train_range": rng, "monotone": mono[:-1]})


# ---------------------------------------------------------------------------- direct (local projection) forecasts
class DirectSet:
    """One equation per horizon h = 1..H, for the cumulative change over the h quarters:

        z_i,t,h = a_i,h + c_h y_i,t + sum_k b_k,h * (x_k,t+1-l + ... + x_k,t+h-l) + e

    (log growth over h quarters for balances, the change for rates). The scenario enters through each driver's
    cumulative change over the same h quarters. The coefficients are estimated jointly for all horizons with a
    penalty on their second differences across h (smooth local projections), and each driver's coefficients are
    bounded to its expected sign.
    """

    def __init__(self, name, drivers, c, B, A, info):
        self.name, self.drivers, self.c, self.B, self.A, self.info = name, drivers, c, B, A, info

    def forecast(self, bank_index, y_last, xpath, start, H):
        Hm = min(H, len(self.c))
        z = np.zeros(Hm)
        cum = np.zeros(len(self.drivers))
        for h in range(1, Hm + 1):
            for k, (v, l) in enumerate(self.drivers):
                cum[k] += np.nan_to_num(xpath[v].get(start + h - l, np.nan))
            z[h - 1] = self.A[h - 1, bank_index] + self.c[h - 1] * y_last + self.B[h - 1] @ cum
        if Hm < H:                       # beyond the longest horizon estimated: keep the last quarterly change
            z = np.r_[z, z[-1] + np.arange(1, H - Hm + 1) * (z[-1] - (z[-2] if Hm > 1 else 0.0))]
        yp = np.diff(np.r_[0.0, z])
        return yp, yp[None, :]


def direct(target, xhist, origin, first, drivers, signs, cfg, H, restr=None, name="direct"):
    from scipy.optimize import lsq_linear

    restr = restr or Restrictions(cfg)
    dc = cfg["methods"].get("direct", {})
    lam_rel = float(dc.get("smooth", 1.0))
    use = restr.uses_signs(name)
    K = len(drivers)
    P = 1 + K                                         # c_h and b_k,h per horizon
    nb = len(target.banks)
    idx = xhist.index
    pos = {q: k for k, q in enumerate(idx)}
    o_pos, f_pos = pos[origin], pos.get(first, 0)

    def csum(a):
        """cumulative sums and cumulative NaN counts, so sums over (t, t+h] are differences."""
        nan = np.isnan(a)
        return np.r_[0.0, np.cumsum(np.where(nan, 0.0, a))], np.r_[0, np.cumsum(nan)]

    XC = [csum(xhist[v].shift(l).values.astype(float)) for v, l in drivers]
    YA, YC = [], []
    for b in target.banks:
        ya = target.y[b].reindex(idx).values.astype(float)
        YA.append(ya)
        YC.append(csum(ya))
    meta = []
    for h in range(1, H + 1):
        Zh, zh, bh = [], [], []
        t = np.arange(f_pos, o_pos - h + 1)
        if len(t) == 0:
            meta.append((np.zeros((0, P)), np.zeros(0), np.zeros(0, int)))
            continue
        cols_x = []
        bad_x = np.zeros(len(t), bool)
        for cs, cn in XC:
            cols_x.append(cs[t + h + 1] - cs[t + 1])           # x at positions t+1 .. t+h
            bad_x |= (cn[t + h + 1] - cn[t + 1]) > 0
        for i in range(nb):
            cs, cn = YC[i]
            z = cs[t + h + 1] - cs[t + 1]
            ok = ~bad_x & ((cn[t + h + 1] - cn[t + 1]) == 0) & ~np.isnan(YA[i][t])
            if not ok.any():
                continue
            Zh.append(np.column_stack([YA[i][t]] + cols_x)[ok])
            zh.append(z[ok])
            bh.append(np.full(int(ok.sum()), i))
        if Zh:
            meta.append((np.vstack(Zh), np.concatenate(zh), np.concatenate(bh)))
        else:
            meta.append((np.zeros((0, P)), np.zeros(0), np.zeros(0, int)))
    Hn = max([h for h, (Zh, _, _) in enumerate(meta, 1) if len(Zh) > P + nb] or [0])
    if Hn == 0:
        raise ValueError(f"{name}: too few observations")
    rows_, rhs = [], []
    means = []
    for h in range(Hn):
        Zh, zh, bh = meta[h]
        cnt = np.bincount(bh, minlength=nb).astype(float)
        c_ = np.where(cnt > 0, cnt, 1)
        mz = np.bincount(bh, zh, minlength=nb) / c_
        mZ = np.column_stack([np.bincount(bh, Zh[:, j], minlength=nb) / c_ for j in range(P)])
        means.append((mz, mZ, cnt))
        Zd, zd = Zh - mZ[bh], zh - mz[bh]
        big = np.zeros((len(zd), Hn * P))
        big[:, h * P:(h + 1) * P] = Zd
        rows_.append(big)
        rhs.append(zd)
    Z = np.vstack(rows_)
    zz = np.concatenate(rhs)
    # smoothness: second differences of every coefficient across horizons, scaled to the data
    if Hn >= 3 and lam_rel > 0:
        scale = np.mean(np.sum(Z ** 2, axis=0)[np.sum(Z ** 2, axis=0) > 0]) if Z.size else 1.0
        lam = lam_rel * scale
        pen = []
        for h in range(Hn - 2):
            for j in range(P):
                r = np.zeros(Hn * P)
                r[h * P + j], r[(h + 1) * P + j], r[(h + 2) * P + j] = 1.0, -2.0, 1.0
                pen.append(np.sqrt(lam) * r)
        Z = np.vstack([Z, np.array(pen)])
        zz = np.concatenate([zz, np.zeros(len(pen))])
    lb = np.full(Hn * P, -np.inf)
    ub = np.full(Hn * P, np.inf)
    for h in range(Hn):
        for k, (v, _) in enumerate(drivers):
            sg = signs.get(v, 0) if use else 0
            if sg > 0:
                lb[h * P + 1 + k] = 0.0
            elif sg < 0:
                ub[h * P + 1 + k] = 0.0
    coef = lsq_linear(Z, zz, bounds=(lb, ub), method="bvls").x.reshape(Hn, P)
    c, Bm = coef[:, 0], coef[:, 1:]
    A = np.zeros((Hn, nb))
    for h in range(Hn):
        mz, mZ, cnt = means[h]
        A[h] = mz - mZ @ coef[h]
        if (cnt == 0).any():
            A[h, cnt == 0] = np.average(A[h, cnt > 0], weights=cnt[cnt > 0])
    info = {"drivers": list(drivers), "horizons": Hn, "smooth": lam_rel,
            "coef_h": {feat_name(d): [float(x) for x in Bm[:, k]] for k, d in enumerate(drivers)},
            "persistence_h": [float(x) for x in c]}
    return DirectSet(name, list(drivers), c, Bm, A, info)
