"""Regime / structural-break battery.

Design notes
------------
* The 60d rolling beta series is a moving average of itself: ~59/60 overlap.
  Break tests applied to it must respect that (min segment >= window;
  bootstrap with blocks; or -- preferred -- test the Kalman beta innovations,
  which are serially uncorrelated under the null).
* sup-F (Quandt-Andrews) and Nyblom-Hansen p-values come from a CIRCULAR
  BLOCK BOOTSTRAP rather than asymptotic tables: self-contained, and robust
  to the serial correlation these series carry.
* HMM: only FILTERED probabilities are reported for signal use. Smoothed
  probabilities condition on the full sample and are not tradeable.
* Effective sample honesty: the report counts detected episodes; that count,
  not the number of daily rows, is the effective sample for regime inference.
"""

import logging

import numpy as np
import pandas as pd
from scipy import stats

logging.getLogger("hmmlearn").setLevel(logging.ERROR)


# --------------------------------------------------------------------------
# stationarity
# --------------------------------------------------------------------------

def adf_kpss(series, name=""):
    from statsmodels.tsa.stattools import adfuller, kpss
    s = pd.Series(series).dropna()
    out = {"series": name, "n": len(s)}
    try:
        a = adfuller(s, autolag="AIC")
        out["adf_stat"], out["adf_p"] = float(a[0]), float(a[1])
    except Exception as e:
        out["adf_p"] = np.nan
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            k = kpss(s, regression="c", nlags="auto")
        out["kpss_stat"], out["kpss_p"] = float(k[0]), float(k[1])
    except Exception:
        out["kpss_p"] = np.nan
    return out


# --------------------------------------------------------------------------
# sup-F (Quandt-Andrews) with circular block bootstrap
# --------------------------------------------------------------------------

def _rss_prefix(y, x):
    """Prefix sums enabling O(1) segment RSS for y = a + b x + e."""
    one = np.ones_like(y)
    P = np.cumsum(np.column_stack([one, x, x * x, y, x * y, y * y]), axis=0)
    P = np.vstack([np.zeros(6), P])
    return P

def _seg_rss(P, i, j):
    n, sx, sxx, sy, sxy, syy = (P[j] - P[i])
    if n < 3:
        return np.inf
    det = n * sxx - sx * sx
    if abs(det) < 1e-12:
        return syy - sy * sy / n
    b = (n * sxy - sx * sy) / det
    a = (sy - b * sx) / n
    return max(syy - a * sy - b * sxy, 0.0)

def sup_f(y, x, trim=0.15):
    y = np.asarray(y, float); x = np.asarray(x, float)
    T = len(y)
    P = _rss_prefix(y, x)
    rss_full = _seg_rss(P, 0, T)
    lo, hi = int(trim * T), int((1 - trim) * T)
    best, best_tau = -np.inf, None
    k = 2
    for tau in range(lo, hi):
        r1 = _seg_rss(P, 0, tau)
        r2 = _seg_rss(P, tau, T)
        denom = (r1 + r2) / max(T - 2 * k, 1)
        if denom <= 0:
            continue
        F = ((rss_full - r1 - r2) / k) / denom
        if F > best:
            best, best_tau = F, tau
    return best, best_tau

def _block_boot_idx(T, block, rng):
    nb = int(np.ceil(T / block))
    starts = rng.integers(0, T, size=nb)
    idx = (starts[:, None] + np.arange(block)[None, :]) % T
    return idx.ravel()[:T]

def sup_f_bootstrap(y, x, trim=0.15, B=199, block=20, seed=0, min_obs=120):
    """H0: constant (a,b). Bootstrap resamples (x, e_hat) in circular blocks,
    rebuilds y* under the null fit, recomputes sup-F."""
    y = np.asarray(y, float); x = np.asarray(x, float)
    mask = np.isfinite(y) & np.isfinite(x)
    y, x = y[mask], x[mask]
    T = len(y)
    if T < min_obs:
        return {"supF": np.nan, "p": np.nan, "break_idx": None, "n": T}
    X = np.column_stack([np.ones(T), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    F0, tau0 = sup_f(y, x, trim)
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(B):
        idx = _block_boot_idx(T, block, rng)
        xs, es = x[idx], e[idx]
        ys = beta[0] + beta[1] * xs + es
        Fb, _ = sup_f(ys, xs, trim)
        if Fb >= F0:
            cnt += 1
    return {"supF": float(F0), "p": (cnt + 1) / (B + 1),
            "break_idx": tau0, "n": T}


# --------------------------------------------------------------------------
# Nyblom-Hansen parameter-constancy (bootstrap p-value)
# --------------------------------------------------------------------------

def nyblom(y, x):
    y = np.asarray(y, float); x = np.asarray(x, float)
    T = len(y)
    X = np.column_stack([np.ones(T), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    s = X * e[:, None]                       # scores per obs (k=2)
    V = s.T @ s
    S = np.cumsum(s, axis=0)
    try:
        Vi = np.linalg.inv(V)
    except np.linalg.LinAlgError:
        return np.nan
    L = np.einsum("ti,ij,tj->", S, Vi, S) / T
    return float(L)

def nyblom_bootstrap(y, x, B=199, block=20, seed=1, min_obs=120):
    y = np.asarray(y, float); x = np.asarray(x, float)
    mask = np.isfinite(y) & np.isfinite(x)
    y, x = y[mask], x[mask]
    T = len(y)
    if T < min_obs:
        return {"L": np.nan, "p": np.nan}
    X = np.column_stack([np.ones(T), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    L0 = nyblom(y, x)
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(B):
        idx = _block_boot_idx(T, block, rng)
        ys = beta[0] + beta[1] * x[idx] + e[idx]
        if nyblom(ys, x[idx]) >= L0:
            cnt += 1
    return {"L": L0, "p": (cnt + 1) / (B + 1)}


# --------------------------------------------------------------------------
# CUSUM (recursive residuals) on the hedge regression
# --------------------------------------------------------------------------

def cusum_recursive(y, x, min_obs=150):
    """Brown-Durbin-Evans CUSUM via statsmodels RecursiveLS on dP ~ dTsy.
    Tests constancy of the hedge relationship directly."""
    import statsmodels.api as sm
    df = pd.concat([pd.Series(y, name="y"), pd.Series(x, name="x")], axis=1).dropna()
    if len(df) < min_obs:
        return {"cusum_exceed_frac": np.nan}
    X = sm.add_constant(df["x"].values)
    try:
        res = sm.RecursiveLS(df["y"].values, X).fit()
        c = res.cusum
        T = len(c)
        k = 2
        a = 0.948
        j = np.arange(T)
        band = a * (np.sqrt(T) + 2 * j / np.sqrt(T))
        exceed = np.abs(c) > band
        first = df.index[np.argmax(exceed)] if exceed.any() else None
        return {"cusum_exceed_frac": float(exceed.mean()),
                "cusum_first_exceed": first,
                "cusum": pd.Series(c, index=df.index[-T:] if T == len(df) else df.index[:T]),
                "band": band}
    except Exception as e:
        return {"cusum_exceed_frac": np.nan, "error": str(e)}


# --------------------------------------------------------------------------
# multiple breaks via PELT (ruptures)
# --------------------------------------------------------------------------

def pelt_breaks(series, index, model="l2", min_size=90, pen_scale=3.0, jump=5):
    import ruptures as rpt
    s = pd.Series(series, index=index).dropna()
    z = (s - s.mean()) / (s.std() + 1e-12)
    x = z.values.reshape(-1, 1)
    T = len(x)
    if T < 2 * min_size:
        return []
    pen = pen_scale * np.log(T)
    algo = rpt.Pelt(model=model, min_size=min_size, jump=jump).fit(x)
    bkps = algo.predict(pen=pen)
    return [s.index[b - 1] for b in bkps[:-1]]


# --------------------------------------------------------------------------
# Kalman time-varying-parameter hedge beta
# --------------------------------------------------------------------------

def kalman_tvp(y, x, q_init=1e-6, estimate=True, init_window=60, min_obs=150):
    """State [alpha_t, beta_t] random walk; obs y_t = a_t + b_t x_t + eps.
    MLE over (log r, log q_a, log q_b) by Nelder-Mead. Returns filtered path
    and standardized one-step innovations (the online regime detector)."""
    df = pd.concat([pd.Series(y, name="y"), pd.Series(x, name="x")], axis=1).dropna()
    yy, xx = df["y"].values, df["x"].values
    T = len(yy)
    if T < min_obs:
        return None

    # init from first `init_window` obs OLS
    n0 = max(10, min(int(init_window), T // 2))
    X0 = np.column_stack([np.ones(n0), xx[:n0]])
    b0, *_ = np.linalg.lstsq(X0, yy[:n0], rcond=None)
    r0 = np.var(yy[:n0] - X0 @ b0)

    def loglik(params, ret_path=False):
        r = np.exp(params[0]); qa = np.exp(params[1]); qb = np.exp(params[2])
        Q = np.diag([qa, qb])
        th = b0.copy()
        P = np.eye(2) * 1.0
        ll = 0.0
        alphas = np.empty(T); betas = np.empty(T)
        innov = np.empty(T); Fs = np.empty(T)
        for t in range(T):
            P = P + Q
            H = np.array([1.0, xx[t]])
            F = H @ P @ H + r
            v = yy[t] - H @ th
            K = P @ H / F
            th = th + K * v
            P = P - np.outer(K, H @ P)
            ll += -0.5 * (np.log(2 * np.pi * F) + v * v / F)
            alphas[t], betas[t] = th
            innov[t], Fs[t] = v, F
        if ret_path:
            return alphas, betas, innov, Fs
        return -ll

    p0 = np.log([max(r0, 1e-8), q_init, q_init])
    if estimate:
        from scipy.optimize import minimize
        res = minimize(loglik, p0, method="Nelder-Mead",
                       options={"maxiter": 400, "xatol": 1e-3, "fatol": 1e-3})
        p = res.x
    else:
        p = p0
    alphas, betas, innov, Fs = loglik(p, ret_path=True)
    z = innov / np.sqrt(Fs)
    return {
        "alpha": pd.Series(alphas, index=df.index),
        "beta": pd.Series(betas, index=df.index),
        "z": pd.Series(z, index=df.index),
        "r": float(np.exp(p[0])), "q_a": float(np.exp(p[1])), "q_b": float(np.exp(p[2])),
        "loglik": float(-loglik(p)),
    }


def gaussian_abs_cutoff(tail_p):
    """|z| cutoff with two-sided Gaussian exceedance probability tail_p.
    This is the 'textbook constant' expressed as a function of the alarm rate:
    tail_p = 4.65e-4 (0.12 flags/yr) returns 3.4906 ~ the conventional 3.5."""
    return float(stats.norm.ppf(1.0 - tail_p / 2.0))


def _gpd_tail_quantile(x, tail_p, q_u=0.95, min_exceed=10):
    """Peaks-over-threshold tail quantile of |z|.

    Why not a raw empirical quantile: the designed alarm rate is 0.12/yr, i.e.
    a per-day tail of 4.65e-4, which is 1-in-2150 days. A 500-day window has
    no order statistic out there -- asking for that quantile just returns the
    window maximum, and the detector goes nearly silent (realized rate far
    BELOW the design, not at it). POT fits a Generalized Pareto to exceedances
    over a moderate threshold that IS well estimated (~25 points above the 95th
    pct in 500 obs) and extrapolates, which is the standard way to reach a
    quantile beyond the sample.

    Moment estimator for (xi, beta) -- closed form, stable on small exceedance
    counts where GPD MLE is erratic. Returns NaN if the fit is unusable.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 50:
        return np.nan
    u = np.quantile(x, q_u)
    exc = x[x > u] - u
    if len(exc) < min_exceed:
        return np.nan
    m, v = exc.mean(), exc.var(ddof=1)
    if not np.isfinite(m) or not np.isfinite(v) or m <= 0 or v <= 0:
        return np.nan
    xi = 0.5 * (1.0 - m * m / v)
    beta = 0.5 * m * (1.0 + m * m / v)
    if beta <= 0:
        return np.nan
    # xi >= 1 => infinite mean; the moment estimator is not valid there and the
    # extrapolation would be meaningless. Cap into the usable range.
    xi = float(np.clip(xi, -0.5, 0.95))
    ratio = tail_p / (1.0 - q_u)          # P(exceed target | exceed u)
    if ratio >= 1.0:
        return float(np.quantile(x, 1.0 - tail_p))
    if abs(xi) < 1e-6:
        return float(u + beta * np.log(1.0 / ratio))
    return float(u + (beta / xi) * (ratio ** (-xi) - 1.0))


def empirical_shewhart_threshold(z, cal, burn=60):
    """Threshold for |z| flags, calibrated to the data's OWN tail instead of a
    Gaussian convention, at a DESIGNED alarm rate you set in config.

    Both modes target the same rate, so switching mode changes the estimator
    and not the operating point:
      fixed      -> Gaussian cutoff for that rate (3.49 at 0.12/yr)
      empirical  -> POT/GPD tail of trailing |z| at that rate

    The cutoffs are therefore directly comparable, and their ratio is the
    fat-tail diagnostic. (The REALIZED flag rate is not: it is pinned near the
    design by construction in either mode, so it carries no information about
    tail shape.)

    Strictly trailing: the threshold in force on day t is fit on |z| up to
    t-1 only. Refit every `shewhart_refit_every` days and held between refits.

    CAVEAT on point-in-time-ness: the THRESHOLD is trailing, but `z` is not.
    kalman_tvp estimates (r, q_a, q_b) by MLE over the whole sample, so the
    innovations being thresholded already embed future information. This layer
    cannot undo that. For a genuinely PIT alarm history the Kalman parameters
    would have to be re-estimated on an expanding window too; as it stands,
    treat the LIVE threshold (the last value) as tradeable and the historical
    flag series as in-sample description.

    Returns (threshold_series, mode_used, diagnostics).
    """
    per_year = float(cal.get("shewhart_alarm_per_year", 0.12))
    tail_p = per_year / 252.0
    gauss_cut = gaussian_abs_cutoff(tail_p)
    diag = {"alarm_per_year": per_year, "tail_p": tail_p,
            "gauss_cutoff": gauss_cut, "estimator": "gaussian"}

    zz = z.iloc[burn:].abs()
    wmin = int(cal.get("shewhart_min_window", 250))
    if cal.get("shewhart_mode") != "empirical" or len(zz) < wmin:
        return pd.Series(gauss_cut, index=z.index), "fixed", diag

    w = int(cal.get("shewhart_window", 500))
    q_u = float(cal.get("shewhart_pot_quantile", 0.95))
    min_exc = int(cal.get("shewhart_min_exceed", 10))
    stride = max(1, int(cal.get("shewhart_refit_every", 21)))
    # Can the window read the target tail off directly? Needs min_exc order
    # statistics beyond it. At 0.12/yr and w=500 this is False by a wide
    # margin, so POT does the work -- but if the user dials the alarm rate up
    # (or the window way out), the direct quantile is preferable: no model.
    direct = (w * tail_p) >= min_exc
    diag["estimator"] = "empirical_quantile" if direct else "pot_gpd"

    v = zz.values
    n = len(v)
    thr_vals = np.full(n, np.nan)
    last = np.nan
    for t in range(n):
        if t >= wmin and (t - wmin) % stride == 0:
            win = v[max(0, t - w):t]          # strictly prior -> PIT
            if direct:
                cand = float(np.quantile(win, 1.0 - tail_p)) if len(win) else np.nan
            else:
                cand = _gpd_tail_quantile(win, tail_p, q_u, min_exc)
            if np.isfinite(cand):
                last = cand
        thr_vals[t] = last

    thr = pd.Series(thr_vals, index=zz.index)
    # Warm-up (and any failed fit) falls back to the Gaussian cutoff for the
    # SAME alarm rate, so the operating point is continuous across the join.
    thr = thr.reindex(z.index).ffill().fillna(gauss_cut)
    fitted = thr_vals[np.isfinite(thr_vals)]
    if len(fitted):
        diag["latest_cutoff"] = float(fitted[-1])
        diag["median_cutoff"] = float(np.median(fitted))
        diag["tail_ratio"] = float(np.median(fitted) / gauss_cut)
        diag["n_refits"] = int(len(np.unique(np.round(fitted, 6))))
    return thr, "empirical", diag


def calibrate_pelt_penalty(series, cal, min_size, model="l2", seed=0, jump=5):
    """Choose the PELT penalty by SIMULATING THE NULL instead of trusting
    pen = c*log(T), which is a modelling convention carrying no error-rate
    guarantee and which over-detects badly on persistent series.

    Returns the smallest penalty whose family-wise false-alarm rate is at or
    below target -- smallest because that maximizes power subject to the error
    constraint. `jump` must match the value used in the real pelt_breaks call,
    or the null is being segmented at a different resolution than the data.

    Returns (pen_scale, achieved_rate, mode, diagnostics)."""
    import ruptures as rpt
    s = pd.Series(series).dropna()
    T = len(s)
    if cal.get("pelt_mode") != "empirical" or T < 4 * min_size:
        return None, None, "fixed", {}
    z = ((s - s.mean()) / (s.std() + 1e-12)).values
    rng = np.random.default_rng(seed)
    B = int(cal.get("pelt_boot_B", 400))
    block = int(cal.get("pelt_boot_block", max(min_size, 120)))
    block = max(2, min(block, max(2, T // 4)))
    null_kind = cal.get("pelt_null", "difference")
    grid = sorted(cal.get("pelt_pen_grid", [1.0, 2.0, 3.0, 6.0]))
    counts = {p: 0 for p in grid}
    n_ok = 0
    logT = np.log(T)

    # AR-sieve setup: fit once, resample residuals per replicate.
    ar_p = int(cal.get("pelt_ar_order", 2))
    phi = res_ar = None
    if null_kind == "ar_sieve" and T > 10 * ar_p:
        Xa = np.column_stack([z[ar_p - 1 - k: T - 1 - k] for k in range(ar_p)])
        ya = z[ar_p:]
        try:
            phi, *_ = np.linalg.lstsq(Xa, ya, rcond=None)
            res_ar = ya - Xa @ phi
        except np.linalg.LinAlgError:
            phi = res_ar = None
    if phi is None and null_kind == "ar_sieve":
        null_kind = "difference"          # fall back if the AR fit failed

    def _draw():
        """One no-break replicate.

        The null has to reproduce how hard the series WANDERS without
        containing any break, and the three options trade off differently:

        'level'     -- block-resample the levels. WRONG for a persistent
                       series: chunks are drawn from unrelated levels and
                       every seam is a genuine mean shift, so the null
                       arrives pre-loaded with the breaks it exists to
                       exclude, and the penalty is pushed up by an artifact.

        'difference'-- resample increments and cumulate. Seam-free, but it
                       imposes an EXACT unit root. A mean-reverting series
                       gets replaced by a true random walk, which wanders
                       more than the truth, so the penalty is pushed up
                       again -- just for the opposite reason.

        'ar_sieve'  -- (default) simulate from an AR(p) fitted to the series
                       with block-resampled residuals. Persistence is
                       estimated rather than assumed, so a near-unit-root
                       series gets a near-unit-root null and a mean-reverting
                       one does not. The block on the residuals still carries
                       the ~59/60 window overlap.
        """
        if null_kind == "ar_sieve":
            e_idx = _block_boot_idx(len(res_ar), block, rng)
            e = res_ar[e_idx]
            path = np.empty(T)
            s0 = rng.integers(0, T - ar_p)
            path[:ar_p] = z[s0:s0 + ar_p]
            for i in range(ar_p, T):
                # lags most-recent-first, matching the column order of the
                # design matrix phi was fitted on. Sliced forward then
                # reversed: the `i-1 : i-1-p : -1` form silently underflows
                # to an empty slice on the first step when i-1-p == -1.
                path[i] = phi @ path[i - ar_p:i][::-1] + e[(i - ar_p) % len(e)]
        elif null_kind == "difference":
            d = np.diff(z)
            if len(d) < block:
                return None
            idx = _block_boot_idx(len(d), block, rng)
            path = np.concatenate([[0.0], np.cumsum(d[idx])])
        else:
            path = z[_block_boot_idx(T, block, rng)]
        sd = path.std()
        if not np.isfinite(sd) or sd <= 0 or not np.isfinite(path).all():
            return None
        # rescale to unit variance: pelt_breaks standardizes the REAL series
        # before applying the penalty, so the null must be on the same scale
        # or the calibrated penalty means something different in use.
        return ((path - path.mean()) / sd).reshape(-1, 1)

    # Monotonicity: the optimal segmentation's changepoint count is
    # non-increasing in the penalty, so once a replicate is clean at p it is
    # clean at every larger p -- ascend and stop at the first clean level.
    # This makes `counts` non-increasing in p by construction, so the rate
    # curve crosses the target at most once, AND it means a penalty skipped
    # because a smaller one was already clean is genuinely clean (not missing
    # data), so the counts stay exact.
    #
    # Pruning: a penalty whose dirty count has already exceeded target*B can
    # never meet the target, and by monotonicity neither can anything below
    # it. Advancing a floor past those is what makes B=400 affordable -- the
    # small penalties are dirty on nearly every replicate, so they are both
    # the most expensive to evaluate (no early break) and the first to be
    # disqualified.
    target = float(cal.get("pelt_target_fa", 0.05))
    budget = target * B
    lo = 0
    for _ in range(B):
        if lo >= len(grid):
            break               # every penalty on the grid is disqualified
        xb = _draw()
        if xb is None:
            continue
        try:
            algo = rpt.Pelt(model=model, min_size=min_size,
                            jump=jump).fit(xb)
        except Exception:
            continue
        hit = {}
        failed = False
        for p in grid[lo:]:
            try:
                hit[p] = (len(algo.predict(pen=p * logT)) - 1) > 0
            except Exception:
                failed = True
                break
            if not hit[p]:
                break           # monotone: all larger penalties also clean
        if failed:
            continue            # drop the whole replicate; never score it as clean
        n_ok += 1
        for p, dirty in hit.items():
            if dirty:
                counts[p] += 1
        while lo < len(grid) and counts[grid[lo]] > budget:
            lo += 1

    if n_ok == 0:
        return None, None, "fixed", {"n_ok": 0}
    # Rates over the whole grid. For penalties below the disqualification
    # floor the count stopped accumulating, so those are LOWER bounds -- fine,
    # since all we ever needed from them was "above target".
    rates = {p: counts[p] / n_ok for p in grid}
    ok = [p for p in grid[lo:] if rates[p] <= target]
    exhausted = not ok
    chosen = min(ok) if ok else max(grid)
    diag = {
        "n_ok": n_ok, "block": block, "null": null_kind,
        "rates": rates, "n_disqualified": lo,
        "se": float(np.sqrt(max(target * (1 - target), 1e-12) / n_ok)),
        "at_grid_top": exhausted,
        "ar_order": ar_p if null_kind == "ar_sieve" else None,
    }
    # Never return None for the rate: when the grid is exhausted the rate at
    # the top penalty is a lower bound, and the caller must be able to print
    # it. `at_grid_top` is what says "this did not meet target".
    return chosen, float(rates[chosen]), "empirical", diag


def innovation_cusum(z, a=0.948, burn=60, threshold=None):
    """CUSUM and CUSUM-of-squares of standardized Kalman innovations.
    Persistent excursions = the linear-Gaussian description broke."""
    z = z.iloc[burn:]
    T = len(z)
    c = z.cumsum() / z.std()
    j = np.arange(T)
    band = a * (np.sqrt(T) + 2 * j / np.sqrt(T))
    exceed = np.abs(c.values) > band
    csq = (z ** 2).cumsum() / (z ** 2).sum()
    line = (j + 1) / T
    dev = np.abs(csq.values - line)
    # ICSS-style locator: argmax of |D_k|; Inclan-Tiao 5% critical value for
    # sqrt(T/2)*max|D_k| is ~1.358 (iid-normal asymptotics -- indicative only
    # for these series, hence 'stat' is reported alongside the date).
    it_stat = float(np.sqrt(T / 2.0) * dev.max())
    sq_date = z.index[int(np.argmax(dev))]
    thr = threshold if threshold is not None else 3.5
    thr_al = (thr.reindex(z.index) if hasattr(thr, "reindex")
              else pd.Series(thr, index=z.index))
    flag_mask = (np.abs(z) > thr_al).fillna(False).values
    return {
        "cusum": pd.Series(c.values, index=z.index),
        "band": band,
        "first_exceed": z.index[np.argmax(exceed)] if exceed.any() else None,
        "cusum_sq": pd.Series(csq.values, index=z.index),
        "sq_break_date": sq_date,
        "sq_it_stat": it_stat,
        "shewhart_dates": list(z.index[flag_mask]),
        "shewhart_threshold_used": (float(thr.iloc[-1])
                                    if hasattr(thr, "iloc") else float(thr)),
        "shewhart_realized_rate": float(flag_mask.mean()),
    }


# --------------------------------------------------------------------------
# multivariate Gaussian HMM (filtered probabilities for signal use)
# --------------------------------------------------------------------------

def fit_hmm(features, n_states_list=(2, 3), seeds=(0, 1, 2, 3, 4), winsor_z=5.0):
    from hmmlearn.hmm import GaussianHMM
    F = features.dropna()
    Xz = (F - F.mean()) / F.std()
    # winsorize: stop EM spending a state on one outlier day
    Xz = Xz.clip(-abs(winsor_z), abs(winsor_z))
    X = Xz.values
    T, d = X.shape
    best = None
    for k in n_states_list:
        for s in seeds:
            try:
                m = GaussianHMM(n_components=k, covariance_type="full",
                                n_iter=300, tol=1e-4, random_state=s)
                m.fit(X)
                ll = m.score(X)
                p = (k - 1) + k * (k - 1) + k * d + k * d * (d + 1) / 2
                bic = -2 * ll + p * np.log(T)
                if best is None or bic < best["bic"]:
                    best = {"model": m, "bic": bic, "k": k, "ll": ll}
            except Exception:
                continue
    if best is None:
        return None
    m = best["model"]
    # FILTERED probabilities (forward pass only) -- tradeable, unlike smoothed
    filt = _forward_filtered(m, X)
    filt = pd.DataFrame(filt, index=F.index,
                        columns=[f"state{i}" for i in range(best["k"])])
    smooth = pd.DataFrame(m.predict_proba(X), index=F.index,
                          columns=filt.columns)
    states = filt.values.argmax(axis=1)
    dwell = _dwell_times(states)
    # state table in ORIGINAL units
    table = []
    for i in range(best["k"]):
        w = smooth.values[:, i]
        w = w / w.sum()
        mu = (F.values * w[:, None]).sum(axis=0)
        table.append(dict(zip(F.columns, mu)))
    table = pd.DataFrame(table)
    table["uncond_freq"] = [float((states == i).mean()) for i in range(best["k"])]
    table["mean_dwell_days"] = [float(np.mean(dwell[i])) if dwell[i] else np.nan
                                for i in range(best["k"])]
    return {"model": m, "k": best["k"], "bic": best["bic"],
            "filtered": filt, "smoothed": smooth, "states": pd.Series(states, index=F.index),
            "state_table": table, "transmat": m.transmat_}


def _forward_filtered(m, X):
    from scipy.stats import multivariate_normal
    T, _ = X.shape
    k = m.n_components
    logB = np.empty((T, k))
    for i in range(k):
        logB[:, i] = multivariate_normal.logpdf(X, m.means_[i], m.covars_[i],
                                                allow_singular=True)
    logA = np.log(m.transmat_ + 1e-300)
    a = np.log(m.startprob_ + 1e-300) + logB[0]
    out = np.empty((T, k))
    out[0] = np.exp(a - a.max()); out[0] /= out[0].sum()
    for t in range(1, T):
        a = logB[t] + _logsumexp_rows(a[:, None] + logA)
        w = np.exp(a - a.max())
        out[t] = w / w.sum()
    return out

def _logsumexp_rows(M):
    mx = M.max(axis=0)
    return mx + np.log(np.exp(M - mx).sum(axis=0))

def _dwell_times(states):
    from collections import defaultdict
    d = defaultdict(list)
    run, cur = 1, states[0]
    for s in states[1:]:
        if s == cur:
            run += 1
        else:
            d[cur].append(run); cur, run = s, 1
    d[cur].append(run)
    return d


def fit_hmm_train_filter_full(features, train_frac=0.6, **kw):
    """Fit HMM parameters on the first train_frac of the sample only, then run
    the forward filter over the FULL sample with those fixed parameters.
    Removes full-sample parameter look-ahead from the regime feature used in
    forecasting (the in-sample fit_hmm output is kept for description only)."""
    F = features.dropna()
    n_tr = int(len(F) * train_frac)
    fit = fit_hmm(F.iloc[:n_tr], **kw)
    if fit is None:
        return None
    mu, sd = F.iloc[:n_tr].mean(), F.iloc[:n_tr].std()
    # Clip to the SAME winsorization the parameters were estimated under. The
    # emission covariances never saw |z| beyond this, so an unclipped outlier
    # gets a wildly negative logpdf and pins the filter on one state for days
    # -- a false regime flag manufactured by a preprocessing mismatch.
    wz = abs(kw.get("winsor_z", 5.0))
    Xfull = ((F - mu) / sd).clip(-wz, wz).values
    filt = _forward_filtered(fit["model"], Xfull)
    filt = pd.DataFrame(filt, index=F.index,
                        columns=[f"state{i}" for i in range(fit["k"])])
    fit["filtered_full_pitparams"] = filt
    fit["train_end"] = F.index[n_tr - 1]
    return fit


def hysteresis_signal(filtered_prob, enter=0.8, exit=0.2):
    """Binary regime flag with hysteresis to suppress whipsaw."""
    sig = np.zeros(len(filtered_prob), dtype=int)
    on = False
    p = filtered_prob.values
    for t in range(len(p)):
        if not on and p[t] > enter:
            on = True
        elif on and p[t] < exit:
            on = False
        sig[t] = int(on)
    return pd.Series(sig, index=filtered_prob.index)


# --------------------------------------------------------------------------
# vol-beta regime: is the basis' loading on dVol itself unstable?
# --------------------------------------------------------------------------

def vol_beta_regime(basis_ret, dvol, cfg):
    rc = cfg["regimes"]
    w = rc["vol_beta_window"]
    mo = max(2, int(w * rc.get("min_obs_frac", 0.75)))
    df = pd.concat([basis_ret.rename("b"), dvol.rename("dv")], axis=1).dropna()
    roll_beta = df["b"].rolling(w, min_periods=mo).cov(df["dv"]) / \
        df["dv"].rolling(w, min_periods=mo).var()
    battery = sup_f_bootstrap(df["b"].values, df["dv"].values,
                              trim=rc["supF_trim"], B=rc["boot_B"],
                              block=rc["boot_block"],
                              min_obs=rc.get("min_obs_break_test", 120))
    if battery["break_idx"] is not None:
        battery["break_date"] = df.index[battery["break_idx"]]
    breaks = pelt_breaks(roll_beta, roll_beta.index, model="l2",
                         min_size=rc["pelt_min_size"],
                         pen_scale=rc["pelt_pen_scale"],
                         jump=rc.get("pelt_jump", 5))
    return {"rolling_beta": roll_beta, "supF": battery, "pelt_break_dates": breaks}
