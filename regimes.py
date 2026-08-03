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

def sup_f_bootstrap(y, x, trim=0.15, B=199, block=20, seed=0):
    """H0: constant (a,b). Bootstrap resamples (x, e_hat) in circular blocks,
    rebuilds y* under the null fit, recomputes sup-F."""
    y = np.asarray(y, float); x = np.asarray(x, float)
    mask = np.isfinite(y) & np.isfinite(x)
    y, x = y[mask], x[mask]
    T = len(y)
    if T < 120:
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

def nyblom_bootstrap(y, x, B=199, block=20, seed=1):
    y = np.asarray(y, float); x = np.asarray(x, float)
    mask = np.isfinite(y) & np.isfinite(x)
    y, x = y[mask], x[mask]
    T = len(y)
    if T < 120:
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

def cusum_recursive(y, x):
    """Brown-Durbin-Evans CUSUM via statsmodels RecursiveLS on dP ~ dTsy.
    Tests constancy of the hedge relationship directly."""
    import statsmodels.api as sm
    df = pd.concat([pd.Series(y, name="y"), pd.Series(x, name="x")], axis=1).dropna()
    if len(df) < 150:
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

def pelt_breaks(series, index, model="l2", min_size=90, pen_scale=3.0):
    import ruptures as rpt
    s = pd.Series(series, index=index).dropna()
    z = (s - s.mean()) / (s.std() + 1e-12)
    x = z.values.reshape(-1, 1)
    T = len(x)
    if T < 2 * min_size:
        return []
    pen = pen_scale * np.log(T)
    algo = rpt.Pelt(model=model, min_size=min_size, jump=5).fit(x)
    bkps = algo.predict(pen=pen)
    return [s.index[b - 1] for b in bkps[:-1]]


# --------------------------------------------------------------------------
# Kalman time-varying-parameter hedge beta
# --------------------------------------------------------------------------

def kalman_tvp(y, x, q_init=1e-6, estimate=True):
    """State [alpha_t, beta_t] random walk; obs y_t = a_t + b_t x_t + eps.
    MLE over (log r, log q_a, log q_b) by Nelder-Mead. Returns filtered path
    and standardized one-step innovations (the online regime detector)."""
    df = pd.concat([pd.Series(y, name="y"), pd.Series(x, name="x")], axis=1).dropna()
    yy, xx = df["y"].values, df["x"].values
    T = len(yy)
    if T < 150:
        return None

    # init from first 60 obs OLS
    X0 = np.column_stack([np.ones(60), xx[:60]])
    b0, *_ = np.linalg.lstsq(X0, yy[:60], rcond=None)
    r0 = np.var(yy[:60] - X0 @ b0)

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


def empirical_shewhart_threshold(z, cal, burn=60):
    """Threshold for |z| flags, calibrated to the data's OWN tail rather than
    a Gaussian convention. Trailing quantile => self-adjusting to residual
    leptokurtosis: if innovations are t-like, a fixed 3.5 fires ~10x more
    often than its nominal rate, while the quantile holds the DESIGNED alarm
    rate whatever the tail shape. Trailing (not full-sample) keeps it PIT.
    Returns (threshold_series, mode_used)."""
    zz = z.iloc[burn:].abs()
    if cal.get("shewhart_mode") != "empirical" or \
            len(zz) < cal.get("shewhart_min_window", 250):
        return pd.Series(cal.get("shewhart_fixed", 3.5), index=z.index), "fixed"
    w = cal.get("shewhart_window", 500)
    q = cal.get("shewhart_quantile", 0.998)
    thr = zz.rolling(w, min_periods=cal.get("shewhart_min_window", 250)) \
            .quantile(q).shift(1)
    thr = thr.reindex(z.index).ffill().fillna(cal.get("shewhart_fixed", 3.5))
    return thr, "empirical"


def calibrate_pelt_penalty(series, cal, min_size, model="l2", seed=0):
    """Choose the PELT penalty by SIMULATING THE NULL instead of trusting
    pen = c*log(T). Circular-block bootstrap destroys any break structure
    while preserving the serial correlation (critical: 60d rolling betas are
    ~59/60 overlapping, so an uncalibrated penalty over-detects wildly).
    Returns the smallest penalty whose family-wise false-alarm rate is at or
    below target -- smallest because that maximizes power subject to the
    error constraint. Also returns the achieved rate for the report."""
    import ruptures as rpt
    s = pd.Series(series).dropna()
    T = len(s)
    if cal.get("pelt_mode") != "empirical" or T < 4 * min_size:
        return None, None, "fixed"
    z = ((s - s.mean()) / (s.std() + 1e-12)).values
    rng = np.random.default_rng(seed)
    B = cal.get("pelt_boot_B", 99)
    block = max(min_size // 2, 20)
    grid = sorted(cal.get("pelt_pen_grid", [1.0, 2.0, 3.0, 6.0]))
    counts = {p: 0 for p in grid}
    # coarse-to-fine: once a replicate is clean at penalty p it is clean at
    # every larger p (PELT is monotone in the penalty), so ascend the grid
    # and stop at the first clean level for that replicate.
    for _ in range(B):
        idx = _block_boot_idx(T, block, rng)
        xb = z[idx].reshape(-1, 1)
        try:
            algo = rpt.Pelt(model=model, min_size=min_size,
                            jump=cal.get("jump", 5)).fit(xb)
        except Exception:
            continue
        for p in grid:
            try:
                if len(algo.predict(pen=p * np.log(T))) - 1 > 0:
                    counts[p] += 1
                else:
                    break   # monotone: all larger penalties also clean
            except Exception:
                pass
    target = cal.get("pelt_target_fa", 0.05)
    rates = {p: counts[p] / max(B, 1) for p in grid}
    ok = [p for p in grid if rates[p] <= target]
    chosen = min(ok) if ok else max(grid)
    return chosen, rates.get(chosen), "empirical"


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

def fit_hmm(features, n_states_list=(2, 3), seeds=(0, 1, 2, 3, 4)):
    from hmmlearn.hmm import GaussianHMM
    F = features.dropna()
    Xz = (F - F.mean()) / F.std()
    Xz = Xz.clip(-5, 5)  # winsorize: stop EM spending a state on one outlier day
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
    Xfull = ((F - mu) / sd).values
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
    w = cfg["regimes"]["vol_beta_window"]
    df = pd.concat([basis_ret.rename("b"), dvol.rename("dv")], axis=1).dropna()
    roll_beta = df["b"].rolling(w, min_periods=int(w * 0.75)).cov(df["dv"]) / \
        df["dv"].rolling(w, min_periods=int(w * 0.75)).var()
    battery = sup_f_bootstrap(df["b"].values, df["dv"].values,
                              trim=cfg["regimes"]["supF_trim"],
                              B=cfg["regimes"]["boot_B"],
                              block=cfg["regimes"]["boot_block"])
    if battery["break_idx"] is not None:
        battery["break_date"] = df.index[battery["break_idx"]]
    breaks = pelt_breaks(roll_beta, roll_beta.index, model="l2",
                         min_size=cfg["regimes"]["pelt_min_size"],
                         pen_scale=cfg["regimes"]["pelt_pen_scale"])
    return {"rolling_beta": roll_beta, "supF": battery, "pelt_break_dates": breaks}
