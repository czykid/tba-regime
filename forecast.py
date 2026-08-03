"""Forecast harness for the hedged basis.

Target discipline: the hedged basis return is already rate-hedged, i.e. the
excess-return object. NEVER score forecasts of the raw forward path -- the
forward is mostly carry/financing and near-deterministic given repo, so
accuracy against it is an identity, not skill.

Evaluation:
  * OOS R^2 vs expanding historical mean.
  * Clark-West for NESTED comparisons (Diebold-Mariano is invalid there).
  * Giacomini-White CONDITIONAL test: does the regime variable predict which
    model wins? That is the actual claim a regime-switching signal makes.
  * Economic eval with turnover costs -- regime signals whipsaw, and hit
    rates are meaningless under asymmetric payoffs.

Walk-forward: expanding window, refit every `refit_every` days, and an
`embargo` purge so no training target overlaps the prediction window.
"""

import numpy as np
import pandas as pd


def build_target(b, h):
    """Next-h-day cumulative hedged basis return, aligned to feature date t."""
    return b.shift(-1).rolling(h).sum().shift(-(h - 1))


def _fit_predict(Xtr, ytr, Xte):
    X = np.column_stack([np.ones(len(Xtr)), Xtr])
    beta, *_ = np.linalg.lstsq(X, ytr, rcond=None)
    return beta[0] + Xte @ beta[1:]


def walk_forward(features, target, cfg):
    """Returns DataFrame of predictions per model + realized target.
    Models:
      mean        : expanding historical mean (benchmark)
      ar          : lagged h-day basis return
      regime_cond : mean + regime dummy interaction (uses filtered state)
      vol_cond    : mean + dVol level feature
    Feature availability is enforced at t; training pairs end >= embargo
    before t."""
    fc = cfg["forecast"]
    h, refit, min_train, emb = (fc["horizon"], fc["refit_every"],
                                fc["min_train"], fc["embargo"])
    df = pd.concat([features, target.rename("y")], axis=1).dropna()
    idx = df.index
    T = len(df)
    preds = {m: np.full(T, np.nan) for m in
             ["mean", "ar", "regime_cond", "vol_cond"]}
    cols_ar = [c for c in features.columns if c.startswith("lag_b")]
    cols_rg = [c for c in features.columns if c.startswith("regime")] + cols_ar
    cols_vl = [c for c in features.columns if c.startswith("dvol")] + cols_ar

    cached = {}
    last_fit = None
    for t in range(min_train, T):
        train_end = t - h - emb
        if train_end < 100:
            continue
        # Refit on cadence, and always on the first usable t. The previous
        # condition tested `t - 1 not in cached`, i.e. whether an INTEGER was
        # a key of a dict keyed by model names -- never true, so the cadence
        # was dead and this refit every single day.
        if last_fit is None or (t - last_fit) >= refit:
            last_fit = t
            ytr = df["y"].values[:train_end]
            cached = {"mean": ytr.mean()}
            for name, cols in (("ar", cols_ar), ("regime_cond", cols_rg),
                               ("vol_cond", cols_vl)):
                if cols:
                    Xtr = df[cols].values[:train_end]
                    X = np.column_stack([np.ones(len(Xtr)), Xtr])
                    beta, *_ = np.linalg.lstsq(X, ytr, rcond=None)
                    cached[name] = (cols, beta)
        preds["mean"][t] = cached["mean"]
        xrow = df.iloc[t]
        for name in ["ar", "regime_cond", "vol_cond"]:
            if name in cached:
                cols, beta = cached[name]
                preds[name][t] = beta[0] + xrow[cols].values @ beta[1:]
    out = pd.DataFrame(preds, index=idx)
    out["y"] = df["y"]
    return out.dropna(subset=["mean"])


# --------------------------------------------------------------------------
# statistical evaluation
# --------------------------------------------------------------------------

def _nw_var(u, lag):
    T = len(u)
    u = u - u.mean()
    g0 = (u @ u) / T
    v = g0
    for l in range(1, lag + 1):
        w = 1 - l / (lag + 1)
        v += 2 * w * (u[:-l] @ u[l:]) / T
    return v

def oos_r2(y, pred, bench):
    e1 = y - pred
    e0 = y - bench
    return 1 - (e1 @ e1) / (e0 @ e0)

def clark_west(y, pred_small, pred_big, hac_lag):
    """CW adjusted MSPE test for nested models. H0: models equal;
    positive significant stat => big model improves."""
    e0 = y - pred_small
    e1 = y - pred_big
    f = e0 ** 2 - e1 ** 2 + (pred_small - pred_big) ** 2
    T = len(f)
    se = np.sqrt(_nw_var(f, hac_lag) / T)
    t = f.mean() / se if se > 0 else np.nan
    from scipy.stats import norm
    return {"cw_stat": float(t), "cw_p_onesided": float(1 - norm.cdf(t))}

def giacomini_white(y, pred_a, pred_b, Z, hac_lag):
    """Conditional predictive ability: regress loss differential
    d = L(a)-L(b) on instruments Z (const + regime vars, lagged).
    Wald that all coefs = 0 (HAC). Rejection + sign of regime coef tells you
    WHEN model b wins -- the tradeable statement."""
    d = (y - pred_a) ** 2 - (y - pred_b) ** 2
    sd = d.std()
    if sd < 1e-300:
        return {"gw_wald": np.nan, "gw_p": np.nan,
                "coefs": [], "note": "zero loss differential"}
    d = d / sd                            # pure rescaling; Wald is invariant
    Z = np.atleast_2d(Z)
    # an instrument must genuinely vary: for near-binary columns require
    # >= 20 obs in the minority value, else the HAC covariance is rank-starved
    keep = []
    for j in range(Z.shape[1]):
        col = Z[:, j]
        if col.std() <= 1e-10:
            keep.append(False); continue
        uniq = np.unique(np.round(col, 8))
        if len(uniq) <= 2:
            minority = min((col == u).sum() for u in uniq)
            keep.append(minority >= 20)
        else:
            keep.append(True)
    keep = np.array(keep)
    if not keep.any():
        return {"gw_wald": np.nan, "gw_p": np.nan,
                "coefs": [], "note": "instruments constant/degenerate"}
    Zc = np.column_stack([np.ones(len(d)), Z[:, keep]])
    beta, *_ = np.linalg.lstsq(Zc, d, rcond=None)
    u = d - Zc @ beta
    T, k = Zc.shape
    Su = Zc * u[:, None]
    S = np.zeros((k, k))
    g0 = Su.T @ Su / T
    S += g0
    for l in range(1, hac_lag + 1):
        w = 1 - l / (hac_lag + 1)
        gl = Su[l:].T @ Su[:-l] / T
        S += w * (gl + gl.T)
    XtX = Zc.T @ Zc / T
    try:
        if np.linalg.cond(XtX) > 1e10:
            raise np.linalg.LinAlgError("ill-conditioned instruments")
        Xi = np.linalg.inv(XtX)
        V = Xi @ S @ Xi / T
        if not np.isfinite(V).all() or np.linalg.cond(V) > 1e12:
            raise np.linalg.LinAlgError("near-singular HAC covariance")
        W = float(beta @ np.linalg.pinv(V, rcond=1e-12) @ beta)
        from scipy.stats import chi2
        p = float(1 - chi2.cdf(W, k))
        if not np.isfinite(W) or W < 0 or W > 1e6:
            W, p = np.nan, np.nan
    except np.linalg.LinAlgError:
        W, p = np.nan, np.nan
    return {"gw_wald": W, "gw_p": p, "coefs": beta.tolist()}


# --------------------------------------------------------------------------
# economic evaluation
# --------------------------------------------------------------------------

def econ_eval(preds, cfg, scale_vol_target=None):
    """Sign-based position in the basis, unit gross, turnover-costed.

    Decisions taken every h days and held for h days (NON-overlapping), so the
    PnL observations are serially independent under the null. Scoring daily
    positions against an overlapping h-day target would inflate Sharpe by
    roughly sqrt(h) through mechanical autocorrelation."""
    fc = cfg["forecast"]
    h = fc["horizon"]
    P = preds.iloc[::h]                   # stride-h decision dates
    out = {}
    ann = np.sqrt(252.0 / h)              # h-day periods per year
    for m in ["mean", "ar", "regime_cond", "vol_cond"]:
        pos = np.sign(P[m]).fillna(0.0)
        pnl_g = pos * P["y"]
        cost = pos.diff().abs().fillna(0.0) * fc["cost_per_unit"]
        pnl_n = pnl_g - cost
        out[m] = {
            "gross_sharpe": float(pnl_g.mean() / (pnl_g.std() + 1e-12) * ann),
            "net_sharpe": float(pnl_n.mean() / (pnl_n.std() + 1e-12) * ann),
            "ann_turnover": float(pos.diff().abs().mean() * (252.0 / h)),
            "hit_rate": float((np.sign(P[m]) == np.sign(P["y"])).mean()),
            "max_dd": float((pnl_n.cumsum().cummax() - pnl_n.cumsum()).max()),
            "n_decisions": int(len(P)),
        }
    return pd.DataFrame(out).T


def evaluate(preds, regime_dummy, cfg):
    fc = cfg["forecast"]
    y = preds["y"].values
    res = {}
    for m in ["ar", "regime_cond", "vol_cond"]:
        if preds[m].notna().sum() < 100:
            continue
        mask = preds[m].notna().values
        r = {"oos_R2_vs_mean": float(oos_r2(y[mask], preds[m].values[mask],
                                            preds["mean"].values[mask]))}
        r.update(clark_west(y[mask], preds["mean"].values[mask],
                            preds[m].values[mask], fc["hac_lag"]))
        if regime_dummy is not None:
            Z = regime_dummy.reindex(preds.index).shift(fc["horizon"]) \
                .fillna(0).values[mask].reshape(-1, 1)
            r.update(giacomini_white(y[mask], preds["mean"].values[mask],
                                     preds[m].values[mask], Z, fc["hac_lag"]))
        res[m] = r
    return pd.DataFrame(res).T
