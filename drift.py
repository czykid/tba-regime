"""Drift regimes of the basis itself: carry-earning vs spread-widening.

The question this module answers: the cumulative hedged basis is normally
upward sloping (carry); occasionally it slopes down (widening episodes --
hiking cycles, risk-off shocks, demand withdrawal). Detect the slope regimes,
date them, and attribute each widening episode to candidate sources.

Statistical honesty, stated once here and repeated in the report:
  * Drift regimes are LOW-SNR objects. With daily vol several times the
    drift differential, return-only detection needs weeks of evidence.
    Vol regimes flag in days. So covariates (vol, funding) are the fast
    trigger; the drift statistics below are the confirmation layer.
  * The attribution components are correlated proxies, not an orthogonal
    decomposition. Hedge leakage and convexity overlap by construction
    (the Kalman beta adapts to the level), so convexity is a memo item.
"""

import warnings

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# carry decomposition
# --------------------------------------------------------------------------

def carry_proxy(drop_ref, index, cycle_days=21):
    """Ex-ante daily roll carry: last observed drop amortized over the roll
    cycle. The panama series realizes the drop as a lump at each roll; this
    spreads the same income evenly so the ex-carry residual is ~zero-mean
    under a pure-carry regime. Proxy only -- financing spread of the hedge
    leg is not netted here."""
    return (drop_ref.shift(1).ffill(limit=5) / cycle_days).reindex(index)


# --------------------------------------------------------------------------
# Markov-switching mean (Hamilton) with switching variance
# --------------------------------------------------------------------------

def ms_mean_regimes(b_ex, k=2, scale=100.0, sigma=None):
    """MS mean+variance model on the carry-adjusted basis return. This is the
    canonical tool for 'upward vs downward sloping': regimes differ in MEAN
    (carry vs widening), with variance allowed to switch because widening
    episodes are also high-vol. FILTERED probabilities are the tradeable
    output; smoothed are descriptive."""
    from statsmodels.tsa.regime_switching.markov_regression import (
        MarkovRegression)
    if sigma is not None:
        # vol-studentize (trailing sigma, lagged): a permanent variance era
        # otherwise hijacks the state allocation and the MEAN regime -- the
        # carry-vs-widening question -- goes undetected
        b_use = (b_ex / sigma.shift(1)).replace([np.inf, -np.inf], np.nan)
        unit = "z"
    else:
        b_use = b_ex
        unit = "pts"
    s = (b_use.dropna() * scale).astype(float)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mod = MarkovRegression(s, k_regimes=k, trend="c",
                                   switching_variance=True)
            res = mod.fit(search_reps=5)
    except Exception as e:
        return None
    names = list(res.params.index) if hasattr(res.params, "index") else \
        list(mod.param_names)
    consts = []
    for i in range(k):
        key = f"const[{i}]"
        consts.append(float(res.params[key]) if key in names else np.nan)
    widening = int(np.nanargmin(consts))
    filt = res.filtered_marginal_probabilities
    smooth = res.smoothed_marginal_probabilities
    if not isinstance(filt, pd.DataFrame):
        filt = pd.DataFrame(np.asarray(filt), index=s.index)
        smooth = pd.DataFrame(np.asarray(smooth), index=s.index)
    tab = pd.DataFrame({
        f"mean_per_day_{unit}": np.array(consts) / scale,
        f"sigma_{unit}": [np.sqrt(float(res.params[f"sigma2[{i}]"])) / scale
                          if f"sigma2[{i}]" in names else np.nan
                          for i in range(k)],
        "uncond_freq": [float((filt.values.argmax(1) == i).mean())
                        for i in range(k)],
    })
    if sigma is not None:
        med = float(sigma.median())
        tab["approx_mean_pts_per_day"] = tab[f"mean_per_day_{unit}"] * med
    # expected dwell from transition params p[i->i]
    dwell = []
    for i in range(k):
        key = f"p[{i}->{i}]"
        dwell.append(1.0 / (1.0 - float(res.params[key]))
                     if key in names and float(res.params[key]) < 1 else np.nan)
    tab["exp_dwell_days"] = dwell
    return {"result": res, "k": k, "widening_state": widening,
            "filtered": filt, "smoothed": smooth, "table": tab,
            "llf": float(res.llf), "bic": float(res.bic)}


# --------------------------------------------------------------------------
# trend segmentation of the cumulative series (slope table)
# --------------------------------------------------------------------------

def drift_segments(b_ex, min_size=40, pen_scale=2.0, widen_t=-1.0,
                   sigma=None, jump=2):
    """PELT mean-shift on returns == piecewise-linear slopes on the level.
    Returns a dated slope table; a segment is labelled 'widening' when its
    per-day mean is negative with t-stat below widen_t.

    Units: when `sigma` is supplied the detection series is vol-studentized,
    so segment slopes are in SIGMA PER DAY, not price points. Both are
    reported -- slope_z_per_day is what the t-stat and the label refer to;
    slope_pts_per_day re-expresses the same segment in points using the raw
    (unstudentized) series over the same dates, which is the number that
    means anything for P&L."""
    import ruptures as rpt
    raw = b_ex
    if sigma is not None:
        b_ex = (b_ex / sigma.shift(1)).replace([np.inf, -np.inf], np.nan)
    s = b_ex.dropna()
    z = ((s - s.mean()) / (s.std() + 1e-12)).values.reshape(-1, 1)
    T = len(z)
    if T < 2 * min_size:
        return pd.DataFrame()
    algo = rpt.Pelt(model="l2", min_size=min_size, jump=jump).fit(z)
    bkps = algo.predict(pen=pen_scale * np.log(T))
    rows, start = [], 0
    for b in bkps:
        seg = s.iloc[start:b]
        mu, sd, n = seg.mean(), seg.std(), len(seg)
        t = mu / (sd / np.sqrt(n) + 1e-12)
        label = ("widening" if (mu < 0 and t < widen_t)
                 else ("carry" if (mu > 0 and t > -widen_t) else "flat"))
        seg_raw = raw.loc[seg.index[0]:seg.index[-1]].dropna()
        mu_pts = float(seg_raw.mean()) if len(seg_raw) else np.nan
        rows.append({"start": seg.index[0], "end": seg.index[-1], "days": n,
                     "slope_z_per_day": mu, "tstat": t, "label": label,
                     "cum_z": mu * n,
                     "slope_pts_per_day": mu_pts,
                     "cum_pts": mu_pts * len(seg_raw) if len(seg_raw) else np.nan})
        start = b
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# per-episode attribution
# --------------------------------------------------------------------------

def build_components(b, carry, kal_beta, ratio_applied, dTsy, dvol=None,
                     dYield=None, beta_slope_at_cc=None, cc_sens=0.9,
                     lam_window=250):
    """Daily candidate-explanation series, all point-in-time:
      leakage  = (kalman beta - applied ratio), lagged, x dTsy
                 -> rate beta leaked while the production hedge lags
      vol_term = trailing-250d loading on dVol, lagged, x dVol
      carry    = amortized roll income (ex ante)
      conv     = 0.5 * (dBeta/dy) * dy * dTsy  [memo: overlaps leakage]
      residual = b - carry - leakage - vol_term  ('demand/spread shock')
    """
    leak = (kal_beta.shift(1) - ratio_applied.shift(1)) * dTsy
    if dvol is not None:
        df = pd.concat([(b - carry).rename("bx"), dvol.rename("dv")],
                       axis=1).dropna()
        lam = (df["bx"].rolling(lam_window, min_periods=100).cov(df["dv"]) /
               df["dv"].rolling(lam_window, min_periods=100).var())
        vol_term = lam.shift(1).reindex(b.index) * dvol.reindex(b.index)
    else:
        vol_term = pd.Series(0.0, index=b.index)
    conv = None
    if dYield is not None and beta_slope_at_cc is not None:
        dbdy = -cc_sens * beta_slope_at_cc  # dBeta/dy via moneyness channel
        conv = 0.5 * dbdy.shift(1) * dYield * dTsy
    comp = pd.DataFrame({
        "b": b, "carry": carry.reindex(b.index),
        "leakage": leak.reindex(b.index),
        "vol_term": vol_term,
    })
    if conv is not None:
        comp["convexity"] = conv.reindex(b.index)
    comp["residual_demand"] = (comp["b"] - comp["carry"].fillna(0)
                               - comp["leakage"].fillna(0)
                               - comp["vol_term"].fillna(0)
                               - (comp["convexity"].fillna(0)
                                  if "convexity" in comp else 0.0))
    return comp


def attribute_episodes(components, episodes, dYield=None):
    """Sum components over each widening episode; add rally/selloff beta
    asymmetry (flight-to-quality signature) when yields are available."""
    rows = []
    for _, ep in episodes.iterrows():
        if ep["label"] != "widening":
            continue
        w = components.loc[ep["start"]:ep["end"]]
        row = {"start": ep["start"].date(), "end": ep["end"].date(),
               "days": int(ep["days"]), "total_pts": float(w["b"].sum())}
        for c in ["carry", "leakage", "vol_term", "convexity",
                  "residual_demand"]:
            if c in w:
                row[c] = float(w[c].sum(skipna=True))
        if dYield is not None:
            dy = dYield.reindex(w.index)
            up, dn = dy.clip(lower=0), dy.clip(upper=0)
            X = np.column_stack([np.ones(len(w)), up.fillna(0), dn.fillna(0)])
            mask = w["b"].notna().values & dy.notna().values
            if mask.sum() > 25:
                beta, *_ = np.linalg.lstsq(X[mask], w["b"].values[mask],
                                           rcond=None)
                row["beta_selloff"] = float(beta[1])   # dy > 0
                row["beta_rally"] = float(beta[2])     # dy < 0
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# event overlay
# --------------------------------------------------------------------------

def event_overlay(events, b, widening_prob, pre=5, post=15):
    """events: DataFrame with columns date,label (user-supplied macro dates:
    FOMC, tariff announcements, geopolitical shocks). Reports basis drift and
    the jump in filtered widening probability around each."""
    rows = []
    for _, ev in events.iterrows():
        d = pd.Timestamp(ev["date"])
        win = b.loc[d - pd.tseries.offsets.BDay(pre):
                    d + pd.tseries.offsets.BDay(post)]
        prb = widening_prob.loc[d - pd.tseries.offsets.BDay(pre):
                                d + pd.tseries.offsets.BDay(post)] \
            if widening_prob is not None else pd.Series(dtype=float)
        pre_p = widening_prob.loc[:d].iloc[-pre - 1:-1].mean() \
            if widening_prob is not None and len(widening_prob.loc[:d]) > pre \
            else np.nan
        rows.append({
            "date": d.date(), "label": ev.get("label", ""),
            "cum_b_window_pts": float(win.sum()),
            "mean_b_per_day": float(win.mean()) if len(win) else np.nan,
            "widen_prob_before": float(pre_p) if pre_p == pre_p else np.nan,
            "widen_prob_peak_after": float(prb.max()) if len(prb) else np.nan,
        })
    return pd.DataFrame(rows)
