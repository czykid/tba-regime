"""Synthetic TBA / Treasury / vol / desk-ratio generator with PLANTED breaks.

Purpose: smoke-test the pipeline and verify break-recovery before real data
arrives. Stylized, not calibrated -- do not use for production inference.

Pricing model (fixed coupon c, moneyness m = c - cc_t):
    P(c,t) = 100 + G(m) - delta(m) * (y_t - ybar) + u_t + eps_{c,t}
  G(m)     = a*m - b*c0*log(cosh(m/c0))    -> slope a - b*tanh(m/c0):
             steep for discounts, flat for premiums (negative convexity).
  delta(m) = direct rate-level sensitivity beyond the CC channel (OAS/level
             effect). This is what a constant-moneyness series retains.
  u_t      = common basis factor, Markov-switching mean/vol (planted break),
             loading on dVol only after vol_link_start (planted vol-beta regime).
  Planted duration break: G-slope and delta scaled by beta_break_mult after
  break_beta_idx.

Desk hedge ratio: reactive -- updates only after cumulative |rate move| since
last update exceeds a threshold, then moves 70% toward the quant value, with a
persistent discretionary offset that switches occasionally.
"""

import numpy as np
import pandas as pd


def _g_slope(m, a=5.25, b=2.75, c0=0.6):
    return a - b * np.tanh(m / c0)


def _g(m, a=5.25, b=2.75, c0=0.6):
    return a * m - b * c0 * np.log(np.cosh(m / c0))


def _delta(m, d0=1.6):
    # larger direct rate sensitivity for discounts
    return d0 * (1.0 + 0.5 / (1.0 + np.exp(2.0 * m)))


def generate(cfg):
    sc = cfg["synth"]
    rng = np.random.default_rng(sc["seed"])
    T = sc["n_days"]
    dates = pd.bdate_range("2019-07-01", periods=T)
    coupons = np.array(sc["coupons"])

    # ---- rates: OU around a drifting mean (low -> high, mimics 2022), MS vol ----
    mu = np.concatenate([
        np.full(500, 2.0),
        np.linspace(2.0, 4.5, 500),
        np.full(T - 1000, 4.5),
    ])
    vol_state = np.zeros(T, dtype=int)
    p_stay = np.array([[0.985, 0.015], [0.03, 0.97]])
    for t in range(1, T):
        vol_state[t] = rng.choice(2, p=p_stay[vol_state[t - 1]])
    sig_y = np.where(vol_state == 0, 0.05, 0.10)  # daily bp/100 (i.e., % units)
    y = np.empty(T)
    y[0] = 2.0
    for t in range(1, T):
        y[t] = y[t - 1] + 0.03 * (mu[t] - y[t - 1]) + sig_y[t] * rng.standard_normal() / np.sqrt(1)
    y = np.clip(y, 0.4, 6.5)

    # continuous (back-adjusted) 10y proxy price
    tsy_price = 100 - sc["tsy_duration"] * (y - y[0])
    tsy_price = tsy_price - tsy_price[0] + 100.0

    # ---- current coupon: 10y + slow-moving spread ----
    spread = np.empty(T)
    spread[0] = 1.4
    for t in range(1, T):
        spread[t] = spread[t - 1] + 0.01 * (1.55 - spread[t - 1]) + 0.008 * rng.standard_normal()
    cc = y + spread

    # ---- swaption vol proxy: loads on realized rate vol + own MS state ----
    rv = pd.Series(np.abs(np.diff(y, prepend=y[0]))).ewm(span=20).mean().values
    vol = 60 + 900 * rv + np.where(vol_state == 1, 18, 0) + \
        np.cumsum(0.4 * rng.standard_normal(T)) * 0.3
    vol = np.clip(vol, 35, None)
    dvol = np.diff(vol, prepend=vol[0])
    # NOTE: stress windows also bump the vol SERIES below (after u uses dvol
    # pre-bump for the vol-link channel), mimicking risk-off co-movement

    # ---- common basis factor u_t: MS mean/vol with planted shift ----
    b_idx = sc["break_basis_idx"]
    u = np.empty(T)
    u[0] = 0.0
    mean_u = np.where(np.arange(T) < b_idx, 0.0, -0.55)
    sig_u = np.where(np.arange(T) < b_idx, 0.035, 0.075)
    lam = np.where(np.arange(T) >= sc["vol_link_start"], -0.010, 0.0)  # vol-beta regime
    stress = np.zeros(T, dtype=bool)
    for a, b in sc.get("stress_windows", []):
        stress[a:b] = True
    sig_u = np.where(stress, sig_u * sc.get("stress_sig_mult", 1.0), sig_u)
    for t in range(1, T):
        u[t] = u[t - 1] + 0.03 * (mean_u[t] - u[t - 1]) \
            + lam[t] * dvol[t] + sig_u[t] * rng.standard_normal()
    # explicit SLOPE channel, outside the OU (an OU absorbs constant drift
    # into its level): spread-carry accrual in calm regimes, sustained
    # widening drift in stress windows. This is what makes the cumulative
    # basis upward-sloping normally and downward-sloping in the planted
    # episodes -- the pattern the drift module must recover.
    trend_incr = np.where(stress, sc.get("stress_drift", -0.045),
                          sc.get("carry_drift", 0.0))
    u = u + np.cumsum(trend_incr)

    # ---- planted duration break ----
    dur_mult = np.where(np.arange(T) < sc["break_beta_idx"], 1.0, sc["beta_break_mult"])

    # ---- coupon-level prices ----
    ybar = y.mean()
    front = pd.DataFrame(index=dates, columns=coupons, dtype=float)
    eps = {c: np.cumsum(0.02 * rng.standard_normal(T)) * 0.4 for c in coupons}
    for c in coupons:
        m = c - cc
        front[c] = (100 + dur_mult * _g(m)
                    - dur_mult * _delta(m) * (y - ybar)
                    + u + eps[c])
    # liquidity mask: quotes only within +-175bp of CC
    for c in coupons:
        front.loc[np.abs(c - cc) > 1.75, c] = np.nan

    # ---- roll calendar + drop (specialness episodes) ----
    months = pd.Series(dates).dt.to_period("M")
    roll_dates = []
    for mth in months.unique():
        md = dates[(months == mth).values]
        if len(md) >= 11:
            roll_dates.append(md[10])
    roll_dates = pd.DatetimeIndex(roll_dates)
    special = 0.02 * (rng.random(T) < 0.06) * rng.random(T)
    drop_base = 0.06 + 0.5 * np.maximum(cc - y - 1.3, 0) * 0.1
    drop = pd.DataFrame(
        {c: drop_base + special + 0.01 * rng.standard_normal(T) for c in coupons},
        index=dates)
    back = front - drop

    # ---- quant "true" fixed-coupon beta at CC (for scoring only) ----
    true_beta_cc = dur_mult * (_g_slope(0.0) + _delta(0.0)) / sc["tsy_duration"]

    # ---- desk ratio: reactive + discretionary offset (built on true beta + noise) ----
    desk = {}
    for c in [3.0, 3.5, 4.0, 4.5, 5.0]:
        m = c - cc
        quant_like = dur_mult * (_g_slope(m) + _delta(m)) / sc["tsy_duration"]
        d = np.empty(T)
        d[0] = quant_like[0]
        cum = 0.0
        offset = 0.0
        for t in range(1, T):
            cum += y[t] - y[t - 1]
            if rng.random() < 0.005:
                offset = rng.choice([-0.05, 0.0, 0.0, 0.07])
            if abs(cum) > 0.15:                      # update only after ~15bp net move
                d[t] = d[t - 1] + 0.6 * (quant_like[t] + offset - d[t - 1])
                cum = 0.0
            else:
                d[t] = d[t - 1]
        desk[c] = d
    desk = pd.DataFrame(desk, index=dates)

    for a, b in sc.get("stress_windows", []):
        vol[a:b] += sc.get("stress_vol_bump", 0.0)
    # ---- raw on-the-run 10y + when-issued around quarterly switches ----
    # each quarter's note has its own price LEVEL (different coupon/maturity);
    # the true continuous changes are those of tsy_price. The splice must
    # recover them exactly from (ct10, wi, calendar).
    qlen = 63
    nq = T // qlen + 2
    J = rng.normal(0, 1.4, nq)
    quarter = np.arange(T) // qlen
    ct10 = tsy_price + J[quarter]
    wi = np.full(T, np.nan)
    tsy_rolls = []
    for q in range(nq - 1):
        b_idx = (q + 1) * qlen - 1          # last day of quarter q
        if b_idx >= T - 1:
            break
        a_idx = max(b_idx - 9, 0)
        wi[a_idx:b_idx + 1] = tsy_price[a_idx:b_idx + 1] + J[q + 1]
        tsy_rolls.append(dates[b_idx])
    data = {
        "dates": dates,
        "tba_front": front,
        "tba_back": back,
        "tsy_price": pd.Series(tsy_price, index=dates, name="tsy_price"),
        "ct10": pd.Series(ct10, index=dates, name="ct10_price"),
        "wi": pd.Series(wi, index=dates, name="wi_price"),
        "tsy_roll_dates": pd.DatetimeIndex(tsy_rolls),
        "tsy_yield": pd.Series(y, index=dates, name="tsy_yield"),
        "cc": pd.Series(cc, index=dates, name="cc"),
        "vol": pd.Series(vol, index=dates, name="vol"),
        "roll_dates": roll_dates,
        "desk_ratio": desk,
        "repo": pd.Series(np.clip(y - 0.4, 0.05, None), index=dates, name="repo"),
    }
    truth = {
        "drift_windows": [(dates[a], dates[b - 1])
                          for a, b in sc.get("stress_windows", [])],
        "break_beta_date": dates[sc["break_beta_idx"]],
        "break_basis_date": dates[sc["break_basis_idx"]],
        "vol_link_date": dates[sc["vol_link_start"]],
        "true_beta_cc": pd.Series(true_beta_cc, index=dates),
        "vol_state": pd.Series(vol_state, index=dates),
    }
    return data, truth


def write_csvs(data, path):
    """Write the exact CSV schema the pipeline expects from real data."""
    import os
    os.makedirs(path, exist_ok=True)
    f = data["tba_front"].stack().rename("price_front")
    b = data["tba_back"].stack().rename("price_back")
    tba = pd.concat([f, b], axis=1).reset_index()
    tba.columns = ["date", "coupon", "price_front", "price_back"]
    tba.to_csv(f"{path}/tba_prices.csv", index=False)
    pd.concat([data["tsy_price"], data["tsy_yield"]], axis=1).reset_index() \
        .rename(columns={"index": "date"}).to_csv(f"{path}/treasury.csv", index=False)
    data["cc"].reset_index().rename(columns={"index": "date"}) \
        .to_csv(f"{path}/current_coupon.csv", index=False)
    data["vol"].reset_index().rename(columns={"index": "date"}) \
        .to_csv(f"{path}/vol.csv", index=False)
    pd.Series(data["roll_dates"], name="roll_date") \
        .to_csv(f"{path}/roll_dates.csv", index=False)
    raw = pd.concat([data["ct10"], data["wi"]], axis=1).reset_index() \
        .rename(columns={"index": "date"})
    raw.to_csv(f"{path}/treasury_raw.csv", index=False)
    pd.Series(data["tsy_roll_dates"], name="tsy_roll_date") \
        .to_csv(f"{path}/tsy_roll_dates.csv", index=False)
    d = data["desk_ratio"].stack().rename("ratio").reset_index()
    d.columns = ["date", "coupon", "ratio"]
    d.to_csv(f"{path}/hedge_ratio_desk.csv", index=False)
    data["repo"].reset_index().rename(columns={"index": "date"}) \
        .to_csv(f"{path}/repo.csv", index=False)
