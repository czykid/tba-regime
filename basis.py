"""Basis-series construction and desk-vs-quant diagnostics.

Roll handling ("panama"): daily changes are always computed on the contract
month actually held over that interval -- long the front month until the roll
date close, then long what was the back month. The drop therefore never
appears as a spurious jump in changes; it is kept as its OWN column because
drop/specialness is a funding signal, not noise to smooth away.

Point-in-time discipline: the hedge applied over day t uses the ratio known at
the close of t-1. Same-day ratios would embed look-ahead (the 60d regression
through t uses day-t returns).
"""

import numpy as np
import pandas as pd


def roll_adjusted_changes(front, back, roll_dates):
    """Per-coupon daily changes with panama roll handling.
    Returns (dP, level_adj, drop). level_adj is the additive back-adjusted
    price level (for plotting); dP is what all regressions use."""
    idx = front.index
    is_roll = idx.isin(roll_dates)
    dP = pd.DataFrame(index=idx, columns=front.columns, dtype=float)
    prev_roll = pd.Series(is_roll, index=idx).shift(1, fill_value=False)
    drop = front - back
    for c in front.columns:
        f, b = front[c], back[c]
        d = f.diff()
        # Day after the roll: at the roll-date close we sold front and bought
        # back; month labels advance next day, so the held contract's change
        # over [t*, t*+1] is front(t*+1) - back(t*). This removes the drop
        # from the change series entirely (it lives in its own column).
        d[prev_roll.values] = (f - b.shift(1))[prev_roll.values]
        dP[c] = d
    # Adjusted LEVELS: front + cumulative roll offset. Crucially NOT anchored
    # to day-0 prices (coupons illiquid at t0 would otherwise be NaN forever).
    # Offset accrues the day-after-roll drop; NaN drops (illiquid at the roll)
    # contribute zero -- that coupon's continuity is broken there anyway and
    # its changes are already NaN via the liquidity mask.
    offset_incr = drop.shift(1).where(
        pd.Series(prev_roll.values, index=idx), 0.0).fillna(0.0)
    level_adj = front + offset_incr.cumsum()
    return dP, level_adj, drop


def hedged_basis(dP_tba, dTsy, ratios, lag=1):
    """b_c(t) = dP_c(t) - ratio_c(t-lag) * dTsy(t), per coupon."""
    out = pd.DataFrame(index=dP_tba.index, columns=dP_tba.columns, dtype=float)
    for c in dP_tba.columns:
        if c in ratios.columns:
            out[c] = dP_tba[c] - ratios[c].shift(lag) * dTsy
    return out


def cc_node_basis(dS_node, dTsy, node_beta_fitted, lag=1, node=0.0):
    """Hedged basis of the current-coupon node itself (the 'CC basis')."""
    if node not in dS_node.columns:
        node = dS_node.columns[len(dS_node.columns) // 2]
    return dS_node[node] - node_beta_fitted[node].shift(lag) * dTsy


# --------------------------------------------------------------------------
# desk-vs-quant diagnostics
# --------------------------------------------------------------------------

def desk_reactivity(desk_ratio, dYield, max_lag=10):
    """Cross-correlation of desk ratio CHANGES with lagged rate changes.
    A reactive desk shows mass at positive lags (ratio moves AFTER rates)."""
    rows = {}
    dr = desk_ratio.diff().abs().mean(axis=1)  # update magnitude across coupons
    for k in range(0, max_lag + 1):
        rows[k] = float(dr.corr(dYield.abs().shift(k)))
    xcorr = pd.Series(rows, name="corr_dDesk_vs_lagged_absRateMove")
    frac_static = float((desk_ratio.diff().abs().sum(axis=1) < 1e-12).mean())
    return xcorr, frac_static


def update_trigger_profile(desk_ratio, yld, thresholds=(0.05, 0.10, 0.14, 0.18)):
    """P(desk updates | cumulative |rate move| since last update > x)."""
    dr = desk_ratio.diff().abs().sum(axis=1)
    updated = dr > 1e-12
    # SIGNED net move since last update, measured BEFORE the day's decision:
    # this is the quantity a reactive desk actually triggers on
    cum = np.zeros(len(yld))
    last = 0.0
    y = yld.values
    upd = updated.values
    for t in range(1, len(y)):
        last += y[t] - y[t - 1]
        cum[t] = last
        if upd[t]:
            last = 0.0
    cum = pd.Series(np.abs(cum), index=yld.index)
    out = {}
    for x in thresholds:
        mask = cum > x
        if mask.sum() > 10:
            out[x] = float(updated[mask.values].mean())
    base = float(updated.mean())
    return pd.Series(out, name="P(update| |net_move| > x)"), base


def hedge_slippage(dP_tba, dTsy, ratio_quant, ratio_desk, lag=1):
    """PnL difference between desk-hedged and quant-hedged books, per coupon,
    plus the ratio spread series (the 'discretion indicator')."""
    common = [c for c in ratio_desk.columns if c in ratio_quant.columns
              and c in dP_tba.columns]
    spread = (ratio_desk[common] - ratio_quant[common])
    slip = pd.DataFrame(index=dP_tba.index, columns=common, dtype=float)
    for c in common:
        # desk basis - quant basis = (quant_ratio - desk_ratio)*dTsy
        slip[c] = (ratio_quant[c] - ratio_desk[c]).shift(lag) * dTsy
    return slip, spread


def carry_columns(drop, repo, cc_series, days_to_settle=30):
    """Drop-implied roll finance vs repo -> specialness proxy (annualized %).
    Rough: implied finance = drop / price * (360/days). Kept as diagnostics."""
    implied = drop.div(100.0) * (360.0 / days_to_settle) * 100.0
    if repo is not None:
        specialness = implied.sub(repo, axis=0) * -1.0
    else:
        specialness = implied * np.nan
    return implied, specialness


def treasury_splice(ct10, wi, tsy_roll_dates):
    """Held-instrument Treasury changes across on-the-run switches.

    Convention: tsy_roll_dates lists the LAST day the outgoing note is the
    hedge. The day after, the held-instrument change is ct10(t) - wi(t-1):
    today's new on-the-run, priced yesterday as the when-issued -- same CUSIP
    on both sides, so no coupon/maturity level artifact enters the changes.

    Returns (dTsy, level, otr_wi_spread, n_gap_days). Gaps occur when the WI
    price is missing on a splice day; those days are NaN and counted -- fill
    the WI series around switches or those changes are lost.
    The CT10-WI spread is kept as a column: the on-the-run liquidity premium
    is itself a flight-to-quality covariate.
    """
    idx = ct10.index
    is_roll = idx.isin(tsy_roll_dates)
    prev_roll = pd.Series(is_roll, index=idx).shift(1, fill_value=False)
    d = ct10.diff()
    d[prev_roll.values] = (ct10 - wi.shift(1))[prev_roll.values]
    n_gaps = int(d[prev_roll.values].isna().sum())
    start = ct10.dropna().iloc[0] if ct10.notna().any() else 0.0
    level = d.fillna(0).cumsum() + start
    level[d.isna() & (np.arange(len(d)) > 0)] = np.nan
    level = pd.Series(level, index=idx).ffill()
    spread = (ct10 - wi).rename("otr_wi_spread")
    return d, level.rename("tsy_price"), spread, n_gaps


def hedge_timing_band(dP_tba, dTsy, ratio):
    """Bracket the two hedge-ratio timing conventions for an EOD-stamped
    desk file: 'new ratio starts next day' (shift 1, conservative, no
    look-ahead) vs 'ratio backfilled to the day's open' (shift 0, what a
    last-value-of-day database implies). Per-day difference = d(ratio) x
    dTsy; the cumulative sum is the band between the two basis histories.

    For a REACTIVE desk this is not mean-zero: updates follow moves, and
    d(ratio) tends to carry the sign that would have helped during the move
    that triggered it -- so the backfilled convention systematically
    flatters historical hedging. A one-signed band with significant t-stat
    means update timing is itself a P&L factor (route it to slippage
    attribution); a small sign-random band means the convention choice is
    immaterial for regime inference."""
    stats = {}
    band = pd.DataFrame(index=dP_tba.index)
    for c in ratio.columns:
        if c not in dP_tba.columns:
            continue
        dhr = ratio[c].diff()
        contrib = dhr * dTsy
        band[c] = contrib
        ev = contrib[dhr.abs() > 1e-12].dropna()
        stats[c] = {
            "n_change_days": int(len(ev)),
            "cum_band_pts": float(contrib.sum(skipna=True)),
            "mean_per_event": float(ev.mean()) if len(ev) else np.nan,
            "tstat_events": (float(ev.mean() / (ev.std() / np.sqrt(len(ev))))
                             if len(ev) > 3 and ev.std() > 0 else np.nan),
            "corr_dratio_dTsy": float(dhr.corr(dTsy)),
        }
    return band, pd.DataFrame(stats).T
