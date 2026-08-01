"""Replication of the quant hedge-ratio pipeline, in two variants.

Step 1: interpolate artificial TBA prices at 10bp moneyness nodes around the
        current coupon, linearly between adjacent actual coupons.
Step 2: 60-day rolling regression of node price changes on 10y Treasury price
        changes -> empirical hedge ratio per node.
Step 3: fit a smooth curve across moneyness daily; read actual-coupon ratios
        off the curve at m = coupon - CC(t).

VARIANT A ("literal"): difference the moneyness-anchored series S_m(t), where
  the anchor CC(t) moves daily. This measures the sensitivity of a portfolio
  that continuously rebalances along the coupon stack to stay at constant
  moneyness. Because moving along the stack partially offsets the rate move
  (dP/dc * dCC/dy has the opposite sign to dP/dy), Variant A systematically
  UNDERSTATES the duration of a fixed coupon you actually hold.

VARIANT B ("fixed-instrument"): compute daily changes per actual coupon first
  (same instrument both days, roll-adjusted), then interpolate the CHANGES to
  moneyness nodes using yesterday's anchor CC(t-1). This is the sensitivity of
  the coupon you hold, mapped onto the moneyness grid.

The A-B gap is approximately dP/dc * dCC/dy scaled by Treasury duration, and
its size is a direct empirical test of how much the literal step-2 procedure
dampens the hedge. If the desk systematically runs ratios above the quant
benchmark after selloffs, this is the first suspect.
"""

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# current coupon
# --------------------------------------------------------------------------

def compute_current_coupon(front, coupon_step=0.5):
    """Fallback par-coupon: linear interpolation in price to P=100 on the front
    month. Post-2022-style stacks entirely below par force EXTRAPOLATION above
    the top liquid coupon -- flagged, because a convention change here is a
    manufactured structural break. Prefer a supplied CC series."""
    out, flag = [], []
    for _, row in front.iterrows():
        r = row.dropna()
        if len(r) < 2:
            out.append(np.nan); flag.append(True); continue
        c = r.index.values.astype(float); p = r.values
        order = np.argsort(c); c, p = c[order], p[order]
        if (p >= 100).any() and (p <= 100).any():
            i = np.where(p >= 100)[0][-1] if p[0] >= 100 else np.where(p <= 100)[0][0]
            # find bracketing pair around 100
            j = None
            for k in range(len(p) - 1):
                if (p[k] - 100) * (p[k + 1] - 100) <= 0:
                    j = k; break
            if j is None:
                out.append(np.nan); flag.append(True); continue
            w = (100 - p[j]) / (p[j + 1] - p[j])
            out.append(c[j] + w * (c[j + 1] - c[j])); flag.append(False)
        else:
            # whole stack one side of par: extrapolate from the two nearest-par coupons
            k = np.argsort(np.abs(p - 100))[:2]
            k = np.sort(k)
            slope = (c[k[1]] - c[k[0]]) / (p[k[1]] - p[k[0]])
            out.append(c[k[1]] + slope * (100 - p[k[1]])); flag.append(True)
    return (pd.Series(out, index=front.index, name="cc"),
            pd.Series(flag, index=front.index, name="cc_extrapolated"))


# --------------------------------------------------------------------------
# interpolation to moneyness nodes
# --------------------------------------------------------------------------

def _interp_row(values_by_coupon, target_coupons, coupon_step):
    """Linear interpolation between adjacent actual coupons; NaN outside the
    quoted range or if either bracket is missing."""
    s = values_by_coupon.dropna()
    out = np.full(len(target_coupons), np.nan)
    if len(s) < 2:
        return out
    cs = s.index.values.astype(float)
    vs = s.values
    for i, tc in enumerate(target_coupons):
        lo = coupon_step * np.floor(tc / coupon_step + 1e-9)
        hi = lo + coupon_step
        if abs(tc - lo) < 1e-9:
            hi = lo
        if lo in s.index and (hi in s.index or hi == lo):
            if hi == lo:
                out[i] = s[lo]
            else:
                w = (hi - tc) / coupon_step
                out[i] = w * s[lo] + (1 - w) * s[hi]
        else:
            # allow interpolation between whatever brackets exist on the 0.5 grid
            below = cs[cs <= tc + 1e-9]
            above = cs[cs >= tc - 1e-9]
            if len(below) and len(above):
                c0, c1 = below.max(), above.min()
                if c1 - c0 <= coupon_step + 1e-9 and c1 > c0:
                    w = (c1 - tc) / (c1 - c0)
                    out[i] = w * s[c0] + (1 - w) * s[c1]
                elif c1 == c0:
                    out[i] = s[c0]
    return out


def build_node_series(front_adj, cc, cfg):
    """Variant A step 1: S_m(t) = interpolated price at CC(t)+m.
    front_adj must already be roll-adjusted per coupon (see basis.py)."""
    hc = cfg["hedge"]
    grid = np.arange(-hc["moneyness_max_bp"], hc["moneyness_max_bp"] + 1,
                     hc["moneyness_grid_bp"]) / 100.0
    S = pd.DataFrame(index=front_adj.index, columns=np.round(grid, 2), dtype=float)
    for t in front_adj.index:
        if pd.isna(cc.loc[t]):
            continue
        S.loc[t] = _interp_row(front_adj.loc[t], cc.loc[t] + grid, hc["coupon_step"])
    return S


def build_node_changes_B(front_adj, cc, cfg):
    """Variant B: per-coupon changes (same instrument), interpolated to nodes
    anchored at CC(t-1) -- the instrument actually held into day t."""
    hc = cfg["hedge"]
    grid = np.arange(-hc["moneyness_max_bp"], hc["moneyness_max_bp"] + 1,
                     hc["moneyness_grid_bp"]) / 100.0
    dP = front_adj.diff()
    cc_lag = cc.shift(1)
    dS = pd.DataFrame(index=front_adj.index, columns=np.round(grid, 2), dtype=float)
    for t in front_adj.index:
        a = cc_lag.loc[t]
        if pd.isna(a):
            continue
        dS.loc[t] = _interp_row(dP.loc[t], a + grid, hc["coupon_step"])
    return dS


# --------------------------------------------------------------------------
# rolling node regressions + daily curve fit
# --------------------------------------------------------------------------

def rolling_node_betas(dS, dTsy, cfg):
    """60d rolling OLS beta of node changes on Treasury price changes.
    Vectorized via rolling moments. Trailing-only."""
    hc = cfg["hedge"]
    w, mo = hc["reg_window"], hc["min_obs"]
    x = dTsy.reindex(dS.index)
    betas = pd.DataFrame(index=dS.index, columns=dS.columns, dtype=float)
    r2 = pd.DataFrame(index=dS.index, columns=dS.columns, dtype=float)
    for col in dS.columns:
        yv = dS[col]
        cov = yv.rolling(w, min_periods=mo).cov(x)
        varx = x.rolling(w, min_periods=mo).var()
        vary = yv.rolling(w, min_periods=mo).var()
        b = cov / varx
        betas[col] = b
        r2[col] = (cov ** 2) / (varx * vary)
    return betas, r2


def fit_curves(betas, cfg):
    """Daily smooth curve beta(m); returns fitted node values + coefficients."""
    hc = cfg["hedge"]
    m = betas.columns.values.astype(float)
    fitted = pd.DataFrame(index=betas.index, columns=betas.columns, dtype=float)
    rmse = pd.Series(index=betas.index, dtype=float)
    coefs = {}
    if hc["curve_fit"] == "spline":
        from scipy.interpolate import UnivariateSpline
    for t, row in betas.iterrows():
        r = row.dropna()
        if len(r) < 5:
            continue
        mm = r.index.values.astype(float); bb = r.values.astype(float)
        try:
            if hc["curve_fit"] == "poly3":
                c = np.polyfit(mm, bb, 3)
                fv = np.polyval(c, m)
                coefs[t] = c
            else:
                sp = UnivariateSpline(mm, bb, s=hc["spline_s"] * len(mm))
                fv = sp(m)
                coefs[t] = sp
            fitted.loc[t] = fv
            rmse.loc[t] = float(np.sqrt(np.mean((np.interp(mm, m, fv) - bb) ** 2)))
        except Exception:
            continue
    return fitted, rmse, coefs


def read_coupon_ratios(coefs, cc, coupons, cfg, index):
    """Step 3 read-off: ratio for actual coupon c at m = c - CC(t)."""
    out = pd.DataFrame(index=index, columns=coupons, dtype=float)
    poly = cfg["hedge"]["curve_fit"] == "poly3"
    mmax = cfg["hedge"]["moneyness_max_bp"] / 100.0
    for t in index:
        if t not in coefs or pd.isna(cc.loc[t]):
            continue
        for c in coupons:
            m = c - cc.loc[t]
            if abs(m) > mmax:
                continue
            out.loc[t, c] = (np.polyval(coefs[t], m) if poly else float(coefs[t](m)))
    return out


def run_hedge_pipeline(front_adj, cc, dTsy, coupons, cfg):
    """Full replication, both variants. Returns dict of panels."""
    # Variant A: literal
    S = build_node_series(front_adj, cc, cfg)
    dS_A = S.diff()
    betas_A, r2_A = rolling_node_betas(dS_A, dTsy, cfg)
    fit_A, rmse_A, coef_A = fit_curves(betas_A, cfg)
    ratio_A = read_coupon_ratios(coef_A, cc, coupons, cfg, front_adj.index)
    # Variant B: fixed-instrument
    dS_B = build_node_changes_B(front_adj, cc, cfg)
    betas_B, r2_B = rolling_node_betas(dS_B, dTsy, cfg)
    fit_B, rmse_B, coef_B = fit_curves(betas_B, cfg)
    ratio_B = read_coupon_ratios(coef_B, cc, coupons, cfg, front_adj.index)
    return {
        "node_betas_A": betas_A, "node_betas_B": betas_B,
        "fit_A": fit_A, "fit_B": fit_B,
        "rmse_A": rmse_A, "rmse_B": rmse_B,
        "ratio_A": ratio_A, "ratio_B": ratio_B,
        "r2_A": r2_A, "r2_B": r2_B,
        "dS_A": dS_A, "dS_B": dS_B,
    }


def ab_divergence_report(res, cc, dTsy_yield=None):
    """Quantify the composition effect: A vs B at the current-coupon node and
    its link to CC mobility. Theory: gap grows with |dCC| activity."""
    bA = res["fit_A"][0.0] if 0.0 in res["fit_A"].columns else res["fit_A"].iloc[:, len(res["fit_A"].columns)//2]
    bB = res["fit_B"][0.0] if 0.0 in res["fit_B"].columns else res["fit_B"].iloc[:, len(res["fit_B"].columns)//2]
    both = pd.concat([bA.rename("A"), bB.rename("B")], axis=1).dropna()
    gap = (both["B"] - both["A"])
    out = {
        "mean_beta_A_cc": float(both["A"].mean()),
        "mean_beta_B_cc": float(both["B"].mean()),
        "mean_gap_B_minus_A": float(gap.mean()),
        "gap_pct_of_B": float(gap.mean() / both["B"].mean()) if both["B"].mean() else np.nan,
        "corr_A_B": float(both["A"].corr(both["B"])),
    }
    cc_act = cc.diff().abs().rolling(60).mean().reindex(gap.index)
    valid = pd.concat([gap, cc_act.rename("ccact")], axis=1).dropna()
    if len(valid) > 100:
        out["corr_gap_vs_ccActivity"] = float(valid.iloc[:, 0].corr(valid["ccact"]))
    return out, both


def production_ratio_check(prod, ratio_A, ratio_B):
    """If the desk exports the PRODUCTION quant ratios, this settles the
    Variant A/B question empirically: correlations, mean abs differences,
    and a pooled horse race prod ~ A + B. Whichever variant production loads
    on tells you whether the composition-effect critique applies to it."""
    common_c = [c for c in prod.columns if c in ratio_A.columns]
    frames = []
    for c in common_c:
        f = pd.concat([prod[c].rename("prod"), ratio_A[c].rename("A"),
                       ratio_B[c].rename("B")], axis=1).dropna()
        frames.append(f)
    if not frames:
        return None
    pool = pd.concat(frames)
    out = {
        "n_obs": len(pool),
        "corr_prod_A": float(pool["prod"].corr(pool["A"])),
        "corr_prod_B": float(pool["prod"].corr(pool["B"])),
        "mad_prod_A": float((pool["prod"] - pool["A"]).abs().mean()),
        "mad_prod_B": float((pool["prod"] - pool["B"]).abs().mean()),
    }
    X = np.column_stack([np.ones(len(pool)), pool["A"].values, pool["B"].values])
    try:
        beta, *_ = np.linalg.lstsq(X, pool["prod"].values, rcond=None)
        out["horse_race_const"] = float(beta[0])
        out["horse_race_loading_A"] = float(beta[1])
        out["horse_race_loading_B"] = float(beta[2])
    except np.linalg.LinAlgError:
        pass
    return out
