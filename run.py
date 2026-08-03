"""End-to-end runner.

  python run.py --synthetic          # smoke test with planted breaks
  python run.py --data-dir ./data    # real data in the documented CSV schema

Outputs land in ./outputs: CSVs, PNGs, report.md, console summary.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import CONFIG
import synth
import hedge
import basis as basis_mod
import regimes
import drift as drift_mod
import forecast as fc_mod


# --------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------

def load_csvs(path):
    import basis as _bm
    tba = pd.read_csv(f"{path}/tba_prices.csv", parse_dates=["date"])
    tba["coupon"] = tba["coupon"].astype(float)
    tba = tba.drop_duplicates(["date", "coupon"], keep="last")
    front = tba.pivot(index="date", columns="coupon", values="price_front")
    back = tba.pivot(index="date", columns="coupon", values="price_back")
    data = {"tba_front": front, "tba_back": back, "dates": front.index,
            "tsy_yield": None, "tsy_roll_dates": None, "otr_wi_spread": None}
    trd = f"{path}/tsy_roll_dates.csv"
    if os.path.exists(trd):
        data["tsy_roll_dates"] = pd.DatetimeIndex(
            pd.read_csv(trd, parse_dates=["tsy_roll_date"])["tsy_roll_date"])
    if os.path.exists(f"{path}/treasury.csv"):
        tsy = pd.read_csv(f"{path}/treasury.csv",
                          parse_dates=["date"]).set_index("date")
        data["tsy_price"] = tsy["tsy_price"]
        if "tsy_yield" in tsy:
            data["tsy_yield"] = tsy["tsy_yield"]
    elif os.path.exists(f"{path}/treasury_raw.csv"):
        if data["tsy_roll_dates"] is None:
            raise ValueError("treasury_raw.csv needs tsy_roll_dates.csv "
                             "(last day each outgoing note is the hedge)")
        raw = pd.read_csv(f"{path}/treasury_raw.csv",
                          parse_dates=["date"]).set_index("date")
        dts, lvl, spr, gaps = _bm.treasury_splice(
            raw["ct10_price"], raw["wi_price"], data["tsy_roll_dates"])
        if gaps:
            print(f"[warn] {gaps} OTR switch days lack a WI price -- those "
                  f"changes are lost; fill wi_price around switches")
        data["tsy_price"] = lvl
        data["otr_wi_spread"] = spr
        if "ct10_yield" in raw:
            data["tsy_yield"] = raw["ct10_yield"]
    else:
        raise ValueError("provide treasury.csv (continuous) or "
                         "treasury_raw.csv + tsy_roll_dates.csv")
    def opt(fname, col=None, pivot=False):
        f = f"{path}/{fname}"
        if not os.path.exists(f):
            return None
        d = pd.read_csv(f, parse_dates=["date"]) if not fname.startswith("roll") \
            else pd.read_csv(f, parse_dates=["roll_date"])
        if fname.startswith("roll"):
            return pd.DatetimeIndex(d["roll_date"])
        d = d.set_index("date")
        if pivot:
            d = d.reset_index()
            d["coupon"] = d["coupon"].astype(float)
            d = d.drop_duplicates(["date", "coupon"], keep="last")
            return d.pivot(index="date", columns="coupon", values="ratio")
        return d[col]
    data["cc"] = opt("current_coupon.csv", "cc")
    data["quant_ratio_prod"] = opt("hedge_ratio_quant.csv", pivot=True)
    data["vol"] = opt("vol.csv", "vol")
    data["roll_dates"] = opt("roll_dates.csv")
    data["desk_ratio"] = opt("hedge_ratio_desk.csv", pivot=True)
    data["repo"] = opt("repo.csv", "repo")
    ev = f"{path}/events.csv"
    data["events"] = pd.read_csv(ev, parse_dates=["date"]) \
        if os.path.exists(ev) else None
    if data["roll_dates"] is None:
        raise ValueError("roll_dates.csv is required for roll adjustment")
    # align auxiliaries to the price calendar; short forward-fills only
    # (uses past data -- PIT-safe) to bridge holiday-calendar mismatches
    for k in ["cc", "vol", "repo"]:
        if data.get(k) is not None:
            data[k] = data[k].reindex(front.index).ffill(limit=3)
    return data


# --------------------------------------------------------------------------
# main pipeline
# --------------------------------------------------------------------------

def main(mode, data_dir=None):
    cfg = CONFIG
    out = cfg["outdir"]
    os.makedirs(out, exist_ok=True)
    truth = None
    if mode == "synthetic":
        data, truth = synth.generate(cfg)
        synth.write_csvs(data, f"{out}/synthetic_data")
        print("[data] synthetic sample generated "
              f"({len(data['dates'])} days, planted breaks: "
              f"beta {truth['break_beta_date'].date()}, "
              f"basis {truth['break_basis_date'].date()}, "
              f"vol-link {truth['vol_link_date'].date()})")
    else:
        data = load_csvs(data_dir)
        print(f"[data] loaded {data_dir}: {len(data['dates'])} days, "
              f"{data['tba_front'].shape[1]} coupons")

    cal = cfg["calibration"]
    front, back = data["tba_front"], data["tba_back"]
    coupons = list(front.columns)
    report = ["# TBA basis regime pipeline -- run report\n"]
    report.append(f"Mode: **{mode}** | days: {len(front)} | coupons: {coupons}\n")

    # ---- current coupon ----
    if data.get("cc") is not None:
        cc = data["cc"]
        cc_flag = pd.Series(False, index=cc.index)
        report.append("Current coupon: supplied series.\n")
    else:
        cc, cc_flag = hedge.compute_current_coupon(front, cfg["hedge"]["coupon_step"])
        report.append(f"Current coupon: computed par coupon; extrapolated on "
                      f"{cc_flag.mean():.1%} of days (whole stack off par -- "
                      f"convention risk, see README).\n")

    # ---- roll adjustment ----
    dP, level_adj, drop = basis_mod.roll_adjusted_changes(front, back, data["roll_dates"])
    dTsy = data["tsy_price"].diff()
    dYield = data["tsy_yield"].diff() if data.get("tsy_yield") is not None else None

    # ---- hedge-ratio replication (A literal / B fixed-instrument) ----
    print("[hedge] running node interpolation + 60d rolling regressions (A & B)...")
    hres = hedge.run_hedge_pipeline(level_adj, cc, dTsy, coupons, cfg)
    ab, ab_series = hedge.ab_divergence_report(hres, cc)
    report.append("## Variant A vs B (composition effect)\n")
    report.append("Literal step-2 (A) differences a moneyness-anchored series; "
                  "as CC moves, sliding along the stack offsets part of the rate "
                  "move, so A measures a constant-moneyness *strategy* beta and "
                  "understates fixed-coupon duration. B interpolates fixed-"
                  "instrument changes instead.\n")
    for k, v in ab.items():
        report.append(f"- {k}: {v:.4f}")
    report.append("")
    for k in ["node_betas_A", "node_betas_B", "ratio_A", "ratio_B"]:
        hres[k].to_csv(f"{out}/{k}.csv")
    prod = data.get("quant_ratio_prod")
    if mode == "synthetic" and prod is None:
        # exercise the diagnostic path: pseudo-production = B + noise, so the
        # horse race should point decisively at Variant B
        rng_p = np.random.default_rng(7)
        prod = hres["ratio_B"] + rng_p.normal(0, 0.02, hres["ratio_B"].shape)
    if prod is not None:
        pchk = hedge.production_ratio_check(prod, hres["ratio_A"],
                                            hres["ratio_B"])
        if pchk:
            report.append("### Production quant ratios vs replication "
                          "(A/B horse race)")
            for k, v in pchk.items():
                report.append(f"- {k}: {v:.4f}" if isinstance(v, float)
                              else f"- {k}: {v}")
            report.append("")

    # ---- treasury splice validation (synthetic) / diagnostics (real) ----
    splice_err = None
    if data.get("ct10") is not None and data.get("tsy_roll_dates") is not None:
        dts_s, _, otr_spr, sp_gaps = basis_mod.treasury_splice(
            data["ct10"], data["wi"], data["tsy_roll_dates"])
        if mode == "synthetic":
            splice_err = float((dts_s - data["tsy_price"].diff())
                               .abs().max())
            report.append(f"## Treasury on-the-run splice\n"
                          f"Held-instrument changes rebuilt from raw CT10 + "
                          f"WI across {len(data['tsy_roll_dates'])} switches: "
                          f"max abs error vs true continuous changes = "
                          f"{splice_err:.2e} ({sp_gaps} gap days). The CT10-WI "
                          f"spread is saved as a liquidity/flight-to-quality "
                          f"covariate.\n")
        otr_spr.to_csv(f"{out}/otr_wi_spread.csv")
    if data.get("otr_wi_spread") is not None:
        data["otr_wi_spread"].to_csv(f"{out}/otr_wi_spread.csv")

    # ---- back-adjusted levels (current-front benchmark convention) ----
    off_end = (level_adj - front).ffill().iloc[-1]
    back_adj = level_adj.sub(off_end, axis=1)
    back_adj.to_csv(f"{out}/tba_backadjusted_levels.csv")

    # ---- basis construction (point-in-time, quant B ratios) ----
    ratio_q = hres["ratio_B"]
    b_panel = basis_mod.hedged_basis(dP, dTsy, ratio_q, lag=cfg["basis"]["ratio_lag"])
    b_panel_A = basis_mod.hedged_basis(dP, dTsy, hres["ratio_A"], lag=1)
    b_cc = basis_mod.cc_node_basis(hres["dS_B"], dTsy, hres["fit_B"], lag=1)
    b_cc.name = "b_cc"
    b_panel.to_csv(f"{out}/basis_quantB_by_coupon.csv")
    # coupon-swap series by identity: duration-neutral swap (long c1 / short
    # c2 / tsy hedge = hr1-hr2) has P&L = b_c1 - b_c2. The ratio-weighted
    # swap is b_c1 - (hr1/hr2)*b_c2, also derivable from this panel.
    if cfg["basis"].get("emit_coupon_swaps", True):
        liq = [c for c in b_panel.columns if b_panel[c].notna().mean() > 0.3]
        if len(liq) >= 2:
            swaps = {f"{c1}_{c2}": b_panel[c1] - b_panel[c2]
                     for c1, c2 in zip(liq[:-1], liq[1:])}
            pd.DataFrame(swaps).to_csv(f"{out}/basis_coupon_swaps.csv")
    b_cc.to_csv(f"{out}/basis_cc_node.csv")
    drop.to_csv(f"{out}/drop_by_coupon.csv")
    if data.get("repo") is not None:
        implied, spec = basis_mod.carry_columns(
            drop, data["repo"], cc, days_to_settle=cfg["drift"]["settle_days"])
        spec.to_csv(f"{out}/specialness_proxy.csv")

    # ---- desk diagnostics ----
    desk = data.get("desk_ratio")
    if desk is not None and dYield is not None:
        print("[desk] reactivity + slippage diagnostics...")
        xcorr, frac_static = basis_mod.desk_reactivity(
            desk, dYield, max_lag=cfg["regimes"]["desk_xcorr_max_lag"])
        trig, base_rate = basis_mod.update_trigger_profile(
            desk, data["tsy_yield"],
            thresholds=cfg["regimes"]["desk_trigger_thresholds"])
        slip, spread = basis_mod.hedge_slippage(dP, dTsy, ratio_q, desk)
        report.append("## Desk vs quant\n")
        report.append(f"- Desk ratio unchanged on {frac_static:.1%} of days "
                      f"(base update rate {base_rate:.1%}).")
        report.append("- P(update | cumulative |move| since last update > x):")
        for x, p in trig.items():
            report.append(f"    - >{x*100:.0f}bp: {p:.1%}")
        peak = xcorr.idxmax()
        # Read the conclusion off the peak instead of asserting it: lag 0 is
        # same-day response, not evidence of a lag, and a flat profile is not
        # evidence of anything.
        rng_x = float(xcorr.max() - xcorr.min())
        if rng_x < 0.05:
            verdict = ("no clear timing signature -- the profile is flat "
                       "across lags, so this does not distinguish reactive "
                       "from forward-looking")
        elif peak == 0:
            verdict = ("same-day response: the desk moves with the market, "
                       "not after it")
        else:
            verdict = (f"reactive -- updates follow rate moves by ~{peak} "
                       f"day(s), not forward-looking")
        report.append(f"- Update-vs-|rate-move| xcorr peaks at lag {peak} "
                      f"days (corr {xcorr.max():.2f}, range across lags "
                      f"{rng_x:.2f}) -> {verdict}.")
        report.append(f"- Mean |desk - quantB| ratio spread: "
                      f"{spread.abs().mean().mean():.3f}; slippage PnL sd "
                      f"{slip.std().mean():.4f}/day. Break tests on the spread "
                      f"below treat discretion itself as a regime.\n")
        band, bstats = basis_mod.hedge_timing_band(dP, dTsy, desk)
        band.cumsum().to_csv(f"{out}/hedge_timing_band_cum.csv")
        report.append("- Timing-convention band (EOD-stamped ratios: "
                      "'starts next day' vs 'backfilled to open'); one-signed "
                      "with |t|>2 means desk update timing is itself a P&L "
                      "factor:")
        report.append("```\n" + bstats.round(4).to_string() + "\n```")
        spread_breaks = regimes.pelt_breaks(spread.mean(axis=1), spread.index,
                                            model="l2",
                                            min_size=cfg["regimes"]["pelt_min_size"],
                                            pen_scale=cfg["regimes"]["pelt_pen_scale"],
                                            jump=cfg["regimes"]["pelt_jump"])
        report.append(f"- PELT breaks in desk-quant spread: "
                      f"{[d.date() for d in spread_breaks]}\n")
    else:
        xcorr = None
        report.append("## Desk vs quant\nDesk ratios not supplied -- skipped.\n")

    # ---- pick reference coupon: nearest to CC on average with coverage ----
    # MEAN OF |m|, not |mean of m|: the latter lets a coupon that sits +100bp
    # for half the sample and -100bp for the other half score 0 -- ranked
    # "nearest" ahead of one that stays a steady 30bp away, when in fact it is
    # never near CC at all.
    cov = dP.notna().mean()
    avg_m = {c: float((c - cc).abs().mean()) for c in coupons}
    elig = [c for c in coupons if cov[c] > 0.7]
    if not elig:
        elig = [max(coupons, key=lambda c: cov[c])]
        print(f"[warn] no coupon has >70% coverage; falling back to {elig[0]} "
              f"({cov[elig[0]]:.0%})")
    ref = min(elig, key=lambda c: avg_m[c])
    y_ref = dP[ref]
    print(f"[regimes] reference coupon {ref} (best coverage nearest CC)")
    report.append(f"## Regime battery (reference coupon {ref}, CC node)\n")

    # ---- stationarity ----
    st = [regimes.adf_kpss(b_cc, "b_cc"),
          regimes.adf_kpss(b_panel[ref], f"b_{ref}"),
          regimes.adf_kpss(hres["fit_B"][0.0], "beta_B_cc")]
    report.append("### Stationarity (ADF p / KPSS p)")
    for r in st:
        report.append(f"- {r['series']}: ADF p={r.get('adf_p', np.nan):.3f}, "
                      f"KPSS p={r.get('kpss_p', np.nan):.3f}")
    report.append("")

    # ---- break battery on the hedge regression itself ----
    print("[regimes] sup-F / Nyblom / CUSUM on hedge regression (bootstrap)...")
    dfh = pd.concat([y_ref.rename("y"), dTsy.rename("x")], axis=1).dropna()
    sf = regimes.sup_f_bootstrap(dfh["y"].values, dfh["x"].values,
                                 trim=cfg["regimes"]["supF_trim"],
                                 B=cfg["regimes"]["boot_B"],
                                 block=cfg["regimes"]["boot_block"],
                                 min_obs=cfg["regimes"]["min_obs_break_test"])
    sf_date = dfh.index[sf["break_idx"]] if sf.get("break_idx") is not None else None
    ny = regimes.nyblom_bootstrap(dfh["y"].values, dfh["x"].values,
                                  B=cfg["regimes"]["boot_B"],
                                  block=cfg["regimes"]["boot_block"],
                                  min_obs=cfg["regimes"]["min_obs_break_test"])
    cu = regimes.cusum_recursive(dfh["y"], dfh["x"],
                                 min_obs=cfg["regimes"]["min_obs_kalman"])
    report.append("### Hedge-regression stability (dP_ref ~ dTsy)")
    report.append(f"- sup-F = {sf['supF']:.1f}, bootstrap p = {sf['p']:.3f}, "
                  f"argmax break ~ {sf_date.date() if sf_date is not None else 'n/a'}")
    report.append(f"- Nyblom L = {ny['L']:.2f}, bootstrap p = {ny['p']:.3f}")
    fe = cu.get("cusum_first_exceed")
    report.append(f"- Recursive CUSUM outside 5% bands {cu.get('cusum_exceed_frac', float('nan')):.1%} "
                  f"of path; first exceed {fe.date() if fe is not None else 'never'}\n")

    # ---- PELT on the beta path (respecting overlap) ----
    beta_path = hres["fit_B"][0.0]
    pelt_jump = cfg["regimes"]["pelt_jump"]
    if cal.get("pelt_mode") == "empirical":
        print(f"[regimes] calibrating PELT penalty by null bootstrap "
              f"(B={cal['pelt_boot_B']}, this is the slow step)...")
    pen_beta, fa_beta, pen_mode, pen_diag = regimes.calibrate_pelt_penalty(
        beta_path, cal, cfg["regimes"]["pelt_min_size"], model="l2",
        jump=pelt_jump)
    pen_beta = pen_beta if pen_beta is not None else cfg["regimes"]["pelt_pen_scale"]
    pelt_beta = regimes.pelt_breaks(beta_path, beta_path.index, model="l2",
                                    min_size=cfg["regimes"]["pelt_min_size"],
                                    pen_scale=pen_beta, jump=pelt_jump)
    pelt_basis = regimes.pelt_breaks(b_cc, b_cc.index, model="rbf",
                                     min_size=cfg["regimes"]["pelt_min_size"],
                                     pen_scale=cfg["regimes"]["pelt_pen_scale"],
                                     jump=pelt_jump)
    # variance specialist: rbf saturates on the single most dramatic
    # distributional event; l2 on squared demeaned returns targets vol shifts
    b_dm = b_cc - b_cc.mean()
    pelt_var = regimes.pelt_breaks(b_dm ** 2, b_cc.index, model="l2",
                                   min_size=cfg["regimes"]["pelt_min_size"],
                                   pen_scale=2.0, jump=pelt_jump)
    report.append(f"### PELT multiple-break dates")
    if pen_mode == "empirical":
        report.append(
            f"- **beta path only**: penalty calibrated by null bootstrap to "
            f"pen_scale={pen_beta} (achieved false-alarm rate {fa_beta:.1%} "
            f"vs {cal['pelt_target_fa']:.0%} target, +/-{pen_diag['se']:.1%} "
            f"MC error on {pen_diag['n_ok']} replicates). Null = "
            f"'{pen_diag['null']}' with residual blocks of "
            f"{pen_diag['block']}d, which estimates the beta path's "
            f"persistence rather than assuming it: resampling LEVELS would "
            f"splice unrelated levels (every seam a real mean shift), while "
            f"differencing would impose an exact unit root. The other PELT "
            f"lines below still use the uncalibrated "
            f"pen_scale={cfg['regimes']['pelt_pen_scale']}.")
        if pen_diag.get("at_grid_top"):
            report.append(
                f"- **CAUTION**: no penalty on the grid reached the "
                f"{cal['pelt_target_fa']:.0%} target, so the top of the grid "
                f"was used and the true false-alarm rate is ABOVE target. "
                f"This series may be too persistent for segmentation to be "
                f"inferential at all -- treat its dates as descriptive and "
                f"lean on the Kalman layer. Widen pelt_pen_grid upward.")
    report.append(f"- beta_B(cc) mean shifts: {[d.date() for d in pelt_beta]}")
    prox = cfg["regimes"]["tsy_roll_proximity_days"]
    if data.get("tsy_roll_dates") is not None and len(pelt_beta):
        near_tsy = [d.date() for d in pelt_beta
                    if min(abs((d - rr).days)
                           for rr in data["tsy_roll_dates"]) <= prox]
        if near_tsy:
            report.append(f"- CAUTION: {near_tsy} fall within {prox} days of an "
                          f"on-the-run switch -> candidate hedge-leg DV01 "
                          f"steps (new note's different coupon/maturity), "
                          f"not TBA duration regimes.")
    report.append(f"- b_cc distributional shifts (rbf): {[d.date() for d in pelt_basis]}")
    report.append(f"- b_cc variance shifts (l2 on squares): {[d.date() for d in pelt_var]}")
    # hedge-lag artifact check: a basis break within ~45d after a beta break
    # is likely leaked rate beta while the 60d rolling hedge catches up,
    # NOT an independent basis regime
    lag_win = cfg["regimes"]["hedge_lag_artifact_days"]
    artifacts = [d for d in (pelt_basis + pelt_var)
                 for db in pelt_beta if 0 <= (d - db).days <= lag_win]
    if artifacts:
        report.append(f"- NOTE: {sorted(set(d.date() for d in artifacts))} occur "
                      f"just after detected beta breaks -> likely hedge-"
                      f"adaptation leak (rolling 60d ratio lags the new "
                      f"duration regime), not independent basis regimes.\n")
    else:
        report.append("")

    # ---- Kalman TVP + innovation CUSUM (the online detector) ----
    print("[regimes] Kalman TVP beta (MLE)...")
    kal = regimes.kalman_tvp(dfh["y"], dfh["x"],
                             q_init=cfg["regimes"]["kalman_q_init"],
                             init_window=cfg["regimes"]["kalman_init_window"],
                             min_obs=cfg["regimes"]["min_obs_kalman"])
    thr_series, thr_mode, thr_diag = regimes.empirical_shewhart_threshold(
        kal["z"], cal, burn=cfg["regimes"]["kalman_burn"])
    ic = regimes.innovation_cusum(kal["z"], a=cfg["regimes"]["cusum_alpha_a"],
                                  burn=cfg["regimes"]["kalman_burn"],
                                  threshold=thr_series)
    # What the Gaussian-derived cutoff would ACTUALLY have fired at, in-sample.
    # Unlike the realized rate under the calibrated threshold, this one is free
    # to differ from the design -- so it does carry tail information.
    _zb = kal["z"].iloc[cfg["regimes"]["kalman_burn"]:].abs()
    fixed_rate_realized = float((_zb > thr_diag["gauss_cutoff"]).mean() * 252)
    report.append("### Kalman time-varying beta")
    report.append(f"- MLE: r={kal['r']:.2e}, q_alpha={kal['q_a']:.2e}, "
                  f"q_beta={kal['q_b']:.2e} (beta half-life of shocks "
                  f"~{'fast' if kal['q_b'] > 1e-5 else 'slow'} adaptation)")
    fe1 = ic["first_exceed"]
    report.append(f"- Innovation CUSUM first exceed: "
                  f"{fe1.date() if fe1 is not None else 'never'}; CUSUM-sq "
                  f"argmax {ic['sq_break_date'].date()} (IT stat "
                  f"{ic['sq_it_stat']:.2f}, 5% iid crit ~1.36)")
    # Fat-tail diagnostic = the CUTOFF ratio at a matched tail probability,
    # NOT the realized flag rate. Both modes pin the realized rate near the
    # design by construction, so that rate is a readback of the config and
    # says nothing about tail shape; the cutoff the data demands to hit that
    # rate is the thing that varies with the tails.
    report.append(
        f"- Shewhart threshold: {thr_mode}/{thr_diag['estimator']}, designed "
        f"alarm rate {thr_diag['alarm_per_year']:.2f}/yr "
        f"(p={thr_diag['tail_p']:.2e}/day). Latest |z| cutoff "
        f"{ic['shewhart_threshold_used']:.2f}; realized "
        f"{ic['shewhart_realized_rate']*252:.2f}/yr (a check that the "
        f"calibration held, not a tail test).")
    if "tail_ratio" in thr_diag:
        tr = thr_diag["tail_ratio"]
        verdict = ("materially fat-tailed" if tr >= 1.25 else
                   "mildly fat-tailed" if tr >= 1.10 else
                   "close to Gaussian in the typical window" if tr >= 0.90 else
                   "thinner-tailed than Gaussian in the typical window")
        realized = ic["shewhart_realized_rate"] * 252
        report.append(
            f"- **Tail diagnostic**: to hit the same {thr_diag['tail_p']:.2e} "
            f"tail the data demands |z| > {thr_diag['median_cutoff']:.2f} "
            f"(median over refits) where a Gaussian needs "
            f"{thr_diag['gauss_cutoff']:.2f} -- ratio {tr:.2f}x, i.e. "
            f"innovations are {verdict}. Cross-check: the Gaussian cutoff "
            f"{thr_diag['gauss_cutoff']:.2f} actually fires at "
            f"{fixed_rate_realized:.2f}/yr on this sample against its nominal "
            f"{thr_diag['alarm_per_year']:.2f}/yr.")
        # A near-1 tail ratio together with heavy over-firing at the Gaussian
        # cutoff cannot both be explained by tail SHAPE -- they reconcile only
        # if the innovation scale is non-stationary. Say so, because the two
        # numbers otherwise look contradictory and the remedy is different.
        over = fixed_rate_realized / max(thr_diag["alarm_per_year"], 1e-9)
        if tr < 1.10 and over >= 2.0:
            report.append(
                f"- **Non-stationarity, not fat tails**: the typical-window "
                f"tail is ~Gaussian ({tr:.2f}x) yet the Gaussian cutoff "
                f"over-fires {over:.1f}x, and the calibrated threshold still "
                f"realizes {realized:.2f}/yr against a {thr_diag['alarm_per_year']:.2f}"
                f"/yr design. Those reconcile only if the innovation SCALE "
                f"shifts between regimes: |z| is well-behaved within a regime "
                f"and clusters across regime changes. Note the circularity "
                f"this creates -- a trailing threshold adapts UP during the "
                f"very episode you want flagged, so treat clustered flags "
                f"(two in five days) as the signal rather than the count, and "
                f"lean on CUSUM-sq for the level shift itself.")
        elif tr >= 1.10:
            report.append(
                f"- Calibrated threshold realizes {realized:.2f}/yr against "
                f"the {thr_diag['alarm_per_year']:.2f}/yr design.")
    report.append(f"- flag dates (first 8): "
                  f"{[d.date() for d in ic['shewhart_dates'][:8]]}\n")
    kal["beta"].to_csv(f"{out}/kalman_beta.csv")

    # ---- vol-beta regime ----
    vb = None
    if data.get("vol") is not None:
        print("[regimes] vol-beta regime test...")
        dvol = data["vol"].diff()
        vb = regimes.vol_beta_regime(b_cc, dvol, cfg)
        bd = vb["supF"].get("break_date")
        report.append("### Basis-on-dVol loading stability")
        report.append(f"- sup-F = {vb['supF']['supF']:.1f}, p = {vb['supF']['p']:.3f}, "
                      f"break ~ {bd.date() if bd is not None else 'n/a'}")
        report.append(f"- PELT on rolling vol-beta: "
                      f"{[d.date() for d in vb['pelt_break_dates']]}\n")

    # ---- multivariate HMM (PIT-parameter version for signals) ----
    print("[regimes] multivariate HMM...")
    feats = pd.concat([
        b_cc.rename("b_cc"),
        kal["beta"].diff().rename("d_kbeta"),
        (data["vol"].diff().rename("dvol") if data.get("vol") is not None
         else pd.Series(dtype=float)),
    ], axis=1).dropna(axis=1, how="all")
    hmm_pit = regimes.fit_hmm_train_filter_full(
        feats, train_frac=cfg["regimes"]["hmm_train_frac"],
        n_states_list=cfg["regimes"]["hmm_states"],
        seeds=cfg["regimes"]["hmm_seeds"],
        winsor_z=cfg["regimes"]["hmm_winsor_z"])
    if hmm_pit is not None:
        report.append(f"### HMM (fit on first 60%, filtered forward; k={hmm_pit['k']}, "
                      f"BIC={hmm_pit['bic']:.0f})")
        report.append("State table (means in original units, weights smoothed in-train):")
        report.append("```\n" + hmm_pit["state_table"].round(4).to_string() + "\n```")
        # pick the state with the most extreme |b_cc| mean as the 'stress' regime
        tbl = hmm_pit["state_table"]
        eligible = tbl[tbl["uncond_freq"] >= cfg["regimes"]["hmm_state_min_freq"]]
        stress = int(eligible["b_cc"].abs().idxmax()) \
            if "b_cc" in tbl.columns and len(eligible) else 0
        filt = hmm_pit["filtered_full_pitparams"][f"state{stress}"]
        regime_dummy = regimes.hysteresis_signal(
            filt, enter=cfg["regimes"]["hmm_min_prob"],
            exit=1 - cfg["regimes"]["hmm_min_prob"])
        n_epi = int((regime_dummy.diff() == 1).sum())
        report.append(f"- Stress state = state{stress}; hysteresis episodes: {n_epi}")
        report.append(f"- **Effective sample honesty**: regime inference rests on "
                      f"~{n_epi + len(pelt_beta) + len(pelt_basis)} episodes, not "
                      f"{len(front)} daily rows. Treat marginal p-values accordingly.\n")
    else:
        regime_dummy = None
        report.append("### HMM failed to fit -- skipped.\n")

    # ---- drift regimes: carry vs widening (the P&L slope question) ----
    print("[drift] MS mean regimes + trend segmentation + attribution...")
    dc = cfg["drift"]
    carry = drift_mod.carry_proxy(drop[ref], b_cc.index, dc["cycle_days"])
    b_ex = (b_cc - carry).rename("b_ex")
    sw = dc["studentize_window"]
    sig_tr = b_ex.rolling(
        sw, min_periods=max(2, int(sw * cfg["regimes"]["min_obs_frac"]))).std()
    ms = drift_mod.ms_mean_regimes(b_ex, k=dc["ms_k"], scale=dc["ms_scale"],
                                   sigma=sig_tr)
    seg = drift_mod.drift_segments(b_ex, min_size=dc["pelt_min_size"],
                                   pen_scale=dc["pelt_pen_scale"],
                                   widen_t=dc["widen_tstat"], sigma=sig_tr,
                                   jump=dc["pelt_jump"])
    report.append("## Drift regimes: carry-earning vs spread-widening\n")
    report.append("Detection is on the CARRY-ADJUSTED, VOL-STUDENTIZED basis "
                  "return: carry removed so the null slope is ~0, trailing-"
                  "sigma studentization so variance eras cannot hijack the "
                  "state allocation. Filtered probabilities only. Latency "
                  "honesty: drift flips are low-SNR; expect ~2-4 weeks of "
                  "confirmation lag from returns alone -- vol/covariates are "
                  "the fast trigger, this layer is the confirmation.\n")
    wp = None
    if ms is not None:
        wp = ms["filtered"].iloc[:, ms["widening_state"]]
        report.append(f"### Markov-switching mean (k={ms['k']}, studentized)")
        report.append("```\n" + ms["table"].round(4).to_string() + "\n```")
        report.append(f"- Widening state = state {ms['widening_state']}; "
                      f"time in widening: {float((wp > 0.5).mean()):.1%}\n")
    if len(seg):
        report.append("### Trend segments (detection in z/day; P&L in pts/day)")
        report.append("`slope_z_per_day`/`tstat`/`label` describe the "
                      "vol-studentized series the segmentation actually ran "
                      "on; `slope_pts_per_day`/`cum_pts` re-express the same "
                      "dates in price points, which is the P&L-relevant "
                      "magnitude.")
        segp = seg.copy()
        segp["start"] = segp["start"].dt.date; segp["end"] = segp["end"].dt.date
        report.append("```\n" + segp.round(3).to_string(index=False) + "\n```\n")
        seg.to_csv(f"{out}/drift_segments.csv", index=False)
    # attribution of widening episodes
    dvol_s = data["vol"].diff() if data.get("vol") is not None else None
    beta_slope = None
    if 0.1 in hres["fit_B"].columns and -0.1 in hres["fit_B"].columns:
        beta_slope = (hres["fit_B"][0.1] - hres["fit_B"][-0.1]) / 0.2
    comp = drift_mod.build_components(
        b_cc, carry, kal["beta"], hres["fit_B"][0.0], dTsy, dvol=dvol_s,
        dYield=dYield, beta_slope_at_cc=beta_slope,
        cc_sens=dc["cc_sens"], lam_window=dc["lam_window"])
    comp.to_csv(f"{out}/drift_components.csv")
    attr = drift_mod.attribute_episodes(comp, seg, dYield=dYield) \
        if len(seg) else pd.DataFrame()
    if len(attr):
        report.append("### Widening-episode attribution (points, PIT proxies)")
        report.append("Components overlap (leakage and convexity both touch "
                      "duration error); read as candidate explanations, with "
                      "'residual_demand' = the pure spread/demand shock.")
        report.append("```\n" + attr.round(3).to_string(index=False) + "\n```\n")
        attr.to_csv(f"{out}/widening_attribution.csv", index=False)
    if data.get("events") is not None and wp is not None:
        ov = drift_mod.event_overlay(data["events"], b_cc, wp,
                                     pre=dc["event_pre_days"],
                                     post=dc["event_post_days"])
        report.append("### Event overlay")
        report.append("```\n" + ov.round(3).to_string(index=False) + "\n```\n")

    # ---- forecast harness ----
    print("[forecast] walk-forward + CW/GW...")
    h = cfg["forecast"]["horizon"]
    features = pd.DataFrame(index=b_cc.index)
    features["lag_b"] = b_cc.rolling(h).sum()
    if data.get("vol") is not None:
        features["dvol_5d"] = data["vol"].diff(5)
    if regime_dummy is not None:
        features["regime"] = regime_dummy.reindex(b_cc.index).ffill().fillna(0)
        features["regime_x_lag"] = features["regime"] * features["lag_b"]
    target = fc_mod.build_target(b_cc, h)
    preds = fc_mod.walk_forward(features, target, cfg)
    stat_eval = fc_mod.evaluate(preds, regime_dummy, cfg)
    econ = fc_mod.econ_eval(preds, cfg)
    report.append("## Forecast evaluation (target: next-%dd cumulative b_cc)" % h)
    report.append("Statistical (vs expanding mean; CW one-sided p; GW conditional "
                  "on lagged regime):")
    report.append("```\n" + stat_eval.round(4).to_string() + "\n```")
    report.append("Economic (sign positions, %.4f pts cost/unit turnover):"
                  % cfg["forecast"]["cost_per_unit"])
    report.append("```\n" + econ.round(3).to_string() + "\n```\n")
    preds.to_csv(f"{out}/forecast_predictions.csv")

    # ---- planted-break recovery (synthetic only) ----
    if truth is not None:
        report.append("## Planted-break recovery (synthetic truth)")
        def near(dates, target, tol=45):
            return any(abs((d - target).days) <= tol for d in dates)
        chk = {
            # 60d-window betas turn a step into a ramp: allow ramp-width slack
            "beta break (PELT on beta_B)": near(pelt_beta, truth["break_beta_date"], 60),
            # sup-F is a TEST with a coarse locator under heteroskedastic x:
            # score = rejects constancy AND argmax in the right neighborhood
            "beta break (sup-F rejects, argmax coarse)": (
                sf["p"] < 0.05 and sf_date is not None and
                abs((sf_date - truth["break_beta_date"]).days) <= 180),
            # Kalman is the sharp online detector: Shewhart within +-30d, or
            # variance locator within 60d
            "beta break (Kalman Shewhart/CUSUM-sq)": (
                near(ic["shewhart_dates"], truth["break_beta_date"], 30) or
                abs((ic["sq_break_date"] - truth["break_beta_date"]).days) <= 60),
            **({"treasury splice exact": splice_err is not None and
                splice_err < 1e-9} if splice_err is not None else {}),
            "basis break (PELT rbf or var-l2 on b_cc)": (
                near(pelt_basis, truth["break_basis_date"], 75) or
                near(pelt_var, truth["break_basis_date"], 75)),
            **({f"drift window {i+1} (MS prob or segment)": (
                (wp is not None and (
                    float(wp.loc[a:b].mean()) >= 0.45 or
                    (wp.mean() > 0 and
                     float(wp.loc[a:b].mean()) >= 3 * float(wp.mean()))))
                or (len(seg) and any(
                    s["label"] == "widening" and
                    (min(s["end"], b) - max(s["start"], a)).days >= 25
                    for _, s in seg.iterrows())))
                for i, (a, b) in enumerate(truth.get("drift_windows", []))}),
            "vol-link break (vol-beta tests)": (vb is not None and (
                near(vb["pelt_break_dates"], truth["vol_link_date"], 90) or
                (vb["supF"].get("break_date") is not None and
                 abs((vb["supF"]["break_date"] - truth["vol_link_date"]).days) <= 90))),
        }
        for k, v in chk.items():
            report.append(f"- {k}: {'RECOVERED' if v else 'missed'}")
        report.append("")
        print("[check] planted-break recovery:",
              {k: ("OK" if v else "MISS") for k, v in chk.items()})

    # ---- plots ----
    print("[plots] writing figures...")
    _plots(out, hres, kal, ic, b_panel, b_cc, ref, truth, pelt_beta, pelt_basis,
           hmm_pit, regime_dummy, vb, xcorr, ab_series, desk, ratio_q,
           drift_extras=(b_ex, wp, seg,
                         truth.get("drift_windows", []) if truth else []))

    with open(f"{out}/report.md", "w") as f:
        f.write("\n".join(report))
    print(f"[done] report at {out}/report.md")
    return report


def _plots(out, hres, kal, ic, b_panel, b_cc, ref, truth, pelt_beta, pelt_basis,
           hmm_pit, regime_dummy, vb, xcorr, ab_series, desk, ratio_q,
           drift_extras=None):
    def vlines(ax, dates, color, label):
        for i, d in enumerate(dates):
            ax.axvline(d, color=color, ls="--", lw=1.2,
                       label=label if i == 0 else None)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ab_series["A"].plot(ax=ax, lw=1, label="beta A (literal / constant-moneyness)")
    ab_series["B"].plot(ax=ax, lw=1, label="beta B (fixed-instrument)")
    kal["beta"].plot(ax=ax, lw=1, label="Kalman TVP beta", alpha=0.8)
    if truth is not None:
        truth["true_beta_cc"].plot(ax=ax, lw=1, ls=":", color="k", label="true beta")
        vlines(ax, [truth["break_beta_date"]], "red", "planted beta break")
    vlines(ax, pelt_beta, "purple", "PELT detected")
    ax.legend(fontsize=8); ax.set_title("Hedge ratio at CC: A vs B vs Kalman")
    fig.tight_layout(); fig.savefig(f"{out}/fig1_beta_AB_kalman.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 3.6))
    ic["cusum"].plot(ax=ax, lw=1)
    xb = np.asarray(ic["cusum"].index)
    ax.plot(xb, ic["band"], "r--", lw=0.8)
    ax.plot(xb, -ic["band"], "r--", lw=0.8)
    if truth is not None:
        vlines(ax, [truth["break_beta_date"]], "red", "planted beta break")
    ax.set_title("CUSUM of standardized Kalman innovations (5% bands)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{out}/fig2_innovation_cusum.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4))
    b_cc.cumsum().plot(ax=ax, lw=1, label="cum b_cc (quant B, PIT)")
    b_panel[ref].cumsum().plot(ax=ax, lw=1, label=f"cum basis coupon {ref}")
    if truth is not None:
        vlines(ax, [truth["break_basis_date"]], "red", "planted basis break")
    vlines(ax, pelt_basis, "purple", "PELT detected")
    ax.legend(fontsize=8); ax.set_title("Cumulative hedged basis")
    fig.tight_layout(); fig.savefig(f"{out}/fig3_cum_basis.png", dpi=110)
    plt.close(fig)

    if hmm_pit is not None:
        fig, ax = plt.subplots(figsize=(11, 3.2))
        hmm_pit["filtered_full_pitparams"].plot(ax=ax, lw=0.9)
        if regime_dummy is not None:
            (regime_dummy * 1.0).plot(ax=ax, lw=1.4, color="k",
                                      label="hysteresis flag")
        ax.axvline(hmm_pit["train_end"], color="gray", ls=":",
                   label="HMM param train end")
        ax.legend(fontsize=7, ncol=4)
        ax.set_title("HMM filtered state probabilities (params fit on train only)")
        fig.tight_layout(); fig.savefig(f"{out}/fig4_hmm_filtered.png", dpi=110)
        plt.close(fig)

    if vb is not None:
        fig, ax = plt.subplots(figsize=(11, 3.2))
        vb["rolling_beta"].plot(ax=ax, lw=1)
        if truth is not None:
            vlines(ax, [truth["vol_link_date"]], "red", "planted vol-link start")
        vlines(ax, vb["pelt_break_dates"], "purple", "PELT detected")
        ax.legend(fontsize=8); ax.set_title("Rolling beta of b_cc on dVol")
        fig.tight_layout(); fig.savefig(f"{out}/fig5_vol_beta.png", dpi=110)
        plt.close(fig)

    if drift_extras is not None:
        b_ex, wp, seg, truth_w = drift_extras
        fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
        b_ex.cumsum().plot(ax=axes[0], lw=1, color="k")
        for (a, b) in truth_w:
            axes[0].axvspan(a, b, color="red", alpha=0.15)
        if seg is not None and len(seg):
            for _, s in seg.iterrows():
                if s["label"] == "widening":
                    axes[0].axvspan(s["start"], s["end"], color="purple",
                                    alpha=0.12)
        axes[0].set_title("Cum carry-adjusted basis (red: planted stress; "
                          "purple: detected widening segments)")
        if wp is not None:
            wp.plot(ax=axes[1], lw=1)
            axes[1].set_title("MS filtered P(widening)")
            axes[1].set_ylim(-0.02, 1.02)
        fig.tight_layout(); fig.savefig(f"{out}/fig7_drift_regimes.png", dpi=110)
        plt.close(fig)

    if xcorr is not None:
        fig, ax = plt.subplots(figsize=(7, 3))
        xcorr.plot(kind="bar", ax=ax)
        ax.set_title("corr( |dDesk ratio| , lagged |rate move| ) by lag (days)")
        ax.set_xlabel("lag k: rate move k days BEFORE ratio change")
        fig.tight_layout(); fig.savefig(f"{out}/fig6_desk_reactivity.png", dpi=110)
        plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--data-dir", type=str, default=None)
    args = ap.parse_args()
    if args.synthetic:
        main("synthetic")
    elif args.data_dir:
        main("real", args.data_dir)
    else:
        print("Use --synthetic or --data-dir PATH"); sys.exit(1)
