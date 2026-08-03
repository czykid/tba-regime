"""Central configuration. Every methodological knob lives here so choices are
explicit and auditable. All rolling computations are TRAILING-ONLY by design.
"""

CONFIG = {
    # ---------------- hedge-ratio replication ----------------
    "hedge": {
        "reg_window": 60,          # rolling window (days) for node regressions, per quant spec
        "min_obs": 45,             # minimum non-NaN obs inside window
        "moneyness_grid_bp": 10,   # node spacing in bp, per quant spec
        "moneyness_max_bp": 150,   # grid spans [-max, +max] around current coupon
        "curve_fit": "poly3",      # 'poly3' or 'spline' smooth curve across moneyness
        "spline_s": 0.02,          # smoothing factor if curve_fit == 'spline'
        "coupon_step": 0.5,        # actual TBA coupon increment
    },

    # ---------------- basis construction ----------------
    "basis": {
        "ratio_lag": 1,            # hedge with ratio known at t-1 (point-in-time discipline)
        "roll_method": "panama",   # additive back-adjustment; changes computed within held month
        "emit_coupon_swaps": True, # b_c1 - b_c2 panel (duration-neutral swap identity)
    },

    # ---------------- regime / break battery ----------------
    "regimes": {
        "pelt_min_size": 90,       # >= reg_window: beta series has ~59/60 overlap autocorrelation
        "pelt_pen_scale": 3.0,     # penalty multiplier on log(T)*var
        "supF_trim": 0.15,         # Andrews interior trimming
        "boot_B": 199,             # bootstrap replications for sup-F / Nyblom p-values
        "boot_block": 20,          # circular block length (handles serial correlation)
        "kalman_q_init": 1e-6,     # initial state innovation variance guess
        "cusum_alpha_a": 0.948,    # Brown-Durbin-Evans 5% band coefficient
        "hmm_states": [2, 3],      # candidate state counts; BIC selects
        "hmm_seeds": [0, 1, 2, 3, 4],
        "hmm_min_prob": 0.8,       # hysteresis: enter state at 0.8, exit at 0.2 (filtered prob)
        "vol_beta_window": 60,     # rolling window for basis-on-dVol beta
        "min_obs_frac": 0.75,      # min_periods as fraction of rolling window
        "kalman_init_window": 60,  # OLS burn-in for Kalman state init
        "kalman_burn": 60,         # innovations discarded before CUSUM
        "pelt_jump": 5,            # ruptures grid stride (speed/resolution)
        "hmm_train_frac": 0.6,     # HMM params fit on this leading fraction
        "hmm_winsor_z": 5.0,       # clip standardized features (outlier states)
        "hmm_state_min_freq": 0.02,# ignore degenerate states when picking stress
        "min_obs_break_test": 120, # min sample for sup-F / Nyblom
        "min_obs_kalman": 150,
        "desk_xcorr_max_lag": 10,
        "desk_trigger_thresholds": [0.05, 0.10, 0.14, 0.18],
        "hedge_lag_artifact_days": 75,  # basis break within N days of beta break
        "tsy_roll_proximity_days": 5,   # DV01-step caution window
    },

    # ---------------- forecast harness ----------------
    "forecast": {
        "horizon": 5,              # target = next-h-day cumulative hedged basis return
        "refit_every": 21,         # walk-forward refit cadence (days)
        "min_train": 252,          # first prediction after this many obs
        "embargo": 5,              # purge: training pairs must end >= embargo before prediction date
        "hac_lag": 5,              # Newey-West lags for CW/GW (>= horizon-1)
        "cost_per_unit": 0.0078125,# roundtrip cost per unit position change, price points (~1/4 of 1/32)
    },

    # ---------------- synthetic data (smoke test only) ----------------
    "synth": {
        "n_days": 1750,
        "seed": 42,
        "coupons": [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0],
        "tsy_duration": 8.5,       # price points per 100bp, for the 10y proxy
        # planted truths (indices into the date range) -- used only to score recovery
        "break_beta_idx": 900,     # duration-regime break: G-slope scaled by beta_break_mult
        "beta_break_mult": 1.35,
        "break_basis_idx": 1300,   # basis mean/vol regime shift
        "vol_link_start": 1100,    # basis loads on dVol only after this index (vol-beta regime)
        # planted DRIFT windows: sustained negative basis drift + vol bump
        # (the 'hiking / risk-off' episodes the drift module must recover)
        "stress_windows": [(620, 700), (1470, 1550)],
        "stress_drift": -0.045,    # extra points/day on the common factor
        "stress_sig_mult": 1.5,
        "stress_vol_bump": 15.0,
        "carry_drift": 0.025,     # calm-regime spread-carry accrual (pts/day)
    },

    # ---------------- drift regimes (carry vs widening) ----------------
    "drift": {
        "ms_k": 2,               # Hamilton MS regimes (2 = carry vs widening)
        "ms_scale": 100.0,       # numerical scaling for MS estimation
        "pelt_min_size": 40,     # returns are not overlapping-window objects
        "pelt_pen_scale": 2.0,
        "widen_tstat": -1.0,     # segment labelled widening below this t
        "cycle_days": 21,        # roll-cycle amortization for carry proxy
        "studentize_window": 60, # trailing sigma window for vol-studentization
        "pelt_jump": 2,
        "lam_window": 250,       # trailing window for basis-on-dVol loading
        "cc_sens": 0.9,          # dCC/dy assumption in the convexity term
        "event_pre_days": 5,
        "event_post_days": 15,
        "settle_days": 30,       # days-to-settle in the carry/specialness proxy
    },

    # ---------------- empirical calibration ----------------
    # Replaces textbook constants with thresholds calibrated to YOUR data's
    # own tails / null behaviour. Each has a "fixed" fallback.
    "calibration": {
        "shewhart_mode": "empirical",   # 'empirical' | 'fixed'
        "shewhart_fixed": 3.5,          # Gaussian-convention fallback
        "shewhart_quantile": 0.998,     # trailing quantile of |z| (~1/2yr)
        "shewhart_window": 500,         # trailing obs for the quantile
        "shewhart_min_window": 250,     # below this, use fixed
        "pelt_mode": "empirical",       # 'empirical' | 'fixed'
        "pelt_target_fa": 0.05,         # P(>=1 spurious break) under the null
        "pelt_boot_B": 40,              # bootstrap reps (penalty search is O(B*|grid|))
        "pelt_pen_grid": [1.0, 2.0, 3.0, 6.0, 12.0, 20.0, 40.0, 80.0, 160.0],
        "report_realized_rates": True,  # print achieved exceedance rates
    },

    "outdir": "outputs",
}
