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
    },

    "outdir": "outputs",
}
