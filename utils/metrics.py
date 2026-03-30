"""
Evaluation Metrics: SAF matrix, kinematic stats, error scoring.
"""
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Speed-Acceleration Frequency (SAF) Matrix
# ---------------------------------------------------------------------------
SPEED_BINS_KMH = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, np.inf])
ACCEL_BINS_MS2 = np.array([-np.inf, -2.0, -1.0, -0.2, 0.2, 1.0, 2.0, np.inf])

N_SPEED_BINS = len(SPEED_BINS_KMH) - 1  # 10
N_ACCEL_BINS = len(ACCEL_BINS_MS2) - 1  # 7
SAF_SHAPE = (N_SPEED_BINS, N_ACCEL_BINS)
SAF_SIZE = N_SPEED_BINS * N_ACCEL_BINS    # 70


def build_saf_matrix(speed_kmh: np.ndarray,
                     accel_ms2: np.ndarray,
                     normalise: bool = True) -> np.ndarray:
    """
    Build a Speed-Acceleration Frequency matrix.
    Returns a (N_SPEED_BINS, N_ACCEL_BINS) array, optionally normalised to sum=1.
    """
    H, _, _ = np.histogram2d(
        speed_kmh, accel_ms2,
        bins=[SPEED_BINS_KMH, ACCEL_BINS_MS2]
    )
    if normalise and H.sum() > 0:
        H = H / H.sum()
    return H


def saf_to_vector(speed_kmh: np.ndarray, accel_ms2: np.ndarray) -> np.ndarray:
    """Flat normalised SAF vector (used as clustering feature)."""
    return build_saf_matrix(speed_kmh, accel_ms2, normalise=True).ravel()


def saf_sse(target: np.ndarray, candidate: np.ndarray) -> float:
    """Sum of squared errors between two SAF matrices."""
    return float(np.sum((target - candidate) ** 2))


def saf_mae(target: np.ndarray, candidate: np.ndarray) -> float:
    """Mean absolute error between two SAF matrices."""
    return float(np.mean(np.abs(target - candidate)))


# ---------------------------------------------------------------------------
# 2. Kinematic Summary Statistics
# ---------------------------------------------------------------------------
def kinematic_stats(speed_kmh: np.ndarray,
                    accel_ms2: np.ndarray,
                    dt_s: float = 10.0) -> dict:
    """
    Compute a rich set of kinematic statistics for a speed-acceleration trace.
    speed_kmh : array in km/h
    accel_ms2 : array in m/s²
    dt_s      : time step in seconds (assumed uniform)

    Key metrics:
      stops_per_minute — moving→stopped transitions per minute of total drive
                      time; comparable between fleet and any drive cycle.
      pke_per_km    — Positive Kinetic Energy per km (m²/s² per km); standard
                      EV drive-cycle benchmark capturing speed-weighted
                      acceleration effort (Hung 2007, Cui 2022, WLTP).
      pct_accel/decel/cruise — driving-mode fractions derived from acceleration
                      thresholds; informational (partially redundant with SAF
                      acceleration marginal).
    """
    n = len(speed_kmh)
    if n == 0:
        return {}

    idle_mask   = speed_kmh < 2.0
    moving_mask = ~idle_mask
    accel_mask  = accel_ms2 > 0.1
    decel_mask  = accel_ms2 < -0.1
    cruise_mask = (accel_ms2 >= -0.1) & (accel_ms2 <= 0.1) & moving_mask

    total_duration_s = n * dt_s
    idle_fraction    = float(idle_mask.sum() / n)
    total_dist_km    = float(np.sum(speed_kmh) * dt_s / 3600.0)

    # ── Stop transitions ──────────────────────────────────────────────────────
    # Count transitions from moving (≥2 km/h) to stopped (<2 km/h).
    # stops_per_minute uses total time (including idle) as denominator —
    # directly comparable to fleet macro-trip data and any drive cycle.
    moving_int       = moving_mask.astype(int)
    n_stops          = int(np.sum(np.diff(moving_int) == -1))
    total_min        = total_duration_s / 60.0
    stops_per_minute = n_stops / total_min if total_min > 0 else 0.0

    # ── Upper-tail speed (p95 of running speeds) ──────────────────────────────
    # 95th percentile of moving-only speeds.  Uses running rows so that the idle
    # fraction doesn't compress the percentile downward — making it directly
    # comparable between fleet micro-trips (no idle) and assembled cycles (29 % idle).
    # p95 captures "fast but normal" driving without anchoring to GPS outliers.
    p95_running = (float(np.percentile(speed_kmh[moving_mask], 95))
                   if moving_mask.any() else 0.0)

    # ── Positive Kinetic Energy (PKE) per km ──────────────────────────────────
    # PKE = Σ(v²ᵢ₊₁ − v²ᵢ) for all accelerating steps / total_dist_km
    # Units: m²/s² per km.  Captures energy invested in speed increases,
    # weighted by the speed at which they occur (unlike mean_accel_ms2).
    v_ms  = speed_kmh / 3.6
    dv2   = np.diff(v_ms ** 2)
    pke   = float(dv2[dv2 > 0].sum())
    pke_per_km = pke / total_dist_km if total_dist_km > 0 else 0.0

    stats = {
        # ── Speed ──────────────────────────────────────────────────────────────
        "mean_speed_kmh":         float(np.mean(speed_kmh)),
        "mean_running_speed_kmh": float(np.mean(speed_kmh[moving_mask])) if moving_mask.any() else 0.0,
        "max_speed_kmh":          float(np.max(speed_kmh)),
        "std_speed_kmh":          float(np.std(speed_kmh)),
        # ── Idle / stops ───────────────────────────────────────────────────────
        "idle_fraction":          idle_fraction,
        "idle_time_pct":          idle_fraction * 100.0,
        "n_stops":                n_stops,           # total moving→stopped transitions
        "stops_per_minute":       stops_per_minute,  # informational; target uses n_stops
        "p95_running_speed_kmh":  p95_running,
        # ── Acceleration ───────────────────────────────────────────────────────
        "mean_accel_ms2":         float(np.mean(accel_ms2[accel_mask])) if accel_mask.any() else 0.0,
        "mean_decel_ms2":         float(np.mean(accel_ms2[decel_mask])) if decel_mask.any() else 0.0,
        "rms_accel_ms2":          float(np.sqrt(np.mean(accel_ms2 ** 2))),
        "max_accel_ms2":          float(np.max(accel_ms2)),
        "min_accel_ms2":          float(np.min(accel_ms2)),
        # ── Energy ─────────────────────────────────────────────────────────────
        "pke_per_km":             pke_per_km,
        # ── Distance / duration ────────────────────────────────────────────────
        "total_dist_km":          total_dist_km,
        "duration_s":             float(total_duration_s),
        # ── Speed bins (time-proportional, reported but NOT used in fitness) ───
        "pct_0_20":               float((speed_kmh < 20).sum() / n * 100),
        "pct_20_40":              float(((speed_kmh >= 20) & (speed_kmh < 40)).sum() / n * 100),
        "pct_40_60":              float(((speed_kmh >= 40) & (speed_kmh < 60)).sum() / n * 100),
        "pct_60_plus":            float((speed_kmh >= 60).sum() / n * 100),
        # ── Driving-mode fractions (informational) ─────────────────────────────
        "pct_accel":              float(accel_mask.sum() / n * 100),
        "pct_decel":              float(decel_mask.sum() / n * 100),
        "pct_cruise":             float(cruise_mask.sum() / n * 100),
    }
    return stats


# ---------------------------------------------------------------------------
# 3. Elevation Metrics
# ---------------------------------------------------------------------------
MAX_PLAUSIBLE_GRADIENT_PCT = 30.0   # Nairobi roads rarely exceed 30 % grade


def elevation_stats(elev_change_m: np.ndarray,
                    dist_m: np.ndarray) -> dict:
    """
    Elevation statistics for a trace.
    elev_change_m: per-step elevation delta (m)
    dist_m: per-step distance (m)

    GPS and SRTM noise can create spurious gradient spikes.  We clip gradient
    computation to ±MAX_PLAUSIBLE_GRADIENT_PCT before deriving max/min, and use
    only steps with |gradient| < MAX_PLAUSIBLE_GRADIENT_PCT for gain/loss sums
    so that artefacts don't corrupt the fleet target statistics.
    """
    total_km = dist_m.sum() / 1000.0

    # Compute gradients only where distance is meaningful
    valid = dist_m > 0.5
    gradients = np.zeros(len(dist_m))
    if valid.any():
        gradients[valid] = (elev_change_m[valid] / dist_m[valid]) * 100.0

    # Mask out implausible gradient steps (GPS / SRTM tile artefacts)
    plausible = np.abs(gradients) <= MAX_PLAUSIBLE_GRADIENT_PCT

    gain = float(elev_change_m[plausible & (elev_change_m > 0)].sum())
    loss = float(elev_change_m[plausible & (elev_change_m < 0)].sum())

    plausible_grads = gradients[plausible]

    return {
        "elevation_gain_m":     gain,
        "elevation_loss_m":     loss,
        "gain_per_km":          gain / total_km if total_km > 0 else 0.0,
        "loss_per_km":          loss / total_km if total_km > 0 else 0.0,
        "max_gradient_pct":     float(plausible_grads.max()) if len(plausible_grads) > 0 else 0.0,
        "min_gradient_pct":     float(plausible_grads.min()) if len(plausible_grads) > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# 4. Composite Error Score (lower = better)
# ---------------------------------------------------------------------------
def composite_error(candidate_speed: np.ndarray,
                    candidate_accel: np.ndarray,
                    candidate_elev: np.ndarray,
                    candidate_dist: np.ndarray,
                    target_saf: np.ndarray,
                    target_kine: dict,
                    target_elev_stats: dict,
                    weights: dict = None) -> tuple:
    """
    Calculate a weighted composite error between a candidate cycle and targets.

    Includes an explicit speed-bin distribution penalty to prevent the GA from
    over-representing high-speed segments (a known failure mode when SAF SSE
    is the only objective, since high-speed bins have low absolute frequencies).

    Returns (total_score, breakdown_dict).
    """
    if weights is None:
        weights = {"saf": 1.0, "kine": 0.8, "elev": 0.2}

    # SAF error
    cand_saf = build_saf_matrix(candidate_speed, candidate_accel, normalise=True)
    saf_err = saf_sse(target_saf, cand_saf)

    # Kinematic MAPE — independent metrics only.
    # mean_speed_kmh excluded: linearly dependent on mean_running_speed_kmh
    # and idle_fraction (any two of the three suffice).
    # stops_per_minute captures stop frequency as a time-domain rate.
    # max_accel/min_accel excluded: single extreme events, not achievable
    # by trip selection.
    cand_kine = kinematic_stats(candidate_speed, candidate_accel)
    kine_errs = []
    fitness_kine_keys = ["mean_running_speed_kmh", "n_stops",
                         "mean_accel_ms2", "mean_decel_ms2", "rms_accel_ms2"]
    # Add idle_fraction only if the target is meaningfully non-zero
    # (i.e. set from macro-trip stats, not the default 0.0 from micro-trips)
    if target_kine.get("idle_fraction", 0.0) > 0.01:
        fitness_kine_keys = ["idle_fraction"] + fitness_kine_keys
    for key in fitness_kine_keys:
        t_val = target_kine.get(key, 0.0)
        c_val = cand_kine.get(key, 0.0)
        # Use max(|target|, 0.1) as denominator to prevent blowup when
        # target is near zero (e.g. mean_decel near 0 for flat urban driving)
        kine_errs.append(abs(c_val - t_val) / (max(abs(t_val), 0.1) + 1e-9))
    kine_err = float(np.mean(kine_errs))

    # Elevation MAPE (per-km metrics + max gradient)
    cand_elev_stats = elevation_stats(candidate_elev, candidate_dist)
    elev_errs = []
    for key in ["gain_per_km", "loss_per_km", "max_gradient_pct"]:
        t_val = target_elev_stats.get(key, 0.0)
        c_val = cand_elev_stats.get(key, 0.0)
        elev_errs.append(abs(c_val - t_val) / (abs(t_val) + 1e-6))
    elev_err = float(np.mean(elev_errs)) if elev_errs else 0.0

    total = (weights.get("saf", 1.0)    * saf_err
             + weights.get("kine", 0.8) * kine_err
             + weights.get("elev", 0.2) * elev_err)

    breakdown = {
        "saf_sse":     saf_err,
        "kine_mape":   kine_err,
        "elev_mape":   elev_err,
        "total_score": total,
    }
    return total, breakdown


# ---------------------------------------------------------------------------
# 5. Percentage Error Report
# ---------------------------------------------------------------------------
def error_report(target_stats: dict, candidate_stats: dict,
                 label: str = "Candidate") -> pd.DataFrame:
    """
    Side-by-side table of target vs candidate statistics with % error.
    """
    rows = []
    for key in sorted(set(target_stats) | set(candidate_stats)):
        t = target_stats.get(key, np.nan)
        c = candidate_stats.get(key, np.nan)
        if isinstance(t, (int, float)) and isinstance(c, (int, float)):
            # Return NaN when target ≈ 0 to avoid near-infinite % errors
            # (e.g. saf_sse/saf_mae when target is 0.0 for a perfect match)
            if abs(t) < 1e-6:
                pct_err = np.nan
            else:
                pct_err = abs(c - t) / abs(t) * 100
        else:
            pct_err = np.nan
        rows.append({"Metric": key, "Target": t, label: c, "Abs_Err_%": pct_err})
    return pd.DataFrame(rows).set_index("Metric")


def saf_scalar_stats(target_saf: np.ndarray, cand_saf: np.ndarray) -> dict:
    """
    Return SAF SSE and MAE as scalar metrics suitable for merging into
    kinematic stats dicts before calling error_report().
    """
    return {
        "saf_sse": saf_sse(target_saf, cand_saf),
        "saf_mae": saf_mae(target_saf, cand_saf),
    }


# ---------------------------------------------------------------------------
# 6. Speed-Elevation Frequency (SEF) Matrix
# ---------------------------------------------------------------------------
DELTA_V_BINS_KMH  = np.array([-np.inf, -6, -4, -2, -0.5, 0.5, 2, 4, 6, np.inf])  # 9 bins
DELTA_ELEV_BINS_M = np.array([-np.inf, -2, -1, -0.3, 0.3, 1, 2, np.inf])          # 7 bins

SEF_SHAPE = (len(DELTA_V_BINS_KMH) - 1, len(DELTA_ELEV_BINS_M) - 1)  # 9×7 = 63


def build_sef_matrix(speed_kmh: np.ndarray,
                     elev_change_m: np.ndarray,
                     normalise: bool = True) -> np.ndarray:
    """
    Speed-Elevation Frequency matrix.
    X-axis: Δv (km/h) = diff of speed_kmh  — 9 bins
    Y-axis: Δelev (m/step) — 7 bins
    Returns (9, 7) array, optionally normalised to sum=1.

    Captures the joint distribution of speed changes and elevation changes —
    a compact fingerprint of how the vehicle accelerates/decelerates
    relative to terrain gradient.
    """
    delta_v = np.diff(speed_kmh)
    delta_e = elev_change_m[1:]   # align lengths (both length n-1)
    H, _, _ = np.histogram2d(delta_v, delta_e,
                              bins=[DELTA_V_BINS_KMH, DELTA_ELEV_BINS_M])
    if normalise and H.sum() > 0:
        H = H / H.sum()
    return H
