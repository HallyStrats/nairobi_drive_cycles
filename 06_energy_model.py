"""
Script 06 — Energy Model & Over/Under Consumption Classification
================================================================
Uses ONLY measured baseline energy (delta_rc per micro-trip) — NO prediction
over trips with missing energy data.

Pipeline:
  1. Compute measured Wh/km for every micro-trip with sufficient rc coverage
  2. Remove energy outliers (IQR × 2.5)
  3. Fit a Ridge regression: Wh/km ~ f(speed features, gradient, accel RMS)
     — used to explain WHICH kinematic factors drive consumption
  4. Compute fleet energy baseline (median Wh/km)
  5. Classify each energy-measured trip as OVER or UNDER consuming vs. baseline
  6. Annotate all N_RUNS cycles from Stage 05a with:
       - model-predicted Wh/km (for all segments, clearly labelled as predicted)
       - over/under flag vs. fleet baseline Q1/Q3

OUTPUT
------
  output/06_energy_baseline.csv           — fleet energy statistics
  output/06_energy_model_coefficients.csv
  output/06_energy_classified_trips.csv   — trip-level with flag
  output/06_energy_distributions.png      — visualisation

  output_full_mape/:
    06_ga_run_00_energy.csv  …  06_ga_run_04_energy.csv
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from utils.energy_utils import (
    compute_delta_rc, has_sufficient_rc_data,
    efficiency_wh_per_km, remove_energy_outliers,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
FEATURES_PATH = os.path.join(HERE, "output", "03_features.csv")
OUTPUT_DIR    = os.path.join(HERE, "output")

BASELINE_PATH   = os.path.join(OUTPUT_DIR, "06_energy_baseline.csv")
COEFF_PATH      = os.path.join(OUTPUT_DIR, "06_energy_model_coefficients.csv")
CLASSIFIED_PATH = os.path.join(OUTPUT_DIR, "06_energy_classified_trips.csv")
PLOT_PATH       = os.path.join(OUTPUT_DIR, "06_energy_distributions.png")

# Output directory and run count (must match Stage 05a)
OUTPUT_DIR_CYCLES = "output_full_mape"
N_RUNS = 1

IQR_K            = 2.5
STEP_SEC         = 10.0

# Regression features (all derived from kinematic summary)
REGRESSION_FEATURES = [
    "mean_speed_kmh",
    "rms_accel_ms2",
    "idle_fraction",
    "gain_per_km",
    "loss_per_km",
]

# Minimum elevation gain (m) to compute wh_per_m_elev (avoid division by noise)
MIN_ELEV_GAIN_M = 5.0

# Over/under consuming thresholds: use Q1/Q3 of the fleet efficiency distribution.
# These are computed from the data at runtime in classify_consumption() rather than
# as fixed multipliers relative to the median.


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD FEATURES & FILTER TO MEASURED ENERGY ONLY
# ─────────────────────────────────────────────────────────────────────────────
def load_energy_trips() -> pd.DataFrame:
    feat_df = pd.read_csv(FEATURES_PATH)

    # Keep only trips with measured energy
    energy_df = feat_df[feat_df["has_energy"] & feat_df["efficiency_wh_km"].notna()].copy()
    print(f"  Trips with measured energy: {len(energy_df)} / {len(feat_df)}")

    # Compute energy-per-metre-elevation (Wh/m of elevation gain)
    # This isolates the gravitational energy contribution from speed/payload effects.
    # Formula: total_wh = efficiency_wh_km × total_dist_km
    #          wh_per_m_elev = total_wh / elevation_gain_m
    if "total_dist_km" in energy_df.columns and "elevation_gain_m" in energy_df.columns:
        total_wh = energy_df["efficiency_wh_km"] * energy_df["total_dist_km"]
        valid = energy_df["elevation_gain_m"] >= MIN_ELEV_GAIN_M
        energy_df["wh_per_m_elev"] = np.where(
            valid,
            total_wh / energy_df["elevation_gain_m"],
            np.nan
        )
        n_valid = valid.sum()
        med_val = energy_df.loc[valid, "wh_per_m_elev"].median()
        print(f"  Energy-per-m-elevation: {med_val:.3f} Wh/m "
              f"(n={n_valid} trips with ≥{MIN_ELEV_GAIN_M}m gain)")
    else:
        energy_df["wh_per_m_elev"] = np.nan
        print("  [WARN] elevation_gain_m or total_dist_km not in features — "
              "wh_per_m_elev not computed")

    if len(energy_df) < 5:
        print("  [WARN] Too few energy trips for modelling.")
        return energy_df

    # IQR outlier removal on Wh/km
    before = len(energy_df)
    energy_df = remove_energy_outliers(energy_df, eff_col="efficiency_wh_km", iqr_factor=IQR_K)
    after = len(energy_df)
    print(f"  After IQR outlier removal: {after} trips ({before - after} removed)")

    return energy_df


# ─────────────────────────────────────────────────────────────────────────────
# 2. FLEET BASELINE STATISTICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_baseline(energy_df: pd.DataFrame) -> dict:
    eff = energy_df["efficiency_wh_km"]
    baseline = {
        "n_trips":          int(len(eff)),
        "median_wh_km":     float(eff.median()),
        "mean_wh_km":       float(eff.mean()),
        "std_wh_km":        float(eff.std()),
        "p10_wh_km":        float(eff.quantile(0.10)),
        "p25_wh_km":        float(eff.quantile(0.25)),
        "p75_wh_km":        float(eff.quantile(0.75)),
        "p90_wh_km":        float(eff.quantile(0.90)),
        "min_wh_km":        float(eff.min()),
        "max_wh_km":        float(eff.max()),
    }
    # Energy-per-metre-elevation baseline
    if "wh_per_m_elev" in energy_df.columns:
        wpe = energy_df["wh_per_m_elev"].dropna()
        if len(wpe) > 0:
            baseline["median_wh_per_m_elev"] = float(wpe.median())
            baseline["mean_wh_per_m_elev"]   = float(wpe.mean())
            baseline["std_wh_per_m_elev"]    = float(wpe.std())
            print(f"  Fleet energy-per-elevation: "
                  f"{baseline['median_wh_per_m_elev']:.3f} Wh/m (median, "
                  f"n={len(wpe)} trips)")
            print(f"  Interpretation: each metre of elevation gain costs "
                  f"~{baseline['median_wh_per_m_elev']:.2f} Wh, isolating "
                  f"gravitational contribution from speed/payload effects")
    return baseline


# ─────────────────────────────────────────────────────────────────────────────
# 3. RIDGE REGRESSION MODEL
# ─────────────────────────────────────────────────────────────────────────────
def fit_energy_model(energy_df: pd.DataFrame) -> tuple:
    """
    Fit a Ridge regression for Wh/km from kinematic features.
    Returns (scaler, model, feature_names, r2, mae).
    """
    avail = [f for f in REGRESSION_FEATURES if f in energy_df.columns]
    if len(avail) < 2:
        print("  [WARN] Insufficient features for energy model.")
        return None, None, avail, 0.0, 0.0

    X = energy_df[avail].fillna(0).values
    y = energy_df["efficiency_wh_km"].values

    scaler = StandardScaler()
    X_std  = scaler.fit_transform(X)

    model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=5)
    model.fit(X_std, y)

    y_pred = model.predict(X_std)
    r2  = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    return scaler, model, avail, r2, mae


# ─────────────────────────────────────────────────────────────────────────────
# 4. OVER/UNDER CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────
def classify_consumption(energy_df: pd.DataFrame,
                          baseline: dict) -> pd.DataFrame:
    """
    Add 'consumption_flag' column using Q1/Q3 of the fleet efficiency distribution:
      over       — Wh/km > Q3 (75th percentile)  → heavy-payload trips
      under      — Wh/km < Q1 (25th percentile)  → light-payload trips
      normal     — between Q1 and Q3
    Thresholds are data-driven (IQR bounds) rather than fixed multipliers of the median.
    """
    df = energy_df.copy()
    over_lim  = baseline["p75_wh_km"]   # Q3
    under_lim = baseline["p25_wh_km"]   # Q1

    conditions = [
        df["efficiency_wh_km"] > over_lim,
        df["efficiency_wh_km"] < under_lim,
    ]
    choices = ["over", "under"]
    df["consumption_flag"] = np.select(conditions, choices, default="normal")

    n_over  = (df["consumption_flag"] == "over").sum()
    n_under = (df["consumption_flag"] == "under").sum()
    n_norm  = (df["consumption_flag"] == "normal").sum()
    print(f"  Consumption classification (Q1={under_lim:.1f}, Q3={over_lim:.1f} Wh/km):")
    print(f"    OVER  (>{over_lim:.1f} Wh/km, Q3) : {n_over:5d}  ({n_over/len(df)*100:.1f} %)")
    print(f"    NORMAL (Q1–Q3)                    : {n_norm:5d}  ({n_norm/len(df)*100:.1f} %)")
    print(f"    UNDER (<{under_lim:.1f} Wh/km, Q1): {n_under:5d}  ({n_under/len(df)*100:.1f} %)")
    print(f"  Note: 'over' trips may reflect higher payload (e.g. passenger + cargo)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. ANNOTATE DRIVE CYCLES
# ─────────────────────────────────────────────────────────────────────────────
def annotate_cycle(cycle_df: pd.DataFrame,
                   scaler, model, avail_features: list,
                   baseline_wh_km: float,
                   label: str = "cycle",
                   over_lim: float | None = None,
                   under_lim: float | None = None) -> pd.DataFrame:
    """
    Annotate a drive cycle with per-step predicted Wh/km.
    Uses a rolling window of N steps to compute local kinematic features,
    then applies the regression model.
    Annotates clearly as 'predicted_wh_km' (not measured).
    """
    df = cycle_df.copy()
    n  = len(df)

    # Build rolling features every 10 steps (~100 s window)
    WINDOW = 10
    predicted = np.full(n, np.nan)

    if scaler is not None and model is not None and len(avail_features) > 0:
        for i in range(WINDOW, n):
            window = df.iloc[i - WINDOW: i]
            spd  = window["gps_speed_kmh"].values
            acc  = window["acceleration_ms2"].values
            elev = window["elevation_change_m"].values if "elevation_change_m" in window.columns else np.zeros(WINDOW)
            dist = window["dist_m"].values if "dist_m" in window.columns else np.zeros(WINDOW)

            local_feat = {}
            local_feat["mean_speed_kmh"] = spd.mean()
            local_feat["rms_accel_ms2"]  = float(np.sqrt(np.mean(acc**2)))
            local_feat["idle_fraction"]  = (spd < 2.0).mean()
            total_d = dist.sum()
            if total_d > 0.5:
                local_feat["gain_per_km"] = elev[elev > 0].sum() / (total_d / 1000)
                local_feat["loss_per_km"] = elev[elev < 0].sum() / (total_d / 1000)
            else:
                local_feat["gain_per_km"] = 0.0
                local_feat["loss_per_km"] = 0.0

            row = np.array([[local_feat.get(f, 0.0) for f in avail_features]])
            predicted[i] = float(model.predict(scaler.transform(row))[0])

        predicted = np.clip(predicted, 0, None)

    df["predicted_wh_km"] = predicted
    df["fleet_baseline_wh_km"] = baseline_wh_km

    # Flag over/under against Q1/Q3 thresholds (passed from baseline dict).
    # Fallback to ±20% of median if thresholds not provided (shouldn't happen).
    if over_lim is None:
        over_lim = baseline_wh_km * 1.20
    if under_lim is None:
        under_lim = baseline_wh_km * 0.80

    flags = np.where(
        df["predicted_wh_km"] > over_lim, "over",
        np.where(df["predicted_wh_km"] < under_lim, "under", "normal")
    )
    flags[pd.isna(df["predicted_wh_km"])] = "no_data"
    df["consumption_flag"] = flags

    pct_over  = (df["consumption_flag"] == "over").mean() * 100
    pct_under = (df["consumption_flag"] == "under").mean() * 100
    print(f"  {label} — predicted over: {pct_over:.1f} %, under: {pct_under:.1f} %")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 6. VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────
def plot_energy(energy_df: pd.DataFrame, baseline: dict, path: str):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    eff = energy_df["efficiency_wh_km"]
    axes[0].hist(eff, bins=40, color="#2980b9", edgecolor="white", alpha=0.8)
    axes[0].axvline(baseline["median_wh_km"], color="red", lw=2, label=f"Median: {baseline['median_wh_km']:.1f}")
    axes[0].axvline(baseline["mean_wh_km"],   color="orange", lw=2, ls="--", label=f"Mean: {baseline['mean_wh_km']:.1f}")
    axes[0].set_title("Energy Efficiency Distribution", fontsize=12)
    axes[0].set_xlabel("Wh/km")
    axes[0].set_ylabel("Count")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Consumption classification
    flags = energy_df["consumption_flag"].value_counts()
    colors_map = {"over": "#e74c3c", "normal": "#27ae60", "under": "#3498db"}
    colors = [colors_map.get(f, "grey") for f in flags.index]
    axes[1].bar(flags.index, flags.values, color=colors, edgecolor="white")
    axes[1].set_title("Consumption Classification", fontsize=12)
    axes[1].set_ylabel("Number of Trips")
    axes[1].grid(True, alpha=0.3, axis="y")

    # Wh/km vs mean speed
    axes[2].scatter(energy_df["mean_speed_kmh"], eff,
                    c=energy_df["consumption_flag"].map({"over": "#e74c3c", "normal": "#27ae60", "under": "#3498db"}),
                    alpha=0.5, s=15)
    axes[2].axhline(baseline["median_wh_km"], color="red", lw=1.5, ls="--")
    axes[2].set_title("Wh/km vs Mean Speed", fontsize=12)
    axes[2].set_xlabel("Mean Speed (km/h)")
    axes[2].set_ylabel("Efficiency (Wh/km)")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Fleet Energy Analysis — Measured Baselines Only", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. Load measured energy trips ────────────────────────────────────────
    print("Loading measured energy trips...")
    energy_df = load_energy_trips()

    if energy_df.empty:
        print("[WARN] No energy data available — annotating cycles with baseline=NaN.")
        baseline = {"median_wh_km": np.nan}
        scaler, model, avail = None, None, []
    else:
        # ── 2. Baseline statistics ────────────────────────────────────────────
        baseline = compute_baseline(energy_df)
        print(f"\n  Fleet energy baseline:")
        for k, v in baseline.items():
            if isinstance(v, float):
                print(f"    {k:20s}: {v:.2f}")
            else:
                print(f"    {k:20s}: {v}")

        # ── 3. Ridge model ────────────────────────────────────────────────────
        print("\nFitting energy regression model...")
        scaler, model, avail, r2, mae = fit_energy_model(energy_df)
        if model is not None:
            print(f"  R²  = {r2:.3f}   MAE = {mae:.1f} Wh/km")
            coeff_df = pd.DataFrame({
                "feature":     avail,
                "coefficient": model.coef_,
            }).sort_values("coefficient", ascending=False)
            print(f"  Model coefficients:")
            print(coeff_df.to_string(index=False))
            coeff_df.to_csv(COEFF_PATH, index=False)

        # ── 4. Classify trips ─────────────────────────────────────────────────
        print("\nClassifying trips (over/under/normal)...")
        energy_df = classify_consumption(energy_df, baseline)
        energy_df.to_csv(CLASSIFIED_PATH, index=False)

        # ── 5. Save baseline ──────────────────────────────────────────────────
        pd.DataFrame([baseline]).to_csv(BASELINE_PATH, index=False)

        # ── 6. Plot ───────────────────────────────────────────────────────────
        plot_energy(energy_df, baseline, PLOT_PATH)

    # ── 7. Annotate all drive cycles (N_RUNS) ────────────────────────────────
    bwk = baseline.get("median_wh_km", np.nan)
    annotated_count = 0
    dir_path = os.path.join(HERE, OUTPUT_DIR_CYCLES)

    if not os.path.isdir(dir_path):
        print(f"  [SKIP] {OUTPUT_DIR_CYCLES}/ not found — run Stage 05a first.")
    else:
        for run_i in range(N_RUNS):
            src  = os.path.join(dir_path, f"05a_ga_run_{run_i:02d}.csv")
            dest = os.path.join(dir_path, f"06_ga_run_{run_i:02d}_energy.csv")
            if not os.path.exists(src):
                continue
            cycle = pd.read_csv(src)
            cycle = annotate_cycle(
                cycle, scaler, model, avail, bwk, label=f"run{run_i:02d}",
                over_lim=baseline.get("p75_wh_km"),
                under_lim=baseline.get("p25_wh_km"),
            )
            cycle.to_csv(dest, index=False)
            annotated_count += 1

    print(f"\n{'='*60}")
    print(f"STAGE 06 COMPLETE")
    if not energy_df.empty:
        print(f"  Fleet baseline Wh/km : {bwk:.1f}  (median, from measured rc data)")
        print(f"  Model R²             : {r2:.3f}" if model is not None
              else "  Model: not fitted")
        print(f"  Classified trips     : {CLASSIFIED_PATH}")
    print(f"  Cycles annotated     : {annotated_count} ({N_RUNS} runs)")
    print("="*60)


if __name__ == "__main__":
    main()
