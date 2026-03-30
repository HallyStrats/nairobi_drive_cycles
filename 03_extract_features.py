"""
Script 03 — Feature Extraction & Outlier Removal
==================================================
For each micro-trip from Stage 02:
  - Builds the 70-element Speed-Acceleration Frequency (SAF) vector
  - Computes rich kinematic summary statistics
  - Computes elevation statistics
  - Validates and records measured energy (Wh/km) — trips without rc data are
    excluded from energy analysis but kept for kinematic clustering
  - Applies per-feature IQR outlier removal at the trip level

OUTPUT
------
  output/03_features.csv      — per-micro-trip feature table (clean)
  output/03_saf_matrix.npy    — (N_trips, 70) SAF matrix array
  output/03_outlier_report.csv — count of features removed per column
"""

import os
import sys
import glob
import warnings
import numpy as np
import pandas as pd
from scipy.stats import zscore

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from utils.metrics import (
    build_saf_matrix, saf_to_vector, kinematic_stats, elevation_stats,
    SAF_SIZE,
)
from utils.energy_utils import has_sufficient_rc_data, efficiency_wh_per_km

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
MICROTRIP_DIR  = os.path.join(HERE, "output", "02_microtrips")
SUMMARY_PATH   = os.path.join(HERE, "output", "02_summary.csv")
OUTPUT_DIR     = os.path.join(HERE, "output")
FEATURES_PATH  = os.path.join(OUTPUT_DIR, "03_features.csv")
SAF_MATRIX_PATH = os.path.join(OUTPUT_DIR, "03_saf_matrix.npy")
OUTLIER_REPORT_PATH = os.path.join(OUTPUT_DIR, "03_outlier_report.csv")

STEP_SEC = 10.0  # resample grid from Stage 02

# IQR fence multiplier for trip-level outlier removal
IQR_K = 2.5


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def iqr_remove(df: pd.DataFrame, col: str, k: float = IQR_K) -> pd.Series:
    """Return boolean mask True = within IQR fence."""
    s = df[col].dropna()
    if len(s) < 10:
        return pd.Series(True, index=df.index)
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return (df[col].isna()) | ((df[col] >= lo) & (df[col] <= hi))


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION PER TRIP
# ─────────────────────────────────────────────────────────────────────────────
def extract_trip_features(trip_df: pd.DataFrame,
                          trip_id: int,
                          imei: str) -> tuple[dict, np.ndarray]:
    """
    Returns (feature_dict, saf_vector) for one micro-trip.
    """
    spd  = trip_df["gps_speed_kmh"].fillna(0).values
    acc  = trip_df["acceleration_ms2"].fillna(0).values
    dist = trip_df["dist_m"].fillna(0).values if "dist_m" in trip_df.columns else np.zeros(len(trip_df))
    elev = trip_df["elevation_change_m"].fillna(0).values if "elevation_change_m" in trip_df.columns else np.zeros(len(trip_df))
    elev_abs = trip_df["elevation_m"].values if "elevation_m" in trip_df.columns else None

    # ── SAF vector ──────────────────────────────────────────────────────────
    saf_vec = saf_to_vector(spd, acc)

    # ── Kinematic stats ──────────────────────────────────────────────────────
    kine = kinematic_stats(spd, acc, dt_s=STEP_SEC)

    # ── Elevation stats ─────────────────────────────────────────────────────
    elev_st = elevation_stats(elev, dist)

    # ── Energy (only measured, no prediction) ───────────────────────────────
    energy_wh_km = np.nan
    total_wh     = np.nan
    has_nrg      = False

    if has_sufficient_rc_data(trip_df):
        has_nrg = True
        if "energy_wh" in trip_df.columns:
            ewh = trip_df["energy_wh"].dropna().clip(lower=0).sum()
        else:
            from utils.energy_utils import compute_delta_rc
            ewh = compute_delta_rc(trip_df).dropna().clip(lower=0).sum()
        total_wh     = ewh
        total_km     = kine.get("total_dist_km", 0)
        energy_wh_km = ewh / total_km if total_km > 0.05 else np.nan

    # ── Absolute elevation profile ──────────────────────────────────────────
    start_elev = float(elev_abs[0]) if elev_abs is not None and not np.isnan(elev_abs[0]) else np.nan
    end_elev   = float(elev_abs[-1]) if elev_abs is not None and not np.isnan(elev_abs[-1]) else np.nan

    feat = {
        "trip_id": trip_id,
        "imei_no": imei,
        **kine,
        **elev_st,
        "has_energy":       has_nrg,
        "total_wh":         total_wh,
        "efficiency_wh_km": energy_wh_km,
        "start_elevation_m": start_elev,
        "end_elevation_m":   end_elev,
    }
    return feat, saf_vec


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    trip_files = sorted(glob.glob(os.path.join(MICROTRIP_DIR, "trip_*.csv")))
    if not trip_files:
        print(f"[ERROR] No micro-trip files found in {MICROTRIP_DIR}. Run 02 first.")
        return

    print(f"Found {len(trip_files)} micro-trips. Extracting features...")

    features_list = []
    saf_list      = []

    for fp in trip_files:
        try:
            df = pd.read_csv(fp, parse_dates=["gps_date"])
            if len(df) < 4 or "gps_speed_kmh" not in df.columns:
                continue

            trip_id = int(os.path.basename(fp).split("_")[1].split(".")[0])
            imei    = df["imei_no"].iloc[0] if "imei_no" in df.columns else "unknown"

            feat, saf_vec = extract_trip_features(df, trip_id, imei)
            features_list.append(feat)
            saf_list.append(saf_vec)

        except Exception as e:
            print(f"  [WARN] {os.path.basename(fp)}: {e}")

    if not features_list:
        print("[ERROR] No features extracted.")
        return

    feat_df  = pd.DataFrame(features_list)
    saf_arr  = np.vstack(saf_list)

    print(f"  Extracted features for {len(feat_df)} trips.")

    # ── OUTLIER REMOVAL ──────────────────────────────────────────────────────
    outlier_counts = {}
    numeric_cols   = feat_df.select_dtypes(include=[np.number]).columns.tolist()
    # Skip IDs and boolean flags
    skip_cols = {"trip_id", "has_energy"}

    original_len = len(feat_df)
    keep_mask    = pd.Series(True, index=feat_df.index)

    # Features where extreme outliers should invalidate the whole trip row
    critical_cols = [
        "max_speed_kmh", "mean_speed_kmh", "rms_accel_ms2",
        "max_accel_ms2", "min_accel_ms2",
        "efficiency_wh_km",    # only removes rows that have energy data
        "gain_per_km", "loss_per_km",
    ]

    for col in critical_cols:
        if col not in feat_df.columns:
            continue
        # For energy columns, only fence rows that have data
        if col == "efficiency_wh_km":
            has_e = feat_df["has_energy"]
            mask  = ~has_e | iqr_remove(feat_df, col, k=IQR_K)
        else:
            mask = iqr_remove(feat_df, col, k=IQR_K)

        n_removed = (~mask).sum()
        if n_removed > 0:
            print(f"  [outlier] {col}: removing {n_removed} trips "
                  f"(IQR×{IQR_K})")
        outlier_counts[col] = int(n_removed)
        keep_mask &= mask

    feat_df  = feat_df[keep_mask].reset_index(drop=True)
    saf_arr  = saf_arr[keep_mask.values]

    removed = original_len - len(feat_df)
    print(f"  Trips after outlier removal: {len(feat_df)} "
          f"({removed} removed, {removed/original_len*100:.1f} %)")

    # ── Energy outlier summary ───────────────────────────────────────────────
    eff = feat_df.loc[feat_df["has_energy"], "efficiency_wh_km"].dropna()
    if len(eff) > 0:
        print(f"\n  Energy efficiency (Wh/km):")
        print(f"    n trips with measured energy : {len(eff)}")
        print(f"    Median  : {eff.median():.1f}")
        print(f"    Mean    : {eff.mean():.1f}")
        print(f"    Std     : {eff.std():.1f}")
        print(f"    IQR     : [{eff.quantile(.25):.1f}, {eff.quantile(.75):.1f}]")
        print(f"    Min/Max : {eff.min():.1f} / {eff.max():.1f}")
    else:
        print("  No trips with measured energy efficiency found.")

    # ── Save outputs ─────────────────────────────────────────────────────────
    feat_df.to_csv(FEATURES_PATH, index=False)
    np.save(SAF_MATRIX_PATH, saf_arr)

    outlier_df = pd.DataFrame([
        {"column": k, "n_removed": v} for k, v in outlier_counts.items()
    ])
    outlier_df.to_csv(OUTLIER_REPORT_PATH, index=False)

    print(f"\n{'='*60}")
    print(f"STAGE 03 COMPLETE")
    print(f"  Features CSV  : {FEATURES_PATH}")
    print(f"  SAF matrix    : {SAF_MATRIX_PATH}  shape={saf_arr.shape}")
    print(f"  Outlier report: {OUTLIER_REPORT_PATH}")
    print("="*60)


if __name__ == "__main__":
    main()
