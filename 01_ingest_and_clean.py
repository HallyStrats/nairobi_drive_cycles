"""
Script 01 — Ingest & Clean
==========================
Loads all raw CSV files, deduplcates, removes outliers per feature, smooths GPS
coordinates, derives GPS-based speed, and enriches elevation (SRTM → GPS fallback).

OUTPUT
------
  output/01_cleaned/  — one CSV per IMEI, sorted by time, with columns:
    imei_no, gps_date, lat, lat_smooth, long, long_smooth,
    altitude_raw, elevation_m,
    vehicle_speed, gps_speed_kmh,
    acceleration_ms2, elevation_change_m, dist_m,
    rc, fc, dc, rpm, ct, mt,
    pack_voltage_v,
    road_name, is_ignition_on

  output/01_fleet_stats.csv — per-IMEI summary for QC
"""

import os
import sys
import glob
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── project root on sys.path so `utils` is importable ──────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from utils.gps_utils import (
    haversine_m, compute_gps_speed_kmh,
    smooth_coords, flag_gps_outliers, get_best_elevation,
)
from utils.energy_utils import calc_pack_voltage

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
RAW_DATA_DIR  = os.path.join(os.path.dirname(HERE), "all_data")
OUTPUT_DIR    = os.path.join(HERE, "output", "01_cleaned")
STATS_PATH    = os.path.join(HERE, "output", "01_fleet_stats.csv")

# Speed sanity bounds
# IQR-based per-IMEI removal handles GPS noise spikes as the primary filter.
# 110 km/h is the Kenyan national highway speed limit — no legitimate boda boda
# operation is expected above this speed, so any surviving GPS record above it
# is a sensor artefact and is clipped.
MAX_SPEED_KMH_ABS  = 110.0   # absolute physical ceiling = Kenyan national speed limit
GPS_SPEED_IQR_K    = 2.5     # IQR fence multiplier for speed outlier removal
                              # k=2.5 used (not 3.0): EV motorcycles in Nairobi rarely
                              # exceed 80 km/h so GPS noise spikes must be removed.
                              # k=2.5 catches these while preserving genuine high-speed
                              # segments; k=1.5 was too aggressive, k=3.0 too permissive.
MAX_ACCEL_MS2      = 6.0     # hard deceleration / acceleration physical limit
MIN_ACCEL_MS2      = -6.0

# RPM
MAX_RPM = 9000
MIN_RPM = 0

# Battery / BMS
RC_MIN   = 0.0
RC_MAX   = 200_000.0   # Wh — generous upper bound
FC_MIN   = 1_000.0
FC_MAX   = 200_000.0
DC_MIN   = 0.0
DC_MAX   = 10_000.0

# Controller / Motor temperatures (°C)
TEMP_MIN = -10.0
TEMP_MAX = 120.0


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def iqr_fence(series: pd.Series, k: float = 3.0) -> pd.Series:
    """Return boolean mask True = within IQR fences (not an outlier)."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return (series >= lo) & (series <= hi)


def sanitise_column(df: pd.DataFrame, col: str,
                    hard_min=None, hard_max=None,
                    iqr_k: float = 3.5) -> pd.DataFrame:
    """
    Replace values outside [hard_min, hard_max] with NaN,
    then additionally fence extreme statistical outliers via IQR.
    """
    if col not in df.columns:
        return df
    s = df[col].astype(float)
    if hard_min is not None:
        s[s < hard_min] = np.nan
    if hard_max is not None:
        s[s > hard_max] = np.nan
    # IQR outlier removal (only on non-NaN values)
    valid = s.dropna()
    if len(valid) > 10:
        in_fence = iqr_fence(valid, k=iqr_k)
        outlier_idx = valid[~in_fence].index
        s.loc[outlier_idx] = np.nan
    df[col] = s
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PER-IMEI PROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def process_imei(df_raw: pd.DataFrame, imei: str) -> pd.DataFrame | None:
    """
    Full cleaning pipeline for one IMEI's data across all days.
    Returns cleaned DataFrame or None if insufficient data.
    """
    df = df_raw.copy()

    # ── 1. Parse timestamps & sort ──────────────────────────────────────────
    df["gps_date"] = pd.to_datetime(df["gps_date"], utc=True, errors="coerce")
    df = df.dropna(subset=["gps_date", "lat", "long"])
    df = df.sort_values("gps_date").reset_index(drop=True)

    # ── 2. Remove exact duplicate (imei, timestamp) rows ────────────────────
    df = df.drop_duplicates(subset=["gps_date"]).reset_index(drop=True)

    if len(df) < 10:
        return None

    # ── 3. GPS coordinate outlier removal (teleportation) ───────────────────
    is_outlier = flag_gps_outliers(df)
    n_teleport = is_outlier.sum()
    if n_teleport > 0:
        # Interpolate over flagged rows rather than drop (preserves time series)
        df.loc[is_outlier, ["lat", "long"]] = np.nan
        df["lat"]  = df["lat"].interpolate(method="linear")
        df["long"] = df["long"].interpolate(method="linear")

    # Save raw coords before any modification (df was already copied and reset)
    df["lat_raw"]  = df["lat"].copy()
    df["long_raw"] = df["long"].copy()

    # ── 4. Smooth GPS coordinates (Savitzky-Golay, window=5) ────────────────
    df["lat_smooth"]  = smooth_coords(df["lat"],  window=5, poly=2).values
    df["long_smooth"] = smooth_coords(df["long"], window=5, poly=2).values

    # ── 5. GPS-derived speed from SMOOTHED coordinates (Haversine) ──────────
    # Build a minimal temporary frame with unambiguous column names
    df_for_speed = pd.DataFrame({
        "lat":      df["lat_smooth"].values,
        "long":     df["long_smooth"].values,
        "gps_date": df["gps_date"].values,
    })
    df["gps_speed_kmh"] = compute_gps_speed_kmh(df_for_speed).values

    # IQR-based per-IMEI speed outlier removal (no hard 120 km/h cap).
    # This lets the real observed fleet maximum emerge from the data rather
    # than being set to an arbitrary ceiling.
    df = sanitise_column(df, "gps_speed_kmh",
                         hard_min=0.0, hard_max=MAX_SPEED_KMH_ABS,
                         iqr_k=GPS_SPEED_IQR_K)

    # ── 6. Sanitise vehicle_speed (internal odometer) ───────────────────────
    df = sanitise_column(df, "vehicle_speed",
                         hard_min=0.0, hard_max=MAX_SPEED_KMH_ABS, iqr_k=4.0)

    # ── 7. Acceleration from GPS speed ──────────────────────────────────────
    dt_s = df["gps_date"].diff().dt.total_seconds()
    dv_ms = (df["gps_speed_kmh"].diff()) / 3.6   # km/h → m/s
    with np.errstate(divide="ignore", invalid="ignore"):
        accel = np.where(dt_s > 0, dv_ms / dt_s, 0.0)
    df["acceleration_ms2"] = pd.Series(accel, index=df.index)
    df["acceleration_ms2"] = df["acceleration_ms2"].clip(MIN_ACCEL_MS2, MAX_ACCEL_MS2)

    # ── 8. Distance per step (Haversine on smoothed coords) ─────────────────
    dist = haversine_m(
        df["lat_smooth"].shift().values, df["long_smooth"].shift().values,
        df["lat_smooth"].values,         df["long_smooth"].values,
    )
    df["dist_m"] = pd.Series(dist, index=df.index).fillna(0.0)

    # ── 9. Elevation ─────────────────────────────────────────────────────────
    df["altitude_raw"] = df["altitude"].copy() if "altitude" in df.columns else np.nan
    df["elevation_m"]  = get_best_elevation(
        df, lat_col="lat_smooth", lon_col="long_smooth",
        alt_col="altitude_raw" if "altitude_raw" in df.columns else "altitude",
        time_col="gps_date"
    ).values
    df["elevation_change_m"] = df["elevation_m"].diff().fillna(0.0)
    # Clip extreme elevation changes (GPS artefacts / SRTM tile boundary)
    df["elevation_change_m"] = df["elevation_change_m"].clip(-50, 50)

    # ── 10. Sanitise BMS/energy columns ─────────────────────────────────────
    df = sanitise_column(df, "rc",  hard_min=RC_MIN,   hard_max=RC_MAX)
    df = sanitise_column(df, "fc",  hard_min=FC_MIN,   hard_max=FC_MAX)
    df = sanitise_column(df, "dc",  hard_min=DC_MIN,   hard_max=DC_MAX)
    df = sanitise_column(df, "rpm", hard_min=MIN_RPM,  hard_max=MAX_RPM, iqr_k=4.0)
    df = sanitise_column(df, "ct",  hard_min=TEMP_MIN, hard_max=TEMP_MAX)
    df = sanitise_column(df, "mt",  hard_min=TEMP_MIN, hard_max=TEMP_MAX)

    # ── 11. Pack voltage from cell voltages ──────────────────────────────────
    df["pack_voltage_v"] = calc_pack_voltage(df).values

    # ── 12. Select & return output columns ──────────────────────────────────
    keep = [
        "imei_no", "gps_date",
        "lat_smooth", "long_smooth",
        "altitude_raw", "elevation_m", "elevation_change_m",
        "vehicle_speed", "gps_speed_kmh", "acceleration_ms2", "dist_m",
        "rc", "fc", "dc", "rpm", "ct", "mt", "pack_voltage_v",
        "road_name", "is_ignition_on",
    ]
    available = [c for c in keep if c in df.columns]
    df_out = df[available].copy()
    if "imei_no" not in df_out.columns:
        df_out.insert(0, "imei_no", imei)

    return df_out


# ─────────────────────────────────────────────────────────────────────────────
# FLEET-LEVEL STATS FOR QC
# ─────────────────────────────────────────────────────────────────────────────
def fleet_stats(df: pd.DataFrame) -> dict:
    n = len(df)
    spd = df["gps_speed_kmh"]
    return {
        "n_rows":           n,
        "n_days":           df["gps_date"].dt.date.nunique(),
        "mean_gps_speed":   round(spd.mean(), 2),
        "max_gps_speed":    round(spd.max(), 2),
        "pct_moving":       round((spd > 2.0).mean() * 100, 1),
        "rc_coverage_pct":  round(df["rc"].notna().mean() * 100, 1) if "rc" in df.columns else 0.0,
        "alt_zero_pct":     round((df["altitude_raw"] <= 1.0).mean() * 100, 1) if "altitude_raw" in df.columns else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(STATS_PATH), exist_ok=True)

    raw_files = sorted(glob.glob(os.path.join(RAW_DATA_DIR, "*.csv")))
    if not raw_files:
        print(f"[ERROR] No CSV files found in {RAW_DATA_DIR}")
        return

    print(f"Found {len(raw_files)} raw data files. Loading...")

    # Load all files in one pass (memory efficient with low_memory=False)
    all_chunks = []
    for fp in raw_files:
        try:
            chunk = pd.read_csv(fp, low_memory=False)
            all_chunks.append(chunk)
        except Exception as e:
            print(f"  [WARN] Could not read {os.path.basename(fp)}: {e}")

    if not all_chunks:
        print("[ERROR] No data loaded.")
        return

    all_raw = pd.concat(all_chunks, ignore_index=True)
    print(f"  Total rows loaded: {len(all_raw):,}")
    print(f"  Unique IMEIs:      {all_raw['imei_no'].nunique()}")

    # ── Per-IMEI processing ─────────────────────────────────────────────────
    imeis = all_raw["imei_no"].unique()
    fleet_rows = []
    processed = 0
    skipped = 0

    for imei in imeis:
        subset = all_raw[all_raw["imei_no"] == imei].copy()
        result = process_imei(subset, imei)

        if result is None or len(result) < 10:
            skipped += 1
            continue

        # Save per-IMEI file (safe filename)
        safe_name = imei.replace("/", "_").replace("=", "").replace("+", "_")[:40]
        out_path  = os.path.join(OUTPUT_DIR, f"{safe_name}.csv")
        result.to_csv(out_path, index=False)

        # Collect fleet stats
        stats = fleet_stats(result)
        stats["imei_no"]   = imei
        stats["file"]      = os.path.basename(out_path)
        fleet_rows.append(stats)
        processed += 1

        if processed % 50 == 0:
            print(f"  Processed {processed}/{len(imeis)} IMEIs ...")

    print(f"\n{'='*60}")
    print(f"STAGE 01 COMPLETE")
    print(f"  IMEIs processed : {processed}")
    print(f"  IMEIs skipped   : {skipped}")
    print(f"  Output dir      : {OUTPUT_DIR}")

    if fleet_rows:
        fleet_df = pd.DataFrame(fleet_rows)
        fleet_df.to_csv(STATS_PATH, index=False)
        print(f"  Fleet stats     : {STATS_PATH}")
        print(f"\n  Top 5 most active IMEIs (by % moving):")
        print(fleet_df.sort_values("pct_moving", ascending=False)
                      [["imei_no", "n_rows", "mean_gps_speed", "pct_moving", "rc_coverage_pct"]]
                      .head(5).to_string(index=False))

    print("="*60)


if __name__ == "__main__":
    main()
