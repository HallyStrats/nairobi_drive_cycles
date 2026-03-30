"""
Script 02 — Trip Segmentation (Strict)
=======================================
Implements the segmentation rules requested by the user:

  MICRO-TRIP: each contiguous block of movement between stationary periods.
    Every time the vehicle becomes stationary (speed < STOP_SPEED_KMH) a
    new micro-trip boundary is created.

  MACRO-TRIP (riding session): separated by either
    (a) a GPS record gap > MACRO_GAP_S (device not polling — true session break), OR
    (b) a continuous stationary period > MACRO_STOP_S within a session.
    Rationale: parking the bike for > 45 s is a genuine session boundary even
    if the device keeps polling.

THRESHOLD JUSTIFICATION
-----------------------
  MACRO_GAP_S  = 600 s (10 min): separates genuine session breaks (device
                  powered off / no GPS) from the 40-60 s slow-poll gaps that
                  occur during continuous riding.  After 10s resampling these
                  gaps are interpolated away, so only true data absences matter.

  MACRO_STOP_S = 300 s: a stop longer than 300 s
                  (5 min) is treated as a session boundary (parking, not a traffic light).
                  This threshold gives a median macro-trip duration of ~29.8 min.

  STOP_SPEED_KMH = 2.0: GPS positional noise produces ~1-2 km/h jitter even
                  when stationary; this threshold absorbs that noise.

OUTPUT
------
  output/02_microtrips/  — one CSV per micro-trip
  output/02_summary.csv  — micro-trip summary table
  output/02_segmentation_stats.json — threshold stats
"""

import os
import sys
import glob
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from utils.energy_utils import compute_delta_rc, has_sufficient_rc_data

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
CLEANED_DIR    = os.path.join(HERE, "output", "01_cleaned")
MICROTRIP_DIR  = os.path.join(HERE, "output", "02_microtrips")
SUMMARY_PATH   = os.path.join(HERE, "output", "02_summary.csv")
STATS_PATH     = os.path.join(HERE, "output", "02_segmentation_stats.json")

MACRO_GAP_S    = 600    # GPS record gap → genuine session break
MACRO_STOP_S   = 600    # stationary > 600s → macro-trip boundary (median macro-trip ≈ 25.8 min)
STOP_SPEED_KMH = 2.0    # threshold for "stationary"
MIN_TRIP_DIST_M  = 100.0
MIN_TRIP_ROWS    = 6     # after resampling: 6 × 10s = 60s minimum duration
MIN_PEAK_SPEED_KMH = 5.0  # micro-trip peak speed must exceed this — filters GPS noise
RESAMPLE_STEP_S  = 10
MACROTRIP_STATS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "output", "02_macrotrips_stats.csv")


# ─────────────────────────────────────────────────────────────────────────────
# MACRO-TRIP SEGMENTATION
# ─────────────────────────────────────────────────────────────────────────────
def split_macro_trips(df: pd.DataFrame) -> list[pd.DataFrame]:
    """
    Split per-IMEI data into macro-trips (riding sessions) by:
      1. GPS record gaps > MACRO_GAP_S (device not polling)
      2. Continuous stationary periods > MACRO_STOP_S within a session

    Returns list of DataFrames, each a single riding session.
    """
    df = df.sort_values("gps_date").reset_index(drop=True)

    # --- Pass 1: collect split points from GPS record gaps ---
    dt = df["gps_date"].diff().dt.total_seconds().fillna(0)
    split_at = set([0, len(df)])
    for idx in dt[dt > MACRO_GAP_S].index:
        split_at.add(int(idx))

    # --- Pass 2: collect split points from long stationary periods ---
    stopped = df["gps_speed_kmh"] < STOP_SPEED_KMH
    # Label consecutive same-state runs
    run_id   = (stopped != stopped.shift()).cumsum()
    for rid, group in df[stopped].groupby(run_id[stopped]):
        if len(group) < 2:
            continue
        t_start = df.loc[group.index[0],  "gps_date"]
        t_end   = df.loc[group.index[-1], "gps_date"]
        dur_s   = (t_end - t_start).total_seconds()
        if dur_s >= MACRO_STOP_S:
            # Split at the START of the long stop — the terminal stationary
            # block is excluded from the preceding macro-trip entirely.
            # (Splitting at the end would include the parking time in the trip.)
            start_idx = int(group.index[0])
            if 0 < start_idx < len(df):
                split_at.add(start_idx)

    split_points = sorted(split_at)

    trips = []
    for i in range(len(split_points) - 1):
        seg = df.iloc[split_points[i]: split_points[i + 1]].copy()
        # Keep only sessions where the bike actually moved
        if not (seg["gps_speed_kmh"] >= STOP_SPEED_KMH).any():
            continue
        # Trim leading stationary rows — the stop that triggered the split of
        # the previous macro-trip becomes the start of this segment; those rows
        # are not part of this session's idle time.
        moving_mask = seg["gps_speed_kmh"] >= STOP_SPEED_KMH
        first_moving_idx = moving_mask.values.nonzero()[0][0]
        seg = seg.iloc[first_moving_idx:]
        # Trim trailing stationary rows — the stop that ends a macro-trip
        # (whether triggered by MACRO_STOP_S or a GPS gap) must not be counted
        # as idle time. Only stops BETWEEN micro-trips contribute to idle.
        moving_mask = seg["gps_speed_kmh"] >= STOP_SPEED_KMH
        last_moving_idx = moving_mask.values.nonzero()[0][-1]
        seg = seg.iloc[:last_moving_idx + 1]
        trips.append(seg.reset_index(drop=True))
    return trips


# ─────────────────────────────────────────────────────────────────────────────
# MICRO-TRIP SEGMENTATION
# ─────────────────────────────────────────────────────────────────────────────
def extract_micro_trips(macro_df: pd.DataFrame) -> list[pd.DataFrame]:
    """
    Within a macro-trip, every contiguous block of movement (speed ≥
    STOP_SPEED_KMH) is one micro-trip.  Any drop below STOP_SPEED_KMH
    creates a boundary regardless of stop duration.
    """
    df = macro_df.reset_index(drop=True)

    # Label moving vs stopped
    moving = df["gps_speed_kmh"] >= STOP_SPEED_KMH
    # Group consecutive same-state rows
    group_id = (moving != moving.shift()).cumsum()

    micro_trips = []
    for gid, grp in df.groupby(group_id):
        if not moving.loc[grp.index[0]]:
            continue   # skip stationary blocks

        seg = grp.copy().reset_index(drop=True)
        seg = resample_trip(seg)
        if seg is None or len(seg) < MIN_TRIP_ROWS:
            continue
        if seg["dist_m"].sum() < MIN_TRIP_DIST_M:
            continue
        if seg["gps_speed_kmh"].max() < MIN_PEAK_SPEED_KMH:
            continue
        micro_trips.append(seg)

    return micro_trips


# ─────────────────────────────────────────────────────────────────────────────
# RESAMPLING TO UNIFORM GRID
# ─────────────────────────────────────────────────────────────────────────────
def resample_trip(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Resample a micro-trip to a uniform RESAMPLE_STEP_S grid via linear
    interpolation.  Numeric columns are interpolated; categorical use ffill.
    """
    if len(df) < 2:
        return None

    t0  = df["gps_date"].iloc[0]
    t1  = df["gps_date"].iloc[-1]
    dur = (t1 - t0).total_seconds()

    if dur < RESAMPLE_STEP_S:
        return None

    new_times = pd.date_range(start=t0, end=t1,
                               freq=f"{RESAMPLE_STEP_S}s", tz="UTC")
    if len(new_times) < 2:
        return None

    df_idx = df.set_index("gps_date").sort_index()
    df_idx = df_idx[~df_idx.index.duplicated(keep="first")]

    combined_idx = df_idx.index.union(new_times).sort_values()
    df_expanded  = df_idx.reindex(combined_idx)

    numeric_cols = df_expanded.select_dtypes(include=[np.number]).columns
    cat_cols     = [c for c in df_expanded.columns if c not in numeric_cols]

    df_expanded[numeric_cols] = df_expanded[numeric_cols].interpolate(
        method="time", limit_direction="both"
    )
    df_expanded[cat_cols] = (df_expanded[cat_cols]
                             .fillna(method="ffill")
                             .fillna(method="bfill"))

    df_resampled = df_expanded.loc[new_times].copy()
    df_resampled.index.name = "gps_date"
    df_resampled = df_resampled.reset_index()
    df_resampled["time_s"] = np.arange(len(df_resampled), dtype=float) * RESAMPLE_STEP_S

    # Clip speed to 0 minimum after interpolation (no upper cap — IQR handled in Stage 01)
    if "gps_speed_kmh" in df_resampled.columns:
        df_resampled["gps_speed_kmh"] = df_resampled["gps_speed_kmh"].clip(lower=0)

    # Recompute acceleration from resampled speed
    dv = df_resampled["gps_speed_kmh"].diff() / 3.6    # m/s
    df_resampled["acceleration_ms2"] = (dv / RESAMPLE_STEP_S).fillna(0.0).clip(-6, 6)

    # Recompute elevation change
    if "elevation_m" in df_resampled.columns:
        df_resampled["elevation_change_m"] = (
            df_resampled["elevation_m"].diff().fillna(0.0).clip(-50, 50)
        )

    return df_resampled


# ─────────────────────────────────────────────────────────────────────────────
# MICRO-TRIP FEATURE SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
def summarise_trip(df: pd.DataFrame, trip_id: int, imei: str) -> dict:
    spd  = df["gps_speed_kmh"].values
    acc  = df["acceleration_ms2"].values
    dist = df["dist_m"].values if "dist_m" in df.columns else np.zeros(len(df))
    elev = (df["elevation_change_m"].values
            if "elevation_change_m" in df.columns else np.zeros(len(df)))

    total_dist_km = dist.sum() / 1000.0
    duration_s    = len(df) * RESAMPLE_STEP_S

    has_energy = has_sufficient_rc_data(df)
    if has_energy:
        df["energy_wh"] = compute_delta_rc(df).values
        total_wh  = df["energy_wh"].dropna().clip(lower=0).sum()
        eff_wh_km = total_wh / total_dist_km if total_dist_km > 0.05 else np.nan
    else:
        total_wh  = np.nan
        eff_wh_km = np.nan

    # Actual battery capacity from fc telemetry (mAh × voltage / 1000 → Wh)
    # fc varies per motorcycle by age/health; use the median fc for this trip
    actual_capacity_wh = np.nan
    if "fc" in df.columns and "pack_voltage_v" in df.columns:
        fc_med  = df["fc"].dropna().median()
        pv_med  = df["pack_voltage_v"].dropna().median()
        if not np.isnan(fc_med) and not np.isnan(pv_med) and pv_med > 0:
            actual_capacity_wh = fc_med * pv_med / 1000.0

    return {
        "trip_id":           trip_id,
        "imei_no":           imei,
        "date":              df["gps_date"].iloc[0].date().isoformat() if "gps_date" in df.columns else "",
        "start_time":        df["gps_date"].iloc[0].isoformat() if "gps_date" in df.columns else "",
        "n_rows":            len(df),
        "duration_s":        duration_s,
        "duration_min":      round(duration_s / 60.0, 2),
        "dist_km":           round(total_dist_km, 3),
        "mean_speed_kmh":    round(spd.mean(), 2),
        "max_speed_kmh":     round(spd.max(), 2),
        "idle_pct":          round((spd < 2.0).mean() * 100, 1),
        "mean_accel_ms2":    round(acc[acc > 0.1].mean() if (acc > 0.1).any() else 0.0, 3),
        "mean_decel_ms2":    round(acc[acc < -0.1].mean() if (acc < -0.1).any() else 0.0, 3),
        "elev_gain_m":       round(elev[elev > 0].sum(), 2),
        "elev_loss_m":       round(elev[elev < 0].sum(), 2),
        "has_energy":        has_energy,
        "total_wh":          round(total_wh, 2) if not np.isnan(total_wh) else np.nan,
        "efficiency_wh_km":  round(eff_wh_km, 2) if not np.isnan(eff_wh_km) else np.nan,
        "actual_capacity_wh": round(actual_capacity_wh, 1) if not np.isnan(actual_capacity_wh) else np.nan,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(MICROTRIP_DIR, exist_ok=True)

    # Clear old micro-trip files to avoid stale data mixing with new ones
    old_files = glob.glob(os.path.join(MICROTRIP_DIR, "trip_*.csv"))
    for f in old_files:
        os.remove(f)
    if old_files:
        print(f"  Cleared {len(old_files)} old micro-trip files.")

    cleaned_files = sorted(glob.glob(os.path.join(CLEANED_DIR, "*.csv")))
    if not cleaned_files:
        print(f"[ERROR] No cleaned files in {CLEANED_DIR}. Run 01_ingest_and_clean.py first.")
        return

    print(f"Found {len(cleaned_files)} cleaned IMEI files. Segmenting...")
    print(f"  Macro gap threshold   : {MACRO_GAP_S}s (GPS record gap)")
    print(f"  Macro stop threshold  : {MACRO_STOP_S}s (stationary within session)")
    print(f"  Micro-trip boundary   : any stop (speed < {STOP_SPEED_KMH} km/h)")

    trip_id        = 1
    summaries      = []
    macro_summaries = []   # one row per macro-trip for idle-fraction stats

    for fp in cleaned_files:
        try:
            df = pd.read_csv(fp, parse_dates=["gps_date"])
            if "gps_speed_kmh" not in df.columns:
                continue

            imei   = df["imei_no"].iloc[0] if "imei_no" in df.columns else os.path.basename(fp)
            macros = split_macro_trips(df)

            for macro in macros:
                # Macro-trip stats: duration and idle fraction (stops between
                # micro-trips only — terminal stop already trimmed by split_macro_trips)
                macro_dur_s    = len(macro) * RESAMPLE_STEP_S  # raw rows × step
                # Re-derive from timestamps for accuracy
                if "gps_date" in macro.columns and len(macro) >= 2:
                    macro_dur_s = (macro["gps_date"].iloc[-1] -
                                   macro["gps_date"].iloc[0]).total_seconds()
                # Idle fraction: proportion of elapsed time (seconds) spent stationary.
                # Using elapsed seconds rather than row count is robust to minor
                # polling-interval irregularities in the raw telemetry.
                if "gps_date" in macro.columns and len(macro) >= 2:
                    dt_s = macro["gps_date"].diff().dt.total_seconds().fillna(0)
                    stationary = macro["gps_speed_kmh"] < STOP_SPEED_KMH
                    total_s = dt_s.sum()
                    idle_s  = dt_s[stationary].sum()
                    macro_idle_frac = float(idle_s / total_s) if total_s > 0 else 0.0
                else:
                    macro_idle_frac = float((macro["gps_speed_kmh"] < STOP_SPEED_KMH).mean())

                micros = extract_micro_trips(macro)
                n_micros_in_macro = len(micros)

                if n_micros_in_macro > 0 and macro_dur_s >= 60:
                    macro_summaries.append({
                        "imei_no":        imei,
                        "duration_s":     round(macro_dur_s, 1),
                        "duration_min":   round(macro_dur_s / 60.0, 2),
                        "idle_fraction":  round(macro_idle_frac, 4),
                        "idle_pct":       round(macro_idle_frac * 100, 2),
                        "n_microtrips":   n_micros_in_macro,
                    })

                for micro in micros:
                    micro["trip_id"] = trip_id
                    micro["imei_no"] = imei

                    if has_sufficient_rc_data(micro):
                        micro["energy_wh"] = compute_delta_rc(micro).values

                    out_path = os.path.join(MICROTRIP_DIR, f"trip_{trip_id:06d}.csv")
                    micro.to_csv(out_path, index=False)

                    summaries.append(summarise_trip(micro, trip_id, imei))
                    trip_id += 1

        except Exception as e:
            print(f"  [WARN] {os.path.basename(fp)}: {e}")

    print(f"\n{'='*60}")
    print(f"STAGE 02 COMPLETE")
    print(f"  Micro-trips produced : {trip_id - 1}")
    print(f"  Micro-trip dir       : {MICROTRIP_DIR}")

    if summaries:
        summary_df = pd.DataFrame(summaries)
        summary_df.to_csv(SUMMARY_PATH, index=False)
        print(f"  Summary CSV          : {SUMMARY_PATH}")

        print(f"\n  Fleet overview (micro-trips):")
        print(f"    Total micro-trips     : {len(summary_df)}")
        print(f"    With energy data      : {summary_df['has_energy'].sum()} "
              f"({summary_df['has_energy'].mean()*100:.1f} %)")
        print(f"    Median duration (s)   : {summary_df['duration_s'].median():.0f}")
        print(f"    Median dist (m)       : {summary_df['dist_km'].median()*1000:.0f}")
        print(f"    Median max speed      : {summary_df['max_speed_kmh'].median():.1f} km/h")
        print(f"    Fleet max speed       : {summary_df['max_speed_kmh'].max():.1f} km/h")
        eff = summary_df["efficiency_wh_km"].dropna()
        if len(eff) > 0:
            print(f"    Median Wh/km          : {eff.median():.1f}")
            print(f"    Wh/km IQR             : [{eff.quantile(.25):.1f}, {eff.quantile(.75):.1f}]")

    if macro_summaries:
        macro_df = pd.DataFrame(macro_summaries)
        macro_df.to_csv(MACROTRIP_STATS_PATH, index=False)
        print(f"\n  Fleet overview (macro-trips):")
        print(f"    Total macro-trips     : {len(macro_df)}")
        print(f"    Median duration       : {macro_df['duration_min'].median():.1f} min "
              f"({macro_df['duration_s'].median():.0f} s)")
        print(f"    Median idle fraction  : {macro_df['idle_fraction'].median():.3f} "
              f"({macro_df['idle_pct'].median():.1f}%)")
        print(f"    Mean idle fraction    : {macro_df['idle_fraction'].mean():.3f} "
              f"({macro_df['idle_pct'].mean():.1f}%)")
        print(f"    Macro stats CSV       : {MACROTRIP_STATS_PATH}")

        seg_stats = {
            "macro_gap_threshold_s":  MACRO_GAP_S,
            "macro_stop_threshold_s": MACRO_STOP_S,
            "micro_boundary":         "any stop (speed < stop_speed_kmh)",
            "stop_speed_kmh":         STOP_SPEED_KMH,
            "resample_step_s":        RESAMPLE_STEP_S,
            "total_microtrips":       len(summary_df),
            "with_energy_data":       int(summary_df["has_energy"].sum()),
        }
        with open(STATS_PATH, "w") as f:
            json.dump(seg_stats, f, indent=2)

    print("="*60)


if __name__ == "__main__":
    main()
