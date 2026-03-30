"""
find_macro_stop.py — Auto-sweep to find MACRO_STOP_S giving median macro-trip ≈ 30 min.

Runs AFTER stage 01 (cleaned data must exist in output/01_cleaned/).
Uses the same split logic as 02_segment_trips.py (leading + trailing trim).
Writes the chosen threshold back into 02_segment_trips.py so stage 02
picks it up automatically.
"""

import os
import sys
import re
import glob
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE        = os.path.dirname(os.path.abspath(__file__))
CLEANED_DIR = os.path.join(HERE, "output", "01_cleaned")
SCRIPT_PATH = os.path.join(HERE, "02_segment_trips.py")

STOP_SPEED_KMH = 2.0
MACRO_GAP_S    = 600
TARGET_MIN     = 30.0   # desired median macro-trip duration in minutes

SWEEP_VALUES = [120, 150, 180, 210, 240, 270, 300, 330, 360, 420, 480, 600]


# ── Core split logic (mirrors 02_segment_trips.py exactly) ───────────────────
def split_macro_trips(df: pd.DataFrame, macro_stop_s: int) -> list:
    df = df.sort_values("gps_date").reset_index(drop=True)

    dt = df["gps_date"].diff().dt.total_seconds().fillna(0)
    split_at = set([0, len(df)])
    for idx in dt[dt > MACRO_GAP_S].index:
        split_at.add(int(idx))

    stopped = df["gps_speed_kmh"] < STOP_SPEED_KMH
    run_id  = (stopped != stopped.shift()).cumsum()
    for rid, group in df[stopped].groupby(run_id[stopped]):
        if len(group) < 2:
            continue
        t_start = df.loc[group.index[0],  "gps_date"]
        t_end   = df.loc[group.index[-1], "gps_date"]
        if (t_end - t_start).total_seconds() >= macro_stop_s:
            start_idx = int(group.index[0])
            if 0 < start_idx < len(df):
                split_at.add(start_idx)

    trips = []
    for i, j in zip(sorted(split_at), sorted(split_at)[1:]):
        seg = df.iloc[i:j].copy()
        moving = seg["gps_speed_kmh"] >= STOP_SPEED_KMH
        if not moving.any():
            continue
        # Trim leading stationary rows
        first_idx = moving.values.nonzero()[0][0]
        seg = seg.iloc[first_idx:]
        moving = seg["gps_speed_kmh"] >= STOP_SPEED_KMH
        # Trim trailing stationary rows
        last_idx = moving.values.nonzero()[0][-1]
        seg = seg.iloc[:last_idx + 1]
        if len(seg) >= 2:
            trips.append(seg)
    return trips


def macro_median_min(cleaned_files: list, macro_stop_s: int) -> tuple:
    durations = []
    for fp in cleaned_files:
        try:
            df = pd.read_csv(fp, parse_dates=["gps_date"])
            if "gps_speed_kmh" not in df.columns:
                continue
            for macro in split_macro_trips(df, macro_stop_s):
                dur_s = (macro["gps_date"].iloc[-1] -
                         macro["gps_date"].iloc[0]).total_seconds()
                # Only count macro-trips that contain at least one moving block
                if dur_s >= 60:
                    durations.append(dur_s)
        except Exception:
            continue
    if not durations:
        return float("nan"), 0
    return float(np.median(durations)) / 60.0, len(durations)


# ── Sweep ─────────────────────────────────────────────────────────────────────
def main():
    cleaned_files = sorted(glob.glob(os.path.join(CLEANED_DIR, "*.csv")))
    if not cleaned_files:
        print(f"[ERROR] No cleaned files found in {CLEANED_DIR}. Run stage 01 first.")
        sys.exit(1)

    print(f"Sweeping MACRO_STOP_S to find median macro-trip duration ≈ {TARGET_MIN} min")
    print(f"  (Using {len(cleaned_files)} cleaned IMEI files, corrected split logic)\n")
    print(f"  {'T_stop (s)':>12}  {'T_stop (min)':>13}  {'N trips':>8}  {'Median (min)':>13}")
    print(f"  {'-'*52}")

    results = []
    for t in SWEEP_VALUES:
        med, n = macro_median_min(cleaned_files, t)
        print(f"  {t:>12}  {t/60:>13.1f}  {n:>8}  {med:>13.1f}")
        results.append((t, med, n))

    # Find T_stop whose median is closest to TARGET_MIN from below
    below = [(t, m, n) for t, m, n in results if m <= TARGET_MIN]
    above = [(t, m, n) for t, m, n in results if m > TARGET_MIN]

    if below:
        chosen_t, chosen_med, chosen_n = max(below, key=lambda x: x[1])
    else:
        # All medians exceed target — pick the smallest T_stop
        chosen_t, chosen_med, chosen_n = min(results, key=lambda x: x[1])

    print(f"\n  → Chosen: MACRO_STOP_S = {chosen_t}s  (median = {chosen_med:.1f} min, "
          f"n = {chosen_n} macro-trips)")

    # Patch 02_segment_trips.py
    with open(SCRIPT_PATH, "r") as f:
        src = f.read()

    new_src = re.sub(
        r"(MACRO_STOP_S\s*=\s*)\d+(\s+#.*)",
        lambda m: (f"MACRO_STOP_S   = {chosen_t}"
                   f"    # stationary > {chosen_t}s → macro-trip boundary "
                   f"(median macro-trip ≈ {chosen_med:.1f} min)"),
        src
    )

    if new_src == src:
        print(f"  [WARN] Could not patch MACRO_STOP_S in {SCRIPT_PATH} — update manually.")
    else:
        with open(SCRIPT_PATH, "w") as f:
            f.write(new_src)
        print(f"  Patched MACRO_STOP_S = {chosen_t} in 02_segment_trips.py")

    print()


if __name__ == "__main__":
    main()
