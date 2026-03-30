"""
Script 05a — Genetic Algorithm Drive Cycle Knitting
====================================================
Fitness mode: full_mape — SAF SSE + mean MAPE across all scale-invariant
metrics — run N_RUNS=5 times with different random seeds.

DESIGN RATIONALE — metric consistency:
  Only metrics that are genuinely comparable between fleet-wide data and a
  single 30-minute drive cycle are included in the objective.

  COMPARABLE (rate / proportion / distributional):
    mean_speed_kmh, mean_running_speed_kmh, std_speed_kmh
    idle_fraction, n_stops (integer target = round(fleet_rate × 30 min)), p95_running_speed_kmh
    mean_accel_ms2, mean_decel_ms2, rms_accel_ms2, pke_per_km
    pct_0_20, pct_20_40, pct_40_60, pct_60_plus
    gain_per_km, loss_per_km
    + SAF SSE (normalised matrix)

  INFO-ONLY — excluded from fitness (scale depends on dataset size):
    max_speed_kmh, max_accel_ms2, min_accel_ms2  — extreme single events;
      fleet has ~2.5M rows so its max is far higher than any 30-min cycle.
    max_gradient_pct, min_gradient_pct           — same issue.
    total_dist_km, duration_s, elevation_gain_m,
    elevation_loss_m, idle_time_pct              — absolute totals.

GA FITNESS:
  score = saf_sse + mean_MAPE(FITNESS_METRICS)

INITIALISATION — speed-proportional seeding:
  Initial chromosomes are seeded proportionally to the fleet speed-bin
  distribution (pct_0_20, pct_20_40, pct_40_60, pct_60+).

OUTPUT (per mode directory)
------
  05a_ga_run_00.csv  …  05a_ga_run_09.csv
"""

import os
import sys
import glob
import random
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from utils.metrics import (
    build_saf_matrix, saf_sse, kinematic_stats, elevation_stats,
)
from utils.energy_utils import compute_delta_rc, has_sufficient_rc_data

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
MICROTRIP_DIR   = os.path.join(HERE, "output", "02_microtrips")
MACROTRIP_STATS = os.path.join(HERE, "output", "02_macrotrips_stats.csv")
FEATURES_PATH   = os.path.join(HERE, "output", "03_features.csv")

TARGET_DURATION_S = 1800      # 30 minutes
STEP_SEC          = 10.0
MIN_TRIP_ROWS     = 7         # 70 s minimum micro-trip (7 × 10 s steps)

# GA hyperparameters
POP_SIZE       = 1000
GENERATIONS    = 2000
CROSSOVER_PROB = 0.80
MUTATION_PROB  = 0.03
ELITE_FRAC     = 0.05
RANDOM_STATE   = 42

# ── Fitness modes, N_RUNS per mode ────────────────────────────────────────────
N_RUNS = 10     # GA runs; seeds = RANDOM_STATE + 0..9

OUTPUT_DIR = "output_full_mape"

# Scale-invariant metrics included in full_mape fitness.
# ALL of these are comparable between fleet-wide data and a single drive cycle
# because they are rates, proportions, or conditional means — not absolute totals.
FITNESS_METRICS = [
    # Speed (rate-based)
    "mean_speed_kmh",
    "mean_running_speed_kmh",
    "std_speed_kmh",
    # Idle / stop frequency (proportions / count)
    "idle_fraction",
    "n_stops",              # integer target set to round(fleet_rate × 30 min)
    "p95_running_speed_kmh",
    # Acceleration (conditional means / RMS)
    "mean_accel_ms2",
    "mean_decel_ms2",
    "rms_accel_ms2",
    "pke_per_km",
    # Speed-bin distribution (proportions)
    "pct_0_20",
    "pct_20_40",
    "pct_40_60",
    "pct_60_plus",
    # Elevation (rates)
    "gain_per_km",
    "loss_per_km",
    # Energy (rate — Wh per km; comparable between fleet and a single cycle)
    "efficiency_wh_km",
]

# Metrics whose values are already percentages (0–100 scale).
# Error = absolute percentage-point difference, normalised by /100 for fitness.
# Avoids "% error of a %" which blows up on small values like pct_60_plus.
PERCENT_METRICS = frozenset([
    "pct_0_20", "pct_20_40", "pct_40_60", "pct_60_plus",
])

# Metrics on 0–1 fraction scale — multiply by 100 to convert to pp before
# computing absolute difference (same scale as PERCENT_METRICS).
FRACTION_METRICS = frozenset([
    "idle_fraction",
])

# Metrics kept in the evaluation table for reference but NOT comparable between
# fleet and a single drive cycle — excluded from all fitness calculations.
INFO_ONLY_METRICS = frozenset([
    "max_speed_kmh",     # extreme value; scales with dataset size
    "max_accel_ms2",     # extreme value
    "min_accel_ms2",     # extreme value
    "max_gradient_pct",  # extreme value
    "min_gradient_pct",  # extreme value
    "total_dist_km",     # absolute total (fleet >> cycle)
    "duration_s",        # absolute total (always 1800 s for cycle)
    "idle_time_pct",     # redundant: = idle_fraction × 100
    "elevation_gain_m",  # absolute total; gain_per_km already in FITNESS_METRICS
    "elevation_loss_m",  # absolute total
    "pct_accel",         # removed from fitness; partially redundant with SAF marginal
    "pct_decel",
    "pct_cruise",
])


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_trips(microtrip_dir: str) -> list[pd.DataFrame]:
    """Load micro-trip CSVs.  Only trips with measured energy data are retained
    so that efficiency_wh_km can be computed for every assembled cycle."""
    trips = []
    n_skipped_energy = 0
    for fp in sorted(glob.glob(os.path.join(microtrip_dir, "trip_*.csv"))):
        try:
            df = pd.read_csv(fp, parse_dates=["gps_date"])
            if len(df) < MIN_TRIP_ROWS or "gps_speed_kmh" not in df.columns:
                continue
            # Require measured energy data — trips without it cannot contribute
            # to the energy fitness term and are excluded from the GA pool.
            if "energy_wh" not in df.columns or df["energy_wh"].sum() <= 0:
                n_skipped_energy += 1
                continue
            if "trip_id" in df.columns and "source_trip_id" not in df.columns:
                df["source_trip_id"] = df["trip_id"]
            trips.append(df)
        except Exception:
            pass
    if n_skipped_energy:
        print(f"  Skipped {n_skipped_energy} trips with no energy data.")
    return trips


# ─────────────────────────────────────────────────────────────────────────────
# FLEET TARGET STATISTICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_fleet_targets(trips: list[pd.DataFrame]) -> tuple:
    """
    Return (target_saf, target_kine, target_elev, fleet_gain_per_km,
            fleet_loss_per_km).

    Sets target_kine["n_stops"] to the integer number of stops expected in a
    30-minute assembled cycle, derived from macro-trip data:
        fleet_stops_per_min × 30  →  round to nearest integer
    This avoids the fractional-target problem: the assembled cycle can only
    achieve a whole number of stops, so the target must also be an integer.
    """
    all_s, all_a, all_e, all_d = [], [], [], []
    for df in trips:
        all_s.extend(df["gps_speed_kmh"].fillna(0).values)
        all_a.extend(df["acceleration_ms2"].fillna(0).values)
        if "elevation_change_m" in df.columns:
            all_e.extend(df["elevation_change_m"].fillna(0).values)
        else:
            all_e.extend(np.zeros(len(df)))
        if "dist_m" in df.columns:
            all_d.extend(df["dist_m"].fillna(0).values)
        else:
            all_d.extend(np.zeros(len(df)))

    spd, acc, elev, dist = map(np.array, [all_s, all_a, all_e, all_d])
    t_saf  = build_saf_matrix(spd, acc, normalise=True)
    t_kine = kinematic_stats(spd, acc, dt_s=STEP_SEC)
    t_elev = elevation_stats(elev, dist)

    fleet_gain_per_km = t_elev.get("gain_per_km", 0.0)
    fleet_loss_per_km = t_elev.get("loss_per_km", 0.0)

    # ── Fleet stops_per_minute from macro-trip data ───────────────────────────
    # n_stops per macro-trip = n_microtrips - 1 (inter-micro-trip stops only).
    # Total duration from macro-trip stats (includes idle time) makes this
    # directly comparable to assembled cycles (time-domain, not distance-domain).
    if os.path.exists(MACROTRIP_STATS):
        macro_df = pd.read_csv(MACROTRIP_STATS)
        if "n_microtrips" in macro_df.columns:
            # Each micro-trip ends with a stop — so total stops = n_microtrips
            # (not n_microtrips-1; the terminal stop of every micro-trip counts,
            # including single-micro-trip macro-trips which were previously 0).
            total_fleet_stops = int(macro_df["n_microtrips"].sum())
            # duration_s column holds total macro-trip time (moving + idle)
            if "duration_s" in macro_df.columns:
                total_fleet_min = macro_df["duration_s"].sum() / 60.0
            else:
                # Fallback: use distance/speed estimate
                total_fleet_dist_km = sum(
                    t["dist_m"].sum() / 1000.0
                    for t in trips if "dist_m" in t.columns
                )
                # assume 13 km/h average as per fleet mean_speed target
                total_fleet_min = (total_fleet_dist_km / 13.0) * 60.0 if total_fleet_dist_km > 0 else 1.0
            if total_fleet_min > 0:
                fleet_stops_per_min = total_fleet_stops / total_fleet_min
                # Convert to an integer target for the 30-minute assembled cycle.
                # The cycle can only achieve whole-number stop counts, so rounding
                # here eliminates the fractional-target mismatch in the fitness.
                cycle_duration_min = TARGET_DURATION_S / 60.0
                target_n_stops = round(fleet_stops_per_min * cycle_duration_min)
                t_kine["n_stops"] = float(target_n_stops)
                print(f"  Fleet stops/min (macro): {fleet_stops_per_min:.4f}  "
                      f"→  target n_stops (30 min): {target_n_stops}")

    # ── Fleet energy efficiency target ────────────────────────────────────────
    # Use 03_features.csv (IQR-filtered trips) for the target — same source as
    # Stage 07 — so both stages use an identical reference value.
    if os.path.exists(FEATURES_PATH):
        feat_df = pd.read_csv(FEATURES_PATH)
        e_trips = feat_df[feat_df["has_energy"] == True]["efficiency_wh_km"].dropna()
        if len(e_trips) > 0:
            fleet_eff = float(e_trips.median())
            t_kine["efficiency_wh_km"] = fleet_eff
            print(f"  Fleet energy efficiency: {fleet_eff:.2f} Wh/km  "
                  f"(median of {len(e_trips)} IQR-filtered trips)")

    return t_saf, t_kine, t_elev, fleet_gain_per_km, fleet_loss_per_km


# ─────────────────────────────────────────────────────────────────────────────
# CHROMOSOME HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def idle_segment(n_steps: int = 1) -> pd.DataFrame:
    n = max(1, n_steps)
    return pd.DataFrame({
        "gps_speed_kmh":      [0.0] * n,
        "acceleration_ms2":   [0.0] * n,
        "elevation_change_m": [0.0] * n,
        "dist_m":             [0.0] * n,
        "energy_wh":          [0.0] * n,   # no propulsion energy while stationary
        "source_trip_id":     [np.nan] * n,
        "gps_date":           [pd.NaT] * n,
    })


def decode_chromosome(chromosome: list[int],
                      trips: list[pd.DataFrame],
                      target_s: float = TARGET_DURATION_S,
                      idle_frac: float = 0.0,
                      ) -> tuple[pd.DataFrame | None, list]:
    """Decode chromosome → (cycle_df, trips_used).

    idle_frac: fleet macro-trip median idle fraction (0–1).
      Moving time target = target_s × (1 − idle_frac).
      The remaining time is distributed as idle gaps between micro-trips so
      that the assembled cycle reproduces the real-world stop pattern.
    """
    moving_target_s = target_s * (1.0 - idle_frac)
    trip_segs, trips_used, current_moving_s = [], [], 0.0

    for idx in chromosome:
        trip   = trips[idx]
        trip_s = len(trip) * STEP_SEC
        if current_moving_s + trip_s > moving_target_s + STEP_SEC * 3 and trip_segs:
            continue
        cols = ["gps_speed_kmh", "acceleration_ms2",
                "elevation_change_m", "dist_m", "energy_wh",
                "source_trip_id", "gps_date"]
        seg = trip[[c for c in cols if c in trip.columns]].copy()
        trip_segs.append(seg)
        trips_used.append(trip)
        current_moving_s += trip_s
        if current_moving_s >= moving_target_s - STEP_SEC * 3:
            break

    if not trip_segs:
        if chromosome:
            t0   = trips[chromosome[0]]
            cols = ["gps_speed_kmh", "acceleration_ms2",
                    "elevation_change_m", "dist_m", "energy_wh",
                    "source_trip_id", "gps_date"]
            seg = t0[[c for c in cols if c in t0.columns]].copy()
            trip_segs.append(seg)
            trips_used.append(t0)
            current_moving_s = len(t0) * STEP_SEC
        else:
            return None, []

    # Distribute idle time evenly between micro-trips (N−1 interior gaps)
    # plus a trailing pad so the total reaches target_s exactly.
    total_idle_s = max(0.0, target_s - current_moving_s)
    n_gaps       = max(1, len(trip_segs) - 1)
    idle_per_gap = total_idle_s / n_gaps
    idle_steps   = max(1, round(idle_per_gap / STEP_SEC))

    parts = []
    for i, seg in enumerate(trip_segs):
        parts.append(seg)
        if i < len(trip_segs) - 1:
            parts.append(idle_segment(idle_steps))

    result = pd.concat(parts, ignore_index=True)

    # Pad trailing idle to reach exactly target_rows
    target_rows = int(target_s / STEP_SEC)
    if len(result) < target_rows:
        result = pd.concat(
            [result, idle_segment(target_rows - len(result))],
            ignore_index=True
        )

    for col in ["gps_speed_kmh", "acceleration_ms2", "elevation_change_m",
                "dist_m", "energy_wh"]:
        if col not in result.columns:
            result[col] = 0.0
        result[col] = result[col].fillna(0.0)
    # source_trip_id: integer where known, NaN for synthetic idle rows
    if "source_trip_id" not in result.columns:
        result["source_trip_id"] = np.nan
    # gps_date: real timestamp from source micro-trip; NaT for idle rows
    if "gps_date" not in result.columns:
        result["gps_date"] = pd.NaT
    return result, trips_used


# ─────────────────────────────────────────────────────────────────────────────
# FITNESS
# ─────────────────────────────────────────────────────────────────────────────
def fitness(chrom, trips, target_saf, target_kine, target_elev, idle_frac,
            mode: str) -> float:
    """
    Two-mode fitness function.

    saf_only:
        score = saf_sse(fleet_saf, candidate_saf)

    full_mape:
        score = saf_sse + mean_error(FITNESS_METRICS)

        Error per metric:
          PERCENT_METRICS  (pct_*, 0–100 scale): |c - t| / 100
          FRACTION_METRICS (idle_fraction, 0–1): |c - t|
          All others:                            |c - t| / max(|t|, 0.1)
        All terms are dimensionless and in [0,1] range.
        Percentage metrics use absolute pp difference — avoids computing
        "% error of a %" which blows up on small values (e.g. pct_60_plus).
    """
    cand, _ = decode_chromosome(chrom, trips, idle_frac=idle_frac)
    if cand is None:
        return float("inf")

    spd  = cand["gps_speed_kmh"].values
    acc  = cand["acceleration_ms2"].values
    elev = cand["elevation_change_m"].fillna(0).values
    dist = cand["dist_m"].fillna(0).values

    # SAF SSE — always active (normalised matrix → dimensionless)
    cand_saf = build_saf_matrix(spd, acc, normalise=True)
    score    = saf_sse(target_saf, cand_saf)

    if mode == "saf_only":
        return score

    # full_mape: add mean MAPE across all scale-invariant scalar metrics
    ck = kinematic_stats(spd, acc, dt_s=STEP_SEC)
    ce = elevation_stats(elev, dist)

    # Measured energy efficiency — Wh/km from source trip energy_wh column.
    # Only computed over moving rows (dist > 0); idle rows contribute 0 to both
    # numerator and denominator so the result is naturally comparable to the
    # micro-trip-derived fleet target.
    total_wh   = cand["energy_wh"].fillna(0).sum()
    total_dist = dist.sum() / 1000.0
    ck["efficiency_wh_km"] = (total_wh / total_dist) if total_dist > 0 else 0.0

    cand_all = {**ck, **ce}

    # Build combined target: target_kine already has idle-adjusted values;
    # target_elev provides elevation rates; kine takes precedence on overlaps.
    target_lookup = {**target_elev, **target_kine}

    error_terms = []
    for metric in FITNESS_METRICS:
        t_val = target_lookup.get(metric)
        if t_val is None:
            continue
        c_val = cand_all.get(metric, 0.0)
        if metric in PERCENT_METRICS:
            # Absolute percentage-point difference, normalised to [0,1] by /100.
            # Avoids computing "% error of a %" on small values like pct_60_plus.
            error_terms.append(abs(c_val - t_val) / 100.0)
        elif metric in FRACTION_METRICS:
            # Fraction on 0-1 scale → multiply by 100 to get pp, then /100 = abs diff.
            error_terms.append(abs(c_val - t_val))
        else:
            denom = max(abs(t_val), 0.1) + 1e-9
            error_terms.append(abs(c_val - t_val) / denom)

    if error_terms:
        score += float(np.mean(error_terms))

    return score


# ─────────────────────────────────────────────────────────────────────────────
# SPEED-PROPORTIONAL CHROMOSOME SEEDING
# ─────────────────────────────────────────────────────────────────────────────
def build_speed_buckets(trips: list[pd.DataFrame]) -> dict:
    """Pre-build index lists per speed bucket for fast seeding."""
    buckets = {"0_20": [], "20_40": [], "40_60": [], "60plus": []}
    for i, t in enumerate(trips):
        m = t["gps_speed_kmh"].mean()
        if m < 20:
            buckets["0_20"].append(i)
        elif m < 40:
            buckets["20_40"].append(i)
        elif m < 60:
            buckets["40_60"].append(i)
        else:
            buckets["60plus"].append(i)
    return buckets


def seeded_chromosome(trips: list[pd.DataFrame],
                      buckets: dict,
                      target_kine: dict,
                      chrom_len: int) -> list[int]:
    """
    Build a chromosome biased towards the fleet speed-bin distribution.
    Each bucket is sampled proportionally to its target percentage.
    """
    proportions = {
        "0_20":   target_kine.get("pct_0_20",  50) / 100,
        "20_40":  target_kine.get("pct_20_40", 33) / 100,
        "40_60":  target_kine.get("pct_40_60", 11) / 100,
        "60plus": target_kine.get("pct_60_plus", 4) / 100,
    }
    n_trips = len(trips)
    chrom   = []

    for bname, frac in proportions.items():
        n_draw = max(1, int(round(frac * chrom_len)))
        pool   = buckets.get(bname, [])
        if pool:
            chrom.extend(random.choices(pool, k=n_draw))
        else:
            chrom.extend(random.choices(range(n_trips), k=n_draw))

    # Pad / trim to exact length, then shuffle
    while len(chrom) < chrom_len:
        chrom.append(random.randint(0, n_trips - 1))
    random.shuffle(chrom)
    return chrom[:chrom_len]


# ─────────────────────────────────────────────────────────────────────────────
# GENETIC ALGORITHM
# ─────────────────────────────────────────────────────────────────────────────
def run_ga(trips: list[pd.DataFrame],
           target_saf,
           target_kine: dict,
           target_elev: dict,
           idle_frac: float = 0.0,
           mode: str = "saf_only",
           label: str = "",
           random_state: int = RANDOM_STATE) -> list[int]:
    """Run GA over `trips` and return the best chromosome."""
    if not trips:
        return []

    random.seed(random_state)
    np.random.seed(random_state)

    n_trips   = len(trips)
    chrom_len = 30
    elite_n   = max(1, int(POP_SIZE * ELITE_FRAC))
    buckets   = build_speed_buckets(trips)

    # Seed population proportionally; half random for diversity
    half = POP_SIZE // 2
    population  = [seeded_chromosome(trips, buckets, target_kine, chrom_len)
                   for _ in range(half)]
    population += [[random.randint(0, n_trips - 1) for _ in range(chrom_len)]
                   for _ in range(POP_SIZE - half)]

    best_score = float("inf")
    best_chrom = population[0]
    no_improve = 0

    tag = f"[{label}] " if label else ""
    print(f"  {tag}GA: pop={POP_SIZE}, gen={GENERATIONS}, trips={n_trips}, "
          f"mode={mode}")

    for gen in range(GENERATIONS):
        scores = [fitness(c, trips, target_saf, target_kine, target_elev,
                          idle_frac, mode)
                  for c in population]

        min_i = int(np.argmin(scores))
        if scores[min_i] < best_score:
            best_score = scores[min_i]
            best_chrom = population[min_i][:]
            no_improve = 0
        else:
            no_improve += 1

        if gen % 100 == 0 or gen == GENERATIONS - 1:
            print(f"    Gen {gen:4d} | best = {best_score:.6f}")

        if no_improve >= 300:
            print(f"    Early stop at gen {gen} (300 gens no improvement)")
            break

        ranked  = sorted(range(POP_SIZE), key=lambda i: scores[i])
        new_pop = [population[ranked[i]][:] for i in range(elite_n)]

        while len(new_pop) < POP_SIZE:
            def tournament():
                c = random.sample(range(POP_SIZE), 4)
                return population[min(c, key=lambda i: scores[i])]
            p1, p2 = tournament(), tournament()
            if random.random() < CROSSOVER_PROB:
                cp    = random.randint(1, chrom_len - 2)
                child = p1[:cp] + p2[cp:]
            else:
                child = p1[:]
            for j in range(chrom_len):
                if random.random() < MUTATION_PROB:
                    child[j] = random.randint(0, n_trips - 1)
            new_pop.append(child)

        population = new_pop

    return best_chrom


# ─────────────────────────────────────────────────────────────────────────────
# BUILD CYCLE
# ─────────────────────────────────────────────────────────────────────────────
def build_cycle(chromosome, trips, idle_frac=0.0):
    cand, trips_used = decode_chromosome(chromosome, trips, idle_frac=idle_frac)
    if cand is None:
        return None, []
    target_rows = int(TARGET_DURATION_S / STEP_SEC)
    cand = cand.iloc[:target_rows].copy().reset_index(drop=True)
    cand["time_s"] = np.arange(len(cand)) * STEP_SEC
    return cand, trips_used


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(os.path.join(HERE, "output"), exist_ok=True)

    # ── Load trips ────────────────────────────────────────────────────────────
    print("Loading micro-trips...")
    trips = load_trips(MICROTRIP_DIR)
    if not trips:
        print("[ERROR] No trips loaded. Run Stage 02 first.")
        return
    print(f"  Loaded {len(trips)} micro-trips.")

    # ── Fleet-wide targets ────────────────────────────────────────────────────
    print("\nComputing fleet-wide targets (SAF + kinematic + elevation)...")
    target_saf, target_kine, target_elev, fleet_gain_per_km, fleet_loss_per_km = \
        compute_fleet_targets(trips)
    print(f"  Fleet elevation gain/km : {fleet_gain_per_km:.2f} m/km")
    print(f"  Fleet elevation loss/km : {fleet_loss_per_km:.2f} m/km")

    # ── Override idle_fraction with macro-trip fleet median ───────────────────
    fleet_idle_frac = 0.0
    if os.path.exists(MACROTRIP_STATS):
        macro_df = pd.read_csv(MACROTRIP_STATS)
        fleet_idle_frac = float(macro_df["idle_fraction"].median())
        target_kine["idle_fraction"] = fleet_idle_frac
        target_kine["idle_time_pct"] = fleet_idle_frac * 100
        print(f"  Fleet idle frac  : {fleet_idle_frac:.3f} "
              f"({fleet_idle_frac*100:.1f}%)  ← from macro-trip stats")

        moving_frac = 1.0 - fleet_idle_frac
        base_mean_speed = target_kine.get("mean_speed_kmh", 23.5)
        target_kine["mean_speed_kmh"] = base_mean_speed * moving_frac

        for key in ["pct_20_40", "pct_40_60", "pct_60_plus"]:
            target_kine[key] = target_kine.get(key, 0) * moving_frac
        target_kine["pct_0_20"] = (target_kine.get("pct_0_20", 50) * moving_frac
                                   + fleet_idle_frac * 100)

        target_saf = target_saf * moving_frac
        target_saf[0, 3] += fleet_idle_frac
        if target_saf.sum() > 0:
            target_saf = target_saf / target_saf.sum()

        # RMS acceleration adjustment for idle fraction.
        # Idle rows (a=0) reduce the cycle RMS relative to micro-trip fleet RMS:
        #   RMS_cycle = RMS_micro × sqrt(n_moving / n_total) = RMS_micro × sqrt(1 - f_idle)
        # Without this adjustment the target is systematically too high, inflating
        # the RMS error term in the fitness function.
        target_kine["rms_accel_ms2"] = (
            target_kine.get("rms_accel_ms2", 0.0) * np.sqrt(moving_frac)
        )

        print(f"  Adjusted mean_speed target: {target_kine['mean_speed_kmh']:.1f} km/h")
        print(f"  Adjusted pct_0_20 target  : {target_kine['pct_0_20']:.1f}%")
        print(f"  Adjusted rms_accel target : {target_kine['rms_accel_ms2']:.4f} m/s²")
    else:
        print("  Fleet idle frac  : 0.000 (macro stats not found — run Stage 02)")

    print(f"  Fleet pct_0_20   : {target_kine.get('pct_0_20',0):.1f}%")
    print(f"  Fleet pct_20_40  : {target_kine.get('pct_20_40',0):.1f}%")
    print(f"  Fleet pct_40_60  : {target_kine.get('pct_40_60',0):.1f}%")
    print(f"  Fleet pct_60+    : {target_kine.get('pct_60_plus',0):.1f}%")
    print(f"  Target n_stops   : {int(target_kine.get('n_stops',0))} stops in 30 min")
    print(f"  Fleet p95 run spd: {target_kine.get('p95_running_speed_kmh',0):.1f} km/h")

    # ── N_RUNS loop ───────────────────────────────────────────────────────────
    out_dir = os.path.join(HERE, OUTPUT_DIR)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"Running {N_RUNS} GA runs  →  {OUTPUT_DIR}/")
    print(f"  FITNESS_METRICS ({len(FITNESS_METRICS)}): {', '.join(FITNESS_METRICS)}")
    print(f"{'='*65}")

    for run_i in range(N_RUNS):
        seed = RANDOM_STATE + run_i
        print(f"\n  Run {run_i:02d}  (seed={seed})")

        best_chrom = run_ga(
            trips, target_saf, target_kine, target_elev,
            idle_frac=fleet_idle_frac,
            mode="full_mape",
            label=f"run{run_i:02d}",
            random_state=seed,
        )

        cycle_df, _ = build_cycle(best_chrom, trips, fleet_idle_frac)
        if cycle_df is None:
            print("     [WARN] No cycle produced — skipping.")
            continue

        out_path = os.path.join(out_dir, f"05a_ga_run_{run_i:02d}.csv")
        cycle_df.to_csv(out_path, index=False)

        e_stats = elevation_stats(
            cycle_df["elevation_change_m"].fillna(0).values,
            cycle_df["dist_m"].fillna(0).values,
        )
        score = fitness(best_chrom, trips, target_saf, target_kine,
                        target_elev, fleet_idle_frac, "full_mape")
        ck = kinematic_stats(cycle_df["gps_speed_kmh"].values,
                             cycle_df["acceleration_ms2"].values, dt_s=STEP_SEC)
        cycle_eff = ck.get("efficiency_wh_km", np.nan)
        print(f"     score={score:.5f}  "
              f"gain/km={e_stats['gain_per_km']:.2f}  "
              f"mean_spd={ck['mean_speed_kmh']:.1f}  "
              f"idle={ck['idle_fraction']:.3f}  "
              f"eff={cycle_eff:.1f} Wh/km")
        print(f"     → {out_path}")

    n_files = len(glob.glob(os.path.join(out_dir, "05a_ga_run_*.csv")))
    print(f"\n{'='*65}")
    print(f"STAGE 05a COMPLETE — {n_files} cycles in {out_dir}/")
    print("="*65)


if __name__ == "__main__":
    main()
