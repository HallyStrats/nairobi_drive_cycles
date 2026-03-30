"""
Script 07 — Final Evaluation & Drive Cycle Dashboard
=====================================================
Evaluates the N_RUNS=3 full_mape GA drive cycles against fleet-wide targets
and produces a comprehensive visual report.

Outputs (saved in output_full_mape/):
  07_run_reports.csv              — one row per run, all metrics
  07_summary_report.csv           — mean ± std vs fleet target, In_Fitness flag
  07_dashboard.png                — 3-row × 3-col: speed / elevation / error%
  07_saf_comparison.png           — fleet SAF + each cycle's SAF (1 row × 4 panels)
  07_sef_cycles.png               — fleet SEF + each cycle's SEF (1 row × 4 panels)
  07_ga_run_XX_speed_ramped.csv   — micro-trips re-ordered by ascending max speed
"""

import os
import sys
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from utils.metrics import (
    build_saf_matrix, saf_sse, saf_mae, saf_scalar_stats,
    build_sef_matrix, DELTA_V_BINS_KMH, DELTA_ELEV_BINS_M,
    kinematic_stats, elevation_stats,
    SPEED_BINS_KMH, ACCEL_BINS_MS2,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
MICROTRIP_DIR   = os.path.join(HERE, "output", "02_microtrips")
MACROTRIP_STATS = os.path.join(HERE, "output", "02_macrotrips_stats.csv")

STEP_SEC = 10.0

OUTPUT_DIR  = "output_full_mape"
N_RUNS      = 10
MODE_LABEL  = "Full-MAPE"
MODE_COLOR  = "#e74c3c"

# Scale-invariant metrics used in the full_mape fitness function.
# Must match FITNESS_METRICS in Stage 05a.
FITNESS_METRICS = [
    "mean_speed_kmh", "mean_running_speed_kmh", "std_speed_kmh",
    "idle_fraction", "n_stops", "p95_running_speed_kmh",
    "mean_accel_ms2", "mean_decel_ms2", "rms_accel_ms2", "pke_per_km",
    "pct_0_20", "pct_20_40", "pct_40_60", "pct_60_plus",
    "gain_per_km", "loss_per_km",
    "efficiency_wh_km",
]

# Short display names for bar charts
METRIC_SHORT = {
    "mean_speed_kmh":         "mean spd",
    "mean_running_speed_kmh": "run spd",
    "std_speed_kmh":          "std spd",
    "idle_fraction":          "idle frac",
    "n_stops":                "n stops",
    "stops_per_minute":       "stops/min",  # informational only
    "p95_running_speed_kmh":  "p95 run spd",
    "mean_accel_ms2":         "mean accel",
    "mean_decel_ms2":         "mean decel",
    "rms_accel_ms2":          "rms accel",
    "pke_per_km":             "PKE/km",
    "pct_0_20":               "pct 0-20",
    "pct_20_40":              "pct 20-40",
    "pct_40_60":              "pct 40-60",
    "pct_60_plus":            "pct 60+",
    "gain_per_km":            "gain/km",
    "loss_per_km":            "loss/km",
    "saf_sse":                "SAF SSE ×100",
    "efficiency_wh_km":       "eff Wh/km",
}

# Metrics reported but excluded from fitness (scale issues or redundant).
INFO_ONLY_METRICS = frozenset([
    "max_speed_kmh", "max_accel_ms2", "min_accel_ms2",
    "max_gradient_pct", "min_gradient_pct",
    "total_dist_km", "duration_s", "idle_time_pct",
    "elevation_gain_m", "elevation_loss_m",
    "pct_accel", "pct_decel", "pct_cruise",
    "stops_per_minute",   # informational; n_stops (integer count) is the fitness metric
])

# Percentage-type metrics: error = absolute pp difference (not relative %).
# Must stay in sync with PERCENT_METRICS / FRACTION_METRICS in Stage 05a.
PERCENT_METRICS  = frozenset(["pct_0_20", "pct_20_40", "pct_40_60", "pct_60_plus"])
FRACTION_METRICS = frozenset(["idle_fraction"])

# Run colours (5 runs)
RUN_COLORS = ["#2980b9", "#27ae60", "#e74c3c", "#8e44ad", "#f39c12"]


# ─────────────────────────────────────────────────────────────────────────────
# SPEED-RAMP POST-PROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def ramp_by_max_speed(cycle_df: pd.DataFrame) -> pd.DataFrame:
    """
    Re-order micro-trip segments within an assembled cycle by ascending
    maximum speed, producing a cycle that ramps up in speed over time.

    The cycle is split into alternating trip / idle segments using the
    source_trip_id column (NaN = synthetic idle row).  Trip segments are
    sorted by their per-segment max gps_speed_kmh (ascending), then
    interleaved back with the idle gaps in their original positions.
    time_s is reset to a uniform 10-s grid.

    Returns a new DataFrame; the input is not modified.
    """
    if "source_trip_id" not in cycle_df.columns:
        return cycle_df.copy()

    is_idle  = cycle_df["source_trip_id"].isna().values
    segments = []          # list of ("trip"|"idle", DataFrame)
    seg_start = 0
    for i in range(1, len(cycle_df)):
        if is_idle[i] != is_idle[i - 1]:
            kind = "idle" if is_idle[seg_start] else "trip"
            segments.append((kind, cycle_df.iloc[seg_start:i].copy()))
            seg_start = i
    kind = "idle" if is_idle[seg_start] else "trip"
    segments.append((kind, cycle_df.iloc[seg_start:].copy()))

    trip_segs = [df for kind, df in segments if kind == "trip"]
    idle_segs = [df for kind, df in segments if kind == "idle"]

    # Sort trips by ascending max speed
    trip_segs_sorted = sorted(trip_segs,
                              key=lambda df: df["gps_speed_kmh"].max())

    # Rebuild in original structural order, substituting sorted trips
    parts      = []
    trip_iter  = iter(trip_segs_sorted)
    idle_iter  = iter(idle_segs)
    for kind, _ in segments:
        if kind == "trip":
            parts.append(next(trip_iter))
        else:
            parts.append(next(idle_iter))

    result = pd.concat(parts, ignore_index=True)
    result["time_s"] = np.arange(len(result)) * STEP_SEC
    return result


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def load_cycle(primary: str, fallback: str) -> pd.DataFrame | None:
    for path in [primary, fallback]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            if len(df) > 10:
                return df
    return None


def fleet_arrays(microtrip_dir: str):
    all_s, all_a, all_e, all_d = [], [], [], []
    for fp in sorted(glob.glob(os.path.join(microtrip_dir, "trip_*.csv"))):
        try:
            df = pd.read_csv(fp)
            if "gps_speed_kmh" not in df.columns:
                continue
            all_s.extend(df["gps_speed_kmh"].fillna(0).values)
            all_a.extend(df["acceleration_ms2"].fillna(0).values)
            all_e.extend(df["elevation_change_m"].fillna(0).values
                         if "elevation_change_m" in df.columns else np.zeros(len(df)))
            all_d.extend(df["dist_m"].fillna(0).values
                         if "dist_m" in df.columns else np.zeros(len(df)))
        except Exception:
            pass
    return map(np.array, [all_s, all_a, all_e, all_d])


def cycle_error_pct(cycle_df: pd.DataFrame,
                    target_kine: dict, target_elev: dict,
                    target_saf: np.ndarray = None) -> dict:
    """
    Return {metric: error} for all FITNESS_METRICS plus SAF SSE.

    Error units:
      "saf_sse"        — saf_sse × 100  (absolute, scaled for chart visibility)
      PERCENT_METRICS  — absolute percentage-point difference (pp)
      FRACTION_METRICS — absolute difference × 100 (pp)
      All others       — relative % error (MAPE × 100)

    SAF SSE is included with ×100 scaling so it sits on a comparable visual
    scale to the scalar errors (typical cycle SAF SSE 0.01–0.05 → 1–5 on chart).
    The "weighted mean" displayed on the dashboard reflects the fitness
    structure: fitness = SAF_SSE + mean(scalar errors), so SAF carries the
    same weight as all scalars combined → weighted_mean = (saf_bar + scalar_mean) / 2.
    """
    spd  = cycle_df["gps_speed_kmh"].values
    acc  = cycle_df["acceleration_ms2"].values
    elev = cycle_df["elevation_change_m"].fillna(0).values
    dist = cycle_df["dist_m"].fillna(0).values
    ck   = kinematic_stats(spd, acc, dt_s=STEP_SEC)
    ce   = elevation_stats(elev, dist)

    # Measured energy efficiency from energy_wh column (carried from source trips)
    if "energy_wh" in cycle_df.columns:
        total_wh   = cycle_df["energy_wh"].fillna(0).sum()
        total_dist = dist.sum() / 1000.0
        ck["efficiency_wh_km"] = (total_wh / total_dist) if total_dist > 0 else 0.0

    cand = {**ck, **ce}
    tgt  = {**target_elev, **target_kine}

    errs = {}

    # SAF SSE — multiplied by 100 so it lands on same visual scale as scalar errors
    if target_saf is not None:
        cand_saf = build_saf_matrix(spd, acc, normalise=True)
        errs["saf_sse"] = saf_sse(target_saf, cand_saf) * 100.0

    for m in FITNESS_METRICS:
        t = tgt.get(m)
        c = cand.get(m)
        if t is None or c is None:
            continue
        if m in PERCENT_METRICS:
            errs[m] = abs(c - t)                              # percentage points
        elif m in FRACTION_METRICS:
            errs[m] = abs(c - t) * 100                        # convert to pp
        else:
            errs[m] = abs(c - t) / (max(abs(t), 0.1) + 1e-9) * 100  # relative %
    return errs


# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD — speed / elevation / error% for all N cycles
# ─────────────────────────────────────────────────────────────────────────────
def plot_dashboard(cycles: list, labels: list, saf_sse_vals: list,
                   baseline_wh_km: float, path: str, title: str,
                   target_kine: dict, target_elev: dict,
                   target_saf: np.ndarray = None):
    """
    3 rows × N cols:
      Row 1 — Speed time-series
      Row 2 — Cumulative elevation profile
      Row 3 — Error bar chart: SAF SSE (×100) + all scalar fitness metrics

    Bar chart title shows two summary numbers:
      Mean         — unweighted mean of all bars
      Wtd (fitness) — fitness-weighted mean: (saf_bar + mean(scalar_bars)) / 2
                      This reflects the actual fitness structure where SAF_SSE
                      and mean(scalar errors) each contribute equally.
    """
    n = len(cycles)
    fig = plt.figure(figsize=(5.5 * n, 18))
    gs  = gridspec.GridSpec(4, n, figure=fig, hspace=0.5, wspace=0.35)

    for col, (label, cycle, sse) in enumerate(zip(labels, cycles, saf_sse_vals)):
        color = RUN_COLORS[col % len(RUN_COLORS)]
        t = cycle["time_s"].values

        # ── Row 1: Speed profile ──────────────────────────────────────────────
        ax0 = fig.add_subplot(gs[0, col])
        ax0.plot(t, cycle["gps_speed_kmh"], color=color, lw=1.3)
        ax0.fill_between(t, cycle["gps_speed_kmh"], alpha=0.10, color=color)

        ck     = kinematic_stats(cycle["gps_speed_kmh"].values,
                                 cycle["acceleration_ms2"].values, dt_s=STEP_SEC)
        mean_e = cycle_error_pct(cycle, target_kine, target_elev, target_saf)

        # Weighted mean: fitness = SAF_SSE + mean(scalars) → each half counts equally
        saf_bar     = mean_e.get("saf_sse", np.nan)
        scalar_bars = [v for k, v in mean_e.items() if k != "saf_sse"]
        scalar_mean = float(np.mean(scalar_bars)) if scalar_bars else np.nan
        overall     = float(np.mean(list(mean_e.values()))) if mean_e else np.nan
        weighted    = (saf_bar + scalar_mean) / 2.0 if not (np.isnan(saf_bar) or np.isnan(scalar_mean)) else overall

        ax0.set_title(
            f"{label}\nMean={overall:.1f}  Wtd={weighted:.1f}",
            fontsize=9, fontweight="bold",
        )
        ax0.set_ylabel("Speed (km/h)", fontsize=9)
        ax0.set_xlabel("Time (s)", fontsize=8)
        ax0.set_xlim(0, t[-1])
        ax0.grid(True, alpha=0.3)

        # Target mean speed line
        t_spd = target_kine.get("mean_speed_kmh", np.nan)
        if not np.isnan(t_spd):
            ax0.axhline(t_spd, color="red", lw=0.8, ls="--", alpha=0.7,
                        label=f"target {t_spd:.1f}")
            ax0.legend(fontsize=7, loc="upper right")

        # ── Row 2: Elevation profile ──────────────────────────────────────────
        ax1 = fig.add_subplot(gs[1, col])
        dist_km  = np.cumsum(cycle["dist_m"].fillna(0).values) / 1000.0
        cum_elev = np.cumsum(cycle["elevation_change_m"].fillna(0).values)
        ax1.plot(dist_km, cum_elev, color="#8e44ad", lw=1.5)
        ax1.fill_between(dist_km, cum_elev, alpha=0.08, color="#8e44ad")

        ce = elevation_stats(cycle["elevation_change_m"].fillna(0).values,
                             cycle["dist_m"].fillna(0).values)
        g_km = ce.get("gain_per_km", 0.0)
        l_km = ce.get("loss_per_km", 0.0)
        t_gain = target_elev.get("gain_per_km", 0.0)

        # Fleet gain/km slope reference
        ref_e = dist_km * t_gain
        ax1.plot(dist_km, ref_e, "r--", lw=0.9, alpha=0.7,
                 label=f"fleet {t_gain:.1f} m/km")
        ax1.set_title(f"gain={g_km:.1f}  loss={l_km:.1f} m/km", fontsize=8)
        ax1.set_ylabel("Cum. elevation (m)", fontsize=9)
        ax1.set_xlabel("Distance (km)", fontsize=8)
        ax1.axhline(0, color="black", lw=0.5, ls="--")
        ax1.legend(fontsize=7, loc="upper left")
        ax1.grid(True, alpha=0.3)

        # ── Row 3: Error % bar chart ──────────────────────────────────────────
        ax2 = fig.add_subplot(gs[2, col])

        if mean_e:
            pairs   = sorted(mean_e.items(), key=lambda x: x[1], reverse=True)
            names   = [METRIC_SHORT.get(m, m) for m, _ in pairs]
            vals    = [v for _, v in pairs]
            bar_colors = [
                "#e74c3c" if v > 20 else "#f39c12" if v > 10 else "#27ae60"
                for v in vals
            ]
            y_pos = range(len(names))
            ax2.barh(list(y_pos), vals, color=bar_colors, alpha=0.8, edgecolor="white")
            ax2.set_yticks(list(y_pos))
            ax2.set_yticklabels(names, fontsize=6.5)
            ax2.axvline(10, color="black", lw=0.6, ls="--", alpha=0.5)
            ax2.axvline(20, color="red",   lw=0.6, ls="--", alpha=0.4)
            ax2.set_xlabel("Error  (SAF ×100 | pp for %s | rel% others)", fontsize=7)
            ax2.set_title(f"Mean {overall:.1f}  |  Wtd {weighted:.1f}",
                          fontsize=9, fontweight="bold")
        ax2.grid(True, alpha=0.2, axis="x")

        # ── Row 4: Energy consumption over time ───────────────────────────────
        ax3 = fig.add_subplot(gs[3, col])

        fleet_eff = target_kine.get("efficiency_wh_km", np.nan)

        # Measured energy: cumulative Wh / cumulative km = running average Wh/km
        has_measured = "energy_wh" in cycle.columns and cycle["energy_wh"].fillna(0).sum() > 0
        if has_measured:
            cum_wh   = np.cumsum(cycle["energy_wh"].fillna(0).values)
            cum_km   = np.cumsum(cycle["dist_m"].fillna(0).values) / 1000.0
            # Running average — only plot where we have meaningful distance
            with np.errstate(divide="ignore", invalid="ignore"):
                running_eff = np.where(cum_km > 0.05, cum_wh / cum_km, np.nan)
            meas_avg = float(cum_wh[-1] / cum_km[-1]) if cum_km[-1] > 0 else np.nan
            ax3.plot(t, running_eff, color="#2980b9", lw=1.3, label=f"Measured (avg {meas_avg:.1f})")
            ax3.axhline(meas_avg, color="#2980b9", lw=0.8, ls=":", alpha=0.7)

        # Predicted energy from Stage 06 Ridge model (per-row predicted_wh_km)
        has_predicted = "predicted_wh_km" in cycle.columns and cycle["predicted_wh_km"].notna().any()
        if has_predicted:
            pred = cycle["predicted_wh_km"].values.astype(float)
            # Smooth with 30-step rolling mean for readability
            pred_series = pd.Series(pred).rolling(30, min_periods=1, center=True).mean().values
            pred_avg = float(np.nanmean(pred[pred > 0])) if np.any(pred > 0) else np.nan
            ax3.plot(t, pred_series, color="#e67e22", lw=1.3, ls="--",
                     label=f"Predicted (avg {pred_avg:.1f})")
            ax3.axhline(pred_avg, color="#e67e22", lw=0.8, ls=":", alpha=0.7)

        if not np.isnan(fleet_eff):
            ax3.axhline(fleet_eff, color="red", lw=1.0, ls="--", alpha=0.8,
                        label=f"Fleet target {fleet_eff:.1f}")

        if not has_measured and not has_predicted:
            ax3.text(0.5, 0.5, "Energy data not available\n(run Stage 06)",
                     ha="center", va="center", transform=ax3.transAxes,
                     fontsize=8, color="grey")

        ax3.set_ylabel("Wh/km", fontsize=9)
        ax3.set_xlabel("Time (s)", fontsize=8)
        ax3.set_xlim(0, t[-1])
        ax3.set_ylim(bottom=0)
        ax3.legend(fontsize=7, loc="upper right")
        ax3.set_title("Energy Consumption", fontsize=9)
        ax3.grid(True, alpha=0.3)

    suptitle = f"Drive Cycle Dashboard — {title}"
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# SAF COMPARISON — fleet + each cycle individually
# ─────────────────────────────────────────────────────────────────────────────
def plot_saf_comparison(target_saf, cycle_safs: list, labels: list, path: str):
    """1 row: Fleet SAF | cycle SAF for each of N cycles."""
    all_safs   = [target_saf] + cycle_safs
    all_labels = ["Fleet"] + labels
    n = len(all_safs)

    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5))
    if n == 1:
        axes = [axes]

    vmax = max(s.max() for s in all_safs) if all_safs else 0.1
    for ax, saf, ttl in zip(axes, all_safs, all_labels):
        im = ax.imshow(saf.T, origin="lower", aspect="auto",
                       cmap="YlOrRd", vmin=0, vmax=vmax)
        ax.set_xticks(range(len(SPEED_BINS_KMH) - 1))
        ax.set_xticklabels([f"{int(b)}" for b in SPEED_BINS_KMH[:-1]],
                           rotation=45, fontsize=7)
        ax.set_yticks(range(len(ACCEL_BINS_MS2) - 1))
        ax.set_yticklabels([f"{b:.1f}" for b in ACCEL_BINS_MS2[:-1]], fontsize=7)
        ax.set_xlabel("Speed (km/h)", fontsize=8)
        ax.set_ylabel("Accel (m/s²)", fontsize=8)
        ax.set_title(ttl, fontsize=10, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("SAF Matrix: Fleet vs Drive Cycles", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# SEF COMPARISON — fleet + each cycle individually (Δv × Δelev heatmaps)
# ─────────────────────────────────────────────────────────────────────────────
def _sef_axis(ax, mat, title, vmax):
    im = ax.imshow(mat.T, origin="lower", aspect="auto",
                   cmap="YlOrRd", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(DELTA_V_BINS_KMH) - 1))
    ax.set_xticklabels(
        [("-∞" if np.isinf(b) and b < 0 else ("+∞" if np.isinf(b) else f"{b:.0f}"))
         for b in DELTA_V_BINS_KMH[:-1]],
        rotation=45, fontsize=6,
    )
    ax.set_yticks(range(len(DELTA_ELEV_BINS_M) - 1))
    ax.set_yticklabels(
        [("-∞" if np.isinf(b) and b < 0 else ("+∞" if np.isinf(b) else f"{b:.1f}"))
         for b in DELTA_ELEV_BINS_M[:-1]],
        fontsize=6,
    )
    ax.set_xlabel("Δv (km/h)", fontsize=7)
    ax.set_ylabel("Δelev (m/step)", fontsize=7)
    ax.set_title(title, fontsize=9, fontweight="bold")
    return im


def plot_sef_all_cycles(fleet_sef: np.ndarray,
                        cycle_sefs: list, labels: list, path: str):
    """1 row: Fleet SEF | individual cycle SEF for each of N cycles."""
    all_sefs   = [fleet_sef] + cycle_sefs
    all_labels = ["Fleet"] + labels
    n = len(all_sefs)

    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5))
    if n == 1:
        axes = [axes]

    vmax = max(s.max() for s in all_sefs) if all_sefs else 0.1
    for ax, mat, ttl in zip(axes, all_sefs, all_labels):
        im = _sef_axis(ax, mat, ttl, vmax)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("SEF Matrix: Fleet vs Drive Cycles  (Δv × Δelev frequency)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)



# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(os.path.join(HERE, "output"), exist_ok=True)

    # ── Fleet targets ─────────────────────────────────────────────────────────
    print("Loading fleet data for targets...")
    f_spd, f_acc, f_elv, f_dst = fleet_arrays(MICROTRIP_DIR)
    if len(f_spd) == 0:
        print("[ERROR] No micro-trip data found. Run Stage 02 first.")
        return

    target_saf  = build_saf_matrix(f_spd, f_acc, normalise=True)
    target_kine = kinematic_stats(f_spd, f_acc, dt_s=STEP_SEC)
    target_elev = elevation_stats(f_elv, f_dst)
    fleet_sef   = build_sef_matrix(f_spd, f_elv, normalise=True)

    # ── Idle fraction adjustment (same as Stage 05a) ──────────────────────────
    fleet_idle_frac = 0.0
    if os.path.exists(MACROTRIP_STATS):
        macro_df = pd.read_csv(MACROTRIP_STATS)
        fleet_idle_frac = float(macro_df["idle_fraction"].median())
        moving_frac = 1.0 - fleet_idle_frac
        target_kine["idle_fraction"] = fleet_idle_frac
        target_kine["idle_time_pct"] = fleet_idle_frac * 100
        target_kine["mean_speed_kmh"] = target_kine.get("mean_speed_kmh", 23.5) * moving_frac
        for key in ["pct_20_40", "pct_40_60", "pct_60_plus"]:
            target_kine[key] = target_kine.get(key, 0) * moving_frac
        target_kine["pct_0_20"] = (target_kine.get("pct_0_20", 50) * moving_frac
                                   + fleet_idle_frac * 100)
        target_saf = target_saf * moving_frac
        target_saf[0, 3] += fleet_idle_frac
        if target_saf.sum() > 0:
            target_saf = target_saf / target_saf.sum()
        print(f"  Fleet idle fraction: {fleet_idle_frac:.3f} ({fleet_idle_frac*100:.1f}%)")

        # RMS acceleration adjustment (mirrors Stage 05a logic).
        # Idle rows (a=0) reduce cycle RMS vs micro-trip RMS:
        #   RMS_cycle = RMS_micro × sqrt(1 − f_idle)
        target_kine["rms_accel_ms2"] = (
            target_kine.get("rms_accel_ms2", 0.0) * np.sqrt(moving_frac)
        )

        # n_stops target (integer count for 30-min assembled cycle).
        # Replaces stops_per_minute in the fitness: the cycle can only achieve
        # a whole number of stops, so the target must also be an integer.
        if "n_microtrips" in macro_df.columns:
            total_stops = int(macro_df["n_microtrips"].sum())
            if "duration_s" in macro_df.columns:
                total_min = macro_df["duration_s"].sum() / 60.0
            else:
                total_km = target_kine.get("total_dist_km", 1.0)
                total_min = (total_km / 13.0) * 60.0 if total_km > 0 else 1.0
            if total_min > 0:
                fleet_stops_per_min = total_stops / total_min
                target_n_stops = round(fleet_stops_per_min * 30.0)  # 30-min cycle
                target_kine["n_stops"] = float(target_n_stops)
                target_kine["stops_per_minute"] = fleet_stops_per_min  # informational
                print(f"  Fleet stops/min: {fleet_stops_per_min:.4f}  "
                      f"→  target n_stops: {target_n_stops}")

    # Fleet energy baseline — from 03_features (IQR-filtered trips with energy)
    baseline_wh_km = np.nan
    features_path = os.path.join(HERE, "output", "03_features.csv")
    if os.path.exists(features_path):
        feat_df = pd.read_csv(features_path)
        e_trips = feat_df[feat_df["has_energy"] == True]["efficiency_wh_km"].dropna()
        if len(e_trips) > 0:
            baseline_wh_km = float(e_trips.median())
            target_kine["efficiency_wh_km"] = baseline_wh_km
            print(f"  Fleet efficiency target: {baseline_wh_km:.2f} Wh/km "
                  f"(median of {len(e_trips)} trips)")
    # Fallback to 06_energy_baseline.csv
    if np.isnan(baseline_wh_km):
        baseline_path = os.path.join(HERE, "output", "06_energy_baseline.csv")
        if os.path.exists(baseline_path):
            bdf = pd.read_csv(baseline_path)
            if "median_wh_km" in bdf.columns:
                baseline_wh_km = float(bdf["median_wh_km"].iloc[0])
                target_kine["efficiency_wh_km"] = baseline_wh_km

    # Combined target dict for summary report
    target_saf_scalars = {"saf_sse": 0.0, "saf_mae": 0.0}
    target_sef_scalars = {"sef_sse": 0.0, "sef_mae": 0.0}
    target_all = {**target_kine, **target_elev,
                  **target_saf_scalars, **target_sef_scalars}

    print(f"\n{'='*65}")
    print(f"Evaluating {N_RUNS} runs in {OUTPUT_DIR}/")
    print(f"{'='*65}")

    out_dir = os.path.join(HERE, OUTPUT_DIR)
    if not os.path.isdir(out_dir):
        print(f"\n[SKIP] {OUTPUT_DIR}/ not found — run Stage 05a first.")
        return

    label = MODE_LABEL
    color = MODE_COLOR
    print(f"\n{'─'*65}")
    print(f"  {label}  ({OUTPUT_DIR}/)")
    print(f"{'─'*65}")

    # ── Load all N_RUNS cycles ────────────────────────────────────────────────
    cycles, run_ids = [], []
    for run_i in range(N_RUNS):
        primary  = os.path.join(out_dir, f"06_ga_run_{run_i:02d}_energy.csv")
        fallback = os.path.join(out_dir, f"05a_ga_run_{run_i:02d}.csv")
        df = load_cycle(primary, fallback)
        if df is None:
            continue
        for col, default in [("elevation_change_m", 0.0), ("dist_m", 0.0)]:
            if col not in df.columns:
                df[col] = default
        if "time_s" not in df.columns:
            df["time_s"] = np.arange(len(df)) * STEP_SEC
        cycles.append(df)
        run_ids.append(run_i)

    if not cycles:
        print(f"  No cycles found — run Stage 05a first.")
        return

    print(f"  Loaded {len(cycles)} cycles.")

    # ── Per-run statistics ────────────────────────────────────────────────────
    run_rows = []
    for run_i, df in zip(run_ids, cycles):
        spd  = df["gps_speed_kmh"].values
        acc  = df["acceleration_ms2"].values
        elev = df["elevation_change_m"].fillna(0).values
        dist = df["dist_m"].fillna(0).values

        ck       = kinematic_stats(spd, acc, dt_s=STEP_SEC)
        ce       = elevation_stats(elev, dist)

        # Measured energy efficiency (from energy_wh column in cycle)
        if "energy_wh" in df.columns:
            total_wh   = df["energy_wh"].fillna(0).sum()
            total_dist = dist.sum() / 1000.0
            ck["efficiency_wh_km"] = (total_wh / total_dist) if total_dist > 0 else 0.0

        cand_saf = build_saf_matrix(spd, acc, normalise=True)
        cs       = saf_scalar_stats(target_saf, cand_saf)
        cand_sef = build_sef_matrix(spd, elev, normalise=True)
        sef_s    = {
            "sef_sse": float(np.sum((fleet_sef - cand_sef) ** 2)),
            "sef_mae": float(np.mean(np.abs(fleet_sef - cand_sef))),
        }
        errs        = cycle_error_pct(df, target_kine, target_elev, target_saf)
        saf_e       = errs.pop("saf_sse", np.nan)   # separate before averaging scalars
        scalar_mean = float(np.mean(list(errs.values()))) if errs else np.nan
        mean_e      = float((saf_e + scalar_mean) / 2.0) if not np.isnan(saf_e) else scalar_mean
        errs["saf_sse"] = saf_e                     # restore for run_df

        m_wh = np.nan
        if "measured_wh_km" in df.columns and df["measured_wh_km"].notna().any():
            m_wh = float(df["measured_wh_km"].dropna().iloc[0])

        row = {"run_id": run_i, "measured_wh_km": m_wh,
               "mean_fitness_mape_pct": mean_e,
               **ck, **ce, **cs, **sef_s}
        run_rows.append(row)

        print(f"  run_{run_i:02d}: saf_sse={cs['saf_sse']:.4f}  "
              f"mean_err={mean_e:.1f}%  "
              f"gain/km={ce['gain_per_km']:.2f}  "
              f"idle={ck['idle_fraction']:.3f}  "
              f"mean_spd={ck['mean_speed_kmh']:.1f}")

    run_df = pd.DataFrame(run_rows)
    run_df.to_csv(os.path.join(out_dir, "07_run_reports.csv"), index=False)
    print(f"  Run reports → {out_dir}/07_run_reports.csv")

    # ── Summary report ────────────────────────────────────────────────────────
    summary_rows = []
    for metric in sorted(target_all.keys()):
        if metric not in run_df.columns:
            continue
        t_val   = target_all[metric]
        vals    = run_df[metric].dropna()
        is_info = metric in INFO_ONLY_METRICS
        if len(vals) == 0:
            continue
        mean_v = float(vals.mean())
        std_v  = float(vals.std())
        if is_info:
            mae_pct = np.nan
        elif metric in PERCENT_METRICS:
            mae_pct = float(np.mean(np.abs(vals - t_val)))         # pp
        elif metric in FRACTION_METRICS:
            mae_pct = float(np.mean(np.abs(vals - t_val)) * 100)   # pp
        elif abs(t_val) < 1e-6:
            mae_pct = np.nan
        else:
            mae_pct = float(np.mean(np.abs(vals - t_val)) / abs(t_val) * 100)  # rel %
        summary_rows.append({
            "Metric": metric, "Fleet_Target": t_val,
            "Mean": mean_v, "Std": std_v,
            "Mean_Abs_Err": mae_pct,   # pp for % metrics; rel% for others
            "In_Fitness": "no" if is_info else "yes",
        })

    sum_df = pd.DataFrame(summary_rows)
    sum_df.to_csv(os.path.join(out_dir, "07_summary_report.csv"), index=False)
    print(f"  Summary report → {out_dir}/07_summary_report.csv")

    fitness_df  = sum_df[sum_df["In_Fitness"] == "yes"].dropna(subset=["Mean_Abs_Err"])
    printable   = fitness_df.sort_values("Mean_Abs_Err", ascending=False)
    overall_mae = fitness_df["Mean_Abs_Err"].mean() if not fitness_df.empty else np.nan
    print(f"\n  Fitness metric errors — mean={overall_mae:.1f}%:")
    for _, r in printable.head(10).iterrows():
        flag = "**" if r["Mean_Abs_Err"] > 20 else "  "
        print(f"    {flag}{r['Metric']:28s}: target={r['Fleet_Target']:8.3f}  "
              f"mean={r['Mean']:8.3f}  err={r['Mean_Abs_Err']:5.1f}%")

    # ── Dashboard — all cycles ────────────────────────────────────────────────
    saf_sse_vals = run_df["saf_sse"].values.tolist()
    run_labels   = [f"{label}\nrun_{r:02d}" for r in run_ids]

    dashboard_path = os.path.join(out_dir, "07_dashboard.png")
    plot_dashboard(
        cycles, run_labels, saf_sse_vals,
        baseline_wh_km, dashboard_path,
        title=label,
        target_kine=target_kine,
        target_elev=target_elev,
        target_saf=target_saf,
    )
    print(f"\n  Dashboard (all {len(cycles)} cycles) → {dashboard_path}")

    # ── SAF comparison — fleet + all cycles ──────────────────────────────────
    cycle_safs = [
        build_saf_matrix(df["gps_speed_kmh"].values, df["acceleration_ms2"].values)
        for df in cycles
    ]
    run_short = [f"run_{r:02d}" for r in run_ids]
    saf_path  = os.path.join(out_dir, "07_saf_comparison.png")
    plot_saf_comparison(target_saf, cycle_safs, run_short, saf_path)
    print(f"  SAF comparison → {saf_path}")

    # ── SEF comparison — fleet + all individual cycles ────────────────────────
    cycle_sefs = [
        build_sef_matrix(df["gps_speed_kmh"].values,
                         df["elevation_change_m"].fillna(0).values)
        for df in cycles
    ]
    sef_cycles_path = os.path.join(out_dir, "07_sef_cycles.png")
    plot_sef_all_cycles(fleet_sef, cycle_sefs, run_short, sef_cycles_path)
    print(f"  SEF cycles     → {sef_cycles_path}")

    # ── Speed-ramp post-processing ────────────────────────────────────────────
    # Re-order each cycle's micro-trip segments by ascending max speed so the
    # assembled cycle increases in pace over time.  Idle gaps stay in place.
    # Saves alongside the originals; does not re-run the GA.
    print(f"\n  Speed-ramp post-processing ({len(cycles)} cycles)...")
    ramped_cycles = []
    for run_i, df in zip(run_ids, cycles):
        ramped = ramp_by_max_speed(df)
        ramp_path = os.path.join(out_dir, f"07_ga_run_{run_i:02d}_speed_ramped.csv")
        ramped.to_csv(ramp_path, index=False)
        ramped_cycles.append(ramped)
        peak = ramped["gps_speed_kmh"].max()
        print(f"    run_{run_i:02d}: peak={peak:.1f} km/h  → {ramp_path}")

    # Dashboard for ramped cycles
    ramp_labels = [f"Ramped\nrun_{r:02d}" for r in run_ids]
    ramp_sse    = [float(saf_sse(target_saf,
                         build_saf_matrix(df["gps_speed_kmh"].values,
                                          df["acceleration_ms2"].values)))
                   for df in ramped_cycles]
    ramp_dash   = os.path.join(out_dir, "07_dashboard_speed_ramped.png")
    plot_dashboard(
        ramped_cycles, ramp_labels, ramp_sse,
        baseline_wh_km, ramp_dash,
        title=f"{label} — Speed Ramped",
        target_kine=target_kine,
        target_elev=target_elev,
        target_saf=target_saf,
    )
    print(f"  Ramped dashboard → {ramp_dash}")

    n_out = len(glob.glob(os.path.join(out_dir, "07_*")))
    print(f"\n{'='*65}")
    print(f"STAGE 07 COMPLETE — {n_out} output files in {OUTPUT_DIR}/")
    print("="*65)


if __name__ == "__main__":
    main()
