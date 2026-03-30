"""
Generate:
1. 2-panel SAF comparison: fleet vs Run 02
2. Predicted Wh/km for all 10 runs (for LaTeX table)
3. Speed-ramped dashboard for Run 02
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import os, warnings
warnings.filterwarnings('ignore')

FIGS_DIR   = "drive_cycle_research_paper/figs"
FEATURES   = "output/03_features.csv"
RUN_RPT    = "output_full_mape/07_run_reports.csv"
BEST_CSV   = "output_full_mape/05a_ga_run_02.csv"
RAMPED_CSV = "output_full_mape/07_ga_run_02_speed_ramped.csv"
os.makedirs(FIGS_DIR, exist_ok=True)

# ── Helpers ────────────────────────────────────────────────────────────────────
SPEED_EDGES = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 1000]
ACCEL_EDGES = [-10, -2.0, -1.0, -0.2, 0.2, 1.0, 2.0, 10]
NS, NA = len(SPEED_EDGES)-1, len(ACCEL_EDGES)-1

def compute_saf(speed_col, accel_col):
    """Compute normalised SAF matrix from speed (km/h) and accel (m/s²) arrays."""
    H = np.zeros((NS, NA))
    for s, a in zip(speed_col, accel_col):
        if np.isnan(s) or np.isnan(a):
            continue
        si = min(np.searchsorted(SPEED_EDGES[1:], s), NS-1)
        ai = min(np.searchsorted(ACCEL_EDGES[1:], a), NA-1)
        H[si, ai] += 1
    total = H.sum()
    return H / total if total > 0 else H

# ── 1. SAF comparison ──────────────────────────────────────────────────────────
feat = pd.read_csv(FEATURES)
feat = feat[feat["efficiency_wh_km"].notna() | feat["efficiency_wh_km"].isna()]  # all rows

# Fleet SAF: aggregate from micro-trips using mean_speed and mean_accel
# Better: load cycle data for fleet; we'll use the raw speed approximation
# Use fleet features as proxy: build SAF from representative sample
# Actually reconstruct from best cycle vs fleet aggregate via run 02 vs fleet 07_saf
# Simpler: load each 05a csv to compute SAF directly

def saf_from_cycle(csv_path):
    df = pd.read_csv(csv_path)
    speed = df["gps_speed_kmh"].values
    accel = df["acceleration_ms2"].values
    return compute_saf(speed, accel)

# Fleet SAF: aggregate over ALL micro-trips in 03_features — we approximate by
# using the speed/accel distribution implied by the fleet SAF matrix in saf_comparison.png.
# Instead, compute directly from best-cycle CSV for that, and load the fleet aggregate
# from a subsample of features.
# Best approach: read raw micro-trip CSVs from output/02_microtrips/*.csv sample.
import glob
micro_files = sorted(glob.glob("output/02_microtrips/*.csv"))
print(f"Found {len(micro_files)} micro-trip files")

# Sample up to 200 files for fleet SAF
fleet_H = np.zeros((NS, NA))
sampled = micro_files[:200]
for fp in sampled:
    try:
        df = pd.read_csv(fp)
        if "gps_speed_kmh" in df.columns and "acceleration_ms2" in df.columns:
            H = compute_saf(df["gps_speed_kmh"].values, df["acceleration_ms2"].values)
            fleet_H += H
    except Exception:
        pass

fleet_saf = fleet_H / fleet_H.sum() if fleet_H.sum() > 0 else fleet_H
cycle_saf = saf_from_cycle(BEST_CSV)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
speed_labels = ["0-10","10-20","20-30","30-40","40-50","50-60","60-70","70-80","80-90","90+"]
accel_labels = ["<-2","-2:-1","-1:-0.2","-0.2:0.2","0.2:1","1:2",">2"]

vmax = max(fleet_saf.max(), cycle_saf.max())
for ax, H, title in zip(axes, [fleet_saf, cycle_saf],
                         ["Fleet-wide SAF matrix", "Assembled cycle SAF matrix\n(Run 3, seed 44)"]):
    im = ax.imshow(H.T, origin='lower', aspect='auto', cmap='YlOrRd',
                   vmin=0, vmax=vmax, interpolation='nearest')
    ax.set_xticks(range(NS))
    ax.set_xticklabels(speed_labels, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(NA))
    ax.set_yticklabels(accel_labels, fontsize=7)
    ax.set_xlabel("Speed (km/h)", fontsize=9)
    ax.set_ylabel("Acceleration (m/s²)", fontsize=9)
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.85, label="Normalised frequency")

plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, "saf_comparison.png"), dpi=150, bbox_inches='tight')
plt.savefig(os.path.join(FIGS_DIR, "saf_comparison.pdf"), bbox_inches='tight')
plt.close()
print("SAF comparison saved.")

# ── 2. Predicted Wh/km for all 10 runs ─────────────────────────────────────────
REGRESSION_FEATURES = ["mean_speed_kmh","rms_accel_ms2","idle_fraction","gain_per_km","loss_per_km"]
energy_df = pd.read_csv(FEATURES)
energy_df = energy_df[energy_df["has_energy"] & energy_df["efficiency_wh_km"].notna()].copy()
q1, q3 = energy_df["efficiency_wh_km"].quantile([0.25, 0.75])
iqr = q3 - q1
energy_df = energy_df[(energy_df["efficiency_wh_km"] >= q1 - 2.5*iqr) &
                      (energy_df["efficiency_wh_km"] <= q3 + 2.5*iqr)].copy()
energy_df = energy_df[REGRESSION_FEATURES + ["efficiency_wh_km"]].dropna()
X = energy_df[REGRESSION_FEATURES].values
y = energy_df["efficiency_wh_km"].values
scaler = StandardScaler(); X_sc = scaler.fit_transform(X)
model = Ridge(alpha=1.0); model.fit(X_sc, y)

reports = pd.read_csv(RUN_RPT)
composite = reports["saf_sse"] + reports["mean_fitness_mape_pct"] / 100.0

print("\n--- All 10 runs table data ---")
print(f"{'Run':>6} {'Seed':>6} {'Composite':>12} {'Gain (m/km)':>12} {'Loss (m/km)':>12} {'Pred Wh/km':>12}")
pred_whkm = []
for i, row in reports.iterrows():
    x = np.array([[row[f] for f in REGRESSION_FEATURES]])
    pred = model.predict(scaler.transform(x))[0]
    pred_whkm.append(pred)
    print(f"  {i:2d}   {42+i:4d}   {composite[i]:10.5f}   {row['gain_per_km']:10.2f}   {row['loss_per_km']:10.2f}   {pred:10.1f}")

best_comp = int(composite.idxmin())
best_wh = int(np.argmin(pred_whkm))
print(f"\nBest composite: Run {best_comp:02d} ({composite[best_comp]:.5f})")
print(f"Best Wh/km:     Run {best_wh:02d} ({pred_whkm[best_wh]:.1f} Wh/km)")

# ── 3. Speed-ramped dashboard ──────────────────────────────────────────────────
df_r = pd.read_csv(RAMPED_CSV)
print(f"\nRamped cycle columns: {list(df_r.columns)}")

speed_r = df_r["gps_speed_kmh"].values
t_r = df_r["time_s"].values / 60.0
elev_r = np.cumsum(df_r["elevation_change_m"].fillna(0).values)
dist_r_km = df_r["dist_m"].cumsum().values / 1000.0

fig = plt.figure(figsize=(10, 6))
gs = gridspec.GridSpec(2, 1, hspace=0.40)

ax1 = fig.add_subplot(gs[0])
ax1.fill_between(t_r, speed_r, alpha=0.15, color='#7B1FA2')
ax1.plot(t_r, speed_r, color='#7B1FA2', linewidth=0.8)
ax1.set_ylabel("Speed (km/h)", fontsize=10)
ax1.set_xlabel("Time (min)", fontsize=10)
ax1.set_xlim([0, t_r[-1]])
ax1.set_ylim(bottom=0)
ax1.set_title("Speed-ramped drive cycle (Run 3, seed 44) — micro-trips ordered by ascending peak speed", fontsize=10)
ax1.grid(True, linestyle='--', alpha=0.3)
ax1.text(0.99, 0.96, f"Peak: {speed_r.max():.0f} km/h  |  Idle: {(speed_r<2).mean()*100:.1f}%",
         transform=ax1.transAxes, ha='right', va='top', fontsize=8.5, color='#333')

ax2 = fig.add_subplot(gs[1])
ax2.plot(dist_r_km, elev_r, color='#388E3C', linewidth=1.0)
ax2.fill_between(dist_r_km, elev_r, alpha=0.12, color='#388E3C')
ax2.set_ylabel("Cumulative elevation change (m)", fontsize=10)
ax2.set_xlabel("Distance (km)", fontsize=10)
ax2.set_xlim([0, dist_r_km[-1]])
ax2.axhline(0, color='gray', linewidth=0.5, linestyle='--')
ax2.set_title("Elevation profile (speed-ramped)", fontsize=10)
ax2.grid(True, linestyle='--', alpha=0.3)

plt.savefig(os.path.join(FIGS_DIR, "ga_cycle_speed_ramped.png"), dpi=150, bbox_inches='tight')
plt.savefig(os.path.join(FIGS_DIR, "ga_cycle_speed_ramped.pdf"), bbox_inches='tight')
plt.close()
print("Speed-ramped dashboard saved.")
