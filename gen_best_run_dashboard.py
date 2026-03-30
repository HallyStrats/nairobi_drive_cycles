"""Generate a clean 2-row dashboard for the best run (Run 02) only."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

FIGS_DIR = "drive_cycle_research_paper/figs"
CYCLE_CSV = "output_full_mape/05a_ga_run_02.csv"
os.makedirs(FIGS_DIR, exist_ok=True)

df = pd.read_csv(CYCLE_CSV)
print(f"Columns: {list(df.columns)}")

speed = df["gps_speed_kmh"].values
t = df["time_s"].values / 60.0  # minutes

# Cumulative elevation from elevation_change_m
elev_cum = np.cumsum(df["elevation_change_m"].fillna(0).values)

# Cumulative distance from dist_m
dist_cum_km = df["dist_m"].cumsum().values / 1000.0

fig = plt.figure(figsize=(10, 6))
gs = gridspec.GridSpec(2, 1, hspace=0.40)

# Row 1: Speed trace
ax1 = fig.add_subplot(gs[0])
idle_mask = speed < 2
ax1.fill_between(t, speed, alpha=0.15, color='#1976D2')
ax1.plot(t, speed, color='#1976D2', linewidth=0.8)
ax1.set_ylabel("Speed (km/h)", fontsize=10)
ax1.set_xlabel("Time (min)", fontsize=10)
ax1.set_xlim([0, t[-1]])
ax1.set_ylim(bottom=0)
ax1.set_title("Representative drive cycle — best of 10 GA runs (seed 44)", fontsize=11)
ax1.grid(True, linestyle='--', alpha=0.3)
idle_frac = idle_mask.mean()
ax1.text(0.99, 0.96, f"Idle: {idle_frac*100:.1f}%  |  Peak: {speed.max():.0f} km/h  |  Duration: 30 min",
         transform=ax1.transAxes, ha='right', va='top', fontsize=8.5, color='#333')

# Row 2: Elevation profile vs distance
ax2 = fig.add_subplot(gs[1])
ax2.plot(dist_cum_km, elev_cum, color='#388E3C', linewidth=1.0)
ax2.fill_between(dist_cum_km, elev_cum, alpha=0.12, color='#388E3C')
ax2.set_ylabel("Cumulative elevation change (m)", fontsize=10)
ax2.set_xlabel("Distance (km)", fontsize=10)
ax2.set_xlim([0, dist_cum_km[-1]])
ax2.axhline(0, color='gray', linewidth=0.5, linestyle='--')
ax2.set_title(f"Elevation profile  (gain: 10.5 m/km | loss: −10.3 m/km)", fontsize=11)
ax2.grid(True, linestyle='--', alpha=0.3)

plt.savefig(os.path.join(FIGS_DIR, "ga_cycle_dashboard.png"), dpi=150, bbox_inches='tight')
plt.savefig(os.path.join(FIGS_DIR, "ga_cycle_dashboard.pdf"), bbox_inches='tight')
plt.close()
print(f"Dashboard saved. Total distance: {dist_cum_km[-1]:.2f} km")
print(f"Elevation range: {elev_cum.min():.1f} to {elev_cum.max():.1f} m")
