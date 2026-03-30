"""Generate bar chart of GA run composite scores and elevation sensitivity analysis."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import os, warnings
warnings.filterwarnings('ignore')

FIGS_DIR = "drive_cycle_research_paper/figs"
FEATURES_PATH = "output/03_features.csv"
BEST_RUN_CSV  = "output_full_mape/05a_ga_run_02.csv"
RUN_REPORTS   = "output_full_mape/07_run_reports.csv"

os.makedirs(FIGS_DIR, exist_ok=True)

# ── 1. Bar chart ──────────────────────────────────────────────────────────────
reports = pd.read_csv(RUN_REPORTS)
n = len(reports)
scores = reports["saf_sse"] + reports["mean_fitness_mape_pct"] / 100.0
best_idx = int(scores.idxmin())

colors = ["#2196F3" if i == best_idx else "#90CAF9" for i in range(n)]
fig, ax = plt.subplots(figsize=(7, 3.5))
bars = ax.bar(range(n), scores, color=colors, edgecolor="white", linewidth=0.5)
ax.set_xticks(range(n))
ax.set_xticklabels([f"Run {i:02d}" for i in range(n)], fontsize=8)
ax.set_ylabel("Composite fitness score", fontsize=9)
ax.set_xlabel("GA run (seed 42–51)", fontsize=9)
ax.set_title("Composite fitness scores across 10 GA runs", fontsize=10)
ax.yaxis.grid(True, linestyle='--', alpha=0.5)
ax.set_axisbelow(True)
# Annotate best
ax.text(best_idx, scores[best_idx] + 0.0003, f"Best\n{scores[best_idx]:.4f}",
        ha='center', va='bottom', fontsize=7.5, color='#1565C0', fontweight='bold')
blue_patch  = mpatches.Patch(color='#2196F3', label='Best run (Run 02)')
other_patch = mpatches.Patch(color='#90CAF9', label='Other runs')
ax.legend(handles=[blue_patch, other_patch], fontsize=8, loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, "ga_run_scores.png"), dpi=150, bbox_inches='tight')
plt.savefig(os.path.join(FIGS_DIR, "ga_run_scores.pdf"), bbox_inches='tight')
plt.close()
print(f"Bar chart saved. Best run: {best_idx}, score: {scores[best_idx]:.5f}")

# ── 2. Elevation sensitivity ──────────────────────────────────────────────────
REGRESSION_FEATURES = ["mean_speed_kmh","rms_accel_ms2","idle_fraction",
                        "gain_per_km","loss_per_km"]
feat = pd.read_csv(FEATURES_PATH)
feat = feat[feat["efficiency_wh_km"].notna() & feat["has_energy"]].copy()

# IQR filter
q1, q3 = feat["efficiency_wh_km"].quantile([0.25, 0.75])
iqr = q3 - q1
feat = feat[(feat["efficiency_wh_km"] >= q1 - 2.5*iqr) &
            (feat["efficiency_wh_km"] <= q3 + 2.5*iqr)].copy()
feat = feat[REGRESSION_FEATURES + ["efficiency_wh_km"]].dropna()

X = feat[REGRESSION_FEATURES].values
y = feat["efficiency_wh_km"].values
scaler = StandardScaler()
X_sc = scaler.fit_transform(X)
model = Ridge(alpha=1.0)
model.fit(X_sc, y)

from sklearn.metrics import r2_score, mean_absolute_error
y_pred = model.predict(X_sc)
r2 = r2_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)
print(f"Ridge R²={r2:.3f}  MAE={mae:.1f} Wh/km")

# Run 02 features from run_reports
r = reports.iloc[best_idx]
cycle_feats_with = {
    "mean_speed_kmh": r["mean_speed_kmh"],
    "rms_accel_ms2":  r["rms_accel_ms2"],
    "idle_fraction":  r["idle_fraction"],
    "gain_per_km":    r["gain_per_km"],
    "loss_per_km":    r["loss_per_km"],
}
cycle_feats_flat = cycle_feats_with.copy()
cycle_feats_flat["gain_per_km"] = 0.0
cycle_feats_flat["loss_per_km"] = 0.0

def predict(feats):
    x = np.array([[feats[f] for f in REGRESSION_FEATURES]])
    return model.predict(scaler.transform(x))[0]

pred_with = predict(cycle_feats_with)
pred_flat = predict(cycle_feats_flat)
delta     = pred_with - pred_flat
pct_diff  = delta / pred_flat * 100

print(f"\nElevation sensitivity — best cycle (Run 02, seed 44):")
print(f"  gain_per_km = {cycle_feats_with['gain_per_km']:.2f} m/km")
print(f"  loss_per_km = {cycle_feats_with['loss_per_km']:.2f} m/km")
print(f"  Predicted Wh/km (with elevation) : {pred_with:.1f}")
print(f"  Predicted Wh/km (flat terrain)   : {pred_flat:.1f}")
print(f"  Delta                             : {delta:.1f} Wh/km")
print(f"  Terrain adds                      : {pct_diff:.1f}% to consumption")

# ── 3. Print all run scores for LaTeX table ────────────────────────────────────
print("\nAll run composite scores:")
for i, s in enumerate(scores):
    mark = " ← best" if i == best_idx else ""
    print(f"  Run {i:02d} (seed {42+i}): {s:.5f}{mark}")
