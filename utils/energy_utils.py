"""
Energy Utilities: Remaining-Charge-based energy derivation, outlier removal,
pack-voltage reconstruction from cell voltages.
"""
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Pack Voltage from Cell Voltages
# ---------------------------------------------------------------------------
CELL_COLS = [f"c{i}v" for i in range(1, 21)]   # c1v … c20v
CELL_MV_MIN = 2_800.0   # mV — LFP cell fully discharged ~2.8 V
CELL_MV_MAX = 3_700.0   # mV — LFP cell fully charged  ~3.65 V (nominal ceiling)


def calc_pack_voltage(df: pd.DataFrame) -> pd.Series:
    """
    Reconstruct total pack voltage (V) by summing individual cell voltages.
    Cell voltages in the dataset are in millivolts.
    Returns a Series named 'pack_voltage_v'.
    """
    present = [c for c in CELL_COLS if c in df.columns]
    if not present:
        return pd.Series(np.nan, index=df.index, name="pack_voltage_v")

    cells = df[present].copy().astype(float)
    # Flag implausible cell readings as NaN
    cells[(cells < CELL_MV_MIN) | (cells > CELL_MV_MAX)] = np.nan

    # Sum valid cells; if more than half are NaN, mark entire row as NaN
    valid_count = cells.notna().sum(axis=1)
    total = cells.sum(axis=1)
    total[valid_count < len(present) / 2] = np.nan

    # Convert mV → V
    return (total / 1000.0).rename("pack_voltage_v")


# ---------------------------------------------------------------------------
# 2. delta_rc Energy Derivation
# ---------------------------------------------------------------------------
RC_OUTLIER_DELTA_MAX = 5_000.0   # mAh — cap on single-step discharge (sanity)
RC_OUTLIER_DELTA_MIN = -500.0    # mAh — small regen / sensor noise tolerance
NOMINAL_PACK_VOLTAGE_V = 75.0   # V  — midpoint operating voltage (full=81V at 20×4.05V,
                                #      empty≈66V at 20×3.3V; 3.24 kWh / 40 Ah = 81V spec)

def compute_delta_rc(df: pd.DataFrame,
                     rc_col: str = "rc",
                     time_col: str = "gps_date") -> pd.Series:
    """
    Compute per-timestep energy consumed in Wh from Remaining Charge (rc).

    UNIT NOTE: rc/fc are in milliamp-hours (mAh) based on the scale of values
    (~40,000 mAh for a full charge = 40 Ah, consistent with an EV motorcycle
    battery of ~2.5-3 kWh at ~70V pack voltage).

    Conversion: Wh = delta_rc (mAh) × pack_voltage (V) / 1000

    Rules:
    - delta_rc = rc[t-1] - rc[t]  (positive = energy discharged)
    - Negative deltas beyond RC_OUTLIER_DELTA_MIN (regen/sensor noise) → 0
    - Positive deltas beyond RC_OUTLIER_DELTA_MAX → NaN (sensor error)
    - If rc is NaN for this row → NaN

    Returns a Series named 'energy_wh' (true Wh using pack voltage).
    """
    rc = df[rc_col].astype(float)
    delta_mah = rc.shift(1) - rc      # positive when discharging (mAh)

    # Clip regen / noise to zero
    delta_mah[delta_mah < RC_OUTLIER_DELTA_MIN] = 0.0
    # Flag implausible large deltas
    delta_mah[delta_mah > RC_OUTLIER_DELTA_MAX] = np.nan
    # Propagate NaN from original rc
    delta_mah[rc.isna()] = np.nan

    # Convert mAh → Wh using pack voltage
    if "pack_voltage_v" in df.columns:
        voltage = df["pack_voltage_v"].fillna(NOMINAL_PACK_VOLTAGE_V)
    else:
        voltage = NOMINAL_PACK_VOLTAGE_V

    energy_wh = delta_mah * voltage / 1000.0

    return energy_wh.rename("energy_wh")


def has_sufficient_rc_data(df: pd.DataFrame,
                           rc_col: str = "rc",
                           min_coverage: float = 0.80) -> bool:
    """
    Returns True if at least min_coverage fraction of rows have non-null rc.
    Used to decide whether a trip has usable energy data.
    """
    if rc_col not in df.columns:
        return False
    coverage = df[rc_col].notna().mean()
    return coverage >= min_coverage


def efficiency_wh_per_km(energy_wh_series: pd.Series,
                          dist_m_series: pd.Series) -> float:
    """
    Compute trip energy efficiency in Wh/km.
    Returns NaN if distance is negligible or energy data is all-NaN.
    """
    total_wh = energy_wh_series.dropna().sum()
    total_km = dist_m_series.sum() / 1000.0
    if total_km < 0.1 or np.isnan(total_wh):
        return np.nan
    return total_wh / total_km


# ---------------------------------------------------------------------------
# 3. Energy Outlier Removal
# ---------------------------------------------------------------------------
def remove_energy_outliers(df: pd.DataFrame,
                            eff_col: str = "efficiency_wh_per_km",
                            iqr_factor: float = 2.5) -> pd.DataFrame:
    """
    Remove micro-trip rows whose energy efficiency is an extreme outlier.
    Uses IQR fencing (more robust than z-score for skewed distributions).
    """
    vals = df[eff_col].dropna()
    if len(vals) < 4:
        return df
    q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
    iqr = q3 - q1
    lo = q1 - iqr_factor * iqr
    hi = q3 + iqr_factor * iqr
    mask = (df[eff_col].isna()) | ((df[eff_col] >= lo) & (df[eff_col] <= hi))
    removed = (~mask).sum()
    if removed > 0:
        print(f"  [energy_utils] Removed {removed} trips with Wh/km outside "
              f"[{lo:.1f}, {hi:.1f}] (IQR×{iqr_factor})")
    return df[mask].copy()
