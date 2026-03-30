"""
GPS Utilities: Haversine distance, coordinate smoothing, SRTM elevation lookup.
"""
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


# ---------------------------------------------------------------------------
# 1. Haversine Distance
# ---------------------------------------------------------------------------
EARTH_RADIUS_M = 6_371_000.0  # metres

def haversine_m(lat1, lon1, lat2, lon2):
    """
    Vectorised Haversine formula — returns distance in metres.
    Works correctly for very short distances (10–100 m).
    lat/lon in decimal degrees (numpy arrays or scalars).
    """
    φ1 = np.radians(lat1)
    φ2 = np.radians(lat2)
    Δφ = np.radians(lat2 - lat1)
    Δλ = np.radians(lon2 - lon1)

    a = np.sin(Δφ / 2.0) ** 2 + np.cos(φ1) * np.cos(φ2) * np.sin(Δλ / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return EARTH_RADIUS_M * c


def compute_gps_speed_kmh(df, lat_col="lat", lon_col="long", time_col="gps_date"):
    """
    Compute GPS-derived speed in km/h from smoothed lat/long and timestamps.
    Returns a Series aligned with df's index.
    """
    dist_m = haversine_m(
        df[lat_col].shift().values,
        df[lon_col].shift().values,
        df[lat_col].values,
        df[lon_col].values,
    )
    dt_s = df[time_col].diff().dt.total_seconds().values
    # Avoid div-by-zero; first row gets 0
    with np.errstate(divide="ignore", invalid="ignore"):
        speed_ms = np.where(dt_s > 0, dist_m / dt_s, 0.0)
    return pd.Series(speed_ms * 3.6, index=df.index, name="gps_speed_kmh")


# ---------------------------------------------------------------------------
# 2. Coordinate Smoothing (Savitzky-Golay)
# ---------------------------------------------------------------------------
def smooth_coords(series: pd.Series, window: int = 5, poly: int = 2) -> pd.Series:
    """
    Apply Savitzky-Golay smoothing to a coordinate series.
    Falls back to original if the series is too short.
    window must be odd and > poly.
    """
    n = len(series)
    if n < window:
        return series
    # Ensure window is odd
    w = window if window % 2 == 1 else window + 1
    w = min(w, n if n % 2 == 1 else n - 1)
    if w <= poly:
        return series
    smoothed = savgol_filter(series.values, window_length=w, polyorder=poly)
    return pd.Series(smoothed, index=series.index, name=series.name)


# ---------------------------------------------------------------------------
# 3. GPS Outlier / Teleportation Detection
# ---------------------------------------------------------------------------
MAX_PLAUSIBLE_SPEED_KMH = 120.0  # physical upper limit for these motorcycles


def flag_gps_outliers(df, lat_col="lat", lon_col="long",
                      time_col="gps_date",
                      max_speed_kmh=MAX_PLAUSIBLE_SPEED_KMH) -> pd.Series:
    """
    Returns a boolean mask (True = outlier row) where consecutive GPS speed
    exceeds max_speed_kmh.  The first row is never flagged.
    """
    dist_m = haversine_m(
        df[lat_col].shift().values,
        df[lon_col].shift().values,
        df[lat_col].values,
        df[lon_col].values,
    )
    dt_s = df[time_col].diff().dt.total_seconds().values
    with np.errstate(divide="ignore", invalid="ignore"):
        implied_kmh = np.where(dt_s > 0, (dist_m / dt_s) * 3.6, 0.0)

    is_outlier = pd.Series(implied_kmh > max_speed_kmh, index=df.index)
    is_outlier.iloc[0] = False
    return is_outlier


# ---------------------------------------------------------------------------
# 4. Altitude Cleaning & SRTM Lookup
# ---------------------------------------------------------------------------
def clean_altitude(alt_series: pd.Series,
                   zero_threshold: float = 1.0,
                   max_gap_s: float = 120.0,
                   time_series: pd.Series = None) -> pd.Series:
    """
    Replace altitude <= zero_threshold with NaN, then interpolate.
    If time_series is provided, only interpolate over gaps < max_gap_s;
    larger gaps are left as NaN.
    """
    alt = alt_series.copy().astype(float)
    alt[alt <= zero_threshold] = np.nan

    if time_series is not None:
        # Limit interpolation across large time gaps
        dt = time_series.diff().dt.total_seconds().fillna(0)
        cum_gap = dt.where(alt.isna()).fillna(0)
        # Mark positions where the gap before them is too large
        large_gap = cum_gap > max_gap_s
        alt = alt.interpolate(method="linear", limit_direction="both")
        alt[large_gap & alt_series.isna()] = np.nan
    else:
        alt = alt.interpolate(method="linear", limit_direction="both")

    return alt


def try_srtm_lookup(lat_arr, lon_arr):
    """
    Fetch SRTM elevation for an array of (lat, lon) pairs.
    Returns a numpy array of elevations (metres), or None if srtm.py is unavailable.

    Uses the `srtm` Python library (pip install srtm.py).  Tiles are downloaded
    once and cached locally; subsequent calls read from in-memory arrays.

    Performance: coordinates are rounded to 3 decimal places (~111 m grid) and
    deduplicated before lookup, reducing Python-loop overhead from N rows to
    N unique grid cells (typically 10–30× fewer calls).
    """
    try:
        import srtm
        data = srtm.get_data()

        lat_f = np.asarray(lat_arr, dtype=float)
        lon_f = np.asarray(lon_arr, dtype=float)

        # Round to 3 dp (~111 m) to collapse near-duplicate points
        lat_r = np.round(lat_f, 3)
        lon_r = np.round(lon_f, 3)

        # Build lookup for unique grid cells only
        unique_keys = np.unique(np.column_stack([lat_r, lon_r]), axis=0)
        elev_map = {}
        for lat, lon in unique_keys:
            e = data.get_elevation(float(lat), float(lon))
            elev_map[(lat, lon)] = float(e) if e is not None else np.nan

        elevations = np.array([
            elev_map.get((lat_r[i], lon_r[i]), np.nan)
            for i in range(len(lat_f))
        ])
        return elevations
    except ImportError:
        return None
    except Exception:
        return None


def get_best_elevation(df, lat_col="lat", lon_col="long",
                       alt_col="altitude",
                       time_col="gps_date") -> pd.Series:
    """
    Pipeline:
    1. Try SRTM lookup (free, accurate to ~30 m).
    2. Fall back to cleaned GPS altitude if SRTM is unavailable.
    Returns a Series named 'elevation_m'.
    """
    srtm_elev = try_srtm_lookup(df[lat_col].values, df[lon_col].values)

    if srtm_elev is not None and not np.all(np.isnan(srtm_elev)):
        elev = pd.Series(srtm_elev, index=df.index, name="elevation_m")
        # Fill any tile gaps with GPS altitude fallback
        gps_alt = clean_altitude(df[alt_col], time_series=df[time_col])
        elev = elev.fillna(gps_alt)
    else:
        # SRTM unavailable — use GPS altitude with cleaning
        gps_alt = clean_altitude(df[alt_col], time_series=df[time_col])
        # Smooth GPS altitude to remove sensor noise
        if len(gps_alt.dropna()) > 5:
            smoothed = smooth_coords(gps_alt.fillna(method="ffill"), window=7, poly=2)
        else:
            smoothed = gps_alt
        elev = smoothed.rename("elevation_m")

    return elev
