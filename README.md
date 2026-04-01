# Nairobi Drive Cycle Pipeline
### Genetic Algorithm Micro-Trip Selection for Electric Motorcycle Taxis in Kenya

This repository contains the pipeline used to build the **Nairobi Electric Boda-Boda Drive Cycle (NEBDC)** — a 30-minute speed–elevation drive cycle for electric motorcycle taxis (*boda bodas*) in Nairobi, Kenya. The cycle is assembled from real GPS telemetry using a Genetic Algorithm that selects and sequences measured micro-trips to match fleet-wide driving statistics.

The pipeline accompanies the paper:

> Stratford, H. (2026). *A Drive Cycle for Electric Motorcycle Taxis in Nairobi, Kenya using Genetic Algorithm Micro-Trip Selection.* [Manuscript submitted for publication]

---

## The NEBDC — Ready to Use

If you just want the drive cycle, it is provided directly in this repository:

**[`NEBDC.csv`](./NEBDC.csv)**

| Column | Unit | Description |
|---|---|---|
| `time_s` | seconds | Elapsed time, 0–1790 s at 10-second resolution |
| `cumulative_dist_km` | km | Distance from start |
| `speed_kmh` | km/h | Vehicle speed at each time step |
| `elevation_m` | m | Cumulative elevation change relative to start |

The NEBDC is a 30-minute, 9.03 km cycle derived from 35,571 real micro-trips collected across 291 electric motorcycles over eight days in Nairobi. It is suitable for:

- Vehicle range estimation and simulation
- Chassis dynamometer testing
- Battery sizing and thermal modelling
- Regulatory homologation adapted to Sub-Saharan urban conditions

---

## How It Works

Raw telemetry from 291 motorcycles is cleaned and segmented into 35,571 micro-trips (short bursts of movement between stops). A feature extraction stage computes a 70-element Speed–Acceleration Frequency (SAF) matrix and seventeen kinematic, elevation, and energy statistics for each micro-trip. The Genetic Algorithm then searches the space of possible micro-trip sequences, selecting and concatenating real trips to build a 30-minute cycle that matches fleet-wide averages as closely as possible. After GA convergence, micro-trips are re-ordered by ascending peak speed (the standard speed-ramp convention), and the best of ten independent GA runs is kept as the NEBDC.

---

## Repository Structure

```
nairobi_drive_cycles/
│
├── NEBDC.csv                   # The final drive cycle (ready to use)
├── run_pipeline.sh             # One-command pipeline runner
│
├── 01_ingest_and_clean.py      # Raw data ingestion and per-device cleaning
├── find_macro_stop.py          # Auto-sweep: macro-trip threshold calibration
├── 02_segment_trips.py         # Micro- and macro-trip segmentation
├── 03_extract_features.py      # Feature extraction and outlier removal
├── 05a_knit_cycle_ga.py        # GA-based drive cycle assembly
├── 06_energy_model.py          # Energy model and consumption analysis
├── 07_evaluate_and_compare.py  # Evaluation, scoring, and dashboard
│
├── gen_best_run_dashboard.py   # Dashboard for the best GA run
├── gen_saf_and_table_data.py   # SAF matrix and table exports
├── run_bar_chart.py            # Metric comparison bar charts
│
└── utils/
    ├── gps_utils.py            # Haversine distance, GPS smoothing, elevation lookup
    ├── energy_utils.py         # Battery RC delta, Wh/km, pack voltage
    └── metrics.py              # SAF/SEF matrix builders, kinematic statistics
```

---

## Input Data

The pipeline expects raw telemetry CSV files placed in `../all_data/` (one level above this repository).

| Column | Description |
|---|---|
| `imei_no` | Device identifier (one per motorcycle) |
| `gps_date` | UTC timestamp (parseable by `pandas.to_datetime`) |
| `lat`, `long` | GPS coordinates (decimal degrees) |
| `altitude` | GPS-reported altitude (metres) — used as fallback elevation |
| `vehicle_speed` | Controller-reported vehicle speed (km/h) |
| `rc` | Remaining battery capacity (Wh) — used to compute energy consumption |
| `fc` | Full battery capacity (Wh) |
| `dc` | Discharge energy counter (Wh) |
| `rpm` | Motor RPM |
| `ct`, `mt` | Controller and motor temperatures (°C) |

> **Elevation:** The pipeline uses SRTM elevation data (via the `srtm` Python package) with GPS altitude as a fallback.

> **Energy:** The `rc` channel is the primary energy source. Micro-trips with fewer than 80% non-null `rc` readings are excluded from energy analysis but retained for kinematic feature extraction and GA selection.

---

## Installation

Python 3.9+ is required.

```bash
pip install numpy pandas scipy scikit-learn matplotlib srtm.py
```

---

## Running the Pipeline

```bash
cd nairobi_drive_cycles
bash run_pipeline.sh
```

This runs all stages in sequence and logs output to a timestamped `pipeline_output_YYYYMMDD_HHMMSS.txt` file.

To run individual stages:

```bash
python3 01_ingest_and_clean.py
python3 find_macro_stop.py
python3 02_segment_trips.py
python3 03_extract_features.py
python3 05a_knit_cycle_ga.py
python3 06_energy_model.py
python3 07_evaluate_and_compare.py
```

---

## Citation

If you use the NEBDC or this pipeline in your work, please cite:

```
Stratford, H. (2026). A Drive Cycle for Electric Motorcycle Taxis in Nairobi,
Kenya using Genetic Algorithm Micro-Trip Selection. [Manuscript submitted for
publication]
```

For the software specifically:

```
Stratford, H. (2026). Nairobi Drive Cycle Pipeline: Genetic Algorithm
Micro-Trip Selection for Electric Motorcycle Taxis in Kenya (Version 1.0).
https://github.com/HallyStrats/nairobi_drive_cycles
```

---

## License

MIT License — see `LICENSE` for details.
