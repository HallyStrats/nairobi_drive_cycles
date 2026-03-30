#!/bin/bash
# ============================================================
# Master pipeline runner — src_git/
# Run from the src_git/ directory:
#   cd src_git && bash run_pipeline.sh
# ============================================================

set -e  # exit on first error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_FILE="pipeline_output_$(date +%Y%m%d_%H%M%S).txt"
exec > >(tee "$LOG_FILE") 2>&1
echo "Logging to $LOG_FILE"
echo ""

echo "========================================"
echo " EV Drive Cycle Pipeline — src_git"
echo "========================================"
echo ""

echo "[01] Ingest & Clean..."
python3 01_ingest_and_clean.py

echo ""
echo "[sweep] Auto-determining MACRO_STOP_S for median macro-trip ≈ 30 min..."
python3 find_macro_stop.py

echo ""
echo "[02] Segment Trips..."
python3 02_segment_trips.py

echo ""
echo "[03] Extract Features..."
python3 03_extract_features.py

echo ""
echo "[05a] GA Knitting (pop=1000, gen=2000, N=10 runs)..."
python3 05a_knit_cycle_ga.py

echo ""
echo "[06] Energy Model..."
python3 06_energy_model.py

echo ""
echo "[07] Evaluate & Compare..."
python3 07_evaluate_and_compare.py

echo ""
echo "========================================"
echo " PIPELINE COMPLETE"
echo "========================================"
