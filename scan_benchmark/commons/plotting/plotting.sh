#!/bin/sh

SCRIPT="scan_benchmark.commons.plotting.plotting"

RESULTS_VLM="scan_benchmark/vlm/performance_surrogate/results"
PLOTS_VLM="scan_benchmark/vlm/performance_surrogate/plots"

echo "Running plotting for VLM benchmark predictors comparison"
python -m  "$SCRIPT" \
  --results_root "$RESULTS_VLM" \
  --plot_root "$PLOTS_VLM"

echo "Done."
