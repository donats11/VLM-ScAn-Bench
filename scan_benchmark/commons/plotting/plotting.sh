#!/bin/sh

SCRIPT="scan_benchmark.commons.plotting.plotting"

RESULTS_VLM="scan_benchmark/vlm/performance_surrogate/results/predictors"
PLOTS_VLM="scan_benchmark/vlm/performance_surrogate/plots"

echo "Running plotting for VLM benchmark predictors comparison"
python -m  "$SCRIPT" \
  --results-root "$RESULTS_VLM" \
  --plot-root "$PLOTS_VLM"

echo "Done."
