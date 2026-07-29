# VLM-ScAn-Bench: A Surrogate Benchmark for Scaling Analysis of Vision-Language Models

This repository provides VLM-ScAn-Bench, a surrogate benchmark for evaluating scaling analysis (ScAn) methodology on
vision-language models (VLMs). The benchmark approximates the mapping from training configurations to performance,
enabling fast evaluation without training full models.

## Requirements

We recommend using a conda environment.

```bash
conda create -n vlm-scan-benchmark python=3.11
conda activate vlm-scan-benchmark
pip install .
```

Install pytorch with CUDA, if you want to utilize the GPU.

## Training and evaluation

To train and get the main performance results for the surrogate benchmark, run the provided shell scripts. Change DEVICE
to 'cuda' in train_surrogates.sh to use the GPU.

### Performance predictor surrogate

```bash
bash scan_benchmark/vlm/performance_surrogate/train/train_surrogates.sh
python -m scan_benchmark.commons.surrogate_evaluation.main_performance
```

### Divergence predictor surrogate

```bash
bash scan_benchmark/vlm/divergence_surrogate/train.sh
```

## Results

### Surrogate Performance

| Model             |            RMSE ↓ |      Spearman ρ ↑ |
|-------------------|------------------:|------------------:|
| AutoGluon         |     0.261 ± 0.081 |     0.978 ± 0.012 |
| Ensemble LightGBM |     0.416 ± 0.068 |     0.955 ± 0.017 |
| Ensemble Mix      |     0.369 ± 0.091 |     0.954 ± 0.021 |
| Ensemble XGBoost  |     0.350 ± 0.087 |     0.956 ± 0.018 |
| **TabPFN**        | **0.208 ± 0.079** | **0.984 ± 0.009** |

## Extrapolation results

To get the extrapolation results, run the following command:

```bash
python -m scan_benchmark.commons.surrogate_evaluation.extrapolation
```

## Query time

To see how much time it takes for the API to return responses, please run the script below:

```bash
python -m scan_benchmark.commons.query_time.query_time
```

## Feature ablation

To run the feature ablation experiments, run the following command:

```bash
python -m scan_benchmark.commons.surrogate_evaluation.leave_one_out_ablation
python -m scan_benchmark.commons.surrogate_evaluation.main_performance --root_dir 'scan_benchmark/vlm/performance_surrogate/ablation_results/' --ablation
python -m scan_benchmark.commons.plotting.ablation_plotting
```

## API usage

Refer to [VLM API](scan_benchmark/vlm/api.py) for API usage.

## Additional

### Data

The repository includes pre-collected configuration-performance datasets used to train the surrogate models.

- **VLM data**:  
  `scan_benchmark/vlm/performance_surrogate/splits`  
  Contains training and test splits for VLM surrogate modeling.
- **VLM divergence data**:  
  `scan_benchmark/vlm/divergence_surrogate/splits`  
  Contains data and models for predicting failed (diverged) configurations.

A separate repository was used to collect the data. For the full information, please visit
the [scaling_studies_vlm repo](https://github.com/automl/scaling_studies_vlm).

### Plotting

For additional plottings on comparing the surrogate predictors, run bash sript:

```bash
bash scan_benchmark/commons/plotting/plotting.sh
```

## Contributing

Contributions are welcome. Please open an issue or a pull request to address specific changes.
