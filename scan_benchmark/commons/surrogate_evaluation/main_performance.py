import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir",
        type=Path,
        default=Path("scan_benchmark/vlm/performance_surrogate/results"),
    )

    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Collect results for ablation experiments.",
    )
    return parser.parse_args()


def valid_files(root: Path, filename: str, is_ablation: bool):
    for file in root.rglob(f"fold*/{filename}"):
        if is_ablation:
            yield file
        elif (
                "fit_with_intermediate" in file.parts
                and "pred_with_intermediate" in file.parts
        ):
            yield file


def get_model_name(file: Path, root: Path) -> str:
    return "/".join(file.relative_to(root).parts[:-5])


def summarize_metrics(root: Path, is_ablation: bool) -> pd.DataFrame:
    rows = []

    for file in valid_files(root, "val_loss.json", is_ablation):
        data = json.loads(file.read_text())
        last_metrics = data[max(data, key=lambda key: int(key))]

        rows.append({
            "model": get_model_name(file, root),
            **last_metrics,
        })

    return pd.DataFrame(rows).groupby("model").agg(["mean", "std"])


def summarize_metrics_by_bin(root: Path, is_ablation: bool) -> pd.DataFrame:
    rows = []

    for file in valid_files(root, "val_loss_by_bin.json", is_ablation):
        data = json.loads(file.read_text())["metrics_by_bin"]

        for bin_name, metrics in data.items():
            rows.append({
                "model": get_model_name(file, root),
                "bin": bin_name,
                **metrics,
            })

    return pd.DataFrame(rows).groupby(["model", "bin"]).agg(["mean", "std"])


def summarize_metrics_by_bin_top_performing(root: Path, is_ablation: bool) -> pd.DataFrame:
    rows = []

    for file in valid_files(root, "val_loss_top_performing_per_bin.json", is_ablation):
        data = json.loads(file.read_text())["metrics_by_bin"]

        for bin_name, metrics in data.items():
            rows.append({
                "model": get_model_name(file, root),
                "bin": bin_name,
                **metrics,
            })

    return pd.DataFrame(rows).groupby(["model", "bin"]).agg(["mean", "std"])


if __name__ == "__main__":
    args = parse_args()
    root = args.root_dir
    is_ablation = args.ablation

    if is_ablation:
        ablations = [
            "without_beta1",
            "without_beta2",
            "without_eps",
            "without_global_batch_size",
            "without_lr",
            "without_lr_ratio",
            "without_text_width",
            "without_total_samples_planned",
            "without_training_progress",
            "without_vision_width",
            "without_warmup_fraction",
            "without_wd",
        ]

        for ablation in ablations:
            ablation_root = root / ablation

            summarize_metrics(ablation_root, is_ablation).to_csv(
                ablation_root / "results_summary.csv"
            )
            summarize_metrics_by_bin(ablation_root, is_ablation).to_csv(
                ablation_root / "results_summary_by_bin.csv"
            )
            summarize_metrics_by_bin_top_performing(ablation_root, is_ablation).to_csv(
                ablation_root / "results_summary_by_bin_top_performing.csv"
            )

    else:
        summarize_metrics(root, is_ablation).to_csv(
            root / "results_summary.csv"
        )
        summarize_metrics_by_bin(root, is_ablation).to_csv(
            root / "results_summary_by_bin.csv"
        )
        summarize_metrics_by_bin_top_performing(root, is_ablation).to_csv(
            root / "results_summary_by_bin_top_performing.csv"
        )
