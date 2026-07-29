from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt


name_map = {
    "without_lr": "Learning rate",
    "without_wd": "Weight decay",
    "without_warmup_fraction": "Warmup fraction",
    "without_beta1": r"$\beta_1$",
    "without_beta2": r"$\beta_2$",
    "without_eps": r"$\epsilon$",
    "without_total_samples_planned": "Train samples",
    "without_vision_width": "Vision width",
    "without_text_width": "Text width",
    "without_global_batch_size": "Global batch size",
    "without_training_progress": "Training progress",
    "without_lr_ratio": "Learning-rate ratio",
}


def build_results_dict(root_dir: Path):
    results = {}

    for folder in root_dir.iterdir():
        if not folder.is_dir():
            continue

        file = folder / "results_summary.csv"
        if not file.exists():
            continue

        df = pd.read_csv(file, header=[0, 1], index_col=0)

        key = name_map.get(folder.name, folder.name)

        results[key] = {
            "rmse": df[("rmse", "mean")].iloc[0],
            "spearman": df[("spearman", "mean")].iloc[0],
        }

    return results


if __name__ == "__main__":
    root_dir = Path("scan_benchmark/vlm/performance_surrogate/ablation_results")

    baseline_rmse = 0.203
    baseline_spearman = 0.985

    results = build_results_dict(root_dir)

    df = pd.DataFrame(results).T.reset_index()
    df.columns = ["Feature", "RMSE", "Spearman"]

    df["Delta RMSE"] = df["RMSE"] - baseline_rmse
    df["Delta Spearman"] = baseline_spearman - df["Spearman"]

    df = df.sort_values("Delta RMSE", ascending=True)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 6),
        sharey=True,
    )

    bar_color = "teal"

    axes[0].barh(
        df["Feature"],
        df["Delta RMSE"],
        color=bar_color,
    )
    axes[0].axvline(
        0,
        color="black",
        linewidth=1.5,
        zorder=5,
    )
    axes[0].set_xlabel(r"$\Delta$ RMSE", fontsize=14)
    axes[0].set_title("RMSE", fontsize=14)

    rmse_max = df["Delta RMSE"].max()
    axes[0].set_xlim(-0.03 * rmse_max, rmse_max * 1.05)

    axes[1].barh(
        df["Feature"],
        df["Delta Spearman"],
        color=bar_color,
    )
    axes[1].axvline(
        0,
        color="black",
        linewidth=1.5,
        zorder=5,
    )
    axes[1].set_xlabel(r"$\Delta$ Spearman $\rho$", fontsize=14)
    axes[1].set_title("Spearman", fontsize=14)

    spearman_max = df["Delta Spearman"].max()
    axes[1].set_xlim(-0.03 * spearman_max, spearman_max * 1.05)

    axes[0].tick_params(axis="both", labelsize=12)
    axes[1].tick_params(axis="both", labelsize=12)

    plt.tight_layout()
    plt.savefig(
        "feature_ablation_side_by_side.pdf",
        bbox_inches="tight",
    )
    plt.show()