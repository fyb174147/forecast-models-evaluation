from __future__ import annotations
import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
import numpy as np
import pandas as pd
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def _one_sided_normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))

def dm_test_pvalue(loss_x: pd.Series, loss_y: pd.Series) -> float:
    # One-sided DM test p-value for H1: model_x has lower expected loss than model_y.
    aligned = pd.concat([loss_x, loss_y], axis=1, join="inner").dropna()
    if len(aligned) < 5:
        return np.nan
    # Loss differential series d_t = L_x,t - L_y,t.
    d_t = aligned.iloc[:, 0].to_numpy() - aligned.iloc[:, 1].to_numpy()
    mean_d = float(np.mean(d_t))
    centered_d = d_t - mean_d
    var_d = float(np.mean(centered_d**2))
    if var_d <= 0.0:
        if mean_d < 0.0:
            return 0.0
        if mean_d > 0.0:
            return 1.0
        return 0.5
    dm_stat = mean_d / math.sqrt(var_d / len(d_t))
    return float(np.clip(_one_sided_normal_cdf(dm_stat), 0.0, 1.0))

def find_model_dirs(dataset_dir: Path) -> List[Path]:
    return sorted([path for path in dataset_dir.iterdir() if path.is_dir()])

def _find_target_file(dir_path: Path, target_filename: str) -> Optional[Path]:
    if not dir_path.exists() or not dir_path.is_dir():
        return None
    for candidate in dir_path.iterdir():
        if candidate.is_file() and candidate.suffix.lower() == ".csv" and candidate.name.lower() == target_filename.lower():
            return candidate
    return None

def _iter_sampling_subdirs(sampling_dir: Path) -> List[Path]:
    if not sampling_dir.exists() or not sampling_dir.is_dir():
        return []
    return sorted([path for path in sampling_dir.iterdir() if path.is_dir()])

def _collect_target_files(model_dir: Path, target_filename: str, sampling: str = "Sampling_100") -> List[Path]:
    sampling_dir = model_dir / sampling
    if not sampling_dir.exists():
        return []
    direct = _find_target_file(sampling_dir, target_filename)
    if direct is not None:
        return [direct]
    run1 = _find_target_file(sampling_dir / "Run_1", target_filename)
    if run1 is not None:
        return [run1]
    target_files: List[Path] = []
    for child in _iter_sampling_subdirs(sampling_dir):
        target_file = _find_target_file(child, target_filename)
        if target_file is not None:
            target_files.append(target_file)
    return target_files

def discover_target_filenames(dataset_dir: Path, sampling: str = "Sampling_100") -> List[str]:
    targets: Set[str] = set()
    for model_dir in find_model_dirs(dataset_dir):
        sampling_dir = model_dir / sampling
        if not sampling_dir.exists() or not sampling_dir.is_dir():
            continue
        search_dirs = [sampling_dir]
        search_dirs.extend(_iter_sampling_subdirs(sampling_dir))
        for search_dir in search_dirs:
            for candidate in search_dir.iterdir():
                if not candidate.is_file() or candidate.suffix.lower() != ".csv":
                    continue
                name_lower = candidate.name.lower()
                if name_lower.endswith("_forecast_vs_actual.csv"):
                    targets.add(name_lower)
    return sorted(targets)

def _read_single_load_file(load_csv: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(load_csv)
    except Exception:
        return None
    normalized_cols = {col.lower().strip(): col for col in df.columns}
    datetime_col = normalized_cols.get("datetime")
    actual_col = normalized_cols.get("actual")
    forecast_col = normalized_cols.get("forecast")
    if actual_col is None or forecast_col is None:
        return None
    if datetime_col is not None:
        dt = pd.to_datetime(df[datetime_col], errors="coerce", utc=True)
    else:
        dt = pd.to_datetime(df.index, errors="coerce", utc=True)
    actual = pd.to_numeric(df[actual_col], errors="coerce")
    forecast = pd.to_numeric(df[forecast_col], errors="coerce")
    valid = (~dt.isna()) & (~actual.isna()) & (~forecast.isna())
    if not valid.any():
        return None
    parsed = pd.DataFrame(
        {
            "datetime": pd.DatetimeIndex(dt[valid]),
            "actual": actual[valid].to_numpy(),
            "forecast": forecast[valid].to_numpy(),
        }
    )
    return parsed.sort_values("datetime")

def read_target_series(
    model_dir: Path,
    target_filename: str,
    sampling: str = "Sampling_100",
) -> Tuple[Optional[pd.Series], Optional[pd.Series], Optional[Path]]:
    target_files = _collect_target_files(model_dir, target_filename, sampling)
    if not target_files:
        return None, None, None
    run_frames: List[pd.DataFrame] = []
    for target_file in target_files:
        parsed = _read_single_load_file(target_file)
        if parsed is not None:
            run_frames.append(parsed)
    if not run_frames:
        return None, None, target_files[0]
    if len(run_frames) == 1:
        merged = run_frames[0].copy()
    else:
        aligned_runs: List[pd.DataFrame] = []
        for idx, frame in enumerate(run_frames):
            aligned = frame.set_index("datetime")[["actual", "forecast"]].rename(
                columns={"actual": f"actual_{idx}", "forecast": f"forecast_{idx}"}
            )
            aligned_runs.append(aligned)
        merged_runs = pd.concat(aligned_runs, axis=1, join="inner").dropna()
        if merged_runs.empty:
            return None, None, target_files[0]
        actual_cols = [col for col in merged_runs.columns if col.startswith("actual_")]
        forecast_cols = [col for col in merged_runs.columns if col.startswith("forecast_")]
        merged = pd.DataFrame(
            {
                "datetime": merged_runs.index,
                "actual": merged_runs[actual_cols].mean(axis=1).to_numpy(),
                "forecast": merged_runs[forecast_cols].mean(axis=1).to_numpy(),
            }
        )
    index = pd.DatetimeIndex(merged["datetime"])
    actual_series = pd.Series(merged["actual"].to_numpy(), index=index).sort_index()
    forecast_series = pd.Series(merged["forecast"].to_numpy(), index=index).sort_index()
    return actual_series, forecast_series, target_files[0]

def build_dm_matrix(losses: Dict[str, Optional[pd.Series]], model_names: Iterable[str]) -> np.ndarray:
    names = list(model_names)
    n_models = len(names)
    matrix = np.full((n_models, n_models), np.nan, dtype=float)
    for i, model_x in enumerate(names):
        for j, model_y in enumerate(names):
            if i == j:
                matrix[i, j] = 1.0
                continue
            loss_x = losses.get(model_x)
            loss_y = losses.get(model_y)
            if loss_x is None or loss_y is None:
                continue
            matrix[i, j] = dm_test_pvalue(loss_x, loss_y)
    return matrix

def _compute_common_timestamp_index(losses: Dict[str, Optional[pd.Series]]) -> pd.DatetimeIndex:
    available = [series.index for series in losses.values() if series is not None and not series.empty]
    if not available:
        return pd.DatetimeIndex([])
    common_index = available[0]
    for idx in available[1:]:
        common_index = common_index.intersection(idx)
    return pd.DatetimeIndex(common_index).sort_values()

def _align_losses_to_index(
    losses: Dict[str, Optional[pd.Series]],
    common_index: pd.DatetimeIndex,
) -> Dict[str, Optional[pd.Series]]:
    aligned: Dict[str, Optional[pd.Series]] = {}
    for model_name, series in losses.items():
        if series is None:
            aligned[model_name] = None
            continue
        aligned_series = series.reindex(common_index).dropna()
        aligned[model_name] = aligned_series if len(aligned_series) == len(common_index) else None
    return aligned

def _build_losses_from_actual_forecast(
    actual_series: Optional[pd.Series],
    forecast_series: Optional[pd.Series],
) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
    if actual_series is None or forecast_series is None:
        return None, None
    aligned = pd.concat(
        [actual_series.rename("actual"), forecast_series.rename("forecast")],
        axis=1,
        join="inner",
    ).dropna()
    if aligned.empty:
        return None, None
    err = aligned["actual"].to_numpy() - aligned["forecast"].to_numpy()
    idx = aligned.index
    mae_losses = pd.Series(np.abs(err), index=idx).sort_index()
    rmse_losses = pd.Series(err**2, index=idx).sort_index()
    return mae_losses, rmse_losses

def _shift_xticklabels_right(ax: plt.Axes, fig: plt.Figure, points: float = 2.0) -> None:
    # Move x tick labels slightly right for readability in dense, rotated labels.
    offset = ScaledTranslation(points / 30.0, 0.0, fig.dpi_scale_trans)
    for label in ax.get_xticklabels():
        label.set_transform(label.get_transform() + offset)

def plot_dm_heatmaps(
    model_names: List[str],
    dm_matrix_mae: np.ndarray,
    dm_matrix_rmse: np.ndarray,
    output_pdf: Path,
    font_size: int = 9,
    legend_names: Optional[Dict[str, str]] = None,
) -> None:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8, 4), constrained_layout=False)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.04], wspace=0.08)
    ax_mae = fig.add_subplot(grid[0, 0])
    ax_rmse = fig.add_subplot(grid[0, 1])
    cax = fig.add_subplot(grid[0, 2])
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_over("yellow")
    cmap.set_bad("white")
    im = ax_mae.imshow(dm_matrix_mae.T, cmap=cmap, vmin=0.0, vmax=0.1, aspect="auto")
    ax_rmse.imshow(dm_matrix_rmse.T, cmap=cmap, vmin=0.0, vmax=0.1, aspect="auto")
    ticks = np.arange(len(model_names))
    display_names = [(legend_names or {}).get(name, name) for name in model_names]
    ax_mae.set_title("DM test (MAE loss)", fontsize=font_size)
    ax_mae.set_xticks(ticks)
    ax_mae.set_yticks(ticks)
    ax_mae.set_xticklabels(display_names, rotation=90, ha="right", fontsize=font_size)
    _shift_xticklabels_right(ax_mae, fig)
    ax_mae.set_yticklabels(display_names, fontsize=font_size)
    ax_mae.text(-0.04, 1.035, "(a)", transform=ax_mae.transAxes, ha="left", va="top", fontsize=font_size)
    ax_rmse.set_title("DM test (RMSE loss)", fontsize=font_size)
    ax_rmse.set_xticks(ticks)
    ax_rmse.set_yticks(ticks)
    ax_rmse.set_xticklabels(display_names, rotation=90, ha="right", fontsize=font_size)
    _shift_xticklabels_right(ax_rmse, fig)
    ax_rmse.set_yticklabels([])
    ax_rmse.tick_params(axis="y", left=False)
    ax_rmse.text(-0.04, 1.035, "(b)", transform=ax_rmse.transAxes, ha="left", va="top", fontsize=font_size)
    cbar = fig.colorbar(im, cax=cax, extend="max")
    cbar.set_label("p-value", rotation=90, labelpad=0, fontsize=font_size)
    cbar.ax.tick_params(labelsize=font_size)
    # Eliminate outer white margins in the saved figure.
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0.12)
    fig.savefig(output_pdf, format="pdf", dpi=300, bbox_inches="tight", pad_inches=0)
    output_png = output_pdf.with_suffix(".png")
    fig.savefig(output_png, format="png", dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

DEFAULT_LEGEND_NAMES = {
    "KNNRegression": "KNN Reg.",
    "NaiveDrift": "Naive Drift",
    "NaiveMovingAverage": "Naive MA",
    "MQCNN": "MQ-CNN",
    "TemporalFusionTransformer": "TFT",
    "MQRNN": "MQ-RNN",
    "VanillaTransformer": "Vanilla Trans.",
    "TimerXL": "Timer-XL",
    "PatchTST": "PatchTST",
    "iTransformer": "iTransformer",
    "TimeXer": "TimeXer",
    "TimesNet": "TimesNet",
    "NBEATS": "N-BEATS",
    "NHITS": "N-HITS",
    "NLinear": "N-Linear",
    "DeepNPTS": "DeepNPTS",
    "DeepAR": "DeepAR",
    "DeepFactor": "DeepFactor",
    "WaveNet": "WaveNet",
    "Chronos": "Chronos",
    "MOIRAI": "MOIRAI",
    "TimeGPT": "TimeGPT",
    "TimesFM": "TimesFM",
    "LightGBM": "LightGBM",
}

def run_dm_for_dataset(
    dataset_name: str,
    base_dir: Path = PROJECT_ROOT,
    output_dir: Optional[Path] = None,
    font_size: int = 9,
    legend_names: Optional[Dict[str, str]] = None,
    sampling: str = "Sampling_100",
) -> List[Path]:
    if legend_names is None:
        legend_names = DEFAULT_LEGEND_NAMES
    dataset_dir = base_dir / "results" / f"results_{dataset_name}"
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset results folder not found: {dataset_dir}")
    output_files: List[Path] = []
    model_dirs = find_model_dirs(dataset_dir)
    if not model_dirs:
        print(f"Warning: no model folders found in {dataset_dir}")
        return output_files
    target_filenames = discover_target_filenames(dataset_dir, sampling)
    if not target_filenames:
        print(f"Warning: no *_forecast_vs_actual.csv files found in {dataset_dir} for {sampling}")
        return output_files
    raw_model_names = [model_dir.name for model_dir in model_dirs]
    for target_filename in target_filenames:
        actual_series_map: Dict[str, Optional[pd.Series]] = {}
        forecast_series_map: Dict[str, Optional[pd.Series]] = {}
        missing_loss_files: List[str] = []
        for model_dir in model_dirs:
            model_name = model_dir.name
            actual_series, forecast_series, _ = read_target_series(model_dir, target_filename, sampling)
            actual_series_map[model_name] = actual_series
            forecast_series_map[model_name] = forecast_series
            if actual_series is None or forecast_series is None:
                missing_loss_files.append(model_name)
        # Filter down to models that actually produced forecast series for this target
        available_models = [m for m in raw_model_names if forecast_series_map.get(m) is not None]
        if len(available_models) < 2:
            continue
        available_forecasts = {m: forecast_series_map[m] for m in available_models}
        common_index = _compute_common_timestamp_index(available_forecasts)
        if len(common_index) < 5:
            print(
                f"Warning ({dataset_dir.name} | {sampling} | {target_filename}): insufficient shared timestamps for DM test "
                f"(shared={len(common_index)}). Skipping target."
            )
            continue
        actual_series_map = _align_losses_to_index(actual_series_map, common_index)
        forecast_series_map = _align_losses_to_index(forecast_series_map, common_index)

        mae_losses: Dict[str, Optional[pd.Series]] = {}
        rmse_losses: Dict[str, Optional[pd.Series]] = {}
        valid_models: List[str] = []
        for model_name in available_models:
            mae_series, rmse_series = _build_losses_from_actual_forecast(
                actual_series_map.get(model_name), forecast_series_map.get(model_name)
            )
            if mae_series is not None and rmse_series is not None:
                mae_losses[model_name] = mae_series
                rmse_losses[model_name] = rmse_series
                valid_models.append(model_name)

        if len(valid_models) < 2:
            continue

        dm_matrix_mae = build_dm_matrix(mae_losses, valid_models)
        dm_matrix_rmse = build_dm_matrix(rmse_losses, valid_models)
        target_name = target_filename.replace("_forecast_vs_actual.csv", "").lower()
        result_dir = output_dir or base_dir / "results" / "dm_test"
        sampling_tag = sampling.lower().replace("sampling_", "samp")
        output_pdf = result_dir / dataset_name / f"dm_test_{target_name}_{sampling_tag}.pdf"
        plot_dm_heatmaps(
            valid_models,
            dm_matrix_mae,
            dm_matrix_rmse,
            output_pdf,
            font_size,
            legend_names,
        )
        output_files.append(output_pdf)
    return output_files

if __name__ == "__main__":
    legend_names = {
        "KNNRegression": "KNN Reg.",
        "NaiveDrift": "Naive Drift",
        "NaiveMovingAverage": "Naive MA",
        "MQCNN": "MQ-CNN",
        "TemporalFusionTransformer": "TFT",
        "MQRNN": "MQ-RNN",
        "VanillaTransformer": "Vanilla Trans.",
        "TimerXL": "Timer-XL",
        "PatchTST": "PatchTST",
        "iTransformer": "iTransformer",
        "TimeXer": "TimeXer",
        "TimesNet": "TimesNet",
        "NBEATS": "N-BEATS",
        "NHITS": "N-HITS",
        "NLinear": "N-Linear",
        "DeepNPTS": "DeepNPTS",
        "DeepAR": "DeepAR",
        "DeepFactor": "DeepFactor",
        "WaveNet": "WaveNet",
        "Chronos": "Chronos",
        "MOIRAI": "MOIRAI",
        "TimeGPT": "TimeGPT",
        "TimesFM": "TimesFM",
        "LightGBM": "LightGBM",
    }
    parser = argparse.ArgumentParser(description="Compute and plot DM test heatmaps for model comparisons.")
    parser.add_argument("--datasets", nargs="+", choices=["belgium", "germany", "london", "zonnedael"], default=["belgium"], help="Datasets to compare.")
    parser.add_argument("--sampling", type=str, default="all", choices=["all", "Sampling_100", "Sampling_25", "Sampling_33", "Sampling_50"], help="Sampling folder name or 'all'.")
    parser.add_argument("--base_dir", type=Path, default=PROJECT_ROOT, help="Project directory containing results.")
    parser.add_argument("--output_dir", type=Path, help="Output root; otherwise <base_dir>/results/dm_test.")
    parser.add_argument("--font_size", type=int, default=8, help="Plot font size.")
    args = parser.parse_args()
    
    samplings_to_run = ["Sampling_25", "Sampling_33", "Sampling_50", "Sampling_100"] if args.sampling == "all" else [args.sampling]

    for dataset_name in args.datasets:
        for s in samplings_to_run:
            output_files = run_dm_for_dataset(dataset_name, args.base_dir, args.output_dir, args.font_size, DEFAULT_LEGEND_NAMES, s)
            for out in output_files:
                print(f"Saved DM test heatmap to: {out} and {out.with_suffix('.png')}")
