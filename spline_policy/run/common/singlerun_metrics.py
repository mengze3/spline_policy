import json
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Dict, Optional


def read_json_log(log_path: str) -> pd.DataFrame:
    """
    Read a single log file and return it as a DataFrame.

    Args:
        log_path (str): Path to the log file.

    Returns:
        pd.DataFrame: DataFrame containing the log data.
    """
    with open(log_path, "r") as f:
        lines = f.readlines()
        data = [json.loads(line) for line in lines if line.strip()]
    return pd.DataFrame(data)


def extract_metrics(
    log_df: pd.DataFrame, key: str, metrics_to_plot: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Extract specified metrics from a single log DataFrame.
    Computes 'max', 'min', 'last_10_avg', and 'top_5_avg' for each metric in metrics_to_plot.
    Combines 'key' with each metric to form the actual column name (e.g., 'test' + 'mean_score' = 'test/mean_score').

    Args:
        log_df (pd.DataFrame): DataFrame containing log data.
        key (str): Prefix key for the metric column (e.g., 'test').
        metrics_to_plot (List[str]): List of metric suffixes to compute (e.g., ['mean_score']).

    Returns:
        Dict[str, Dict[str, float]]: Dictionary of computed metrics, including 'max', 'min', 'last_10_avg', and 'top_5_avg'.
    """
    result = {}

    for metric in metrics_to_plot:
        column_name = f"{key}/{metric}" if key else metric

        if column_name in log_df.columns:
            # Drop NaN values and convert to a NumPy array for efficient computation
            key_data = log_df[column_name].dropna().to_numpy()
            
            if len(key_data) > 0:
                # --- New Metric Calculation: top_5_avg ---
                # Sort the data in descending order to easily find the largest values
                sorted_data_desc = np.sort(key_data)[::-1]
                
                # Calculate the average of the top 5 values.
                # If there are fewer than 5 data points, average all available points.
                if len(sorted_data_desc) >= 5:
                    top_5_avg = float(np.mean(sorted_data_desc[:5]))
                else:
                    top_5_avg = float(np.mean(sorted_data_desc))

                result[metric] = {
                    "max": float(np.max(key_data)),
                    "min": float(np.min(key_data)),
                    "last_10_avg": float(np.mean(key_data[-10:]))
                    if len(key_data) >= 10
                    else float(np.mean(key_data)),
                    "top_5_avg": top_5_avg,  # Add the new metric to the result
                }
            else:
                # Handle case where the column exists but is empty after dropping NaNs
                result[metric] = {"max": 0.0, "min": 0.0, "last_10_avg": 0.0, "top_5_avg": 0.0}
        else:
            # Handle case where the column does not exist in the DataFrame
            result[metric] = {"max": 0.0, "min": 0.0, "last_10_avg": 0.0, "top_5_avg": 0.0}
            print(f"Warning: Column {column_name} not found in log_df columns")

    return result


def collect_benchmark_data(
    metrics_dir: str,
    key: str,
    metrics_to_plot: List[str],
    sort_by: Optional[str] = "name",
    name_pattern: Optional[str] = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Collect metrics data from all experiments in the metrics directory.
    Returns a dictionary with method names as keys and nested metric dictionaries as values.

    Args:
        metrics_dir (str): Directory containing log files (json.txt format).
        key (str): Prefix key for the metric column (e.g., 'test').
        metrics_to_plot (List[str]): List of metric suffixes to compute.

    Returns:
        Dict[str, Dict[str, Dict[str, float]]]: Dictionary of benchmark data for each method.
    """
    metrics_dir = pathlib.Path(metrics_dir)
    assert metrics_dir.is_dir(), f"Metrics directory {metrics_dir} does not exist!"

    benchmark_data = {}
    # Use custom pattern if provided, otherwise default to "*.json.txt"
    pattern = name_pattern if name_pattern else "*.json.txt"
    log_files = list(metrics_dir.glob(pattern))
    if not log_files:
        print(f"No log files found in {metrics_dir} with pattern {pattern}!")
        return benchmark_data

    # Sort log files based on the specified criterion
    if sort_by == "name":
        log_files.sort(key=lambda x: x.stem.lower())  # Sort alphabetically by filename
    elif sort_by == "number":
        # Sort by numerical value if filename contains numbers
        log_files.sort(
            key=lambda x: float("".join(filter(str.isdigit, x.stem)))
            if any(c.isdigit() for c in x.stem)
            else float("inf")
        )
    # Add more sorting criteria if needed (e.g., by date, custom key)

    for log_file in log_files:
        method_name = log_file.stem  # Use filename as method name
        log_df = read_json_log(str(log_file))
        metrics = extract_metrics(log_df, key, metrics_to_plot)
        benchmark_data[method_name] = metrics

    return benchmark_data


def plot_benchmark(
    benchmark_data: Dict[str, Dict[str, Dict[str, float]]],
    output_dir: str,
    key: str,
    metrics: List[str],
):
    """
    Plot benchmark results with methods on x-axis and metric values on y-axis, and save as images.
    Plots separate charts for 'max', 'min', and 'last_10_avg' for each metric.

    Args:
        benchmark_data (Dict[str, Dict[str, Dict[str, float]]]): Dictionary of benchmark data.
        output_dir (str): Directory to save the plots.
        key (str): Prefix key for the metric column (used in display).
        metrics (List[str]): List of metric suffixes to plot.
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    stat_types = ["max", "min", "last_10_avg", "top_5_avg"]

    for metric in metrics:
        for stat in stat_types:
            methods = list(benchmark_data.keys())
            values = [benchmark_data[method][metric][stat] for method in methods]

            plt.figure(figsize=(10, 6))
            plt.bar(methods, values, color="skyblue")
            plt.xlabel("Methods")
            display_name = f"{key}/{metric}" if key else metric
            plt.ylabel(f"{stat.capitalize()} {display_name}")
            plt.title(f"Benchmark Results for {display_name} ({stat})")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            output_path = (
                output_dir / f'benchmark_{display_name.replace("/", "_")}_{stat}.png'
            )
            plt.savefig(output_path)
            plt.close()
            print(f"Saved plot to {output_path}")


def generate_benchmark_table(
    benchmark_data: Dict[str, Dict[str, Dict[str, float]]],
    output_dir: str,
    key: str,
    metrics: List[str],
):
    """
    Generate a benchmark table from the collected data and save it as a CSV file.

    Args:
        benchmark_data (Dict[str, Dict[str, Dict[str, float]]]): Dictionary of benchmark data.
        output_dir (str): Directory to save the table.
        key (str): Prefix key for the metric column (used in display).
        metrics_to_plot (List[str]): List of metric suffixes to include in the table.
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Prepare data for the table
    stat_types = ["max", "min", "last_10_avg", "top_5_avg"]
    table_data = []
    methods = list(benchmark_data.keys())

    # Create header
    header = ["Method"]
    for metric in metrics:
        display_name = f"{key}/{metric}" if key else metric
        for stat in stat_types:
            header.append(f"{display_name} ({stat})")
    table_data.append(header)

    # Fill data rows
    for method in methods:
        row = [method]
        for metric in metrics:
            for stat in stat_types:
                value = benchmark_data[method][metric][stat]
                row.append(value)
        table_data.append(row)

    # Convert to DataFrame and save as CSV
    table_df = pd.DataFrame(table_data[1:], columns=table_data[0])
    output_path = output_dir / "benchmark_table.csv"
    table_df.to_csv(output_path, index=False)
    print(f"Saved benchmark table to {output_path}")

    return table_df


def generate_benchmark_results(
    metrics_dir: str,
    key: str = "test",
    metrics: List[str] = ["mean_score", "mean_accel", "mean_jerk"],
) -> None:
    """
    Generate benchmark plots and tables from log files and save them.
    This function collects data once and generates both visualizations and tables.

    Args:
        metrics_dir (str): Directory containing log files (json.txt format).
        output_dir (str): Directory to save the results, default is "benchmark_results".
        key (str): Prefix key for metric columns, default is "test".
        metrics (List[str]): List of metric suffixes to process, default matches user requirement.
    """
    # Collect data once
    benchmark_data = collect_benchmark_data(metrics_dir, key, metrics)
    if not benchmark_data:
        print("No data to process!")
        return

    # Generate plots
    plot_benchmark(benchmark_data, metrics_dir, key, metrics)

    # Generate table
    generate_benchmark_table(benchmark_data, metrics_dir, key, metrics)

    print(f"Benchmark results (plots and table) saved to {metrics_dir}")
