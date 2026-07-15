import os
import pandas as pd
import re
from collections import defaultdict
from pathlib import Path
import shutil
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import argparse
from typing import List, Optional

color_purple = '#B2A2D9'
color_cyan = '#AEEEEE'
default_color = '#CCCCCC'

color_map = {
    'pure_horizon16.json': color_purple,
    'spline_horizon16_nbSeg6.json': color_cyan,
    'Some Other Method': default_color
}

DP3ENCODER_GFLOPS = 0.044
DP3UNET16_GFLOPS = 0.1654
DP3UNET08_GFLOPS = 0.0838

DPENCODER_GFLOPS = 0.4944
DPUNET16_GFLOPS = 0.1654
DPUNET8_GFLOPS = 0.0838

FLOPS_MAP = {
    'adroit_door': {
        'pure':   {'16': 1.0},
        'spline': {'16': (DP3ENCODER_GFLOPS+DP3UNET08_GFLOPS)/(DP3ENCODER_GFLOPS+DP3UNET16_GFLOPS)}
    },
    'adroit_pen': {
        'pure':   {'16': 1.0},
        'spline': {'16': (DP3ENCODER_GFLOPS+DP3UNET08_GFLOPS)/(DP3ENCODER_GFLOPS+DP3UNET16_GFLOPS)}
    },
    'dexart_laptop': {
        'pure':   {'16': 1.0},
        'spline': {'16': (DP3ENCODER_GFLOPS+DP3UNET08_GFLOPS)/(DP3ENCODER_GFLOPS+DP3UNET16_GFLOPS)}
    },
    'pusht_image': {
        'pure':   {'16': 1.0},
        'spline': {'16': (DPENCODER_GFLOPS+DPUNET8_GFLOPS)/(DPENCODER_GFLOPS+DPUNET16_GFLOPS)}
    },
    'can_image_abs': {
        'pure':   {'16': 1.0},
        'spline': {'16': (DPENCODER_GFLOPS+DPUNET8_GFLOPS)/(DPENCODER_GFLOPS+DPUNET16_GFLOPS)}
    },
    # Default values for any other task not listed above
    'default': {
        'pure':   {'16': 1.0},
        'spline': {'16': DPUNET8_GFLOPS/DPUNET16_GFLOPS}
    }
}


# Helper functions (get_base_method_name, analyze_all_tasks) remain the same
def get_base_method_name(method_str: str) -> str:
    """
    Creates a canonical method name by removing the prefix ('diffusion_' or 'flowmatching_')
    and the seed information.

    Examples:
    - 'diffusion_pure_seed42_horizon16.json' -> 'pure_horizon16.json'
    - 'flowmatching_spline_seed42_horizon16_nbSeg6.json' -> 'spline_horizon16_nbSeg6.json'
    """
    # Step 1: Remove the prefix if it exists at the start of the string.
    # The pattern ^(diffusion_|flowmatching_) matches either prefix only at the beginning.
    name_without_prefix = re.sub(r'^(diffusion_|flowmatching_)', '', method_str)

    # Step 2: From the result, remove the seed information.
    final_name = re.sub(r'_seed\d+', '', name_without_prefix)

    return final_name


def get_method_type(method_str: str) -> str:
    return "spline" if "spline" in method_str else "pure"

def analyze_all_tasks(root_dir: Path) -> pd.DataFrame:
    """
    Reads all benchmark CSVs and compiles them into a single, long-format DataFrame
    without any aggregation. Preserves task-level information.
    """
    print("\n--- Step 1: Reading all raw data into a long-format table ---")
    all_records = []
    if not root_dir.is_dir():
        print(f"Error: Root directory '{root_dir}' not found.")
        return pd.DataFrame()

    for task_path in root_dir.iterdir():
        if not task_path.is_dir(): continue
        csv_path = task_path / 'benchmark_table.csv'
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            for record in df.to_dict('records'):
                method_full_name = record.pop('Method')
                base_method = get_base_method_name(method_full_name)
                for metric_name, value in record.items():
                    if pd.notna(value) and isinstance(value, (int, float)):
                        all_records.append({
                            'Task': task_path.name,
                            'Method': base_method,
                            'Original_Method': method_full_name, # Keep original name for feature extraction
                            'Metric': metric_name,
                            'Value': value
                        })
    print(f"  - Compiled {len(all_records)} data points from all tasks.")
    return pd.DataFrame(all_records)

def add_relative_flops(long_format_df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a long-format DataFrame, calculates task-specific relative flops for each
    original data point, and adds them as new rows.
    """
    if long_format_df.empty:
        return long_format_df

    print("\n--- Step 2: Calculating and adding 'relative flops' ---")
    
    # We only need to calculate flops once for each unique (Task, Method) combination
    unique_entries = long_format_df.drop_duplicates(subset=['Task', 'Original_Method'])
    
    flops_rows = []
    for _, row in unique_entries.iterrows():
        task_name = row['Task']
        method_full_name = row['Original_Method']
        base_method = row['Method']
        
        # Extract features to look up in the map
        method_type = get_method_type(method_full_name)
        horizon_match = re.search(r'horizon(\d+)', method_full_name)
        horizon = horizon_match.group(1) if horizon_match else '16'

        # Look up the value
        task_config = FLOPS_MAP.get(task_name, FLOPS_MAP['default'])
        type_config = task_config.get(method_type, {})
        flops_value = type_config.get(horizon, 1.0) # Default to 1.0 if specific horizon not found
        
        flops_rows.append({
            'Task': task_name,
            'Method': base_method,
            'Original_Method': method_full_name,
            'Metric': 'test/flops(relative)',
            'Value': flops_value
        })

    flops_df = pd.DataFrame(flops_rows)
    print(f"  - Generated {len(flops_df)} new rows for relative flops.")

    # Combine the original data with the new flops data
    return pd.concat([long_format_df, flops_df], ignore_index=True)


def print_task_relative_flops_summary(long_format_df: pd.DataFrame) -> None:
    print("\n" + "=" * 50)
    print("      Relative FLOPs By Task")
    print("=" * 50)

    if long_format_df.empty:
        print("No relative flops data was processed.")
        return

    task_names = sorted(long_format_df["Task"].dropna().unique())
    for task_name in task_names:
        task_config = FLOPS_MAP.get(task_name, FLOPS_MAP["default"])
        pure_value = task_config.get("pure", {}).get("16", 1.0)
        spline_value = task_config.get("spline", {}).get("16", 1.0)
        print(f"{task_name}: pure={pure_value:.4f}, spline={spline_value:.4f}")

def plot_metric_comparison(results_df: pd.DataFrame, metric: str, save_dir: Path, methods_to_plot: Optional[List[str]] = None):
    """
    Generates and saves a bar chart for a single metric, showing only specified methods
    or the top 4 by default.

    Args:
        results_df (pd.DataFrame): The DataFrame with all analysis results.
        metric (str): The name of the metric to plot.
        save_dir (Path): The directory to save the plot image.
        methods_to_plot (Optional[List[str]]): A specific list of methods to plot.
                                               If None, plots the top 4 methods for this metric.
    """
    print(f"  - Generating plot for: '{metric}'")

    # 1. Filter by metric
    metric_data = results_df[results_df['Metric'] == metric]

    if metric_data.empty:
        print(f"    - Warning: No data found for metric '{metric}'. Skipping plot.")
        return

    # 2. Determine which methods to show
    if methods_to_plot:
        # User has specified methods
        plot_data_filtered = metric_data[metric_data['Method'].isin(methods_to_plot)]
        print(f"    - Plotting specified methods: {methods_to_plot}")
    else:
        # Default behavior: get the top 4 performing methods for this metric
        plot_data_filtered = metric_data.sort_values(by='Mean_across_tasks', ascending=False).head(4)
        print("    - No methods specified, plotting top 4 performing methods.")

    # 3. Check if there's anything left to plot
    if plot_data_filtered.empty:
        print(f"    - Warning: None of the specified methods were found for metric '{metric}'. Skipping plot.")
        return

    # 4. Sort the final data for a clean visualization (highest bar first)
    plot_data_sorted = plot_data_filtered.sort_values(by='Mean_across_tasks', ascending=False)
    bar_colors = [color_map.get(method, default_color) for method in plot_data_sorted['Method']]

    # 5. Create the plot
    fig, ax = plt.subplots(figsize=(10, 13))
    bars = ax.bar(plot_data_sorted['Method'], plot_data_sorted['Mean_across_tasks'], color=bar_colors, width=0.5, alpha=0.8)
    for bar in bars:
        yval = bar.get_height()
        
        # Format the label text. '.3f' means format as a float with 3 decimal places.
        # You can change this to '.2f' for 2 decimal places, etc.
        label_text = f'{yval * 100:.1f}'
        
        ax.text(
            bar.get_x() + bar.get_width()/2.0,  # X position: center of the bar
            yval,                              # Y position: top of the bar
            label_text,                        # The text to display
            ha='center',                       # Horizontal alignment: center
            va='bottom',                       # Vertical alignment: place text bottom edge at yval
            fontsize=60                         # Adjust font size for readability
        )
        
    if not plot_data_sorted.empty:
        max_y = plot_data_sorted['Mean_across_tasks'].max()
        ax.set_ylim(0, max_y * 1.2)
    
    # Remove top and right borders (spines)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Thicken the remaining left and bottom borders
    ax.spines['left'].set_linewidth(4.0)
    ax.spines['bottom'].set_linewidth(2.0)
    # Thicken and lengthen the axis ticks
    ax.tick_params(axis='y', which='major', width=1.0, length=10)
    
    # Set titles and labels using the 'ax' object
    ax.set_title(f"Comparison of Methods for Metric:\n{metric}", fontsize=16, pad=20)
    ax.set_ylabel("Mean Value", fontsize=14, labelpad=15)
    
    # Set tick label properties
    ax.set_xticks([])
    ax.set_xlabel('')
    # Use fig.tight_layout() for object-oriented plotting
    fig.tight_layout()

    # Save the plot
    safe_filename = re.sub(r'[^a-zA-Z0-9_-]', '', metric)
    save_path = save_dir / f"{safe_filename}_comparison.png"
    fig.savefig(save_path, dpi=200) # Use fig.savefig and increase DPI for quality
    plt.close(fig) # Close the figure object

    print(f"    - Plot saved to: {save_path}")

    print(f"    - Plot saved to: {save_path}")
    
def main(metrics_dir: Path, metrics_to_plot: Optional[List[str]] = None, methods_to_plot: Optional[List[str]] = None):
    """
    Main execution function with the new, correct data flow.
    """
    # 1. Read all data into a long-format DataFrame (no aggregation)
    long_df = analyze_all_tasks(metrics_dir)
    
    # 2. Add 'relative flops' rows based on task and method details
    long_df_with_flops = add_relative_flops(long_df)
    
    print_task_relative_flops_summary(long_df_with_flops)

    # 3. Perform the final aggregation to get the mean values
    print("\n--- Step 3: Aggregating data and calculating final means ---")
    if long_df_with_flops.empty:
        final_results_df = pd.DataFrame()
    else:
        final_results_df = long_df_with_flops.groupby(['Method', 'Metric'])['Value'].mean().reset_index()
        final_results_df.rename(columns={'Value': 'Mean_across_tasks'}, inplace=True)
        print("  - Aggregation complete.")

    # --- Text Report & Plotting (use final_results_df) ---
    print("\n\n" + "="*50)
    print("      Final Consolidated Report (Across Tasks)")
    print("="*50)
    if final_results_df.empty:
        print("No data was processed.")
    else:
        # The rest of the reporting and plotting logic remains the same
        grouped_by_method = final_results_df.groupby('Method')
        for method_name, method_df in grouped_by_method:
            print(f"\n----- Method: {method_name} -----")
            display_table = method_df.drop(columns=['Method']).set_index('Metric')
            print(display_table)
            print("-" * (len(method_name) + 12))

    if metrics_to_plot:
        print("\n" + "="*50)
        print("      Generating Metric Comparison Plots")
        print("="*50)
        if final_results_df.empty:
            print("No data to plot.")
        else:
            for metric in metrics_to_plot:
                plot_metric_comparison(
                    results_df=final_results_df,
                    metric=metric,
                    save_dir=metrics_dir,
                    methods_to_plot=methods_to_plot
                )

    print("\nAnalysis complete.")


# ===================================================================
# COMMAND-LINE INTERFACE WITH THE NEW ARGUMENT
# ===================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze benchmark results, print a report, and generate plots.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=Path("metrics"),
        help="Path to the root directory containing the metrics.\n(default: %(default)s)"
    )
    parser.add_argument(
        "--plot-metrics",
        nargs='+',
        metavar="METRIC_NAME",
        default=["test/mean_score (top_5_avg)", "test/flops(relative)"],
        help="One or more metric names to plot.\n(default: %(default)s)"
    )
    # The new argument for specifying methods
    parser.add_argument(
        "--plot-methods",
        nargs='+',
        metavar="METHOD_NAME",
        default=["pure_horizon16.json", "spline_horizon16_nbSeg6.json"],
        help="A list of specific method names to include in the plots.\nIf not provided, the top 4 performing methods will be plotted by default."
    )
    args = parser.parse_args()
    main(
        metrics_dir=args.metrics_dir, 
        metrics_to_plot=args.plot_metrics,
        methods_to_plot=args.plot_methods # Pass the new argument to main
    )
