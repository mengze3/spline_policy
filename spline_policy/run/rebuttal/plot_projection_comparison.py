"""Paper strip: Numerical on the left, Analytical on the right, with raw position traces."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
import numpy as np
import torch

from fig6_projection import ROOT, Config, Spline, field, make_projector

# FIGURE_STYLE.md: PPT slides 9 and 22, with one color per method.
PURPLE, CYAN = "#AD6AEA", "#6DD2D9"
REFERENCE = ["#F1454F", "#E388B0", "#C0B4FF", "#7296FF"]


def background(ax, curve, reference, bounds, config):
    low, high = bounds
    xx, yy = np.meshgrid(np.linspace(low[0], high[0], 75), np.linspace(low[1], high[1], 90))
    grid = np.c_[xx.ravel(), yy.ravel()]
    velocity, _, _ = field(curve, grid, curve.project, config)
    ax.streamplot(
        xx,
        yy,
        velocity[:, 0].reshape(xx.shape),
        velocity[:, 1].reshape(xx.shape),
        color="#8AC6ED",
        density=0.65,
        linewidth=0.6,
        arrowsize=0.65,
        minlength=0.15,
        maxlength=2.0,
        zorder=0,
    )
    colors = LinearSegmentedColormap.from_list("reference", REFERENCE)(
        np.linspace(0, 1, len(reference) - 1)
    )
    ax.add_collection(
        LineCollection(
            np.stack([reference[:-1], reference[1:]], axis=1),
            colors=colors,
            linewidths=2.6,
            zorder=2,
        )
    )
    ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), aspect="equal")
    ax.set_axis_off()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "outputs/m1_projection")
    args = parser.parse_args()
    results = json.loads((args.input / "results.json").read_text())
    config = Config(**results["config"])
    saved = np.load(args.input / "rollouts.npz")
    torch.set_num_threads(1)
    curve = Spline(saved["demonstration"], config.segments)
    reference, _ = curve.evaluate(np.linspace(0, 1, 4000))
    counts = [config.samples, 0]
    paths = [saved[f"n{n}_dt{config.dt:g}_position"] for n in counts]
    time = saved[f"n0_dt{config.dt:g}_time"]
    metrics = [
        next(m for m in results["metrics"] if m["samples"] == n and m["dt"] == config.dt)
        for n in counts
    ]
    all_points = np.vstack([reference, *paths])
    bounds = all_points.min(0) - 0.045, all_points.max(0) + 0.045
    begin, end = metrics[0]["longest_reversal_window"]["steps"]
    query_steps = [begin, begin + 1]
    queries = paths[0][query_steps]
    plt.rcParams.update(
        {
            "font.family": "Arimo",
            "font.size": 10,
            "mathtext.fontset": "stix",
            "axes.labelweight": "bold",
            "axes.labelsize": 10,
            "text.color": "black",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.65,
        }
    )
    fig, axes = plt.subplots(
        1, 4, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1, 1.15, 1, 1.15]}
    )
    xmax = np.ceil(max(m["goal_time"] for m in metrics))
    projected = {}
    for group, (n, path, color, metric) in enumerate(zip(counts, paths, [PURPLE, CYAN], metrics)):
        spatial, temporal = axes[2 * group : 2 * group + 2]
        background(spatial, curve, reference, bounds, config)
        spatial.plot(*path.T, color=color, lw=1.7, zorder=3)
        spatial.scatter(
            *path[0], marker="*", color=PURPLE, edgecolors="white", linewidths=0.5, s=60, zorder=5
        )
        spatial.scatter(*reference[-1], color="0.25", s=18, zorder=5)
        nearest, _ = curve.evaluate(make_projector(curve, n)(queries))
        projected[f"n{n}"] = nearest
        for point, target in zip(queries, nearest):
            spatial.annotate(
                "",
                xy=target,
                xytext=point,
                arrowprops=dict(
                    arrowstyle="-|>", color=color, lw=1.2, mutation_scale=9, shrinkA=2, shrinkB=3
                ),
                zorder=4,
            )
        spatial.scatter(*queries.T, color="0.2", s=12, zorder=6)
        spatial.scatter(
            *nearest.T, facecolors="white", edgecolors=color, linewidths=1.2, s=22, zorder=6
        )
        temporal.plot(time, path[:, 0], color=color, lw=1.3)
        temporal.scatter(metric["goal_time"], path[-1, 0], s=24, color=color, zorder=5)
        temporal.annotate(
            f"{metric['goal_time']:.2f} s",
            xy=(metric["goal_time"], path[-1, 0]),
            xytext=(-3, -13) if group == 0 else (5, -13),
            textcoords="offset points",
            ha="right" if group == 0 else "left",
            fontsize=10,
            fontweight="bold",
        )
        temporal.set(
            xlim=(0, xmax),
            ylim=(all_points[:, 0].min() - 0.025, all_points[:, 0].max() + 0.025),
            xlabel="Time (s)",
            ylabel="$x(t)$",
        )
        temporal.set_xticks([0, xmax / 3, 2 * xmax / 3, xmax])
        temporal.set_yticks([0.2, 0.5, 0.8])
        temporal.spines[["top", "right"]].set_visible(False)
        temporal.spines[["left", "bottom"]].set_color("0.3")
        temporal.tick_params(labelsize=9, width=0.65, length=2.5, pad=2, color="0.3")
    handles = [
        tuple(Line2D([], [], color=c, lw=3.0) for c in REFERENCE),
        Line2D([], [], color="0.2", marker="o", ls="", markersize=5, markerfacecolor="white"),
        Line2D([], [], color=PURPLE, marker="*", ls="", markersize=8),
    ]
    fig.legend(
        handles,
        ["Reference", "Closest point", "Initial state"],
        ncol=3,
        frameon=False,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        prop={"size": 9.5, "weight": "bold"},
        handlelength=1.8,
        handletextpad=0.5,
        columnspacing=1.5,
    )
    fig.subplots_adjust(left=0.015, right=0.99, bottom=0.245, top=0.79, wspace=0.53)
    for group, name in enumerate(["Numerical", "Analytical"]):
        pair = axes[group * 2 : group * 2 + 2]
        center = (pair[0].get_position().x0 + pair[1].get_position().x1) / 2
        fig.text(center, 0.96, name, ha="center", va="top", fontsize=14, fontweight="bold")
    for extension in ["pdf", "png", "svg"]:
        fig.savefig(args.input / f"comparison.{extension}", dpi=360, facecolor="white")
    plt.close(fig)
    np.savez_compressed(
        args.input / "projection_queries.npz", steps=query_steps, queries=queries, **projected
    )
    caption = (
        "Numerical (left) and analytical (right) closest-point estimation on the same S curve. "
        f"Numerical selects the nearest of {config.samples} uniform parameter samples, including endpoints. "
        "Analytical minimizes continuous curve distance via cubic roots and segment endpoints. "
        f"Both use start={config.start}, attraction={config.attraction:g}, dt={config.dt:g} s, "
        f"speed={config.speed:g}, duration={config.duration:g} s, and goal tolerance={config.goal_tolerance:g}. "
        "Each half shows the full spatial rollout and the raw x-coordinate versus time on identical scales. "
        f"All {time[-1]:g} s of spatial execution are retained; temporal panels show the approach and arrival, "
        "after which both trajectories remain at their final positions. "
        f"Projection arrows use the same numerical states at steps {query_steps}. "
        "The pale-blue streamlines visualize the analytical reference field in both spatial panels. "
        "Labels at the filled markers in the time plots report goal-arrival times. "
        f"The numerical run's longest consecutive reversal window is {time[begin]:g}–{time[end]:g} s. "
        "The oscillation is localized back-and-forth motion followed by recovery; it is not a "
        "large-amplitude traveling wave. No position offsets, smoothing, or artificial oscillations are applied. "
        "Figure size: 7.2 by 2.7 inches. Text uses Arimo with STIX mathematical symbols; PDF fonts are embedded.\n"
    )
    (args.input / "caption.txt").write_text(caption)
    print(args.input / "comparison.pdf")


if __name__ == "__main__":
    main()
