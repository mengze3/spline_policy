"""Plot the preserved release training logs without smoothing or retraining."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullLocator, MaxNLocator
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TASKS = {"pusht_image": "Push-T (Image)", "can_image_abs": "Can (Image)",
         "transport_lowdim_abs": "Transport (Lowdim)", "adroit_door": "Adroit Door (Point Cloud)",
         "adroit_pen": "Adroit Pen (Point Cloud)", "dexart_laptop": "DexArt Laptop (Point Cloud)"}
METHODS = {"pure": ("Baseline", "#AD92E3"), "spline": ("Spline Policy", "#6DD2D9")}


def read_log(path):
    epochs, batch_losses, step_counts = {}, {}, {}
    nonfinite, count = 0, 0
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            if not line.strip():
                continue
            record = json.loads(line)
            epoch = record["epoch"]
            epochs[epoch] = record  # The last row stores the epoch mean, not its last batch loss.
            step_counts[epoch] = step_counts.get(epoch, 0) + 1
            if "bc_loss" in record:
                batch_losses.setdefault(epoch, []).append(record["bc_loss"])
            nonfinite += sum(not np.isfinite(v) for k, v in record.items()
                             if "loss" in k and isinstance(v, (int, float)))
            count += 1
    assert sorted(epochs) == list(range(len(epochs))), path
    # Historical Adroit/DexArt logs also retain every batch's bc_loss.
    if batch_losses:
        for epoch, values in batch_losses.items():
            np.testing.assert_allclose(epochs[epoch]["train_loss"], np.mean(values), rtol=1e-6)
    records = [dict(epoch=e, global_step=r["global_step"], train_loss=r["train_loss"],
                    val_loss=r.get("val_loss", ""), test_mean_score=r.get("test/mean_score", ""))
               for e, r in sorted(epochs.items())]
    audit = dict(source=str(path.relative_to(ROOT)), sha256=digest.hexdigest(), rows=count,
                 epochs=len(epochs), nonfinite_loss=nonfinite,
                 validation_epochs=sum(r["val_loss"] != "" for r in records),
                 epoch_average_checked_against_bc_loss=bool(batch_losses),
                 records_per_epoch=sorted(set(step_counts.values())))
    return records, audit


def plot(curves, tasks, metric, output, name):
    fig, axes = plt.subplots(2, len(tasks), figsize=(7.2, 3.8), squeeze=False)
    fig.subplots_adjust(left=.10, right=.99, bottom=.13, top=.81, hspace=.28, wspace=.55)
    for row, backbone in enumerate(("diffusion", "flowmatching")):
        for column, task in enumerate(tasks):
            ax = axes[row, column]
            for method, (_, color) in METHODS.items():
                records = [r for r in curves[task, backbone, method] if r[metric] != ""]
                ax.plot([r["epoch"] for r in records], [r[metric] for r in records],
                        color=color, lw=1.35, zorder=2 if method == "pure" else 3)
            if row == 0:
                ax.set_title(TASKS[task].replace(" (", "\n("), fontsize=10, fontweight="bold", pad=7)
            ax.set_yscale("log")
            ax.set_xlim(0, curves[task, backbone, "pure"][-1]["epoch"])
            ax.xaxis.set_major_locator(MaxNLocator(3, integer=True))
            ax.yaxis.set_major_locator(LogLocator(base=10, numticks=4))
            ax.yaxis.set_minor_locator(NullLocator())
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color="#EEEEEE", linewidth=.6, zorder=0)
            ax.tick_params(labelsize=9, length=3)
            if column == 0:
                ax.set_ylabel(("Diffusion" if row == 0 else "Flow Matching") + "\nLoss", weight="bold", fontsize=9)
            if row == 1:
                ax.set_xlabel("Epoch", weight="bold")
    fig.legend([Line2D([], [], color=color, lw=2) for _, color in METHODS.values()],
               [name for name, _ in METHODS.values()], loc="upper center", bbox_to_anchor=(.54, 1.015),
               ncol=2, frameon=False, prop={"weight": "bold", "size": 10})
    for ext in ("pdf", "png", "svg"):
        fig.savefig(output / f"{name}.{ext}", dpi=300, bbox_inches="tight", pad_inches=.06)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/m8_training_stability")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "Arimo", "mathtext.fontset": "stix", "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    curves, audit, rows = {}, [], []
    for task in TASKS:
        for backbone in ("diffusion", "flowmatching"):
            for method in METHODS:
                suffix = "_nbSeg6" if method == "spline" else ""
                path = ROOT / "metrics" / task / f"{backbone}_{method}_seed42_horizon16{suffix}.json.txt"
                records, check = read_log(path)
                curves[task, backbone, method] = records
                audit.append(check)
                rows.extend(dict(task=task, backbone=backbone, method=method, **r) for r in records)
    with (args.output / "epoch_metrics.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "audit.json").write_text(json.dumps(dict(
        runs=audit, seed=42, epoch_selection="Last logged row per epoch; no additional averaging or smoothing.",
        limitations="Single seed; logs use historical method names. Loss magnitudes across objectives are not a common performance metric. Three tasks have no validation loss. Cloud W&B history not verified.",
        workspace_source="policy/diffusion_policy/workspace/train_diffusion_unet_image_workspace.py",
        plot_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2) + "\n")
    plot(curves, ["pusht_image", "can_image_abs", "transport_lowdim_abs", "adroit_pen"],
         "train_loss", args.output, "train_loss")
    plot(curves, ["adroit_door", "dexart_laptop"], "train_loss", args.output, "train_loss_additional")
    plot(curves, list(TASKS)[:3], "val_loss", args.output, "val_loss")
    print(f"Plotted {len(audit)} preserved runs; {len(rows)} epoch records; output: {args.output}")


if __name__ == "__main__":
    main()
