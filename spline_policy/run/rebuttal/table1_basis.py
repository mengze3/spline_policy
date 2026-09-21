"""LASA representation error table with matched real coefficient counts."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.io import loadmat
import torch

from basis_reconstruction import METHODS, PROJECTION, ROOT, reconstruct, reconstruction_errors


TABLE_METHODS = {
    "piecewise": "Piecewise",
    "bernstein": "B.P.",
    "rbf": "RBF",
    "fast": "Fourier / DCT (FAST-cont.)",
    "spline_rest": "Q.S. (ours)",
}


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_lasa(directory):
    names, trajectories, sources = [], [], {}
    for path in sorted(directory.glob("*.mat")):
        demos = np.atleast_1d(loadmat(path, squeeze_me=True, struct_as_record=False)["demos"])
        for index, demo in enumerate(demos):
            names.append((path.stem, index))
            trajectories.append(demo.pos.T.astype(np.float64))
        sources[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return names, np.stack(trajectories), sources


def summarize(rows, parameters):
    summary = []
    metrics = ["mean_l2", "rmse", "max_l2", "frobenius_per_point", "mean_l2_over_bbox_percent"]
    for method in METHODS:
        for count in parameters:
            selected = [row for row in rows if row["method"] == method and row["parameters_per_dim"] == count]
            entry = {"method": method, "parameters_per_dim": count, "demonstrations": len(selected)}
            for metric in metrics:
                values = [row[metric] for row in selected]
                entry[metric + "_mean"] = float(np.mean(values))
                entry[metric + "_std"] = float(np.std(values))
            summary.append(entry)
    return summary


def render_table(output, summary, parameters, trajectories):
    lookup = {(row["method"], row["parameters_per_dim"]): row for row in summary}
    cells, tex_rows = [], []
    for method, label in TABLE_METHODS.items():
        values = [lookup[method, count] for count in parameters]
        cells.append([label] + [f'{v["mean_l2_mean"]:.2f} ± {v["mean_l2_std"]:.2f}' for v in values] + [PROJECTION[method]])
        tex_rows.append(label + " & " + " & ".join(
            f'${v["mean_l2_mean"]:.2f} \\pm {v["mean_l2_std"]:.2f}$' for v in values
        ) + " & " + PROJECTION[method] + r" \\")
    header = ["Method"] + [str(k) for k in parameters] + ["Closest-point query"]
    caption = (
        f"Reconstruction error on {len(trajectories)} LASA demonstrations "
        f"({trajectories.shape[1]} original samples each). "
        "Entries are the mean and population standard deviation of per-trajectory mean pointwise "
        "Euclidean errors in original LASA coordinate units; lower is better. "
        "K is the number of independent real coefficients per coordinate (2K scalars in 2D). "
        "Q.S. uses K-1 C1-continuous quadratic segments with zero terminal velocity, "
        "retaining K free coefficients per coordinate after eliminating one coefficient. "
        "Fourier/DCT is grouped as one cosine-basis family; "
        "the row reports FAST's continuous orthonormal DCT with K retained low frequencies, without "
        "quantization or BPE. The RCFS mirrored-Fourier fit is retained in the raw results. "
        "Closest-point query describes "
        "the construction, not a measured runtime: B.P. has a stationarity polynomial of degree "
        "2K-3; Q.S. requires at most cubic roots per segment, with endpoints also checked. "
        "Smooth alternative bases also admit numerical flow-field construction."
    )
    md = "| " + " | ".join(header) + " |\n| " + " | ".join(["---"] * len(header)) + " |\n"
    md += "\n".join("| " + " | ".join(row) + " |" for row in cells) + "\n\n" + caption + "\n"
    (output / "table1.md").write_text(md)
    tex = [r"\begin{table*}[t]", r"\centering", "\\caption{" + caption + "}", r"\label{tab:m1_basis}",
           r"\small", "\\begin{tabular}{l" + "c" * (len(parameters) + 1) + "}", r"\hline",
           "Method & " + " & ".join(f"$K={k}$" for k in parameters) + r" & Closest-point query \\", r"\hline",
           *tex_rows, r"\hline", r"\end{tabular}", r"\end{table*}"]
    (output / "table1.tex").write_text("\n".join(tex) + "\n")
    (output / "table1_caption.txt").write_text(caption + "\n")

    plt.rcParams.update({"font.family": "serif", "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(12.5, 0.32 * (len(cells) + 1)))
    ax.axis("off")
    table = ax.table(cellText=cells, colLabels=header, cellLoc="center", loc="center",
                     colWidths=[0.21] + [0.118] * len(parameters) + [0.20])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.65)
    for (row, column), cell in table.get_celld().items():
        cell.set_linewidth(0)
        if row == 0:
            cell.visible_edges = "TB"
            cell.set_linewidth(0.8)
            cell.set_text_props(weight="bold")
        if row == len(cells):
            cell.visible_edges = "B"
            cell.set_linewidth(0.8)
        if column == 0:
            cell.set_text_props(ha="left")
    fig.tight_layout(pad=0.3)
    for extension in ("pdf", "png", "svg"):
        fig.savefig(output / f"table1.{extension}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Directory containing the original LASA .mat files")
    parser.add_argument("--parameters", type=int, nargs="+", default=[3, 7, 12, 17, 22])
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/m1_table1")
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    names, trajectories, sources = load_lasa(args.data)
    rows, reconstructions = [], {"reference": trajectories, "shape": np.array([name for name, _ in names]),
                                "demo": np.array([demo for _, demo in names])}
    for method in METHODS:
        for count in args.parameters:
            fitted, weights = reconstruct(trajectories, method, count)
            errors = reconstruction_errors(trajectories, fitted)
            for index, (shape, demo) in enumerate(names):
                rows.append({"shape": shape, "demo": demo, "method": method,
                             "parameters_per_dim": count, "total_parameters": count * trajectories.shape[2],
                             **{key: float(value[index]) for key, value in errors.items()}})
            reconstructions[f"{method}_{count}"] = fitted
            reconstructions[f"{method}_{count}_weights"] = weights
        print(f"Fitted {METHODS[method]}: {len(names)} demonstrations", flush=True)
    summary = summarize(rows, args.parameters)
    write_csv(args.output / "per_demo.csv", rows)
    write_csv(args.output / "summary.csv", summary)
    np.savez_compressed(args.output / "reconstructions.npz", **reconstructions)
    render_table(args.output, summary, args.parameters, trajectories)
    manifest = {
        "dataset": str(args.data.resolve()), "shapes": len(sources), "demonstrations": len(names),
        "samples_per_demo": trajectories.shape[1], "dimensions": trajectories.shape[2],
        "parameters_per_dim": args.parameters, "preprocessing": "None: original positions and sample order",
        "table_rows": TABLE_METHODS,
        "fit": "float64 least squares with RCFS ridge=1e-8; FAST uses scipy.fft.dct/idct(norm='ortho'), no ridge",
        "primary_metric": "mean_t ||reconstruction[t] - reference[t]||_2",
        "std": "population standard deviation across all demonstrations (ddof=0)",
        "rcfs": "https://github.com/idiap/rcfs/blob/f4c96d66cab69f8612a3d380861a909bc6904ad0/python/MP.py",
        "fast": "https://huggingface.co/physical-intelligence/fast/blob/ec4d7aa71691cac0b8bed6942be45684db2110f4/processing_action_tokenizer.py",
        "dataset_sha256": sources,
        "code_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in
                        [Path(__file__), Path(__file__).with_name("basis_reconstruction.py"),
                         ROOT / "policy/diffusion_policy/planning/quadratic_spline.py"]},
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__, "torch": torch.__version__},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print((args.output / "table1.md").read_text())
    print(f"Saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
