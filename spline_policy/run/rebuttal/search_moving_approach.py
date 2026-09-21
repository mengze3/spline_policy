"""Search sampled-neighbor rollouts; report oscillation and spatial progress separately."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import torch

from fig6_projection import ROOT, Config, field, load_curve, make_projector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, nargs="+", default=[16, 20, 24, 32])
    parser.add_argument("--attraction", type=float, nargs="+", default=[0.1, 0.2, 0.4, 0.6])
    parser.add_argument("--dt", type=float, nargs="+", default=[0.04, 0.08, 0.12])
    parser.add_argument("--xlim", type=float, nargs=2, default=[0.55, 1.0])
    parser.add_argument("--ylim", type=float, nargs=2, default=[0.8, 1.15])
    parser.add_argument("--spacing", type=float, default=0.05)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/m1_projection/search.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = Config()
    curve, _ = load_curve()
    reference, _ = curve.evaluate(np.linspace(0, 1, 16384))
    tree = cKDTree(reference)
    x, y = np.meshgrid(
        np.arange(args.xlim[0], args.xlim[1] + args.spacing / 2, args.spacing),
        np.arange(args.ylim[0], args.ylim[1] + args.spacing / 2, args.spacing),
    )
    starts = np.round(np.c_[x.ravel(), y.ravel()], 10)
    starts = starts[tree.query(starts)[0] > 0.08]
    summaries, candidates = [], []
    for n in args.samples:
        project = make_projector(curve, n)
        for gain in args.attraction:
            config.attraction = gain
            for dt in args.dt:
                positions = [starts]
                for _ in range(round(config.horizon / dt)):
                    p = positions[-1]
                    velocity, _, _ = field(curve, p, project, config)
                    velocity[np.linalg.norm(p - reference[-1], axis=1) <= 0.005] = 0
                    positions.append(p + dt * velocity)
                positions = np.stack(positions, axis=1)
                distance = tree.query(positions.reshape(-1, 2))[0].reshape(positions.shape[:2])
                delta = np.diff(positions, axis=1)
                bound = (distance[:, :-1] + distance[:, 1:] - np.linalg.norm(delta, axis=2)) / 2
                before = np.minimum.accumulate(bound - 0.0002, axis=1)[:, 1:] > 0.01
                turns = (np.sum(delta[:, :-1] * delta[:, 1:], axis=2) < 0) & before
                goal_error = np.linalg.norm(positions - reference[-1], axis=2)
                reached = goal_error[:, -1] <= config.goal_tolerance
                count = 0
                for i in np.flatnonzero(reached & (turns.sum(1) >= 10)):
                    indices = np.flatnonzero(turns[i]) + 1
                    groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
                    group = max(groups, key=len)
                    if len(group) < 10:
                        continue
                    begin, end = group[0] - 1, group[-1] + 1
                    net = np.linalg.norm(positions[i, end] - positions[i, begin])
                    candidates.append(
                        dict(
                            start=starts[i].tolist(),
                            samples=n,
                            attraction=gain,
                            dt=dt,
                            reversals=len(group),
                            window=[int(begin), int(end)],
                            net_displacement=float(net),
                            moving_window=bool(net >= 2 * config.speed * dt),
                            goal_time=float(np.flatnonzero(goal_error[i] <= 0.005)[0] * dt),
                        )
                    )
                    count += 1
                summaries.append(
                    dict(
                        samples=n,
                        attraction=gain,
                        dt=dt,
                        starts=len(starts),
                        recovered_oscillations=count,
                    )
                )
        print(f"Completed N={n}", flush=True)
    settings = {key: value for key, value in vars(args).items() if key != "output"}
    result = dict(
        search=settings, dynamics=asdict(Config()), summary=summaries, candidates=candidates
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"{len(candidates)} recovered oscillations; "
        f"{sum(c['moving_window'] for c in candidates)} pass the spatial-progress threshold"
    )


if __name__ == "__main__":
    main()
