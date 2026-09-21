"""Continuous quadratic-spline projection and sampled-neighbor flow comparison."""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys

import numpy as np
import torch
import zarr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from policy.diffusion_policy.planning.quadratic_spline import QuadraticSpline

PURPLE = "#8A2BE2"
CYAN = "#65CDD2"


@dataclass
class Config:
    start: tuple = (0.75, 1.05)
    samples: int = 20
    attraction: float = 0.1
    dt: float = 0.08
    duration: float = 6.0
    speed: float = 0.5
    horizon: float = 16.0
    goal_tolerance: float = 0.005
    unnormalized: bool = False
    segments: int = 6


def load_curve(segments=6):
    data = zarr.open(str(ROOT / "data/demonstrations/S.zarr"), mode="r")
    demonstration = np.asarray(data["data/action"][: int(data["meta/episode_ends"][0])]) / 512
    return Spline(demonstration, segments), demonstration


class Spline:
    def __init__(self, demonstration, segments):
        self.spline = QuadraticSpline(nbSeg=segments, device="cpu")
        self.segments = segments
        _, _, phi = self.spline.computePsiList1D(torch.linspace(0, 1, len(demonstration)))
        control_map = self.spline.C.numpy().astype(float)
        # Eliminate the last parameter to impose equal terminal control points.
        terminal_map = np.eye(segments + 2)[:, :-1]
        terminal_map[-1] = control_map[-2, :-1]
        weights = np.linalg.lstsq(phi.numpy() @ terminal_map, demonstration, rcond=None)[0]
        self.control = (control_map @ terminal_map @ weights).reshape(segments, 3, 2)
        self.a = self.control[:, 0] - 2 * self.control[:, 1] + self.control[:, 2]
        self.b = 2 * (self.control[:, 1] - self.control[:, 0])
        self.c = self.control[:, 0]

    def evaluate(self, phase):
        scaled = np.asarray(phase) * self.segments
        segment = np.minimum(scaled.astype(int), self.segments - 1)
        t = (scaled - segment)[..., None]
        point = (self.a[segment] * t + self.b[segment]) * t + self.c[segment]
        tangent = (2 * self.a[segment] * t + self.b[segment]) * self.segments
        return point, tangent

    def coefficients(self, points):
        residual = self.c[None] - points[:, None]
        leading = 2 * np.sum(self.a**2, axis=-1)
        quadratic = 3 * np.sum(self.a * self.b, axis=-1)
        linear = np.sum(self.b**2, axis=-1) + 2 * np.sum(residual * self.a, axis=-1)
        constant = np.sum(residual * self.b, axis=-1)
        return (
            np.stack([np.broadcast_to(quadratic, linear.shape), linear, constant], -1)
            / leading[None, :, None]
        )

    def project(self, points):
        coefficients = self.coefficients(points)
        roots = self.spline.solve_cubic(*torch.from_numpy(coefficients.reshape(-1, 3)).T)
        roots = roots.numpy().reshape(len(points), self.segments, 3)
        # Both endpoints must be candidates, including when interior roots exist.
        t = np.concatenate(
            [roots.clip(0, 1), np.zeros_like(roots[..., :1]), np.ones_like(roots[..., :1])], -1
        )
        candidates = (self.a[None, :, None] * t[..., None] + self.b[None, :, None]) * t[
            ..., None
        ] + self.c[None, :, None]
        distance2 = np.sum((candidates - points[:, None, None]) ** 2, axis=-1)
        best = distance2.reshape(len(points), -1).argmin(axis=-1)
        segment = best // t.shape[-1]
        local = t[np.arange(len(points)), segment, best % t.shape[-1]]
        return (segment + local) / self.segments


def make_projector(curve, samples):
    if samples == 0:
        return curve.project
    phase = np.linspace(0, 1, samples)
    points, _ = curve.evaluate(phase)

    def project(query):
        nearest = np.sum((query[:, None] - points[None]) ** 2, axis=-1).argmin(-1)
        return phase[nearest]

    return project


def field(curve, points, project, args):
    phase = project(points)
    nearest, tangent = curve.evaluate(phase)
    residual = nearest - points
    distance = np.linalg.norm(residual, axis=-1)
    # Eq. (18), written without dividing the residual by its norm at the curve.
    raw = (args.attraction * residual + tangent / args.duration) / (
        1 + args.attraction * distance[:, None]
    )
    velocity = raw.copy()
    if not args.unnormalized:
        # Match the public implementation's normalization before taking a fixed step.
        velocity = args.speed * torch.nn.functional.normalize(torch.from_numpy(raw), dim=-1).numpy()
    return velocity, phase, raw


def rollout(curve, project, start, args, dt):
    time = np.arange(round(args.horizon / dt) + 1) * dt
    position = np.empty((len(time), 2))
    position[0] = start
    goal, _ = curve.evaluate(1.0)
    for i in range(len(time) - 1):
        # Apply the same goal stopping rule to both projection methods.
        if np.linalg.norm(position[i] - goal) <= args.goal_tolerance:
            position[i + 1 :] = position[i]
            break
        velocity, _, _ = field(curve, position[i : i + 1], project, args)
        position[i + 1] = position[i] + dt * velocity[0]
    velocity, phase, raw = field(curve, position, project, args)
    active = np.linalg.norm(position - goal, axis=-1) > args.goal_tolerance
    velocity[~active] = 0
    return {
        "time": time,
        "position": position,
        "velocity": velocity,
        "raw_velocity": raw,
        "phase": phase,
        "active": active,
    }


def verify_projection(curve, queries):
    phase = curve.project(queries)
    projected, _ = curve.evaluate(phase)
    closed_distance2 = np.sum((projected - queries) ** 2, axis=-1)
    numerical_distance2 = []
    for point, coefficients in zip(queries, curve.coefficients(queries)):
        candidates = []
        for segment, coefficient in enumerate(coefficients):
            roots = np.roots(np.r_[1.0, coefficient])
            roots = roots.real[(np.abs(roots.imag) < 1e-9) & (roots.real >= 0) & (roots.real <= 1)]
            candidates.extend((segment + np.r_[0.0, roots, 1.0]) / curve.segments)
        points, _ = curve.evaluate(candidates)
        numerical_distance2.append(np.min(np.sum((points - point) ** 2, axis=-1)))
    error = float(np.max(np.abs(closed_distance2 - numerical_distance2)))
    np.testing.assert_allclose(closed_distance2, numerical_distance2, atol=1e-9, rtol=1e-7)
    np.testing.assert_allclose(curve.evaluate(1.0)[1], 0, atol=1e-12)
    return {"max_squared_distance_error_vs_numerical_roots": error, "queries": len(queries)}


def summarize(curve, run, config):
    position = run["position"]
    nearest, _ = curve.evaluate(curve.project(position))
    distance = np.linalg.norm(position - nearest, axis=1)
    delta = np.diff(position, axis=0)
    angles = np.arctan2(np.cross(delta[:-1], delta[1:]), np.sum(delta[:-1] * delta[1:], axis=1))
    clearance = (distance[:-1] + distance[1:] - np.linalg.norm(delta, axis=1)) / 2
    before = np.minimum.accumulate(clearance)[1:] > 0.01
    turns = np.flatnonzero((np.abs(angles) > np.pi / 2) & before) + 1
    arrival = np.flatnonzero(distance <= 0.01)
    goal_error = np.linalg.norm(position - curve.evaluate(1.0)[0], axis=1)
    reached = np.flatnonzero(goal_error <= config.goal_tolerance)
    run["distance"] = distance
    groups = np.split(turns, np.flatnonzero(np.diff(turns) > 1) + 1)
    longest = max(groups, key=len)
    window = None
    if len(longest):
        begin, end = longest[0] - 1, longest[-1] + 1
        window = {
            "steps": [int(begin), int(end)],
            "reversals": len(longest),
            "duration": float(run["time"][end] - run["time"][begin]),
            "net_displacement": float(np.linalg.norm(position[end] - position[begin])),
            "prefix_clearance_bound": float(clearance[:end].min()),
        }
    return {
        "goal_time": float(run["time"][reached[0]]) if len(reached) else None,
        "goal_error": float(goal_error[-1]),
        "curve_arrival": float(run["time"][arrival[0]]) if len(arrival) else None,
        "pre_curve_reversal_steps": turns.tolist(),
        "longest_reversal_window": window,
        "path_length": float(np.linalg.norm(delta, axis=1).sum()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=float, nargs=2, default=Config.start)
    parser.add_argument("--samples", type=int, default=Config.samples)
    parser.add_argument("--attraction", type=float, default=Config.attraction)
    parser.add_argument("--dt", type=float, default=Config.dt)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/m1_projection")
    args = parser.parse_args()
    config = Config(tuple(args.start), args.samples, args.attraction, args.dt)
    torch.set_num_threads(1)
    curve, demonstration = load_curve(config.segments)
    arrays = {"demonstration": demonstration, "control_points": curve.control}
    metrics = []
    for dt in [config.dt, config.dt / 2]:
        for n in [config.samples, config.samples * 2, 0]:
            project = make_projector(curve, n)
            run = rollout(curve, project, config.start, config, dt)
            metrics.append({"samples": n, "dt": dt, **summarize(curve, run, config)})
            key = f"n{n}_dt{dt:g}"
            arrays.update({f"{key}_{name}": value for name, value in run.items()})
    queries = np.vstack([arrays[f"n{n}_dt{config.dt:g}_position"] for n in [config.samples, 0]])
    verification = verify_projection(curve, queries)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "rollouts.npz", **arrays)
    results = {"config": asdict(config), "metrics": metrics, "verification": verification}
    (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
