"""A fixed spline flow folds SoftGym cloth under two-picker disturbances."""

import argparse
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT.parent
UPSTREAM = RELEASE / "data_local/softgym"
sys.path.insert(0, str(ROOT))

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from policy.diffusion_policy.planning.quadratic_spline import (
    QuadraticSpline,
    dynamical_system_single_step,
)

DT, STEPS, RELEASE_STEP = 0.01, 1700, 1500
SPEED, ATTRACTION = 0.15, 5.0
PULSE_STEPS, RECOVERY_DISTANCE, RECOVERY_HOLD = 20, 0.015, 10


def write_json(path, data):
    path.write_text(
        json.dumps(data, indent=2, default=lambda x: np.asarray(x).tolist())
    )


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit_flow(start, opposite):
    """Fold the grasped edge over the fixed edge along a semicircular path."""
    t = np.linspace(0, 1, 501)
    center = (start + opposite) / 2
    radius = np.linalg.norm((opposite - start)[:, [0, 2]], axis=1) / 2
    horizontal = start - center
    horizontal[:, 1] = 0
    reference = center[None] + np.cos(np.pi * t[:, None, None]) * horizontal[None]
    reference[:, :, 1] = (
        start[None, :, 1]
        + np.sin(np.pi * t[:, None]) * radius[None]
        + 0.015 * t[:, None]
    )
    reference = reference.reshape(-1, 6)
    curve = QuadraticSpline(nbSeg=12, nbDim=6, device="cpu")
    phi = (
        curve.computePsiList1D(torch.tensor(t, dtype=torch.float32))[2]
        .numpy()
        .astype(float)
    )
    control = curve.C.numpy().astype(float)
    constraints = np.vstack([phi[0], phi[-1], control[-1] - control[-2]])
    values = np.vstack([reference[0], reference[-1], np.zeros(6)])
    system = np.block([[phi.T @ phi, constraints.T], [constraints, np.zeros((3, 3))]])
    weights = np.linalg.solve(system, np.vstack([phi.T @ reference, values]))[:-3]
    np.testing.assert_allclose(constraints @ weights, values, atol=1e-9)
    return curve, torch.tensor(weights.reshape(1, -1), dtype=torch.float32), reference


def command(curve, weights, position, goal):
    point = torch.tensor(position.reshape(1, 1, 6), dtype=torch.float32)
    with torch.no_grad():
        _, direction = dynamical_system_single_step(
            curve, point, weights, lambda_dist=ATTRACTION
        )
        distance, _, phase = curve.sdf_batch(point, weights)
    speed = min(SPEED, 5 * np.linalg.norm(position.reshape(-1) - goal))
    return (
        direction.numpy().reshape(2, 3) * speed * DT,
        float(distance),
        float(phase) / curve.nbSeg,
    )


def make_env():
    from softgym.envs.cloth_fold import ClothFoldEnv

    class FixedSizeCloth(ClothFoldEnv):
        def generate_env_variation(self, num_variations=1, **kwargs):
            return super().generate_env_variation(num_variations, vary_cloth_size=False)

    random.seed(0)
    np.random.seed(0)
    torch.set_num_threads(1)
    return FixedSizeCloth(
        observation_mode="key_point",
        action_mode="picker",
        num_picker=2,
        picker_radius=0.015,
        render_mode="cloth",
        headless=True,
        render=True,
        num_variations=1,
        use_cached_states=False,
        save_cached_states=False,
        camera_width=640,
        camera_height=640,
        action_repeat=1,
        horizon=STEPS,
    )


def run_case(env, case, folder):
    import pyflex

    folder.mkdir()
    env.reset(config_id=0)
    camera = dict(pos=[0, 1.1, 1.2], angle=[0, -0.72, 0], width=640, height=640)
    env.update_camera("default_camera", camera)
    vertices = pyflex.get_positions().reshape(-1, 4)
    corners = env._get_key_point_idx()[:2]
    env.action_tool.set_picker_pos(vertices[corners, :3])
    env.action_tool.step(np.array([[0, 0, 0, 1]] * 2))
    np.testing.assert_array_equal(env.action_tool.picked_particles, corners)
    start = pyflex.get_shape_states().reshape(-1, 14)[:, :3].copy()
    opposite = vertices[env._get_key_point_idx()[2:], :3].copy()
    curve, weights, reference = fit_flow(start, opposite)
    goal = reference[-1]
    np.savez_compressed(
        folder / "reference.npz",
        reference=reference,
        weights=weights.numpy(),
        initial_particles=env.init_pos,
        group_a=env.fold_group_a,
        group_b=env.fold_group_b,
    )
    write_json(
        folder / "config.json",
        dict(
            case=case,
            camera=camera,
            scene=env.current_config,
            initial_performance=env.performance_init,
            picked_ids=corners,
        ),
    )
    writer = (
        imageio.get_writer(folder / "rollout.mp4", fps=25, quality=8)
        if case["video"]
        else None
    )
    positions, actions, displacements, distances, phases, performance, scores = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    fold_errors, fixation_errors = [], []
    particles, mesh_steps, held = [], [], []
    for step in range(STEPS + 1):
        position = pyflex.get_shape_states().reshape(-1, 14)[:, :3].copy()
        delta, distance, phase = command(curve, weights, position, goal)
        disturbance = np.zeros((2, 3))
        if case["onset"] <= step < case["onset"] + PULSE_STEPS:
            disturbance[:, case["axis"]] = (
                case["sign"] * case["amplitude"] / PULSE_STEPS
            )
        grasp = step < RELEASE_STEP
        if not grasp:
            delta[:] = 0
        action = np.column_stack([delta + disturbance, np.full(2, grasp)]).reshape(-1)
        info = env._get_info()
        positions.append(position)
        actions.append(action)
        displacements.append(disturbance)
        distances.append(distance)
        phases.append(phase)
        performance.append(info["performance"])
        fold_errors.append(-info["neg_group_dist"])
        fixation_errors.append(-info["neg_fixation_dist"])
        scores.append(info["normalized_performance"])
        held.append(env.action_tool.picked_particles.copy())
        if step % 10 == 0:
            particles.append(pyflex.get_positions().reshape(-1, 4).copy())
            mesh_steps.append(step)
        if writer is not None and step % 4 == 0:
            writer.append_data(env.render())
        if step < STEPS:
            env.step(action)
    if writer is not None:
        writer.close()
    data = dict(
        positions=positions,
        actions=actions,
        disturbances=displacements,
        distance=distances,
        phase=phases,
        performance=performance,
        normalized_performance=scores,
        particles=particles,
        fold_error=fold_errors,
        fixation_error=fixation_errors,
        mesh_steps=mesh_steps,
        held=np.array(held, dtype=object),
    )
    np.savez_compressed(
        folder / "rollout.npz",
        **{k: v for k, v in data.items() if k != "held"},
        held=np.array([[-1 if x is None else x for x in row] for row in held]),
    )
    end = case["onset"] + PULSE_STEPS
    recovery = None
    for k in range(end, RELEASE_STEP - RECOVERY_HOLD):
        if np.all(np.asarray(distances[k : k + RECOVERY_HOLD]) < RECOVERY_DISTANCE):
            recovery = k
            break
    lowering = np.flatnonzero(np.asarray(phases) >= 0.5)
    lower_step = int(lowering[0]) if len(lowering) else None
    result = dict(
        **case,
        final_performance=performance[-1],
        normalized_performance=scores[-1],
        fold_error=fold_errors[-1],
        fixation_error=fixation_errors[-1],
        recovery_step=recovery,
        lowering_step=lower_step,
        recovered_before_lowering=recovery is not None
        and lower_step is not None
        and recovery + RECOVERY_HOLD <= lower_step,
        endpoint_error=float(
            np.linalg.norm(np.asarray(positions)[RELEASE_STEP].reshape(-1) - goal)
        ),
        perturbation_peak_distance=float(max(distances[case["onset"] : end + 1])),
        raw_sha256=digest(folder / "rollout.npz"),
        reference_sha256=digest(folder / "reference.npz"),
    )
    write_json(folder / "metrics.json", result)
    print((folder / "metrics.json").read_text(), flush=True)
    return result


def run(output, suite, pilot=False, amplitudes=(0.04, 0.08)):
    selected = [f"{axis}_pos_{round(max(amplitudes) * 1000):03d}_250" for axis in "xy"]
    output.mkdir(parents=True, exist_ok=False)
    (output / "source_snapshot.py").write_bytes(Path(__file__).read_bytes())
    cases = [dict(name="nominal", axis=0, sign=0, amplitude=0.0, onset=250, video=True)]
    if suite or pilot:
        for axis, axis_name in enumerate("xyz"):
            for sign, sign_name in [(1, "pos"), (-1, "neg")]:
                for amplitude in amplitudes:
                    for onset in [150, 250]:
                        name = f"{axis_name}_{sign_name}_{int(amplitude * 1000):03d}_{onset}"
                        cases.append(
                            dict(
                                name=name,
                                axis=axis,
                                sign=sign,
                                amplitude=amplitude,
                                onset=onset,
                                video=name in selected,
                            )
                        )
    if pilot:
        cases = [
            case
            for case in cases
            if case["name"] == "nominal" or case["name"] in selected
        ]
    paths = [
        Path(__file__),
        ROOT / "policy/diffusion_policy/planning/quadratic_spline.py",
        UPSTREAM / "softgym/envs/cloth_fold.py",
        UPSTREAM / "softgym/envs/cloth_env.py",
        UPSTREAM / "softgym/action_space/action_space.py",
        UPSTREAM / "PyFlex/bindings/main.cpp",
        UPSTREAM / "PyFlex/bindings/softgym_scenes/softgym_cloth.h",
    ]
    write_json(
        output / "manifest.json",
        dict(
            task="SoftGym ClothFold",
            commit=subprocess.check_output(
                ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True
            ).strip(),
            cases=cases,
            dt=DT,
            steps=STEPS,
            release_step=RELEASE_STEP,
            pulse_steps=PULSE_STEPS,
            speed=SPEED,
            attraction=ATTRACTION,
            recovery_distance=RECOVERY_DISTANCE,
            recovery_hold=RECOVERY_HOLD,
            lowering_phase=0.5,
            disturbance="Same additive displacement at both kinematic pickers through the original action interface; not a force pulse.",
            metric="Unmodified upstream paired-particle folding distance plus 1.2 times fixed-half displacement, and normalized performance; no binary task-success threshold.",
            scenario="Seed 0, default cloth size, two adjacent corner grasps; prescribed trajectory, no training or replanning.",
            hashes={str(p.resolve()): digest(p) for p in paths},
            selected=selected,
        ),
    )
    env = make_env()
    results = []
    try:
        for case in cases:
            write_json(
                output / "run_status.json",
                dict(completed=len(results), total=len(cases), current=case["name"]),
            )
            results.append(run_case(env, case, output / case["name"]))
            write_json(output / "results.json", results)
    finally:
        env.close()
    write_json(
        output / "run_status.json",
        dict(completed=len(results), total=len(cases), status="complete"),
    )


def audit(folder):
    """Recompute task metrics from cloth particles, separately from the simulator."""
    import csv
    from scipy.spatial import cKDTree

    manifest = json.loads((folder / "manifest.json").read_text())
    results = json.loads((folder / "results.json").read_text())
    assert len(results) == len(manifest["cases"])
    for path, expected in manifest["hashes"].items():
        source = (
            folder / "source_snapshot.py"
            if Path(path).name == Path(__file__).name
            else Path(path)
        )
        assert digest(source) == expected, path
    checks = []
    initial_states, coefficients = [], []
    for result in results:
        case = folder / result["name"]
        data = np.load(case / "rollout.npz")
        ref = np.load(case / "reference.npz")
        config = json.loads((case / "config.json").read_text())
        assert digest(case / "rollout.npz") == result["raw_sha256"]
        assert digest(case / "reference.npz") == result["reference_sha256"]
        for key in data.files:
            assert np.isfinite(data[key]).all(), (case.name, key)
        assert len(data["positions"]) == STEPS + 1
        xyz = data["particles"][..., :3]
        ga, gb = ref["group_a"], ref["group_b"]
        # Preserve upstream's per-frame float32 reduction order.
        folding = np.array(
            [
                float(np.linalg.norm(frame[ga] - frame[gb], axis=1).mean())
                for frame in xyz
            ]
        )
        fixation = np.array(
            [
                float(
                    np.linalg.norm(
                        frame[gb] - ref["initial_particles"][gb], axis=1
                    ).mean()
                )
                for frame in xyz
            ]
        )
        performance = -folding - 1.2 * fixation
        normalized = (performance - config["initial_performance"]) / -config[
            "initial_performance"
        ]
        for key, values in [
            ("fold_error", folding),
            ("fixation_error", fixation),
            ("performance", performance),
            ("normalized_performance", normalized),
        ]:
            np.testing.assert_allclose(data[key][data["mesh_steps"]], values, atol=1e-9)
            np.testing.assert_allclose(
                result["final_performance" if key == "performance" else key],
                values[-1],
                atol=1e-9,
            )
        assert np.all(data["held"][: RELEASE_STEP + 1] == config["picked_ids"])
        assert np.all(data["held"][RELEASE_STEP + 1 :] == -1)
        assert np.all(data["particles"][-1, :, 3] > 0), "cloth must be fully released"
        assert np.max(np.abs(data["actions"][:, [0, 1, 2, 4, 5, 6]])) <= 0.01
        motion = np.diff(data["positions"], axis=0)
        requested = data["actions"][:-1].reshape(-1, 2, 4)[:, :, :3]
        np.testing.assert_allclose(motion, requested, atol=1e-7)
        np.testing.assert_array_equal(
            data["disturbances"][:, 0], data["disturbances"][:, 1]
        )
        expected_push = np.zeros_like(data["disturbances"])
        end = result["onset"] + PULSE_STEPS
        expected_push[result["onset"] : end, :, result["axis"]] = (
            result["sign"] * result["amplitude"] / PULSE_STEPS
        )
        np.testing.assert_array_equal(data["disturbances"], expected_push)
        curve = QuadraticSpline(nbSeg=12, nbDim=6, device="cpu")
        weights = torch.tensor(ref["weights"])
        replay = []
        with torch.no_grad():
            for i in range(0, RELEASE_STEP, 100):
                positions = data["positions"][i : i + 100].reshape(1, -1, 6)
                _, direction = dynamical_system_single_step(
                    curve, torch.tensor(positions), weights, lambda_dist=ATTRACTION
                )
                speed = np.minimum(
                    SPEED,
                    5 * np.linalg.norm(positions[0] - ref["reference"][-1], axis=1),
                )
                replay.extend(
                    direction.numpy().reshape(-1, 2, 3) * speed[:, None, None] * DT
                )
            phi = curve.computePsiList1D(torch.linspace(0, 1, 50001))[2].numpy()
        expected = (
            data["actions"][:RELEASE_STEP].reshape(-1, 2, 4)[:, :, :3]
            - data["disturbances"][:RELEASE_STEP]
        )
        replay_error = float(np.max(np.abs(np.asarray(replay) - expected)))
        assert replay_error < 3e-6, replay_error
        dense = phi @ ref["weights"].reshape(-1, 6)
        dense_distance = cKDTree(dense).query(data["positions"].reshape(-1, 6))[0]
        distance_error = float(np.max(np.abs(dense_distance - data["distance"])))
        independent_recovery = next(
            (
                k
                for k in range(end, RELEASE_STEP - RECOVERY_HOLD)
                if np.all(dense_distance[k : k + RECOVERY_HOLD] < RECOVERY_DISTANCE)
            ),
            None,
        )
        assert independent_recovery == result["recovery_step"]
        recovered = next(
            (
                k
                for k in range(end, RELEASE_STEP - RECOVERY_HOLD)
                if np.all(data["distance"][k : k + RECOVERY_HOLD] < RECOVERY_DISTANCE)
            ),
            None,
        )
        lowering = np.flatnonzero(data["phase"] >= manifest["lowering_phase"])
        lower_step = int(lowering[0]) if len(lowering) else None
        assert (
            result["recovery_step"] == recovered
            and result["lowering_step"] == lower_step
        )
        assert result["recovered_before_lowering"] == (
            recovered is not None
            and lower_step is not None
            and recovered + RECOVERY_HOLD <= lower_step
        )
        initial_states.append(data["positions"][0])
        coefficients.append(ref["weights"])
        checks.append(
            dict(
                name=case.name,
                command_replay_max_error=replay_error,
                independent_distance_max_error=distance_error,
                released_particles=int(np.sum(data["particles"][-1, :, 3] > 0)),
                final_ground_near_particles=int(np.sum(xyz[-1, :, 1] < 0.02)),
            )
        )
    disturbed = [r for r in results if r["name"] != "nominal"]
    summary = dict(
        cases=len(results),
        disturbances=len(disturbed),
        recovered_before_lowering=sum(
            r["recovered_before_lowering"] for r in disturbed
        ),
        nominal=results[0],
        statistics={},
    )
    for key in [
        "normalized_performance",
        "fold_error",
        "fixation_error",
        "endpoint_error",
    ]:
        values = np.array([r[key] for r in disturbed])
        if len(values):
            summary["statistics"][key] = dict(
                mean=float(values.mean()),
                sample_sd=float(values.std(ddof=1)),
                min=float(values.min()),
                max=float(values.max()),
            )
    write_json(folder / "summary.json", summary)
    with (folder / "results.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    write_json(
        folder / "audit.json",
        dict(
            passed=True,
            checks=checks,
            initial_position_max_difference=float(
                np.max(np.abs(np.array(initial_states) - initial_states[0]))
            ),
            weights_max_difference=float(
                np.max(np.abs(np.array(coefficients) - coefficients[0]))
            ),
        ),
    )
    print(json.dumps(summary, indent=2))


def project(points, camera):
    """SoftGym's OpenGL camera, with the same 45-degree vertical field of view."""
    a = -camera["angle"][1]
    rotation = np.array(
        [[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]]
    )
    view = (points - np.asarray(camera["pos"])) @ rotation.T
    xy = view[:, :2] / -view[:, 2, None] * (320 / np.tan(np.pi / 8))
    return np.column_stack([320 + xy[:, 0], 320 - xy[:, 1]])


def planar_flow(reference, weights, camera, crop):
    """The fixed folding plane, sampled on a regular image grid for streamplot."""
    origin = reference[0].reshape(2, 3)
    horizontal = (reference[-1] - reference[0]).reshape(2, 3).copy()
    horizontal[:, 1] = 0
    horizontal /= np.linalg.norm(horizontal, axis=1, keepdims=True)
    basis = np.column_stack([horizontal.ravel(), np.tile([0, 1, 0], 2)])
    front = int(np.argmax(origin[:, 2]))
    plane = basis.reshape(2, 3, 2)[front]
    normal = np.cross(plane[:, 0], plane[:, 1])
    x, y = np.linspace(crop[0], crop[2], 141), np.linspace(crop[1], crop[3], 121)
    xx, yy = np.meshgrid(x, y)
    focal = 320 / np.tan(np.pi / 8)
    rays = np.stack(
        [(xx - 320) / focal, (320 - yy) / focal, -np.ones_like(xx)], axis=-1
    )
    a = -camera["angle"][1]
    rotation = np.array(
        [[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]]
    )
    rays = rays @ rotation
    camera_pos = np.asarray(camera["pos"])
    depth = np.dot(origin[front] - camera_pos, normal) / (rays @ normal)
    world = camera_pos + depth[..., None] * rays
    coordinates = (world - origin[front]) @ np.linalg.pinv(plane).T
    points = (origin.ravel() + coordinates @ basis.T).reshape(*xx.shape, 2, 3)
    curve = QuadraticSpline(nbSeg=12, nbDim=6, device="cpu")
    with torch.no_grad():
        _, directions = dynamical_system_single_step(
            curve,
            torch.tensor(points.reshape(1, -1, 6), dtype=torch.float32),
            torch.tensor(weights),
            lambda_dist=ATTRACTION,
        )
    vectors = directions.numpy().reshape(points.shape)
    tangent = vectors.reshape(*xx.shape, 6) @ np.linalg.pinv(basis).T @ plane.T
    view = (world - camera_pos) @ rotation.T
    motion = tangent @ rotation.T
    dx = (
        focal
        * (-view[..., 2] * motion[..., 0] + view[..., 0] * motion[..., 2])
        / view[..., 2] ** 2
    )
    dy = (
        -focal
        * (-view[..., 2] * motion[..., 1] + view[..., 1] * motion[..., 2])
        / view[..., 2] ** 2
    )
    return dict(
        x=x,
        y=y,
        points=points,
        vectors=vectors,
        velocity=np.stack([dx, dy], axis=-1),
        origin=origin,
        basis=basis,
        front=front,
    )


def paint_flow(frame, field):
    """Use Matplotlib's native antialiased streamlines and arrow patches."""
    from matplotlib import pyplot as plt

    fig = plt.figure(figsize=(6.4, 6.4), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.streamplot(
        field["x"],
        field["y"],
        field["velocity"][..., 0],
        field["velocity"][..., 1],
        color="#8AC6ED",
        linewidth=2.0,
        density=1.0,
        arrowsize=1.2,
        minlength=0.15,
        maxlength=2.0,
    )
    ax.set(xlim=(0, 640), ylim=(640, 0), aspect="equal")
    ax.set_axis_off()
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    fig.canvas.draw()
    overlay = Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy())
    plt.close(fig)
    overlay = overlay.resize(frame.size, Image.Resampling.LANCZOS)
    return Image.alpha_composite(frame.convert("RGBA"), overlay)


def paint_trajectory(frame, trace, color=None, width=12.5):
    """Paper's reference gradient, or a solid color for actual execution."""
    anchors = np.array(
        [[241, 69, 79], [227, 136, 176], [192, 180, 255], [114, 150, 255]]
    )
    phase = np.linspace(0, 1, len(trace) - 1)
    colors = np.column_stack(
        [np.interp(phase, np.linspace(0, 1, 4), anchors[:, c]) for c in range(3)]
    )
    if color is not None:
        colors[:] = color
    scale = 4
    radius = width * scale / 2
    halo = radius + 2 * scale
    overlay = Image.new("RGBA", (frame.width * scale, frame.height * scale))
    paint = ImageDraw.Draw(overlay)
    xy = trace * scale
    paint.line(
        [tuple(p) for p in xy], fill="white", width=round(2 * halo), joint="curve"
    )
    for point in [xy[0], xy[-1]]:
        paint.ellipse([tuple(point - halo), tuple(point + halo)], fill="white")
    for start, end, color in zip(xy[:-1], xy[1:], colors):
        color = tuple(np.rint(color).astype(int)) + (255,)
        paint.line([tuple(start), tuple(end)], fill=color, width=round(2 * radius))
        paint.ellipse([tuple(start - radius), tuple(start + radius)], fill=color)
        paint.ellipse([tuple(end - radius), tuple(end + radius)], fill=color)
    overlay = overlay.resize(frame.size, Image.Resampling.LANCZOS)
    return Image.alpha_composite(frame.convert("RGBA"), overlay).convert("RGB")


def plot(folder, output=None):
    """Two predefined disturbances, five stages, all frames from recorded physics."""
    from matplotlib import pyplot as plt
    from matplotlib.font_manager import fontManager

    output = folder if output is None else output
    output.mkdir(parents=True, exist_ok=True)
    font_path = "/usr/share/fonts/truetype/croscore/Arimo-Bold.ttf"
    fontManager.addfont(font_path)
    font = ImageFont.truetype(font_path, 42)
    title_font = ImageFont.truetype(font_path, 46)
    names = json.loads((folder / "manifest.json").read_text())["selected"]
    crop = (140, 100, 540, 450)
    width, height, gap = 400, 350, 8
    canvas = Image.new("RGB", (5 * width + 4 * gap, 2 * height + gap + 62), "white")
    draw = ImageDraw.Draw(canvas)
    for j, title in enumerate(["Before", "Disturbance", "Recovery", "Fold", "Released"]):
        draw.text(
            (j * (width + gap) + width / 2, 26),
            title,
            font=title_font,
            fill="black",
            anchor="mm",
        )
    selections, movies, flow_records = [], [], []
    torch.set_num_threads(1)
    for row, name in enumerate(names):
        case = folder / name
        data = np.load(case / "rollout.npz")
        config = json.loads((case / "config.json").read_text())
        result = json.loads((case / "metrics.json").read_text())
        reader = imageio.get_reader(case / "rollout.mp4")
        movies.append((reader, data, config, result))
        reference = np.load(case / "reference.npz")
        weights = reference["weights"]
        curve = QuadraticSpline(nbSeg=12, nbDim=6, device="cpu")
        with torch.no_grad():
            phi = curve.computePsiList1D(torch.linspace(0, 1, 501))[2].numpy()
        global_path = (phi @ weights.reshape(-1, 6)).reshape(-1, 2, 3)
        field = planar_flow(reference["reference"], weights, config["camera"], crop)
        background = paint_flow(Image.new("RGBA", (640, 640)), field)
        recovery = result["recovery_step"]
        times = [
            result["onset"] - 50,
            result["onset"] + PULSE_STEPS,
            recovery + RECOVERY_HOLD if recovery is not None else 750,
            950,
            STEPS,
        ]
        for col, step in enumerate(times):
            step = int(step) // 4 * 4
            frame = Image.fromarray(reader.get_data(step // 4))
            frame = Image.alpha_composite(frame.convert("RGBA"), background).convert(
                "RGB"
            )
            flow_records.append(
                dict(
                    case=name,
                    step=step,
                    state=data["positions"][step],
                    global_path=global_path,
                    **field,
                )
            )
            picker = field["front"]
            trace = project(
                data["positions"][: step + 1, picker],
                config["camera"],
            )
            frame = paint_trajectory(
                frame, project(global_path[:, picker], config["camera"])
            )
            frame = paint_trajectory(frame, trace, color=[138, 43, 226], width=6)
            frame = frame.crop(crop)
            canvas.paste(frame, (col * (width + gap), 62 + row * (height + gap)))
            selections.append(dict(case=name, stage=col, step=step, time=step * DT))
        draw.text(
            (12, 74 + row * (height + gap)),
            "AB"[row],
            font=font,
            fill="black",
            stroke_width=3,
            stroke_fill="white",
        )
    canvas.save(output / "disturbance_sequence.png")
    fig = plt.figure(figsize=(7.2, 7.2 * canvas.height / canvas.width))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(canvas)
    ax.axis("off")
    fig.savefig(output / "disturbance_sequence.pdf", dpi=300)
    plt.close(fig)
    write_json(output / "figure_frames.json", selections)
    np.savez_compressed(
        output / "planar_flow.npz",
        cases=[r["case"] for r in flow_records],
        steps=[r["step"] for r in flow_records],
        states=[r["state"] for r in flow_records],
        x=[r["x"] for r in flow_records],
        y=[r["y"] for r in flow_records],
        points=[r["points"] for r in flow_records],
        vectors=[r["vectors"] for r in flow_records],
        velocity=[r["velocity"] for r in flow_records],
        front=[r["front"] for r in flow_records],
        global_path=[r["global_path"] for r in flow_records],
        origin=[r["origin"] for r in flow_records],
        basis=[r["basis"] for r in flow_records],
    )
    with imageio.get_writer(
        output / "comparison.mp4", fps=25, quality=8, macro_block_size=1
    ) as writer:
        for frame_id in range(STEPS // 4 + 1):
            image = Image.new("RGB", (808, 406), "white")
            d = ImageDraw.Draw(image)
            for row, (reader, data, config, result) in enumerate(movies):
                image.paste(
                    Image.fromarray(reader.get_data(frame_id)).crop(crop),
                    (row * 408, 56),
                )
                d.text(
                    (row * 408 + 12, 10),
                    "AB"[row],
                    font=font,
                    fill="black",
                )
                if result["onset"] <= frame_id * 4 < result["onset"] + PULSE_STEPS:
                    d.text(
                        (row * 408 + 14, 65),
                        "Disturbance",
                        font=font,
                        fill="#F1454F",
                        stroke_width=2,
                        stroke_fill="white",
                    )
            writer.append_data(np.asarray(image))
    for reader, *_ in movies:
        reader.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--amplitudes", type=float, nargs=2, default=[0.04, 0.08])
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--figure-dir", type=Path)
    args = parser.parse_args()
    if args.audit:
        audit(args.output)
    elif args.plot:
        plot(args.output, args.figure_dir)
    else:
        run(args.output, args.suite, args.pilot, args.amplitudes)
