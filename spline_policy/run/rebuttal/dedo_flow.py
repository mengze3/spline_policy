"""DEDO cloth hanging with the release spline flow field, without training."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT.parent
SOURCE = RELEASE / "data_local/dedo"
sys.path[:0] = [str(ROOT), str(SOURCE), str(RELEASE / "data_local/dedo_deps")]

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pybullet as bullet
import torch
from dedo.envs.deform_env import DeformEnv
from dedo.utils.args import get_args
from dedo.utils.mesh_utils import get_mesh_data
from dedo.utils.preset_info import preset_traj
from policy.diffusion_policy.planning.quadratic_spline import (
    QuadraticSpline,
    dynamical_system_single_step,
)


class ClothScene(DeformEnv):
    """Keep DEDO physics and control; add a logged external force pulse."""

    pulse = False
    records = None
    settling = False

    def do_action(self, action, unscaled):
        super().do_action(action, unscaled)
        if self.pulse:
            for anchor in self.anchor_ids:
                self.sim.applyExternalForce(
                    anchor,
                    -1,
                    self.force,
                    self.sim.getBasePositionAndOrientation(anchor)[0],
                    bullet.WORLD_FRAME,
                )

    def make_final_steps(self):
        self.settling = True
        return super().make_final_steps()

    def get_obs(self):
        result = super().get_obs()
        if self.settling:
            self.capture(np.zeros(6), False)
        return result

    def position(self):
        return np.array(
            [self.sim.getBasePositionAndOrientation(i)[0] for i in self.anchor_ids]
        ).reshape(-1)

    def capture(self, command, pulse):
        vertices = np.asarray(get_mesh_data(self.sim, self.deform_id)[1])
        contacts = self.sim.getContactPoints(bodyA=self.deform_id)
        counts = [sum(c[2] == body for c in contacts) for body in self.rigid_ids]
        self.records.append(
            (
                self.position(),
                command,
                vertices,
                len(contacts),
                self.get_reward(),
                pulse,
                self.settling,
                counts,
            )
        )
        if self.writer is not None and len(self.records) % 2 == 0:
            camera = self.sim.computeViewMatrixFromYawPitchRoll([0, 1.6, 7.2], 10, 314, -16, 0, 2)
            projection = self.sim.computeProjectionMatrixFOV(55, 1, 0.1, 100)
            rgb = self.sim.getCameraImage(
                640,
                640,
                viewMatrix=camera,
                projectionMatrix=projection,
                renderer=bullet.ER_TINY_RENDERER,
                lightDirection=[-3, -4, 8],
                shadow=1,
            )[2][:, :, :3]
            canvas = Image.new("RGB", (640, 698), "white")
            canvas.paste(Image.fromarray(rgb), (0, 58))
            draw = ImageDraw.Draw(canvas)
            draw.text((22, 14), self.title, fill="black", font=self.font)
            if pulse:
                draw.text(
                    (22, 78),
                    "External push",
                    fill="#F1454F",
                    font=self.font,
                    stroke_width=2,
                    stroke_fill="white",
                )
            self.writer.append_data(np.asarray(canvas))
            self.last_frame = canvas
            if len(self.records) in [2, 192, 208, 320, 480]:
                canvas.save(self.folder / f"frame_{len(self.records):03d}.png")


def public_trajectory(env):
    """Interpolate the upstream preset positions and segment durations."""
    points = preset_traj[env.deform_obj]["waypoints"]
    paths = []
    for i, key in enumerate(["a", "b"]):
        waypoints = np.array(points[key])
        start = env.position().reshape(2, 3)[i]
        times = np.r_[0, np.cumsum(waypoints[:, 3])]
        positions = np.vstack([start, waypoints[:, :3]])
        paths.append((times, positions))
    duration = max(t[-1] for t, _ in paths)
    time = np.linspace(0, duration, 401)
    reference = np.column_stack(
        [np.interp(time, t, x[:, axis]) for t, x in paths for axis in range(3)]
    )
    return time, reference


def fit_flow(reference, segments=12):
    curve = QuadraticSpline(nbSeg=segments, nbDim=6, device="cpu")
    _, _, phi = curve.computePsiList1D(torch.linspace(0, 1, len(reference)))
    phi, control = phi.numpy().astype(float), curve.C.numpy().astype(float)
    constraints = np.vstack([phi[0], phi[-1], control[-1] - control[-2]])
    values = np.vstack([reference[0], reference[-1], np.zeros(6)])
    system = np.block([[phi.T @ phi, constraints.T], [constraints, np.zeros((3, 3))]])
    weights = np.linalg.solve(system, np.vstack([phi.T @ reference, values]))[:-3]
    np.testing.assert_allclose(constraints @ weights, values, atol=1e-8)
    return curve, torch.tensor(weights.reshape(1, -1), dtype=torch.float32), phi @ weights


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "source_snapshot.py").write_bytes(Path(__file__).read_bytes())
    settings = get_args()
    settings.env = "HangGarment-v1"
    settings.task, settings.version = "HangGarment", 1
    settings.cam_resolution = 0
    settings.max_episode_len = args.steps
    settings.seed = 0
    settings.device = "cpu"
    env = ClothScene(settings)
    env.seed(0)
    env.reset()
    env.force = args.force
    env.folder = args.output
    env.font = ImageFont.truetype("/usr/share/fonts/truetype/croscore/Arimo-Bold.ttf", 28)
    env.title = (
        "Published preset"
        if args.mode == "preset"
        else ("Spline flow + push" if args.perturb else "Spline flow")
    )
    env.writer = (
        None
        if args.no_video
        else imageio.get_writer(
            args.output / "rollout.mp4", fps=31.25, codec="libx264", quality=8, macro_block_size=1
        )
    )
    env.records = []
    time, reference = public_trajectory(env)
    curve, weights, fitted = fit_flow(reference)
    goal = fitted[-1]
    dt = settings.sim_steps_per_action / settings.sim_freq
    speed = np.linalg.norm(np.diff(fitted, axis=0), axis=1).sum() / time[-1]
    np.savez_compressed(
        args.output / "reference.npz",
        time=time,
        reference=reference,
        fitted=fitted,
        weights=weights.numpy(),
    )
    protocol = dict(
        upstream="https://github.com/contactrika/dedo",
        commit=subprocess.check_output(
            ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True
        ).strip(),
        task=settings.env,
        seed=0,
        mode=args.mode,
        perturb=args.perturb,
        force=args.force,
        pulse_seconds=[args.pulse_time, args.pulse_time + args.pulse_duration],
        sim_hz=settings.sim_freq,
        control_hz=1 / dt,
        speed=float(speed),
        lambda_dist=args.attraction,
        flow_source="policy/diffusion_policy/planning/quadratic_spline.py:dynamical_system_single_step",
        pybullet="3.2.7",
        release=vars(args).copy(),
        settings=vars(settings),
    )
    protocol["release"]["output"] = str(args.output)
    protocol["hashes"] = {
        str(path.relative_to(RELEASE)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [
            Path(__file__).resolve(),
            ROOT / "policy/diffusion_policy/planning/quadratic_spline.py",
            SOURCE / "dedo/utils/preset_info.py",
            SOURCE / "dedo/data/cloth/apron_0.obj",
        ]
    }
    (args.output / "protocol.json").write_text(
        json.dumps(protocol, indent=2, default=lambda x: np.asarray(x).tolist())
    )
    phases = []
    if args.mode == "preset":
        from dedo.demo_preset import build_traj, merge_traj

        preset = preset_traj[env.deform_obj]["waypoints"]
        velocities = merge_traj(
            *(
                build_traj(env, preset, key, anchor_idx=i, ctrl_freq=1 / dt, robot=None)[1]
                for i, key in enumerate(["a", "b"])
            )
        )
    pulse_start = int(np.ceil(args.pulse_time / dt - 1e-10))
    pulse_stop = pulse_start + int(round(args.pulse_duration / dt))
    for step in range(args.steps + 1):
        position = env.position()
        p = torch.tensor(position[None, None], dtype=torch.float32)
        with torch.no_grad():
            _, direction = dynamical_system_single_step(
                curve, p, weights, lambda_dist=args.attraction, step_size=dt
            )
            distance, _, phase = curve.sdf_batch(p, weights)
        if args.mode == "preset":
            command = velocities[step] if step < len(velocities) else np.zeros(6)
        else:
            command = direction.numpy().reshape(-1) * min(
                speed, 8 * np.linalg.norm(position - goal)
            )
        env.pulse = args.perturb and pulse_start <= step < pulse_stop
        env.capture(command.copy(), env.pulse)
        phases.append([float(distance), float(phase) / curve.nbSeg])
        _, _, done, info = env.step(command, unscaled=True)
        if step % 100 == 0:
            print(
                step,
                "distance",
                float(distance),
                "phase",
                float(phase) / curve.nbSeg,
                "reward",
                env.get_reward(),
                flush=True,
            )
        if done:
            break
    positions, commands, vertices, contacts, rewards, pulses, settling, rigid_contacts = map(
        np.asarray, zip(*env.records)
    )
    np.savez_compressed(
        args.output / "rollout.npz",
        positions=positions,
        commands=commands,
        vertices=vertices,
        contacts=contacts,
        rewards=rewards,
        pulse=pulses,
        settling=settling,
        projection=phases,
        goal=goal,
        rigid_contacts=rigid_contacts,
        rigid_ids=env.rigid_ids,
    )
    summary = dict(
        success=bool(info["is_success"]),
        final_reward=float(env.get_reward()),
        final_goal_distance=float(np.linalg.norm(env.position() - goal)),
        controlled_goal_distance=float(np.linalg.norm(positions[~settling][-1] - goal)),
        max_rigid_contacts=int(rigid_contacts.sum(axis=1).max()),
        max_reported_contacts=int(contacts.max()),
        fitting_rmse=float(np.sqrt(np.mean((fitted - reference) ** 2))),
        samples=len(positions),
    )
    (args.output / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(summary, flush=True)
    if env.writer is not None:
        env.writer.close()
        env.last_frame.save(args.output / "final.png")
    env.close()
    return summary


def compare(folders, output):
    """Render the saved physical rollouts and check the recorded flow commands."""
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import fontManager
    from scipy.spatial import cKDTree

    output.mkdir(parents=True, exist_ok=False)
    font_path = "/usr/share/fonts/truetype/croscore/Arimo-Bold.ttf"
    fontManager.addfont(font_path)
    plt.rcParams.update({"font.family": "Arimo", "font.size": 12, "axes.labelweight": "bold"})
    font = ImageFont.truetype(font_path, 27)
    colors = ["#6DD2D9", "#AD6AEA"]
    names = ["Flow", "Flow + push"]
    datasets, audits = [], []
    view = np.array(
        bullet.computeViewMatrixFromYawPitchRoll([0, 1.6, 7.2], 10, 314, -16, 0, 2)
    ).reshape(4, 4, order="F")
    projection = np.array(bullet.computeProjectionMatrixFOV(55, 1, 0.1, 100)).reshape(
        4, 4, order="F"
    )

    def screen(points):
        q = np.column_stack([points, np.ones(len(points))]) @ (projection @ view).T
        q = q[:, :2] / q[:, 3:4]
        return np.column_stack([(q[:, 0] + 1) * 320, (1 - q[:, 1]) * 320 + 58])

    for folder in folders:
        data = np.load(folder / "rollout.npz")
        reference = np.load(folder / "reference.npz")
        protocol = json.loads((folder / "protocol.json").read_text())
        n = len(data["projection"])
        assert n == 501 and np.isfinite(data["vertices"]).all()
        curve = QuadraticSpline(nbSeg=12, nbDim=6, device="cpu")
        weights = torch.tensor(reference["weights"])
        with torch.no_grad():
            replay = []
            for position in data["positions"][:n]:
                p = torch.tensor(position[None, None], dtype=torch.float32)
                _, direction = dynamical_system_single_step(
                    curve, p, weights, lambda_dist=protocol["lambda_dist"], step_size=0.016
                )
                speed = min(protocol["speed"], 8 * np.linalg.norm(position - data["goal"]))
                replay.append(direction.numpy().reshape(-1) * speed)
            replay = np.asarray(replay)
            np.testing.assert_array_equal(replay, data["commands"][:n])
            _, _, phi = curve.computePsiList1D(torch.linspace(0, 1, 20001))
        dense = phi.numpy() @ weights.numpy().reshape(-1, 6)
        distances = cKDTree(dense).query(data["positions"][:n])[0]
        assert np.max(np.abs(distances - data["projection"][:, 0])) < 0.002
        for path, digest in protocol["hashes"].items():
            source = (
                folder / "source_snapshot.py" if path.endswith("dedo_flow.py") else RELEASE / path
            )
            assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
        pulse = data["pulse"][:n]
        expected = protocol["perturb"] & (
            (np.arange(n) * 0.016 >= 3) & (np.arange(n) * 0.016 < 3.16)
        )
        np.testing.assert_array_equal(pulse, expected)
        rigid = data["rigid_contacts"][:n]
        metrics = json.loads((folder / "metrics.json").read_text())
        metrics.update(
            trajectory_distance_final=float(distances[-1]),
            trajectory_distance_peak=float(distances.max()),
            rigid_contact_samples=int(np.count_nonzero(rigid.sum(axis=1))),
            hanger_contact_samples=int(np.count_nonzero(rigid[:, 0])),
            command_replay_max_error=float(np.max(np.abs(replay - data["commands"][:n]))),
            source_and_asset_hashes_verified=True,
            controlled_samples=n,
            passive_samples=int(data["settling"].sum()),
        )
        if pulse.any():
            end = np.flatnonzero(pulse)[-1]
            recovered = next(
                (i for i in range(end + 1, n - 30) if np.all(distances[i : i + 31] < 0.05)), None
            )
            metrics["recovery_to_005_seconds"] = (
                None if recovered is None else (recovered - end - 1) * 0.016
            )
        audits.append(metrics)
        datasets.append((data, reference, distances))
    np.testing.assert_array_equal(datasets[0][1]["weights"], datasets[1][1]["weights"])
    np.testing.assert_allclose(
        datasets[0][0]["positions"][:188], datasets[1][0]["positions"][:188], atol=1e-8
    )
    (output / "audit.json").write_text(
        json.dumps(
            dict(
                runs=audits,
                units="upstream simulation units",
                distance="6D grasp-point distance to spline",
                recovery_threshold=0.05,
                recovery_hold_samples=31,
                shared_reference_and_pre_push_states=True,
            ),
            indent=2,
        )
    )

    readers = [imageio.get_reader(folder / "rollout.mp4") for folder in folders]
    writer = imageio.get_writer(
        output / "comparison.mp4", fps=31.25, codec="libx264", quality=8, macro_block_size=1
    )
    selected = {}
    for frame_index, frames in enumerate(zip(*readers)):
        canvas = Image.new("RGB", (1020, 602), "white")
        for side, frame in enumerate(frames):
            data = datasets[side][0]
            index = min(2 * frame_index + 1, len(data["positions"]) - 1)
            image = Image.fromarray(frame)
            draw = ImageDraw.Draw(image)
            if not data["settling"][index]:
                for hand in range(2):
                    points = data["positions"][
                        max(0, index - 110) : index + 1, hand * 3 : hand * 3 + 3
                    ]
                    pixels = screen(points)
                    draw.line([tuple(p) for p in pixels], fill=colors[side], width=5)
                    x, y = pixels[-1]
                    draw.ellipse(
                        (x - 5, y - 5, x + 5, y + 5), fill=colors[side], outline="white", width=2
                    )
                if data["pulse"][index]:
                    x, y = screen(data["positions"][index, :3][None])[0]
                    draw.line((x - 65, y - 25, x - 10, y - 3), fill="#F1454F", width=7)
                    draw.polygon([(x - 5, y), (x - 26, y - 19), (x - 25, y + 2)], fill="#F1454F")
            image = image.crop((100, 150, 610, 698))
            canvas.paste(image, (side * 510, 54))
            label = names[side] if not data["settling"][index] else names[side] + " · released"
            ImageDraw.Draw(canvas).text((side * 510 + 18, 12), label, font=font, fill="black")
        if datasets[1][0]["pulse"][index]:
            ImageDraw.Draw(canvas).text((528, 66), "External push", font=font, fill="#F1454F")
        writer.append_data(np.asarray(canvas))
        if frame_index in [95, 159, 239]:
            selected[frame_index] = canvas.copy()
    writer.close()
    for reader in readers:
        reader.close()
    for index, frame in selected.items():
        frame.save(output / f"comparison_{index:03d}.png")

    still = Image.new("RGB", (700, 460), "white")
    for side, name in enumerate(names):
        panel = selected[159].crop((side * 510 + 170, 150, side * 510 + 500, 570))
        still.paste(panel, (side * 350 + 10, 40))
        ImageDraw.Draw(still).text((side * 350 + 12, 5), name, font=font, fill="black")
    fig, (a, b) = plt.subplots(
        2, 1, figsize=(8, 7), gridspec_kw={"height_ratios": [1.7, 1]}, layout="constrained"
    )
    a.imshow(still)
    a.axis("off")
    for color, name, (_, _, distance) in zip(colors, names, datasets):
        b.plot(np.arange(len(distance)) * 0.016, distance, color=color, lw=2.8, label=name)
    b.axvspan(3.008, 3.168, color="#F1454F", alpha=0.15)
    b.set(xlabel="Time (s)", ylabel="Distance to trajectory\n(simulation units)", xlim=(0, 8))
    b.spines[["top", "right"]].set_visible(False)
    b.legend(frameon=False, loc="upper right", ncol=2)
    fig.savefig(output / "flow_contact.png", dpi=200)
    fig.savefig(output / "flow_contact.pdf")
    plt.close(fig)


def trial_metrics(folder):
    """Separate pre-contact recovery, continuation, and the upstream task criterion."""
    from scipy.spatial import cKDTree

    data = np.load(folder / "rollout.npz")
    reference = np.load(folder / "reference.npz")
    protocol = json.loads((folder / "protocol.json").read_text())
    result = json.loads((folder / "metrics.json").read_text())
    dt = 1 / protocol["control_hz"]
    n = len(data["projection"])
    curve = QuadraticSpline(nbSeg=12, nbDim=6, device="cpu")
    weights = torch.tensor(reference["weights"])
    with torch.no_grad():
        _, _, phi = curve.computePsiList1D(torch.linspace(0, 1, 20001))
        dense = phi.numpy() @ weights.numpy().reshape(-1, 6)
        for i, position in enumerate(data["positions"][:n]):
            p = torch.tensor(position[None, None], dtype=torch.float32)
            _, direction = dynamical_system_single_step(
                curve, p, weights, lambda_dist=protocol["lambda_dist"], step_size=dt
            )
            command = direction.numpy().reshape(-1) * min(
                protocol["speed"], 8 * np.linalg.norm(position - data["goal"])
            )
            np.testing.assert_array_equal(command, data["commands"][i])
    distance = cKDTree(dense).query(data["positions"][:n])[0]
    assert np.max(np.abs(distance - data["projection"][:, 0])) < 0.002
    assert np.isfinite(data["vertices"]).all()
    for name, digest in protocol["hashes"].items():
        path = folder / "source_snapshot.py" if name.endswith("dedo_flow.py") else RELEASE / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    pulse = np.flatnonzero(data["pulse"][:n])
    contact = np.flatnonzero(data["rigid_contacts"][:n, 0] > 0)
    lowering = np.flatnonzero(data["projection"][:, 1] >= 1 / 3)
    recovered = None
    if len(pulse):
        recovered = next(
            (i for i in range(pulse[-1] + 1, n - 10) if np.all(distance[i : i + 11] < 0.05)), None
        )
    before_contact = recovered is not None and len(contact) and recovered + 10 < contact[0]
    before_lowering = recovered is not None and len(lowering) and recovered + 10 < lowering[0]
    result.update(
        case=folder.name,
        force=protocol["force"],
        pulse_start=protocol["pulse_seconds"][0],
        peak_distance=float(distance.max()),
        final_distance=float(distance[-1]),
        recovery_seconds=None if recovered is None else float((recovered - pulse[-1] - 1) * dt),
        recovery_before_contact=bool(before_contact),
        recovery_before_lowering=bool(before_lowering),
        endpoint_reached=bool(result["controlled_goal_distance"] < 0.1),
        force_before_contact=bool(len(pulse) and len(contact) and pulse[-1] < contact[0]),
        first_contact_seconds=None if not len(contact) else float(contact[0] * dt),
        lowering_seconds=None if not len(lowering) else float(lowering[0] * dt),
        recovered_index=recovered,
        command_replay_exact=True,
        hashes_verified=True,
        rollout_sha256=hashlib.sha256((folder / "rollout.npz").read_bytes()).hexdigest(),
    )
    (folder / "audit.json").write_text(json.dumps(result, indent=2))
    return result


def sweep(args):
    """Run a fixed 24-case perturbation grid; keep every outcome."""
    args.output.mkdir(parents=True, exist_ok=False)
    cases = []
    for axis, name in enumerate("xyz"):
        for sign, label in [(-1, "neg"), (1, "pos")]:
            for magnitude in [12, 16]:
                for onset in [0.4, 0.8]:
                    force = [0.0, 0.0, 0.0]
                    force[axis] = sign * magnitude
                    cases.append(
                        dict(
                            name=f"{name}_{label}_f{magnitude}_t{int(onset*100):03d}",
                            force=force,
                            onset=onset,
                        )
                    )
    examples = ["x_pos_f16_t040", "z_pos_f16_t040"]
    manifest = dict(
        cases=cases,
        figure_examples=examples,
        pulse_duration=0.08,
        seed=0,
        recovery_threshold=0.05,
        recovery_hold_samples=11,
        note="Both anchors receive the same force simultaneously. Fixed trajectory; no score-based retries.",
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    results = []
    for index, case in enumerate(cases):
        options = vars(args).copy()
        options.update(
            output=args.output / case["name"],
            sweep=False,
            perturb=True,
            force=case["force"],
            pulse_time=case["onset"],
            pulse_duration=0.08,
            no_video=case["name"] not in examples,
        )
        (args.output / "run_status.json").write_text(
            json.dumps(dict(completed=index, total=len(cases), current=case["name"]))
        )
        run(argparse.Namespace(**options))
        results.append(trial_metrics(args.output / case["name"]))
        (args.output / "results.json").write_text(json.dumps(results, indent=2))
        print(
            "SWEEP",
            index + 1,
            len(cases),
            case["name"],
            results[-1]["recovery_before_lowering"],
            flush=True,
        )
    import csv

    with (args.output / "results.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    keys = [
        "force_before_contact",
        "recovery_before_contact",
        "recovery_before_lowering",
        "endpoint_reached",
        "success",
    ]
    summary = {key: sum(row[key] for row in results) for key in keys}
    summary.update(
        total=len(results),
        recovery_seconds=[row["recovery_seconds"] for row in results],
        all_commands_replayed_exactly=True,
        all_hashes_verified=True,
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.output / "run_status.json").write_text(
        json.dumps(dict(completed=len(cases), total=len(cases), status="complete"))
    )
    sequence(args.output)


def sequence(folder):
    """Two prespecified disturbances, five physical frames per row."""
    manifest = json.loads((folder / "manifest.json").read_text())
    width, height, left, top = 420, 500, 105, 90
    canvas = Image.new("RGB", (left + 5 * width, top + 2 * height), "white")
    font = ImageFont.truetype("/usr/share/fonts/truetype/croscore/Arimo-Bold.ttf", 46)
    small = ImageFont.truetype("/usr/share/fonts/truetype/croscore/Arimo-Bold.ttf", 35)
    draw = ImageDraw.Draw(canvas)
    for column, text in enumerate(["Approach", "Push", "Recovery", "Continue", "Released"]):
        draw.text(
            (left + column * width + width / 2, 24), text, font=font, fill="black", anchor="mt"
        )
    view = np.array(
        bullet.computeViewMatrixFromYawPitchRoll([0, 1.6, 7.2], 10, 314, -16, 0, 2)
    ).reshape(4, 4, order="F")
    projection = np.array(bullet.computeProjectionMatrixFOV(55, 1, 0.1, 100)).reshape(
        4, 4, order="F"
    )

    def project(points):
        p = np.column_stack([points, np.ones(len(points))]) @ (projection @ view).T
        q = p[:, :2] / p[:, 3:4]
        return np.column_stack([(q[:, 0] + 1) * 320, (1 - q[:, 1]) * 320 + 58])

    selection = []
    for row, name in enumerate(manifest["figure_examples"]):
        path = folder / name
        data = np.load(path / "rollout.npz")
        audit = json.loads((path / "audit.json").read_text())
        pulse = np.flatnonzero(data["pulse"])
        descent = np.flatnonzero(data["projection"][:, 1] >= 2 / 3)
        lowering = int(descent[0]) if len(descent) else len(data["projection"]) - 1
        indices = [
            pulse[0] - 2,
            pulse[-1],
            audit["recovered_index"] + 10,
            int(lowering),
            len(data["positions"]) - 1,
        ]
        video = imageio.get_reader(path / "rollout.mp4")
        draw.text(
            (26, top + row * height + height / 2), "AB"[row], font=font, fill="black", anchor="lm"
        )
        for column, requested in enumerate(indices):
            frame_index = max(0, min((requested - 1) // 2, video.count_frames() - 1))
            actual = 2 * frame_index + 1
            image = Image.fromarray(video.get_data(frame_index))
            overlay = ImageDraw.Draw(image)
            for hand in range(2):
                points = data["positions"][
                    max(0, actual - 28) : actual + 1, hand * 3 : hand * 3 + 3
                ]
                pixels = project(points)
                overlay.line([tuple(p) for p in pixels], fill="#AD6AEA", width=5)
                x, y = pixels[-1]
                overlay.ellipse(
                    (x - 5, y - 5, x + 5, y + 5), fill="#AD6AEA", outline="white", width=2
                )
                if column == 1:
                    force = np.asarray(audit["force"])
                    anchor = data["positions"][actual, hand * 3 : hand * 3 + 3]
                    tail = project((anchor - force / np.linalg.norm(force) * 0.8)[None])[0]
                    vector = np.array([x, y]) - tail
                    unit = vector / max(np.linalg.norm(vector), 1e-6)
                    normal = np.array([-unit[1], unit[0]])
                    overlay.line([tuple(tail), (x, y)], fill="#F1454F", width=6)
                    overlay.polygon(
                        [
                            tuple([x, y]),
                            tuple(np.array([x, y]) - 15 * unit + 7 * normal),
                            tuple(np.array([x, y]) - 15 * unit - 7 * normal),
                        ],
                        fill="#F1454F",
                    )
            tile = image.crop((100, 150, 610, 698)).resize((width - 8, height - 40))
            canvas.paste(tile, (left + column * width, top + row * height))
            if column == 3 and not len(descent):
                draw.text(
                    (left + column * width + 16, top + row * height + 12),
                    "Early stop",
                    font=small,
                    fill="#F1454F",
                    stroke_width=2,
                    stroke_fill="white",
                )
            draw.text(
                (left + column * width + width / 2, top + row * height + height - 35),
                f"{actual*.016:.2f} s",
                font=small,
                fill="black",
                anchor="mt",
            )
            selection.append(
                dict(
                    case=name,
                    column=column,
                    requested_index=int(requested),
                    video_frame=int(frame_index),
                    record_index=int(actual),
                )
            )
        video.close()
    canvas.save(folder / "disturbance_sequence.png")
    canvas.save(folder / "disturbance_sequence.pdf", resolution=300)
    (folder / "figure_frames.json").write_text(json.dumps(selection, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--sequence", type=Path)
    parser.add_argument("--compare", type=Path, nargs=2, metavar=("NOMINAL", "PUSH"))
    parser.add_argument("--mode", choices=["flow", "preset"], default="flow")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--attraction", type=float, default=5)
    parser.add_argument("--perturb", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--force", type=float, nargs=3, default=[16, 0, 0])
    parser.add_argument("--pulse-time", type=float, default=3)
    parser.add_argument("--pulse-duration", type=float, default=0.16)
    args, _ = parser.parse_known_args()
    torch.set_num_threads(1)
    if args.sweep:
        sweep(args)
    elif args.sequence:
        sequence(args.sequence)
    elif args.compare:
        compare(args.compare, args.output)
    else:
        run(args)
