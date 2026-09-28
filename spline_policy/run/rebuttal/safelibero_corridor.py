"""SafeLIBERO Level I: public demonstration -> quadratic spline -> safe corridor.

This is a trajectory-adaptation demo, not a trained SafeLIBERO policy benchmark.
The simulator executes OSC actions; saved states are used only to initialize it.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tarfile
import urllib.request

import h5py
import cvxpy as cp
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from scipy.spatial.transform import Rotation, Slerp
from scipy.optimize import least_squares
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT.parent
TASK = 'pick_up_the_orange_juice_and_place_it_in_the_basket'
UPSTREAM = '2457feed5968ae803926e178c8ce8243b9ecdcf9'
STRETCH = 3
REFERENCE_COLORS = ['#F1454F', '#E388B0', '#C0B4FF', '#7296FF']
DEMO_URL = ('https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/resolve/main/'
            f'libero_object/{TASK}_demo.hdf5')
DEMO_SHA = '53e40490c62e94c678233eefbfd874a61c93d92c8088f59d426ae6c3fd90d288'


def fetch(source):
    source.mkdir(parents=True, exist_ok=True)
    archive = source / 'upstream.tar.gz'
    if not (source / 'safelibero/LICENSE').exists():
        urllib.request.urlretrieve(f'https://codeload.github.com/THU-RCSCT/vlsa-aegis/tar.gz/{UPSTREAM}', archive)
        with tarfile.open(archive) as tar:
            for item in tar:
                parts = Path(item.name).parts[1:]
                if item.isfile() and parts and (parts[0] == 'safelibero' or parts == ('LICENSE',)):
                    if '..' in parts:
                        raise ValueError('Invalid archive path')
                    target = source.joinpath(*parts)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(tar.extractfile(item).read())
    demo = source / 'orange_juice_demo.hdf5'
    if not demo.exists():
        urllib.request.urlretrieve(DEMO_URL, demo)
    if hashlib.sha256(demo.read_bytes()).hexdigest() != DEMO_SHA:
        raise ValueError('Public demonstration SHA256 mismatch')


def configure(source):
    """Keep upstream paths and caches local to this experiment."""
    package = source / 'safelibero'
    assets = package / 'libero/libero'
    config = source / 'config'
    config.mkdir(exist_ok=True)
    paths = dict(benchmark_root=str(assets), bddl_files=str(assets / 'bddl_files'),
                 init_states=str(assets / 'init_files'), assets=str(assets / 'assets'),
                 datasets=str(source))
    (config / 'config.yaml').write_text(yaml.safe_dump(paths))
    os.environ['LIBERO_CONFIG_PATH'] = str(config)
    sys.path.insert(0, str(package))


def make_env(source, episode, resolution=512):
    configure(source)
    from libero.libero.envs import OffScreenRenderEnv
    base = source / 'safelibero/libero/libero'
    bddl = base / f'bddl_files/safelibero_object/{TASK}.bddl'
    initial = base / f'init_files/safelibero_object/{TASK}_level_I.pruned_init'
    states = torch.load(initial, map_location='cpu')
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_names=['agentview'],
                            camera_heights=resolution, camera_widths=resolution,
                            ignore_done=True)
    env.seed(7)
    env.reset()
    obs = env.set_init_state(states[episode])
    for _ in range(10):
        obs, _, _, _ = env.step([0.] * 6 + [-1.])
    return env, obs


def demonstration(path, index):
    with h5py.File(path, 'r') as data:
        demo = data[f'data/demo_{index}']
        return dict(position=demo['obs/ee_pos'][:], rotation=demo['obs/ee_ori'][:],
                    actions=demo['actions'][:], initial=demo['states'][0])


def spline_basis(segments, samples):
    path = ROOT / 'policy/diffusion_policy/planning/quadratic_spline.py'
    spec = importlib.util.spec_from_file_location('release_spline', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    spline = module.QuadraticSpline(nbSeg=segments, nbDim=3, device='cpu')
    phase = torch.linspace(0, 1, samples)
    _, _, basis = spline.computePsiList1D(phase)
    return basis.numpy().astype(float), spline.C.numpy().astype(float)


def reference(demo, obs, stretch=STRETCH):
    """Retarget the demonstration's start, grasp and placement translations."""
    n = len(demo['position'])
    transitions = np.flatnonzero(np.diff(demo['actions'][:, -1]) != 0) + 1
    grasp, release = transitions[[0, -1]]
    source_object = demo['initial'][10:13]
    source_basket = demo['initial'][17:20]
    offsets = np.array([obs['robot0_eef_pos'] - demo['position'][0],
                        obs['orange_juice_1_pos'] - source_object,
                        obs['basket_1_pos'] - source_basket])
    # Keep the approach / grasp rigid in the target frame; blend during transport.
    knots = [0, max(1, grasp - 20), grasp + 12, release - 12, n - 1]
    translations = offsets[[0, 1, 1, 2, 2]]
    shift = np.column_stack([np.interp(np.arange(n), knots, translations[:, j])
                             for j in range(3)])
    position = demo['position'] + shift
    t = np.linspace(0, n - 1, n * stretch)
    position = np.column_stack([np.interp(t, np.arange(n), position[:, j])
                                for j in range(3)])
    rotation = Slerp(np.arange(n), Rotation.from_rotvec(demo['rotation']))(t).as_rotvec()
    grip = demo['actions'][np.floor(t).astype(int), -1]
    return position, rotation, grip


def obstacle_bounds(env, obs):
    """World AABBs of collision meshes of active, added benchmark obstacles."""
    model, data = env.sim.model, env.sim.data
    boxes, names = [], []
    for key, value in obs.items():
        if not key.endswith('_pos') or 'obstacle' not in key or '_to_' in key:
            continue
        if value[2] < 0 or np.max(np.abs(value[:2])) > .6:
            continue
        name = key[:-4]
        points = []
        for g in range(model.ngeom):
            geom_name = model.geom_id2name(g) or ''
            if not geom_name.startswith(name) or not model.geom_contype[g]:
                continue
            mesh = model.geom_dataid[g]
            if mesh >= 0:
                start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
                vertices = model.mesh_vert[start:start + count]
            else:
                signs = np.array([[x, y, z] for x in [-1, 1]
                                  for y in [-1, 1] for z in [-1, 1]])
                vertices = signs * model.geom_size[g]
            points.extend(vertices @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g])
        points = np.asarray(points)
        boxes.append([points.min(axis=0), points.max(axis=0)])
        names.append(name)
    return np.array(boxes), names


def fit_reference(position, segments=48):
    basis, controls = spline_basis(segments, len(position))
    w = cp.Variable((segments + 2, 3))
    loss = cp.sum_squares(basis @ w - position)
    problem = cp.Problem(cp.Minimize(loss),
                         [basis[0] @ w == position[0], basis[-1] @ w == position[-1]])
    problem.solve(solver='CLARABEL')
    if problem.status != 'optimal':
        raise RuntimeError(f'Spline fit: {problem.status}')
    return basis, controls, w.value


def build_corridor(controls, nominal, boxes, padding):
    """Build bounded 3D cells from the nominal path and obstacles, before solving."""
    points = (controls @ nominal).reshape(-1, 3, 3)
    inflated = boxes + np.array([-1, 1])[None, :, None]*padding
    faces = []
    for segment, curve in enumerate(points):
        for obstacle, (lo, hi) in enumerate(inflated):
            costs = np.r_[np.maximum(curve-lo, 0).max(axis=0),
                          np.maximum(hi-curve, 0).max(axis=0)]
            face = int(costs.argmin())
            axis, side = face % 3, -1 if face < 3 else 1
            faces.append([segment, obstacle, axis, side, lo[axis] if side < 0 else hi[axis]])
    faces = np.array(faces)

    def clip_bounds(bounds, segment):
        for _, _, axis, side, bound in faces[faces[:, 0] == segment]:
            axis = int(axis)
            if side > 0:
                bounds[0, axis] = max(bounds[0, axis], bound)
            else:
                bounds[1, axis] = min(bounds[1, axis], bound)
        return bounds

    cells, intervals = [], []
    first = 0
    while first < len(points):
        selected = faces[faces[:, 0] == first, 1:]
        last = first+1
        while last < min(first+4, len(points)) and np.array_equal(faces[faces[:, 0] == last, 1:], selected):
            last += 1
        seed = points[first:last].reshape(-1, 3).copy()
        for _, axis, side, bound in selected:
            axis = int(axis)
            seed[:, axis] = np.maximum(seed[:, axis], bound) if side > 0 else np.minimum(seed[:, axis], bound)
        cell = np.array([seed.min(axis=0)-[.06, .06, .05], seed.max(axis=0)+[.06, .06, .05]])
        cells.append(clip_bounds(cell, first))
        intervals.append([first, last])
        first = last
    cells, intervals = np.array(cells), np.array(intervals)
    # Shared exterior connectors make adjacent bounded cells overlap at turns.
    for k in range(len(cells)-1):
        join = points[intervals[k, 1]-1, -1].copy()
        for segment in intervals[k:k+2, 0]:
            for _, _, axis, side, bound in faces[faces[:, 0] == segment]:
                axis = int(axis)
                join[axis] = max(join[axis], bound+.04) if side > 0 else min(join[axis], bound-.04)
        for j in [k, k+1]:
            cells[j, 0] = np.minimum(cells[j, 0], join-.035)
            cells[j, 1] = np.maximum(cells[j, 1], join+.035)
            cells[j] = clip_bounds(cells[j], intervals[j, 0])
    # Tighten every face for tracking error. Preserve the fixed start and end.
    inner = cells.copy()
    for k, ((lo, hi), (first, last)) in enumerate(zip(cells, intervals)):
        margin = np.full((2, 3), .025)
        endpoints = ([points[0, 0]] if first == 0 else []) + ([points[-1, -1]] if last == len(points) else [])
        for point in endpoints:
            margin = np.minimum(margin, .8*np.array([point-lo, hi-point]))
        if np.any(margin <= 0):
            raise ValueError('Fixed endpoint is outside the finite corridor')
        inner[k] = [lo+margin[0], hi-margin[1]]
    overlap = np.minimum(inner[:-1, 1], inner[1:, 1])-np.maximum(inner[:-1, 0], inner[1:, 0])
    if np.any(inner[:, 1] <= inner[:, 0]) or np.any(overlap < 0):
        raise ValueError('Finite corridor is disconnected or empty')
    return cells, inner, intervals, faces, inflated


def constrain_reference(basis, controls, nominal, cells, inner, intervals):
    """Impose all six faces on every Bezier control, including tightening."""
    w = cp.Variable(nominal.shape)
    bezier = controls @ w
    constraints = [basis[0] @ w == basis[0] @ nominal, basis[-1] @ w == basis[-1] @ nominal]
    for (first, last), outer, tightened in zip(intervals, cells, inner):
        curve = bezier[3*first:3*last]
        for lo, hi in [outer, tightened]:
            constraints.extend([curve >= np.tile(lo, (3*(last-first), 1)),
                                curve <= np.tile(hi, (3*(last-first), 1))])
    problem = cp.Problem(cp.Minimize(cp.sum_squares(basis @ (w-nominal))), constraints)
    problem.solve(solver='CLARABEL')
    if problem.status != 'optimal':
        raise RuntimeError(f'Finite corridor optimization: {problem.status}')
    return w.value


def joint_feasibility(env, obs, position, rotation):
    """Bounded IK in a separate MjData; never move the rollout robot here.

    Joint bounds apply to these discrete IK solutions. Continuous Cartesian
    corridor certification does not imply continuous joint or whole-arm safety.
    """
    import mujoco
    robot = env.robots[0]
    model = env.sim.model._model
    data = mujoco.MjData(model)
    data.qpos[:] = env.sim.data.qpos
    indices = robot._ref_joint_pos_indexes
    bounds = model.jnt_range[robot._ref_joint_indexes] + np.array([.02, -.02])
    site = robot.eef_site_id
    robot.controller.update(force=True)
    offset = Rotation.from_quat(obs['robot0_eef_quat']).inv() * Rotation.from_matrix(robot.controller.ee_ori_mat)
    targets = (Rotation.from_rotvec(rotation) * offset).as_matrix()
    previous = data.qpos[indices].copy()
    solutions, errors = [], []
    for point, target in zip(position, targets):
        def residual(q):
            data.qpos[indices] = q
            mujoco.mj_kinematics(model, data)
            angle = Rotation.from_matrix(target @ data.site_xmat[site].reshape(3, 3).T).as_rotvec()
            return np.r_[20*(data.site_xpos[site] - point), angle, .0001*(q - previous)]
        fit = least_squares(residual, previous, bounds=(bounds[:, 0], bounds[:, 1]),
                            max_nfev=60, ftol=1e-8, xtol=1e-8, gtol=1e-8)
        error = residual(fit.x)
        errors.append([np.linalg.norm(error[:3])/20, np.linalg.norm(error[3:6])])
        solutions.append(fit.x)
        previous = fit.x
    errors = np.array(errors)
    if errors[:, 0].max() > .002 or errors[:, 1].max() > .02:
        raise RuntimeError(f'Bounded IK inaccurate: {errors.max(axis=0)}')
    return np.array(solutions), errors


def pixels(points, camera):
    local = (points - camera['position']) @ camera['rotation']
    depth = -local[:, 2]
    return np.column_stack([camera['size']/2 + camera['focal']*local[:, 0]/depth,
                            camera['size']/2 - camera['focal']*local[:, 1]/depth])


def camera_info(env, size):
    index = env.sim.model.camera_name2id('agentview')
    return dict(position=env.sim.data.cam_xpos[index].copy(),
                rotation=env.sim.data.cam_xmat[index].reshape(3, 3).copy(), size=size,
                focal=.5*size/np.tan(np.deg2rad(env.sim.model.cam_fovy[index])/2))


def rollout(env, obs, position, rotation, grip, names, output, label):
    """Execute both references with the same original OSC, using absolute goals."""
    env.robots[0].controller.use_delta = False
    controller = env.robots[0].controller
    controller.update(force=True)
    hand_to_site = Rotation.from_quat(obs['robot0_eef_quat']).inv() * Rotation.from_matrix(controller.ee_ori_mat)
    site_rotation = (Rotation.from_rotvec(rotation) * hand_to_site).as_rotvec()
    model = env.sim.model
    joints = env.robots[0]._ref_joint_indexes
    limits = model.jnt_range[joints]
    initial_obstacles = np.array([obs[n + '_pos'] for n in names])
    rows, frames, contacts, obstacle_positions, obstacle_quaternions = [], [], [], [], []
    success = False
    for i in range(len(position) + 30):
        t = min(i, len(position) - 1)
        # The gripper integrates a binary command. Retiming the arm also needs
        # retiming these increments, otherwise it closes before reaching the grasp.
        grip_command = grip[t] if i % STRETCH == 0 else 0.
        action = np.r_[position[t], site_rotation[t], grip_command]
        obs, _, _, _ = env.step(action)
        success |= bool(env.check_success())
        pairs = []
        for contact in env.sim.data.contact[:env.sim.data.ncon]:
            pair = [model.geom_id2name(g) or '' for g in [contact.geom1, contact.geom2]]
            if any(any(p.startswith(n) for n in names) for p in pair) and any(
                    p.startswith(('robot0_', 'gripper0_', 'orange_juice_1')) for p in pair):
                pairs.append(pair)
        q = obs['robot0_joint_pos']
        joint_margin = np.minimum(q - limits[:, 0], limits[:, 1] - q).min()
        displacement = np.abs(np.array([obs[n + '_pos'] for n in names]) - initial_obstacles).sum(axis=1).max()
        rows.append(np.r_[obs['robot0_eef_pos'], obs['orange_juice_1_pos'], q,
                          joint_margin, displacement, bool(pairs), env.check_success()])
        contacts.append(pairs)
        obstacle_positions.append([obs[n + '_pos'] for n in names])
        obstacle_quaternions.append([obs[n + '_quat'] for n in names])
        frames.append(np.ascontiguousarray(obs['agentview_image'][::-1]))
    rows = np.asarray(rows)
    metrics = dict(task_success=success, final_success=bool(env.check_success()),
                   obstacle_contact_frames=int(rows[:, -2].sum()),
                   obstacle_displacement_max_m=float(rows[:, -3].max()),
                   official_displacement_collision=bool(rows[:, -3].max() > .001),
                   joint_margin_min_rad=float(rows[:, -4].min()),
                   tracking_error_max_m=float(np.linalg.norm(rows[:len(position), :3] - position, axis=1).max()))
    np.savez_compressed(output / f'{label}.npz', reference=position, rotation=rotation,
                        gripper=grip, rollout=rows, obstacle_positions=obstacle_positions,
                        obstacle_quaternions=obstacle_quaternions)
    (output / f'{label}_contacts.json').write_text(json.dumps(contacts))
    print(label, json.dumps(metrics), flush=True)
    return metrics, frames, rows


def comparison_video(frames, results, output, paths, camera):
    font_path = '/usr/share/fonts/truetype/croscore/Arimo-Bold.ttf'
    font = ImageFont.truetype(font_path, 21)
    small = ImageFont.truetype(font_path, 15)
    side = frames[0][0].shape[0]
    writer = imageio.get_writer(output / 'comparison.mp4', fps=20, codec='libx264',
                                quality=8, macro_block_size=1)
    selected = []
    colors = ['#6DD2D9', '#AD6AEA']
    screen_paths = [pixels(path[:, :3], camera) for path in paths]
    for i in range(max(map(len, frames))):
        canvas = Image.new('RGB', (side*2, side+130), 'white')
        draw = ImageDraw.Draw(canvas)
        for j, (sequence, title) in enumerate(zip(frames, ['Unconstrained spline', 'With safe corridor'])):
            frame = Image.fromarray(sequence[min(i, len(sequence)-1)])
            trail = ImageDraw.Draw(frame)
            if i > 1:
                trail.line([tuple(p) for p in screen_paths[j][:i+1]], fill=colors[j], width=3)
            canvas.paste(frame, (j*side, 56))
            draw.text((j*side+18, 16), title, fill='#333333', font=font)
            status = 'Orange juice placed' if paths[j][min(i, len(paths[j])-1), -1] else 'Executing'
            if i >= len(sequence)-1 and not paths[j][-1, -1]:
                status = 'Task not completed'
            if paths[j][:i+1, -2].any():
                status += '  |  Obstacle contact'
            draw.text((j*side+18, side+67), status, fill='#333333', font=small)
        draw.text((18, side+100), 'SafeLIBERO Level I  |  Public demonstration adaptation',
                  fill='#333333', font=small)
        writer.append_data(np.asarray(canvas))
        if i in [0, len(frames[0])//3, 2*len(frames[0])//3, len(frames[0])-1]:
            selected.append(canvas.copy())
    writer.close()
    selected[-1].save(output / 'comparison.png')
    for j, frame in enumerate(selected):
        frame.save(output / f'frame_{j}.png')


def path_figure(paths, boxes, inflated, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib import font_manager
    font_manager.fontManager.addfont('/usr/share/fonts/truetype/croscore/Arimo-Regular.ttf')
    font_manager.fontManager.addfont('/usr/share/fonts/truetype/croscore/Arimo-Bold.ttf')
    plt.rcParams.update({'font.family': 'Arimo', 'font.size': 11, 'pdf.fonttype': 42})
    fig, ax = plt.subplots(figsize=(5, 4))
    for lo, hi in boxes:
        ax.add_patch(Rectangle(lo[:2], *(hi-lo)[:2], facecolor='#999999', alpha=.7))
    # Show the two active corridor faces at approach height, without a workspace box.
    lo, hi = inflated[0]
    ax.plot([lo[0], hi[0], hi[0]], [hi[1], hi[1], lo[1]], '--', color='#8AC6ED', lw=2,
            label='Reference corridor')
    for path, name, color in zip(paths, ['Unconstrained', 'Constrained'], ['#6DD2D9', '#AD6AEA']):
        approach = path[:150]
        ax.plot(approach[:, 0], approach[:, 1], color=color, lw=2.5, label=name)
        ax.scatter(*approach[0, :2], s=18, color=color)
    ax.set_aspect('equal')
    ax.set_title('Approach to the orange juice — top view', weight='bold', pad=45)
    ax.axis('off')
    ax.legend(frameon=False, loc='lower center', bbox_to_anchor=(.5, 1.01), ncol=2,
              fontsize=9)
    ax.text(*boxes[0].mean(axis=0)[:2], 'Wine\nbottle', ha='center', va='center', fontsize=10)
    fig.tight_layout()
    for suffix in ['png', 'pdf', 'svg']:
        fig.savefig(output / f'paths.{suffix}', dpi=250, bbox_inches='tight')
    plt.close(fig)


def corridor_cells(plan):
    """Finite, overlapping portions of the saved halfspaces, for display only."""
    if 'corridor_bounds' in plan:
        return plan['corridor_bounds'], plan['corridor_intervals']
    controls = (plan['control_map'] @ plan['constrained']).reshape(-1, 3, 3)
    faces = plan['faces']
    cells, intervals = [], []
    first = 0
    while first < len(controls):
        restrictions = faces[faces[:, 0] == first, 1:]
        last = first + 1
        while last < min(first + 4, len(controls)):
            if not np.array_equal(faces[faces[:, 0] == last, 1:], restrictions):
                break
            last += 1
        points = controls[first:last].reshape(-1, 3)
        lo, hi = points.min(axis=0)-[.035, .035, .028], points.max(axis=0)+[.035, .035, .028]
        for _, axis, side, bound in restrictions:
            axis = int(axis)
            if side > 0:
                lo[axis] = max(lo[axis], bound)
            else:
                hi[axis] = min(hi[axis], bound)
        assert np.all(points >= lo-1e-9) and np.all(points <= hi+1e-9)
        cells.append([lo, hi])
        intervals.append([first, last])
        first = last
    return np.array(cells), np.array(intervals)


def corridor_surface(cells):
    """Outer quadrilaterals of the cell union, without internal overlapping walls."""
    axes = [np.unique(cells[:, :, i]) for i in range(3)]
    centers = [(a[:-1]+a[1:])/2 for a in axes]
    grid = np.stack(np.meshgrid(*centers, indexing='ij'), axis=-1)
    occupied = np.zeros(grid.shape[:-1], dtype=bool)
    for lo, hi in cells:
        occupied |= ((grid >= lo) & (grid <= hi)).all(axis=-1)
    polygons = []
    for axis in range(3):
        remaining = [i for i in range(3) if i != axis]
        boundary = np.diff(np.pad(occupied.astype(int), [(int(i == axis),)*2 for i in range(3)]), axis=axis)
        for k in range(len(axes[axis])):
            plane = np.take(boundary, k, axis=axis)
            for row in range(plane.shape[0]):
                col = 0
                while col < plane.shape[1]:
                    sign = plane[row, col]
                    if not sign:
                        col += 1
                        continue
                    end = col+1
                    while end < plane.shape[1] and plane[row, end] == sign:
                        end += 1
                    a, b = remaining
                    quad = np.zeros((4, 3))
                    quad[:, axis] = axes[axis][k]
                    quad[:, a] = axes[a][[row, row, row+1, row+1]]
                    quad[:, b] = axes[b][[col, end, end, col]]
                    polygons.append(quad)
                    col = end
    return polygons


def phase_colors(count):
    colors = np.array([[int(c[i:i+2], 16) for i in [1, 3, 5]] for c in REFERENCE_COLORS])
    return np.column_stack([np.interp(np.linspace(0, 3, count), np.arange(4), colors[:, i])
                            for i in range(3)]).round().astype(np.uint8)


def saved_camera(run):
    if (run / 'camera.npz').exists():
        return dict(np.load(run / 'camera.npz'))
    # Fixed agentview in the pinned SafeLIBERO floor scene, for legacy recordings.
    q = np.array([.6182166934013367, .3432307541370392,
                  .3432314395904541, .6182177066802979])
    return dict(position=np.array([.8965773716836134, 5.216182733499864e-7, .65]),
                rotation=Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix(),
                size=512, focal=256/np.tan(np.deg2rad(45)/2))


def corridor_view(run, output, label='constrained', paper=False):
    """Render a volumetric corridor from saved geometry; never rerun physics."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
    from mpl_toolkits.mplot3d import proj3d

    plan = np.load(run / 'planning.npz')
    result = np.load(run / f'{label}.npz')
    reference, actual = result['reference'], result['rollout'][:, :3]
    cells, intervals = corridor_cells(plan)
    finite = 'corridor_bounds' in plan
    surface = corridor_surface(cells) if label == 'constrained' else []
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / 'display_geometry.npz', cells=cells, segment_intervals=intervals)
    camera = saved_camera(run)
    colors = phase_colors(len(reference)-1)
    font_path = '/usr/share/fonts/truetype/croscore/Arimo-Regular.ttf'
    font_manager.fontManager.addfont(font_path)
    font_manager.fontManager.addfont('/usr/share/fonts/truetype/croscore/Arimo-Bold.ttf')
    plt.rcParams.update({'font.family': 'Arimo', 'font.size': 12})
    background_color = 'white' if paper else '#F6F9FB'
    fig = plt.figure(figsize=(7, 6.6), dpi=150, facecolor=background_color)
    ax = fig.add_axes([-.02, -.04, 1.04, 1.09], projection='3d', facecolor=background_color)
    lo, hi = cells.min(axis=(0, 1)), cells.max(axis=(0, 1))
    floor = .06
    if not paper:
        for x in np.arange(-.25, .26, .05):
            ax.plot([x, x], [-.24, .30], [floor, floor], color='#DFE6EB', lw=.55)
        for y in np.arange(-.24, .31, .05):
            ax.plot([-.25, .25], [y, y], [floor, floor], color='#DFE6EB', lw=.55)
        ax.plot(reference[:, 0], reference[:, 1], np.full(len(reference), floor),
                color='#BCCBD1', lw=2, alpha=.6)
    for box in plan['boxes']:
        ax.add_collection3d(Poly3DCollection(corridor_surface(box[None]),
                            facecolor='#CC9B65', edgecolor='#98714D', linewidth=.5, alpha=.55))
        center = box.mean(axis=0)
        ax.text(center[0], center[1], box[1, 2]+.025, 'Wine bottle',
                ha='center', color='#333333' if paper else '#775334',
                fontsize=20 if paper else 11, weight='bold' if paper else 'normal')
    ax.add_collection3d(Poly3DCollection(surface, facecolor='#4BC3CB',
                                        edgecolor='none', alpha=.12, zsort='average'))
    # Thin cell frames make depth and the changing corridor cross-section legible.
    edges = [(a, a ^ (1 << d)) for a in range(8) for d in range(3) if not a & (1 << d)]
    bits = np.array([[bool(n & (1 << d)) for d in range(3)] for n in range(8)])
    for a, b in cells if surface else []:
        vertices = np.where(bits, b, a)
        for j, k in edges:
            ax.plot(*vertices[[j, k]].T, color='#168B9C', lw=.65, alpha=.35)
    if paper:
        ax.add_collection3d(Line3DCollection(np.stack([reference[:-1], reference[1:]], axis=1),
                                           colors=colors/255, linewidth=8))
    else:
        ax.plot(*reference.T, color='#176D85', lw=2.8)
    ax.scatter(*reference[0], color=REFERENCE_COLORS[0] if paper else '#176D85', s=24, depthshade=False)
    ax.scatter(*reference[-1], color=REFERENCE_COLORS[-1] if paper else '#176D85', s=28, marker='s', depthshade=False)
    ax.set(xlim=(lo[0]-.045, hi[0]+.025), ylim=(lo[1]-.035, hi[1]+.025),
           zlim=(floor, hi[2]+.045))
    ax.set_box_aspect((hi-lo)*[1, 1, 1.15])
    ax.view_init(elev=27, azim=-53)
    ax.set_axis_off()
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    background = Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert('RGB')

    def project(points):
        x, y, _ = proj3d.proj_transform(*points.T, ax.get_proj())
        p = ax.transData.transform(np.c_[x, y])
        return np.c_[p[:, 0], height-p[:, 1]]

    projected = project(actual)
    background.save(output / 'corridor_geometry.png')
    fig.savefig(output / 'corridor_geometry.pdf', facecolor=fig.get_facecolor())
    plt.close(fig)
    font = ImageFont.truetype(font_path, 26)
    small = ImageFont.truetype(font_path, 18)
    side = 640
    surface_order = sorted(surface, key=lambda q: ((q.mean(axis=0)-camera['position']) @ camera['rotation'])[2])
    shell = Image.new('RGBA', (512, 512))
    for quad in surface_order:
        layer = Image.new('RGBA', shell.size)
        ImageDraw.Draw(layer).polygon([tuple(p) for p in pixels(quad, camera)], fill=(41, 191, 204, 17))
        shell = Image.alpha_composite(shell, layer)
    draw = ImageDraw.Draw(shell)
    for a, b in cells if surface else []:
        vertices = np.where(bits, b, a)
        for j, k in edges:
            draw.line([tuple(p) for p in pixels(vertices[[j, k]], camera)], fill=(23, 132, 155, 115), width=1)
    screen_reference = pixels(reference, camera)
    screen_actual = pixels(actual, camera)
    brighten = np.round(255*(np.arange(256)/255)**.90).astype(np.uint8).tolist()*3
    reader = imageio.get_reader(run / 'comparison.mp4')
    writer = imageio.get_writer(output / 'corridor.mp4', fps=20, codec='libx264', quality=8, macro_block_size=1)
    for i, recorded in enumerate(reader):
        column = 512 if label == 'constrained' else 0
        scene = Image.fromarray(recorded[56:568, column:column+512])
        if paper:
            scene = scene.point(brighten)
        scene = scene.convert('RGBA')
        scene = Image.alpha_composite(scene, shell).convert('RGB')
        draw = ImageDraw.Draw(scene)
        if paper:
            for a, b, color in zip(screen_reference[:-1], screen_reference[1:], colors):
                draw.line([tuple(a), tuple(b)], fill=tuple(color), width=8)
        else:
            draw.line([tuple(p) for p in screen_reference], fill='#FFFFFF', width=2)
        if (paper or label == 'nominal') and i:
            draw.line([tuple(p) for p in screen_actual[:i+1]], fill='#AD6AEA', width=4 if paper else 3)
        view = background.copy()
        draw = ImageDraw.Draw(view)
        if i:
            draw.line([tuple(p) for p in projected[:i+1]], fill='#AD6AEA', width=6 if paper else 5)
        x, y = projected[i]
        draw.ellipse((x-7, y-7, x+7, y+7), fill='#AD6AEA', outline='white', width=2)
        canvas = Image.new('RGB', (1280, 780), '#FFFFFF')
        canvas.paste(scene.resize((side, side), Image.Resampling.LANCZOS), (0, 64))
        canvas.paste(view.resize((side, side), Image.Resampling.LANCZOS), (side, 64))
        draw = ImageDraw.Draw(canvas)
        draw.text((24, 19), 'Safe corridor in the scene' if surface else 'Without safe corridor', font=font, fill='#263C4B')
        draw.text((664, 19), '3D corridor · spatial view' if surface else 'Unconstrained spline · spatial view', font=font, fill='#263C4B')
        legend = [(310, '#176D85', 'Reference (white in scene)'), (760, '#AE58D5', 'Executed trajectory')]
        if surface:
            legend.insert(0, (24, '#229CAC', 'Corridor volume'))
        for x, color, caption in legend:
            draw.line((x, 726, x+28, 726), fill=color, width=4)
            draw.text((x+40, 714), caption, font=small, fill='#263C4B')
        footer = (('Saved finite corridor: every boundary was imposed during optimization.' if finite else
                   'Finite view of saved halfspaces; outer display bounds were not imposed.')
                  if surface else 'Unconstrained reference and recorded execution; obstacle envelope shown at its initial position.')
        draw.text((24, 750), footer,
                  font=small, fill='#637782')
        draw.text((1170, 716), f'{i/20:.1f} s', font=small, fill='#263C4B')
        writer.append_data(np.asarray(canvas))
        if i in [65, 120, 260, len(actual)-1]:
            canvas.save(output / f'corridor_{i:03d}.png')
    reader.close()
    writer.close()
    (output / 'provenance.json').write_text(json.dumps(dict(
        source_sha256={name: hashlib.sha256((run/name).read_bytes()).hexdigest()
                       for name in ['comparison.mp4', 'planning.npz', f'{label}.npz']},
        display=('Exact pre-optimization corridor bounds; all six faces constrained, with inner tracking reserve.'
                 if finite else 'Legacy finite display crops of halfspace constraints.'),
        physics_rerun=False), indent=2))


def outcome_callout(scene, center, text, font, radii=(62, 53)):
    """PPT-style white ring, colored glow and white-outlined black label."""
    x, y = center
    rx, ry = radii
    bounds = (x-rx, y-ry, x+rx, y+ry)
    tint = (146, 208, 80, 200) if text in ['Success', 'No collision'] else (255, 50, 65, 175)
    glow = Image.new('RGBA', scene.size)
    ImageDraw.Draw(glow).ellipse(bounds, outline=tint, width=14)
    scene = Image.alpha_composite(scene.convert('RGBA'), glow.filter(ImageFilter.GaussianBlur(5)))
    draw = ImageDraw.Draw(scene)
    draw.ellipse(bounds, outline='white', width=5)
    left = x > scene.width/2
    tx = x-rx-17 if left else x+rx+17
    draw.text((tx, y), text, font=font, fill='black', anchor='rm' if left else 'lm',
              stroke_width=3, stroke_fill='white')
    return scene.convert('RGB')


def corridor_comparison(run, output):
    """Paper-style paired views; experiment data and approved renders stay intact."""
    labels = ['nominal', 'constrained']
    videos = []
    for label in labels:
        folder = output / label
        print(f'Rendering {label} view', flush=True)
        corridor_view(run, folder, label=label, paper=True)
        videos.append(folder / 'corridor.mp4')
    records = [np.load(run / f'{name}.npz') for name in labels]
    paths = [record['rollout'] for record in records]
    camera = saved_camera(run)
    objects = [pixels(rows[:, 3:6], camera)*1.25 for rows in paths]
    box = np.load(run / 'planning.npz')['boxes'][0]
    bottle_center = pixels(box.mean(axis=0)[None], camera)[0]*1.25
    # Image-space bottle annotations from the archived video, not measured poses.
    # After tipping, the visible bottle settles at the left edge of the image.
    bottle_marks = np.array([[24, 220, 225, 45, 95], [29, 220, 225, 45, 95],
                             [39, 217, 244, 60, 90], [49, 190, 325, 65, 42],
                             [69, 92, 382, 87, 38], [407, 92, 382, 87, 38]])
    measured_bottles = 'obstacle_positions' in records[0]
    if measured_bottles:
        scene = json.loads((run / 'scene.json').read_text())
        name = json.loads((run / 'metrics.json').read_text())['obstacle_names'][0]
        bits = np.array([[bool(n & (1 << d)) for d in range(3)] for n in range(8)])
        corners = np.where(bits, box[1], box[0])
        local = (corners-np.array(scene[name+'_pos'])) @ Rotation.from_quat(scene[name+'_quat']).as_matrix()
        bottle_boxes = []
        for record in records:
            rotations = Rotation.from_quat(record['obstacle_quaternions'][:, 0]).as_matrix()
            world = np.einsum('tij,kj->tki', rotations, local)+record['obstacle_positions'][:, :1]
            screen = pixels(world.reshape(-1, 3), camera).reshape(-1, 8, 2)*1.25
            lower, upper = screen.min(axis=1), screen.max(axis=1)
            bottle_boxes.append(np.c_[(lower+upper)/2, (upper-lower)/2+10])
    readers = [imageio.get_reader(path) for path in videos]
    writer = imageio.get_writer(output / 'comparison.mp4', fps=20, codec='libx264', quality=8, macro_block_size=1)
    font_path = '/usr/share/fonts/truetype/croscore/Arimo-Bold.ttf'
    title = ImageFont.truetype(font_path, 42)
    heading = ImageFont.truetype(font_path, 34)
    note = ImageFont.truetype(font_path, 32)
    contact = np.flatnonzero(paths[0][:, -2])
    selected = {int(contact[0])+5, 260, len(paths[0])-1} if len(contact) else {65, 260, len(paths[0])-1}
    for i, frames in enumerate(zip(*readers)):
        canvas = Image.new('RGB', (1352, 1380), 'white')
        draw = ImageDraw.Draw(canvas)
        draw.text((24, 72), 'A  Task execution', font=heading, fill='#000000')
        draw.text((24, 661), 'B  3D trajectories', font=heading, fill='#000000')
        draw.line((24, 641, 1328, 641), fill='#D0CECE', width=4)
        for j, (frame, name, rows) in enumerate(zip(frames, ['Without safe corridor', 'With safe corridor'], paths)):
            x = 24 + j*664
            image = Image.fromarray(frame)
            scene = image.crop((0, 64, 640, 704))
            collided = bool(rows[:i+1, -2].any())
            if measured_bottles:
                marker = bottle_boxes[j][i]
                scene = outcome_callout(scene, marker[:2], 'Collision' if collided else 'No collision', heading, marker[2:])
            elif collided:
                marker = [np.interp(i, bottle_marks[:, 0], bottle_marks[:, k]) for k in range(1, 5)]
                scene = outcome_callout(scene, marker[:2], 'Collision', heading, marker[2:])
            else:
                scene = outcome_callout(scene, bottle_center, 'No collision', heading, (45, 95))
            if rows[i, -1]:
                scene = outcome_callout(scene, objects[j][i], 'Success', heading)
            elif i == len(rows)-1:
                scene = outcome_callout(scene, objects[j][i], 'Failure', heading)
            # Identical crops remove unused margins without changing spatial scale.
            spatial = image.crop((745, 195, 1205, 630)).resize((640, 605), Image.Resampling.LANCZOS)
            canvas.paste(scene.crop((0, 50, 640, 550)), (x, 117))
            canvas.paste(spatial, (x, 708))
            draw.text((x+320, 31), name, font=title, fill='#000000', anchor='mm')
            draw.rectangle((x, 117, x+639, 617), outline='#D0CECE', width=2)
        for x, color, name in [(65, '#229CAC', 'Safe corridor'), (470, '#176D85', 'Reference'), (865, '#AD6AEA', 'Execution')]:
            if name == 'Reference':
                for offset, rgb in enumerate(phase_colors(46)):
                    draw.line((x+offset, 1338, x+offset, 1350), fill=tuple(rgb), width=1)
            else:
                draw.line((x, 1344, x+45, 1344), fill=color, width=10)
            draw.text((x+61, 1342), name, font=note, fill='#000000', anchor='lm')
        writer.append_data(np.asarray(canvas))
        if i in selected:
            canvas.save(output / f'comparison_{i:03d}.png', dpi=(300, 300))
    for reader in readers:
        reader.close()
    writer.close()
    (output / 'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (output / 'provenance.json').write_text(json.dumps(dict(
        source_sha256={name: hashlib.sha256((run/name).read_bytes()).hexdigest()
                       for name in ['comparison.mp4', 'planning.npz', 'nominal.npz', 'constrained.npz', 'source_snapshot.py']},
        physics_rerun=False, frames=i+1, fps=20,
        comparison='Archived paired runs; common camera, crop, scale and time axis. Approved corridor_3d files preserved.',
        style='FIGURE_STYLE.md; paper figures 8/9; generative_fig.pptx slides 9, 22, 27. Arimo Bold substitutes Myriad.',
        appearance=dict(reference_colors=REFERENCE_COLORS, execution_color='#AD6AEA',
                        scene_gamma=.90, trajectory_width_multiplier=2,
                        bottle_annotations='Recorded 3D poses of initial collision envelope' if measured_bottles else bottle_marks.tolist(),
                        callouts='Task outcome and cumulative recorded collision indicated separately; '
                                 'Measured poses are used when saved; legacy recordings use image annotations.'),
        display=('Exact pre-optimization finite bounds, imposed on all Bezier controls; initial obstacle envelope.'
                 if 'corridor_bounds' in np.load(run / 'planning.npz') else 'Legacy halfspace display crops.')),  indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=RELEASE / 'data_local/safelibero')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/safelibero_finite3d')
    parser.add_argument('--episode', type=int, default=0)
    parser.add_argument('--demo', type=int, default=0)
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--fetch', action='store_true', help='Download pinned public assets and demonstration')
    parser.add_argument('--padding', type=float, default=.08)
    parser.add_argument('--visualize', type=Path, help='Overlay the corridor on an existing recording')
    parser.add_argument('--compare', type=Path, help='Compare archived runs, preserving the approved 3D corridor render')
    args = parser.parse_args()
    if args.compare:
        corridor_comparison(args.compare, args.output)
        return
    if args.visualize:
        corridor_view(args.visualize, args.output)
        return
    if args.fetch:
        fetch(args.source)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    env, obs = make_env(args.source.resolve(), args.episode)
    imageio.imwrite(args.output / 'scene.png', obs['agentview_image'][::-1])
    objects = {k: np.asarray(v).tolist() for k, v in obs.items()
               if k.endswith('_pos') or k.endswith('_quat')}
    (args.output / 'scene.json').write_text(json.dumps(objects, indent=2))
    print(f'SafeLIBERO Level I, episode {args.episode}, demonstration {args.demo}', flush=True)
    if args.preview:
        env.close()
        return
    demo = demonstration(args.source / 'orange_juice_demo.hdf5', args.demo)
    position, rotation, grip = reference(demo, obs)
    boxes, names = obstacle_bounds(env, obs)
    basis, controls, nominal = fit_reference(position)
    cells, inner, intervals, faces, inflated = build_corridor(controls, nominal, boxes, args.padding)
    definition = args.output / 'corridor_definition.npz'
    np.savez_compressed(definition, bounds=cells, tightened_bounds=inner,
                        segment_intervals=intervals, obstacle_faces=faces, inflated_obstacles=inflated)
    definition_sha = hashlib.sha256(definition.read_bytes()).hexdigest()
    constrained = constrain_reference(basis, controls, nominal, cells, inner, intervals)
    np.savez_compressed(args.output / 'planning.npz', basis=basis, control_map=controls,
                        nominal=nominal, constrained=constrained, faces=faces,
                        boxes=boxes, inflated=inflated, adapted_demo=position,
                        corridor_bounds=cells, tightened_bounds=inner, corridor_intervals=intervals)
    camera = camera_info(env, 512)
    np.savez(args.output / 'camera.npz', **camera)
    ik, ik_error = joint_feasibility(env, obs, basis @ constrained, rotation)
    np.savez_compressed(args.output / 'joint_feasibility.npz', q=ik, errors=ik_error)
    results, movies, paths = {}, [], []
    for label, weights in [('nominal', nominal), ('constrained', constrained)]:
        if label != 'nominal':
            env, obs = make_env(args.source.resolve(), args.episode)
        results[label], frames, rows = rollout(env, obs, basis @ weights, rotation, grip,
                                             names, args.output, label)
        movies.append(frames)
        paths.append(rows)
        env.close()
    results.update(episode=args.episode, demo=args.demo, upstream_commit=UPSTREAM,
                   corridor_definition_sha256=definition_sha, obstacle_names=names,
                   corridor='Finite bounded 3D cells, saved before optimization; all six faces imposed on Bezier controls',
                   padding_m=args.padding, retiming_factor=STRETCH,
                   bounded_ik_max_errors=ik_error.max(axis=0).tolist(),
                   source='LIBERO public demonstration; no learned policy')
    results['provenance'] = dict(demo_url=DEMO_URL, demo_sha256=hashlib.sha256(
        (args.source / 'orange_juice_demo.hdf5').read_bytes()).hexdigest(),
        entry_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output / 'metrics.json').write_text(json.dumps(results, indent=2))
    comparison_video(movies, results, args.output, paths, camera)
    print('Paired physical rollouts complete; raw data saved.', flush=True)


if __name__ == '__main__':
    main()
