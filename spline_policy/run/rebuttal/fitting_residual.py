"""CPU-only Table III representation audit: fit every complete 16-action window.

Six quadratic segments / eight coefficients per coordinate, free endpoints.
No policy inference, training, padding, resampling, or episode-boundary crossing.
"""
import argparse
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import urllib.request
import zlib

import h5py
import numpy as np
from scipy.spatial.transform import Rotation
import torch
import zarr

ROOT = Path(__file__).resolve().parents[2]
DP3 = ROOT.parents[1] / 'others/3D-Diffusion-Policy/3D-Diffusion-Policy'
DECODER = ROOT / 'policy/diffusion_policy/planning/quadratic_spline.py'
URL = 'https://diffusion-policy.cs.columbia.edu/data/training/robomimic_lowdim.zip'
TASKS = ('transport', 'can', 'pusht', 'adroit_door', 'adroit_pen', 'dexart_laptop')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_range(start, end=''):
    request = urllib.request.Request(URL, headers={'Range': f'bytes={start}-{end}'})
    with urllib.request.urlopen(request, timeout=120) as response:
        if response.status != 206:
            raise RuntimeError('Server did not honor Range; refusing full-archive download')
        return response.read()


def fetch_actions(path):
    """Read two ZIP members in RAM; retain only actions, boundaries, provenance."""
    if path.exists():
        return
    tail = read_range('', 65536)
    position = tail.find(b'PK\x01\x02')
    arrays, provenance = {}, {}
    while position >= 0 and tail[position:position + 4] == b'PK\x01\x02':
        entry = struct.unpack_from('<4s6H3L5H2L', tail, position)
        n, extra, comment = entry[10:13]
        name = tail[position + 46:position + 46 + n].decode()
        position += 46 + n + extra + comment
        if name not in [f'robomimic/datasets/{t}/ph/low_dim_abs.hdf5' for t in TASKS[:2]]:
            continue
        size, raw_size, offset = entry[8], entry[9], entry[-1]
        data = read_range(offset, offset + size + 1023)
        header = struct.unpack_from('<4s5H3L2H', data)
        start = 30 + header[-2] + header[-1]
        raw = zlib.decompress(data[start:start + size], -15)
        assert len(raw) == raw_size and zlib.crc32(raw) == entry[7]
        with h5py.File(io.BytesIO(raw), 'r') as source:
            names = sorted(source['data'], key=lambda k: int(k.split('_')[-1]))
            episodes = [source['data'][k]['actions'][:] for k in names]
        task = name.split('/')[2]
        arrays[task + '_actions'] = np.concatenate(episodes)
        arrays[task + '_episode_ends'] = np.cumsum([len(a) for a in episodes])
        provenance[task] = dict(url=URL, member=name, zip_crc32=entry[7],
                               source_hdf5_sha256=sha(raw))
        print('Extracted', task, flush=True)
    assert set(provenance) == set(TASKS[:2])
    arrays['provenance'] = np.asarray(json.dumps(provenance))
    np.savez_compressed(path, **arrays)


def decoder_class(path):
    spec = importlib.util.spec_from_file_location('fitting_decoder', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.QuadraticSpline


def load_actions(task, archive):
    if task in TASKS[:2]:
        with np.load(archive, allow_pickle=False) as source:
            raw = source[task + '_actions']
            ends = source[task + '_episode_ends']
            provenance = json.loads(str(source['provenance']))[task]
        # Match the benchmark's absolute-position + 6D-rotation + gripper actions.
        arms = raw.astype(np.float32).reshape(len(raw), -1, 7)
        rotation = Rotation.from_rotvec(arms[..., 3:6].reshape(-1, 3)).as_matrix()
        rotation = rotation[:, :2, :].reshape(len(raw), -1, 6)
        actions = np.concatenate([arms[..., :3], rotation, arms[..., 6:]], axis=-1)
        actions = actions.reshape(len(raw), -1).astype(np.float32)
        path = archive
    else:
        path = (ROOT / 'data/pusht/pusht_cchi_v7_replay.zarr' if task == 'pusht'
                else DP3 / f'data/{task}_expert.zarr')
        source = zarr.open(str(path), mode='r')
        raw = source['data/action'][:]
        ends = source['meta/episode_ends'][:]
        actions, provenance = raw.astype(np.float32), {}
    assert ends[-1] == len(actions) and np.all(np.diff(np.r_[0, ends]) > 0)
    assert np.isfinite(actions).all()
    provenance.update(path=str(path), raw_action_sha256=sha(raw.tobytes()),
                      episode_ends_sha256=sha(ends.tobytes()), action_shape=list(actions.shape),
                      raw_dtype=str(raw.dtype), raw_shape=list(raw.shape))
    return actions.astype(np.float64), ends, provenance



def plot_examples(output, audit, records, spline, phi, inverse):
    """Show task residuals and low/high Adroit Door fits in the same action coordinate."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D

    for filename in ('Arimo-Regular.ttf', 'Arimo-Bold.ttf'):
        font_manager.fontManager.addfont('/usr/share/fonts/truetype/croscore/' + filename)
    plt.rcParams.update({'font.family': 'Arimo', 'font.size': 10, 'mathtext.fontset': 'stix',
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    task = 'adroit_door'
    actions, ends, _ = load_actions(task, ROOT / 'data/m3_robomimic_actions.npz')
    starts = np.r_[0, ends[:-1]]
    offset = np.asarray(audit['sources'][task]['offset'])
    scale = np.asarray(audit['sources'][task]['range'])
    candidates = sorted((row for row in records if row['task'] == task),
                        key=lambda row: (row['rmse_pct'], row['episode'], row['start']))

    def normalized_window(row):
        i = starts[row['episode']] + row['start']
        return (actions[i:i + 16] - offset) / scale

    worst = candidates[-1]
    normalized = normalized_window(worst)
    residual = phi @ inverse @ normalized - normalized
    coordinate = int(np.mean(residual**2, axis=0).argmax())
    # Select a moving low-error case in the same coordinate, not a stationary fit.
    moving = [row for row in candidates
              if np.ptp(normalized_window(row)[:, coordinate]) >= .5]
    best = moving[0]
    dense = spline.computePsiList1D(torch.linspace(0, 1, 301))[2].numpy().astype(np.float64)
    examples = []
    for row in (best, worst):
        normalized = normalized_window(row)
        weights = inverse @ normalized
        residual = phi @ weights - normalized
        error = 100 * np.sqrt(np.mean(residual**2))
        np.testing.assert_allclose(error, row['rmse_pct'], atol=1e-12)
        examples.append(dict(**row, normalized_samples=normalized.tolist(),
                             normalized_curve=(dense @ weights).tolist(),
                             normalized_weights=weights.tolist(),
                             displayed_coordinate=coordinate,
                             coordinate_rmse_pct=float(100 * np.sqrt(np.mean(residual[:, coordinate]**2)))))
    fig = plt.figure(figsize=(7.2, 2.6))
    summary_ax = fig.add_axes([.13, .21, .43, .58])
    names = ('Transport', 'Can', 'Push-T', 'Adroit Door', 'Adroit Pen', 'DexArt Laptop')
    means = [audit['summary'][t]['mean_episode_rmse_pct'] for t in TASKS]
    sds = [audit['summary'][t]['sample_sd_episode_rmse_pct'] for t in TASKS]
    p95 = [audit['summary'][t]['p95_window_rmse_pct'] for t in TASKS]
    y = np.arange(len(TASKS))
    summary_ax.set_facecolor('#EDF5F6')
    summary_ax.errorbar(means, y - .21, xerr=sds, fmt='o', color='#AD6AEA', ms=6,
                        elinewidth=1.9, capsize=3.2, label='Mean ± SD', zorder=3)
    summary_ax.scatter(p95, y + .21, marker='d', color='#537F89', s=30, label='P95', zorder=3)
    summary_ax.set(yticks=y, yticklabels=names, xlim=(0, 11.3), ylim=(5.65, -.65),
                   xticks=[0, 5, 10])
    summary_ax.tick_params(axis='both', labelsize=9, length=0, pad=4)
    summary_ax.spines[['top', 'right', 'left']].set_visible(False)
    summary_ax.spines['bottom'].set_color('#AAAAAA')
    summary_ax.grid(axis='x', color='#E7E7E7', lw=.6)
    summary_ax.set_axisbelow(True)
    summary_ax.set_xlabel('RMSE (% of observed action range)', fontsize=9, labelpad=5)
    fig.text(.13, .975, 'A  Fitting error', ha='left', va='top', fontsize=10, weight='bold')
    fig.text(.615, .975, 'B  Adroit Door fits', ha='left', va='top', fontsize=10, weight='bold')
    fig.add_artist(Line2D([.587, .587], [.16, .95], transform=fig.transFigure,
                         color='#B3ABBF', lw=2, linestyle=(0, (5, 4))))
    summary_handles, summary_labels = summary_ax.get_legend_handles_labels()
    order = [summary_labels.index(label) for label in ('Mean ± SD', 'P95')]
    summary_ax.legend([summary_handles[i] for i in order], ['Mean ± SD', 'P95'],
                      loc='lower center', bbox_to_anchor=(.52, 1.01), ncol=2,
                      frameon=False, fontsize=8, handletextpad=.4, columnspacing=.9)
    axes = [fig.add_axes([.675, .24, .135, .48]),
            fig.add_axes([.855, .24, .135, .48])]
    shown = [np.asarray(e[k])[:, coordinate] * 100 for e in examples
             for k in ('normalized_samples', 'normalized_curve')]
    lower, upper = min(x.min() for x in shown), max(x.max() for x in shown)
    padding = .08 * (upper - lower)
    for index, (ax, example) in enumerate(zip(axes, examples)):
        samples = np.asarray(example['normalized_samples'])[:, coordinate] * 100
        curve = np.asarray(example['normalized_curve'])[:, coordinate] * 100
        ax.plot(np.arange(16), samples, '-o', color='#AD92E3', lw=2.4, ms=3.4,
                markeredgewidth=0, zorder=2)
        ax.plot(np.linspace(0, 15, 301), curve, color='#6DD2D9', lw=1.8, zorder=3)
        ax.set(xlim=(0, 15), ylim=(lower - padding, upper + padding),
               xticks=[0, 15], yticks=[0, 100])
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_color('#AAAAAA')
        ax.tick_params(labelsize=8, length=2)
        ax.set_xlabel('Step', fontsize=8, labelpad=2)
        if index == 0:
            ax.set_ylabel('Action range (%)', fontsize=8, labelpad=2)
        else:
            ax.tick_params(labelleft=False)
    handles = [Line2D([], [], color='#AD92E3', marker='o', lw=2.6, ms=4),
               Line2D([], [], color='#6DD2D9', lw=1.8)]
    fig.legend(handles, ['Demonstration', 'Spline fit'], loc='upper center', ncol=2,
               bbox_to_anchor=(.805, .90), frameon=False, handletextpad=.4,
               columnspacing=.8, prop={'size': 8.5, 'weight': 'bold'})
    for ext in ('pdf', 'png', 'svg'):
        fig.savefig(output / ('fitting_examples.' + ext), dpi=300, bbox_inches='tight', pad_inches=.03)
    plt.close(fig)
    audit['examples'] = dict(task=task, eligible_windows=len(candidates),
        selection='Worst full 28D window RMSE; display its largest-residual coordinate. Low-error case minimizes full-window RMSE among windows spanning >=50% of the observed range in that same coordinate. Illustrative selected cases, not policy outcomes.',
        moving_low_error_candidates=len(moving), displayed_coordinate=coordinate,
        cases=examples)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/m3_fitting')
    parser.add_argument('--fetch-robomimic', action='store_true')
    parser.add_argument('--plot-examples', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    archive = ROOT / 'data/m3_robomimic_actions.npz'
    if args.fetch_robomimic:
        fetch_actions(archive)
    spline_class = decoder_class(DECODER)
    spline = spline_class(nbSeg=6, nbDim=1, device='cpu')
    times = torch.linspace(0, 1, 16)
    phi = spline.computePsiList1D(times)[2].numpy().astype(np.float64)
    other_path = DP3 / 'diffusion_policy_3d/planning/quadratic_spline.py'
    other = decoder_class(other_path)(nbSeg=6, nbDim=1, device='cpu')
    np.testing.assert_array_equal(phi, other.computePsiList1D(times)[2].numpy())
    inverse = np.linalg.pinv(phi)
    projection = phi @ inverse
    assert np.linalg.matrix_rank(phi) == 8
    np.testing.assert_allclose(projection @ projection, projection, atol=1e-12)
    np.testing.assert_allclose(projection, projection.T, atol=1e-12)
    np.testing.assert_allclose(projection @ np.ones(16), 1, atol=2e-6)
    # A known in-span trajectory reconstructs; the basis does not force terminal rest.
    np.testing.assert_allclose(projection @ phi, phi, atol=1e-12)
    assert torch.linalg.norm(spline.C[-1] - spline.C[-2]).item() > 0
    args.output.mkdir(parents=True, exist_ok=True)
    records, summaries, sources = [], {}, {}
    for task in TASKS:
        actions, ends, source = load_actions(task, archive)
        offset = actions.min(axis=0)
        scale = np.ptp(actions, axis=0)
        scale[scale < 1e-12] = 1  # Constant coordinates reconstruct exactly.
        actions = (actions - offset) / scale
        episode_errors, begin, excluded = [], 0, []
        max_orthogonality, max_decode_difference = 0., 0.
        task_errors = []
        for episode, end in enumerate(ends):
            data = actions[begin:end]
            begin = end
            if len(data) < 16:
                excluded.append(episode)
                continue
            windows = np.lib.stride_tricks.sliding_window_view(data, 16, axis=0).transpose(0, 2, 1)
            fitted = np.einsum('ij,bjd->bid', projection, windows)
            residual = fitted - windows
            # Normal equations independently check every least-squares residual.
            orthogonality = np.einsum('ik,bid->bkd', phi, residual)
            max_orthogonality = max(max_orthogonality, float(np.abs(orthogonality).max()))
            assert max_orthogonality < 1e-10
            if episode == 0:
                actual = spline_class(nbSeg=6, nbDim=data.shape[-1], device='cpu')
                _, decoded = actual.encode_trajectory(torch.tensor(windows[:1], dtype=torch.float32))
                max_decode_difference = float(np.abs(decoded.numpy() - fitted[:1]).max())
                assert max_decode_difference < 3e-5
            errors = 100 * np.sqrt(np.mean(residual**2, axis=(1, 2)))
            curvature = np.sqrt(np.mean(np.diff(windows, n=2, axis=1)**2, axis=(1, 2)))
            task_errors.extend(errors.tolist())
            episode_errors.append(float(np.sqrt(np.mean(errors**2))))
            records.extend(dict(task=task, episode=episode, start=i, rmse_pct=float(e),
                                second_difference_rms=float(c)) for i, (e, c) in enumerate(zip(errors, curvature)))
        summary = dict(episodes=len(ends), fitted_episodes=len(episode_errors),
                       excluded_short_episodes=excluded, windows=len(task_errors), dimensions=actions.shape[1],
                       mean_episode_rmse_pct=float(np.mean(episode_errors)),
                       sample_sd_episode_rmse_pct=float(np.std(episode_errors, ddof=1)),
                       p95_window_rmse_pct=float(np.percentile(task_errors, 95)),
                       episode_rmse_pct=episode_errors,
                       max_normal_equation_residual=max_orthogonality,
                       max_float32_decoder_difference=max_decode_difference)
        summaries[task], sources[task] = summary, dict(**source, offset=offset.tolist(), range=scale.tolist())
        print(task, {k:v for k,v in summary.items() if k != 'episode_rmse_pct'}, flush=True)
    with (args.output / 'windows.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    audit = dict(protocol=dict(horizon=16, segments=6, coefficients_per_dimension=8,
        sampling='Every complete stride-1 window of every available demonstration; no padding.',
        endpoints='Free; inter-segment C1 continuity built into the original decoder.',
        metric='RMSE over time and action coordinates after per-task per-coordinate full-data range normalization, multiplied by 100.',
        aggregation='Per-episode RMS over all window errors, then equal-episode mean and sample SD (ddof=1). P95 pools windows.',
        limitations='Offline best-fit representation diagnostic, not a learned-policy score or a causal explanation of Table III. Public PH lowdim absolute-action data used for Can/Transport; no original image dataset hash available.',
        device='CPU, one torch/BLAS thread; no policy weights loaded'),
        source_sha256={str(p):sha(p.read_bytes()) for p in [Path(__file__).resolve(), DECODER, other_path]},
        sources=sources, summary=summaries, windows_csv_sha256=sha((args.output/'windows.csv').read_bytes()))
    if args.plot_examples:
        plot_examples(args.output, audit, records, spline, phi, inverse)
    (args.output / 'results.json').write_text(json.dumps(audit, indent=2) + '\n')


if __name__ == '__main__':
    main()
