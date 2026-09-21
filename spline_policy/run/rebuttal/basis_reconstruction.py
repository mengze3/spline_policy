"""Matched-size trajectory bases for M1.

Basis definitions: idiap/rcfs, python/MP.py (f4c96d6).
Continuous FAST transform: physical-intelligence/fast (ec4d7aa).
See TABLE1.md for equations, source links, and the evaluation protocol.
"""

from pathlib import Path
import sys

import numpy as np
from scipy.fft import dct, idct
from scipy.special import comb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from policy.diffusion_policy.planning.quadratic_spline import QuadraticSpline


METHODS = {
    "piecewise": "Piecewise",
    "bernstein": "B.P.",
    "rbf": "RBF",
    "fourier": "Fourier",
    "fast": "FAST (continuous DCT)",
    "spline": "Q.S. (ours)",
    "spline_rest": "Q.S. + terminal rest",
}
PROJECTION = {
    "piecewise": "Discrete lookup",
    "bernstein": "Polynomial",
    "rbf": "Numerical",
    "fourier": "Numerical",
    "fast": "Numerical",
    "spline": "Cubic / segment",
    "spline_rest": "Cubic / segment",
}


def spline_maps(parameters, terminal_rest=False):
    """Return power/control maps with exactly K independent columns per axis."""
    segments = parameters - 2 + int(terminal_rest)
    spline = QuadraticSpline(nbSeg=segments, device="cpu")
    control = spline.C.numpy().astype(np.float64)
    power = spline.BC.numpy().astype(np.float64)
    if terminal_rest:
        # One more segment supplies the degree of freedom used by f'(1) = 0.
        reduction = np.eye(parameters + 1)[:, :parameters]
        reduction[-1] = control[-2, :parameters]
        control, power = control @ reduction, power @ reduction
    return power.reshape(segments, 3, parameters), control.reshape(segments, 3, parameters)


def spline_basis(phase, parameters, terminal_rest=False):
    power, _ = spline_maps(parameters, terminal_rest)
    scaled = np.asarray(phase) * len(power)
    segment = np.minimum(scaled.astype(int), len(power) - 1)
    local = scaled - segment
    monomial = np.stack([np.ones_like(local), local, local**2], axis=-1)
    return np.einsum("ti,tik->tk", monomial, power[segment])


def basis_matrix(method, points, parameters):
    phase = np.linspace(0, 1, points)[:, None]
    index = np.arange(parameters)[None, :]
    if method == "piecewise":
        return (np.arange(points)[:, None] // int(np.ceil(points / parameters)) == index).astype(float)
    if method == "bernstein":
        return comb(parameters - 1, index) * phase**index * (1 - phase)**(parameters - 1 - index)
    if method == "rbf":
        return np.exp(-100 * (phase - np.linspace(0, 1, parameters)) ** 2)
    if method == "fourier":
        return np.cos(2 * np.pi * phase * index)
    if method in ("spline", "spline_rest"):
        return spline_basis(phase[:, 0], parameters, terminal_rest=method == "spline_rest")
    raise ValueError(method)


def reconstruct(trajectories, method, parameters, ridge=1e-8):
    """Fit [demonstration, time, dimension]; retain K real coefficients per axis."""
    points = trajectories.shape[1]
    if method == "fast":
        # Official FAST uses this orthonormal DCT-II, then quantization and BPE.
        # This continuous comparison keeps K low frequencies and omits both steps.
        weights = dct(trajectories, axis=1, norm="ortho")[:, :parameters]
        padded = np.zeros_like(trajectories)
        padded[:, :parameters] = weights
        return idct(padded, axis=1, norm="ortho"), weights

    target = trajectories
    if method == "fourier":
        target = np.concatenate([target, target[:, ::-1]], axis=1)
    phi = basis_matrix(method, target.shape[1], parameters)
    flat = target.transpose(1, 0, 2).reshape(len(phi), -1)
    # Match RCFS's 1e-8 ridge without forming the ill-conditioned normal matrix.
    design = np.vstack([phi, np.sqrt(ridge) * np.eye(parameters)])
    target_flat = np.vstack([flat, np.zeros((parameters, flat.shape[1]))])
    weights = np.linalg.lstsq(design, target_flat, rcond=None)[0]
    fitted = (phi[:points] @ weights).reshape(points, len(target), -1).transpose(1, 0, 2)
    return fitted, weights.reshape(parameters, len(target), -1).transpose(1, 0, 2)


def reconstruction_errors(reference, fitted):
    residual = fitted - reference
    distance = np.linalg.norm(residual, axis=-1)
    mean_l2 = distance.mean(axis=1)
    return {
        "mean_l2": mean_l2,
        "rmse": np.sqrt(np.mean(distance**2, axis=1)),
        "max_l2": distance.max(axis=1),
        "frobenius_per_point": np.linalg.norm(residual, axis=(1, 2)) / reference.shape[1],
        "mean_l2_over_bbox_percent": 100 * mean_l2 / np.linalg.norm(np.ptp(reference, axis=1), axis=1),
    }
