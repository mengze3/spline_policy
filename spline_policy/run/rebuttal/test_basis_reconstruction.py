"""Numerical checks for parameter budgets, spline constraints, and transforms."""

import numpy as np
import pytest
from scipy.fft import dct, idct
from scipy.interpolate import BSpline

from basis_reconstruction import basis_matrix, reconstruct, reconstruction_errors, spline_basis, spline_maps


@pytest.mark.parametrize("parameters", [3, 7, 12, 17, 22])
@pytest.mark.parametrize("rest", [False, True])
def test_spline_space_and_parameter_budget(parameters, rest):
    phase = np.linspace(0, 1, 1000)
    power, control = spline_maps(parameters, rest)
    phi = spline_basis(phase, parameters, rest)
    assert phi.shape == (1000, parameters)
    assert np.linalg.matrix_rank(phi) == parameters
    np.testing.assert_allclose(control[:-1, -1], control[1:, 0])
    np.testing.assert_allclose(control[:-1, 2] - control[:-1, 1], control[1:, 1] - control[1:, 0])
    if rest:
        np.testing.assert_array_equal(control[-1, -1], control[-1, -2])
    else:
        # An independent B-spline implementation must span the same C1 space.
        knots = np.r_[np.zeros(3), np.arange(1, len(power)) / len(power), np.ones(3)]
        independent = BSpline(knots, np.eye(parameters), 2)(phase)
        reconstructed, _ = reconstruct(independent[None], "spline", parameters, ridge=0)
        np.testing.assert_allclose(reconstructed[0], independent, atol=2e-13)


def test_fast_matches_official_transform_and_retains_low_frequencies():
    reference = np.random.default_rng(7).normal(size=(3, 100, 2))
    reconstructed, weights = reconstruct(reference, "fast", 12)
    official = dct(reference, axis=1, norm="ortho")
    np.testing.assert_array_equal(weights, official[:, :12])
    np.testing.assert_allclose(dct(reconstructed, axis=1, norm="ortho")[:, 12:], 0, atol=1e-15)
    all_frequencies, _ = reconstruct(reference, "fast", 100)
    np.testing.assert_allclose(all_frequencies, reference, atol=2e-15)
    np.testing.assert_allclose(idct(official, axis=1, norm="ortho"), reference, atol=2e-15)


def test_fourier_mirrors_before_fitting():
    reference = np.random.default_rng(3).normal(size=(2, 100, 2))
    reconstructed, weights = reconstruct(reference, "fourier", 7)
    phi = basis_matrix("fourier", 200, 7)
    target = np.concatenate([reference, reference[:, ::-1]], axis=1)
    for i in range(2):
        expected = np.linalg.solve(phi.T @ phi + 1e-8 * np.eye(7), phi.T @ target[i])
        np.testing.assert_allclose(weights[i], expected, atol=2e-15)
        np.testing.assert_allclose(reconstructed[i], phi[:100] @ expected, atol=2e-15)


def test_metric_is_mean_pointwise_distance():
    reference = np.array([[[0., 0.], [3., 4.]]])
    fitted = reference + np.array([[[3., 4.], [0., 0.]]])
    errors = reconstruction_errors(reference, fitted)
    np.testing.assert_allclose(errors["mean_l2"], 2.5)
    np.testing.assert_allclose(errors["rmse"], 5 / np.sqrt(2))
    np.testing.assert_allclose(errors["mean_l2_over_bbox_percent"], 50)
