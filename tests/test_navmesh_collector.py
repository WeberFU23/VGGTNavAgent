"""Pure-geometry tests for the oracle navmesh mapping collector."""

import math

import numpy as np

from scripts.diagnostics.collect_navmesh_mapping_sequence import (
    _navigation_action, _path_stays_on_floor, _similarity_align)


def test_same_floor_path_checks_every_corner():
    flat = [[0, 1.0, 0], [1, 1.1, 0], [2, 1.0, 0]]
    stairs = [[0, 1.0, 0], [1, 1.6, 0], [2, 1.0, 0]]
    assert _path_stays_on_floor(flat, 1.0, 0.2)
    assert not _path_stays_on_floor(stairs, 1.0, 0.2)


def test_heading_controller_turns_then_moves():
    forward = np.array([0.0, 0.0, -1.0])
    action, error = _navigation_action(forward, [-1.0, 0.0, 0.0])
    assert action == 2 and error > 0
    action, error = _navigation_action(forward, [1.0, 0.0, 0.0])
    assert action == 3 and error < 0
    action, error = _navigation_action(forward, [0.05, 0.0, -1.0])
    assert action == 1 and abs(math.degrees(error)) < 15


def test_similarity_alignment_recovers_known_transform():
    source = np.array([[0., 0., 0.], [1., 0., 0.],
                       [0., 0., 1.], [1., .5, 1.]])
    rotation = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
    target = 2.5 * source @ rotation.T + [4., 1., -3.]
    aligned, rmse = _similarity_align(source, target)
    assert np.allclose(aligned, target, atol=1e-8)
    assert rmse is not None and rmse < 1e-8
