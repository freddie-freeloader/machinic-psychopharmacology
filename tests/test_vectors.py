"""Unit tests for residual-stream vector helpers."""

from __future__ import annotations

import numpy as np
import pytest

from machinic_psychopharmacology.vectors import (
    as_1d,
    cosine_similarity,
    gram_schmidt,
    inner,
    l2_normalize,
    unit,
)
from machinic_psychopharmacology.vectors import mix_overlap


def test_l2_normalize_sets_norm() -> None:
    v, original = l2_normalize([3.0, 4.0], target_norm=2.0)
    assert original == pytest.approx(5.0)
    assert np.linalg.norm(v) == pytest.approx(2.0)


def test_unit_rejects_zero() -> None:
    with pytest.raises(ValueError):
        unit(np.zeros(4))


def test_as_1d_rejects_matrix() -> None:
    with pytest.raises(ValueError):
        as_1d(np.ones((2, 2)))


def test_inner_matches_dot() -> None:
    h = np.array([1.0, 2.0, 3.0])
    v = np.array([0.0, 1.0, 0.0])
    assert inner(h, v) == pytest.approx(2.0)


def test_inner_batches() -> None:
    h = np.arange(6.0).reshape(2, 3)
    v = np.array([1.0, 0.0, 0.0])
    np.testing.assert_allclose(inner(h, v), [0.0, 3.0])


def test_gram_schmidt_collinear() -> None:
    v = np.array([1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="collinear"):
        gram_schmidt(2.0 * v, v)


def test_cosine_self() -> None:
    v = np.array([1.2, -3.4, 0.5])
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_mix_overlap_exact_cosine() -> None:
    g = np.random.default_rng(0)
    anchor = g.normal(size=12)
    extra = g.normal(size=12)
    mixed = mix_overlap(anchor, extra, 0.6)
    assert cosine_similarity(mixed, unit(anchor)) == pytest.approx(0.6, abs=1e-10)
