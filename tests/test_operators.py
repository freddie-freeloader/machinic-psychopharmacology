"""Algebraic contracts for the drug → steering mapping."""

from __future__ import annotations

import numpy as np
import pytest

from machinic_psychopharmacology.operators import (
    Receptor,
    amphetamine,
    antagonist,
    aripiprazole,
    combination,
    methylphenidate,
)
from machinic_psychopharmacology.vectors import cosine_similarity, gram_schmidt, inner, unit


def rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def basis(d: int = 8, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = rng(seed)
    v = unit(g.normal(size=d))
    # Independent-ish second direction.
    extra = g.normal(size=d)
    v_f = unit(extra - np.dot(extra, v) * v)
    h = g.normal(size=d)
    return h, v_f, v


def test_aripiprazole_fixed_point_at_kappa_one() -> None:
    h, _, v = basis()
    p_star = 0.4
    h2 = aripiprazole(h, v, kappa=1.0, p_star=p_star)
    assert inner(h2, unit(v)) == pytest.approx(p_star, abs=1e-10)


def test_aripiprazole_is_bidirectional() -> None:
    _, _, v = basis()
    v = unit(v)
    high = 3.0 * v
    low = -2.0 * v
    p_star = 0.2
    down = aripiprazole(high, v, kappa=1.0, p_star=p_star)
    up = aripiprazole(low, v, kappa=1.0, p_star=p_star)
    assert inner(high, v) > p_star
    assert inner(low, v) < p_star
    assert inner(down, v) == pytest.approx(p_star)
    assert inner(up, v) == pytest.approx(p_star)


def test_methylphenidate_noop_when_projection_zero() -> None:
    _, v_f, v = basis()
    # Residual orthogonal to v_f.
    h = v  # v was built orthogonal to v_f
    assert inner(h, unit(v_f)) == pytest.approx(0.0, abs=1e-10)
    h2 = methylphenidate(h, v_f, gamma=1.7)
    np.testing.assert_allclose(h2, h)


def test_methylphenidate_amplifies_existing_signal() -> None:
    _, v_f, _ = basis()
    v_f = unit(v_f)
    h = 0.5 * v_f
    h2 = methylphenidate(h, v_f, gamma=1.0)
    # h' = h + γ ⟨h,v⟩ v = 0.5 v + 1.0 * 0.5 v = 1.0 v
    assert inner(h2, v_f) == pytest.approx(1.0)


def test_antagonist_unbounded_shift() -> None:
    h, _, v = basis()
    v = unit(v)
    h2 = antagonist(h, v, lambda_=2.5)
    np.testing.assert_allclose(h2, h - 2.5 * v)


def test_amphetamine_injects_regardless_of_state() -> None:
    _, v_f, v = basis()
    v_f = unit(v_f)
    off_state = v  # orthogonal to v_f
    h2 = amphetamine(off_state, v_f, lambda_s=1.3)
    assert inner(off_state, v_f) == pytest.approx(0.0, abs=1e-10)
    assert inner(h2, v_f) == pytest.approx(1.3)


def test_combination_matches_sequential_ops() -> None:
    h, v_f, v = basis()
    sequential = aripiprazole(methylphenidate(h, v_f, 0.7), v, 0.5, 0.1)
    composed = combination(h, v_f, v, 0.7, 0.5, 0.1, orthogonalize=False)
    np.testing.assert_allclose(sequential, composed)


def test_orthogonalize_zeros_shared_component() -> None:
    g = rng(1)
    v = unit(g.normal(size=16))
    v_f = mix_overlap(v, g.normal(size=16), 0.8)
    assert cosine_similarity(v_f, v) == pytest.approx(0.8, abs=1e-10)
    v_f_perp = gram_schmidt(v_f, v)
    assert cosine_similarity(v_f_perp, v) == pytest.approx(0.0, abs=1e-10)


def test_raw_combo_blunts_gain_when_vectors_overlap() -> None:
    """Zeni-like: clamp eats stimulant when v_f shares a component with v."""
    g = rng(2)
    d = 24
    v = unit(g.normal(size=d))
    v_f = mix_overlap(v, g.normal(size=d), 0.8)
    v_f_unique = gram_schmidt(v_f, v)
    h = 1.2 * v_f + 0.5 * v

    raw = combination(h, v_f, v, gamma=1.5, kappa=1.0, p_star=0.0, orthogonalize=False)
    ortho = combination(h, v_f, v, gamma=1.5, kappa=1.0, p_star=0.0, orthogonalize=True)

    # The clamp lands hallucination projection near 0 in both arms.
    assert abs(inner(raw, unit(v))) == pytest.approx(0.0, abs=1e-8)
    assert abs(inner(ortho, unit(v))) == pytest.approx(0.0, abs=1e-8)
    # Stimulant-only on overlapping v_f leaks into the hallucination axis;
    # the combo arms zero that axis. That is the safety cap.
    stim = methylphenidate(h, v_f, 1.5)
    assert abs(inner(stim, unit(v))) > 0.1
    assert abs(inner(raw, unit(v))) < abs(inner(stim, unit(v)))
    assert abs(inner(ortho, unit(v))) < abs(inner(stim, unit(v)))
    # Unique-focus energy is finite and present in both combo arms.
    assert inner(ortho, v_f_unique) != 0.0
    assert inner(raw, v_f_unique) != 0.0


def test_batch_shape_preserved() -> None:
    g = rng(3)
    h = g.normal(size=(4, 6, 10))
    v = unit(g.normal(size=10))
    out = aripiprazole(h, v, kappa=0.5, p_star=0.0)
    assert out.shape == h.shape


def test_rejects_bad_gains() -> None:
    h, v_f, v = basis()
    with pytest.raises(ValueError):
        methylphenidate(h, v_f, gamma=-0.1)
    with pytest.raises(ValueError):
        aripiprazole(h, v, kappa=float("nan"), p_star=0.0)


def test_receptor_dispatch() -> None:
    h, v_f, v = basis()
    r = Receptor("d2", v, kind="clamp", gain=1.0, setpoint=0.25)
    out = r.apply(h)
    assert inner(out, unit(v)) == pytest.approx(0.25)
    g = Receptor("dat", v_f, kind="gain", gain=1.0)
    gained = g.apply(0.4 * unit(v_f))
    assert inner(gained, unit(v_f)) == pytest.approx(0.8)
from machinic_psychopharmacology.vectors import cosine_similarity, gram_schmidt, inner, mix_overlap, unit
from machinic_psychopharmacology.vectors import (
    cosine_similarity,
    gram_schmidt,
    inner,
    mix_overlap,
    unit,
)
