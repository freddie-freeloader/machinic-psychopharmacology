"""Vector utilities for residual-stream steering."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.floating]


def as_1d(v: Array | list[float]) -> np.ndarray:
    """Return a 1-D float64 vector, rejecting scalars and rank-0 arrays."""
    arr = np.asarray(v, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"expected a 1-D vector, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("vector must be non-empty")
    return arr


def hidden_as_array(h: Array) -> np.ndarray:
    """Accept a residual of shape ``(..., d)``."""
    arr = np.asarray(h, dtype=np.float64)
    if arr.ndim < 1:
        raise ValueError("hidden state must have at least one axis")
    return arr


def l2_normalize(
    v: Array,
    *,
    target_norm: float = 1.0,
    eps: float = 1e-12,
) -> tuple[np.ndarray, float]:
    """Scale a 1-D vector to ``target_norm``. Returns ``(scaled, original_norm)``."""
    vec = as_1d(v)
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        raise ValueError("cannot normalize a near-zero vector")
    return vec * (target_norm / norm), norm


def unit(v: Array, *, eps: float = 1e-12) -> np.ndarray:
    """Return a unit-length copy of ``v``."""
    scaled, _ = l2_normalize(v, target_norm=1.0, eps=eps)
    return scaled


def inner(h: Array, v: Array) -> np.ndarray:
    """Dot product of trailing axis of ``h`` with 1-D ``v``.

    ``h`` may be ``(d,)`` or ``(..., d)``. The result keeps all leading
    axes and has no trailing singleton unless ``h`` is 1-D, in which case
    the result is a 0-D array (a scalar numpy value).
    """
    hidden = hidden_as_array(h)
    vec = as_1d(v)
    if hidden.shape[-1] != vec.shape[0]:
        raise ValueError(
            f"trailing dim {hidden.shape[-1]} does not match vector {vec.shape[0]}"
        )
    return np.sum(hidden * vec, axis=-1)


def inner_keepdims(h: Array, v: Array) -> np.ndarray:
    """Like :func:`inner` but keeps a trailing axis of size 1 for broadcasting."""
    return inner(h, v)[..., None]


def cosine_similarity(a: Array, b: Array, *, eps: float = 1e-12) -> float:
    """Cosine similarity of two 1-D vectors."""
    ua = unit(a, eps=eps)
    ub = unit(b, eps=eps)
    return float(np.dot(ua, ub))


def gram_schmidt(v_f: Array, v: Array, *, eps: float = 1e-8) -> np.ndarray:
    """Return the unit component of ``v_f`` orthogonal to ``v``.

    This is the receptor-selectivity step in the design notes: if the
    stimulant (focus) and clamp (safety) directions share a component,
    the clamp eats the shared component of the stimulant. Orthogonalizing
    applies gain only on the unique-focus axis, leaving the clamp intact.
    """
    focus = as_1d(v_f)
    clamp = unit(v, eps=eps)
    residual = focus - float(np.dot(focus, clamp)) * clamp
    norm = float(np.linalg.norm(residual))
    if norm < eps:
        raise ValueError(
            "focus and clamp vectors are collinear; cannot orthogonalize"
        )
    return residual / norm


def project(h: Array, v: Array) -> np.ndarray:
    """Project ``h`` onto the span of unit ``v`` (same shape as ``h``)."""
    vec = unit(v)
    return inner_keepdims(h, vec) * vec


def mix_overlap(anchor: Array, extra: Array, cosine: float, *, eps: float = 1e-8) -> np.ndarray:
    """Unit vector with exact cosine overlap against ``anchor``.

    ``extra`` supplies the orthogonal complement. ``cosine`` must lie in
    ``(-1, 1)``. This is the controlled non-orthogonality used by the
    Pan/Zeni protocol arms.
    """
    if not -1.0 < cosine < 1.0:
        raise ValueError(f"cosine must be in (-1, 1), got {cosine}")
    a = unit(anchor, eps=eps)
    complement = gram_schmidt(extra, a, eps=eps)
    s = float(np.sqrt(1.0 - cosine * cosine))
    return unit(cosine * a + s * complement, eps=eps)
