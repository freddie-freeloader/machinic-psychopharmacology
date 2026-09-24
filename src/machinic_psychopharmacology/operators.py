"""Pharmacology-inspired residual-stream operators.

Each operator is a pure function of a residual ``h`` and one or more
directions. Shapes: ``h`` is ``(..., d)``; vectors are ``(d,)``. The
returned residual has the same shape as ``h``.

The mapping from drugs to operators is the design claim of this repo:

* **Haloperidol-like**  — fixed subtractive shift (D2 antagonist).
* **Aripiprazole-like** — bidirectional setpoint clamp (D2 partial agonist).
* **Methylphenidate-like** — multiplicative, state-dependent gain (DAT/NET
  reuptake blocker).
* **Amphetamine-like**  — additive injection independent of current
  projection (contrast with MPH).
* **Combination**       — MPH gain + aripiprazole clamp, with optional
  Gram–Schmidt orthogonalization of the two directions.

None of these is a clinical model. They are algebraic analogues for
inference-time activation steering, following the AISI self-steering
setup and the activation-addition literature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .vectors import as_1d, gram_schmidt, hidden_as_array, inner_keepdims, unit

Array = NDArray[np.floating]


class Operator(Protocol):
    """Callable that maps a residual to a steered residual."""

    def __call__(self, h: Array) -> np.ndarray: ...


def _require_positive(name: str, value: float) -> None:
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative number, got {value}")


def antagonist(h: Array, v: Array, lambda_: float) -> np.ndarray:
    """Haloperidol-like D2 antagonist: ``h' = h − λ v``.

    Unbounded suppression. At large ``λ`` this is the analogue of
    extrapyramidal side effects / blunted affect: the residual is
    translated off-manifold along ``v`` regardless of its current
    projection.
    """
    _require_positive("lambda_", lambda_)
    hidden = hidden_as_array(h)
    vec = unit(v)
    return hidden - lambda_ * vec


def aripiprazole(h: Array, v: Array, kappa: float, p_star: float) -> np.ndarray:
    """Aripiprazole-like D2 partial agonist: ``h' = h + κ (p* − ⟨h, v⟩) v``.

    Bidirectional setpoint controller. ``p*`` is the intrinsic-activity
    floor (partial agonism is not full blockade). The update pulls the
    projection of ``h`` onto unit ``v`` toward ``p*`` with gain ``κ``.

    Fixed point: ``⟨h', v⟩ = ⟨h, v⟩ + κ (p* − ⟨h, v⟩)``, so at ``κ = 1``
    the projection lands exactly on ``p*``.
    """
    _require_positive("kappa", kappa)
    hidden = hidden_as_array(h)
    vec = unit(v)
    proj = inner_keepdims(hidden, vec)
    return hidden + kappa * (p_star - proj) * vec


def methylphenidate(h: Array, v_f: Array, gamma: float) -> np.ndarray:
    """Methylphenidate-like DAT/NET reuptake block: ``h' = h + γ ⟨h, v_f⟩ v_f``.

    State-dependent gain. If the current residual has no projection on
    the focus direction, the operator is a no-op — reuptake blockade
    only prolongs existing signal; it does not inject one.
    """
    _require_positive("gamma", gamma)
    hidden = hidden_as_array(h)
    vec = unit(v_f)
    proj = inner_keepdims(hidden, vec)
    return hidden + gamma * proj * vec


def amphetamine(h: Array, v_f: Array, lambda_s: float) -> np.ndarray:
    """Amphetamine-like additive injection: ``h' = h + λ_s v_f``.

    Independent of current projection. The design notes contrast this
    with MPH: amphetamine dumps transmitter regardless of firing, with
    higher toxicity (stimulant-psychosis analogue = off-state injection
    plus stereotypy).
    """
    _require_positive("lambda_s", lambda_s)
    hidden = hidden_as_array(h)
    vec = unit(v_f)
    return hidden + lambda_s * vec


def combination(
    h: Array,
    v_f: Array,
    v: Array,
    gamma: float,
    kappa: float,
    p_star: float,
    *,
    orthogonalize: bool = False,
) -> np.ndarray:
    """Combined regimen: MPH gain plus aripiprazole clamp.

    ``h' = h + γ ⟨h, v_f⟩ v_f + κ (p* − ⟨h, v⟩) v``

    If ``orthogonalize`` is true, ``v_f`` is replaced by its component
    orthogonal to ``v`` before the gain is applied. That is the
    Pan-like arm: receptor selectivity that keeps the safety cap
    without letting the clamp eat the stimulant.
    """
    focus = gram_schmidt(v_f, v) if orthogonalize else unit(v_f)
    gained = methylphenidate(h, focus, gamma)
    return aripiprazole(gained, v, kappa, p_star)


def apply_stack(h: Array, v: Array, receptors: dict[str, float]) -> np.ndarray:
    """Apply an aripiprazole-like receptor stack as sequential clamps.

    Keys are labels only. Values are ``(κ, p*)`` packed as a 2-tuple, or
    a single ``p*`` with ``κ = 1``. The vector ``v`` is shared; callers
    who want distinct directions should call :func:`aripiprazole` per
    receptor instead.

    This helper exists for the D2 / 5-HT1A / 5-HT2A sketch in the
    design notes. Prefer explicit per-direction calls in experiments.
    """
    residual = hidden_as_array(h)
    for name, spec in receptors.items():
        if isinstance(spec, tuple):
            if len(spec) != 2:
                raise ValueError(f"receptor {name!r} expected (kappa, p_star)")
            kappa, p_star = spec
        else:
            kappa, p_star = 1.0, float(spec)
        residual = aripiprazole(residual, v, float(kappa), float(p_star))
    return residual


@dataclass(frozen=True)
class Receptor:
    """One direction in a combination, with its own operator kind."""

    name: str
    vector: np.ndarray
    kind: str  # "clamp" | "gain" | "antagonist" | "inject"
    gain: float
    setpoint: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "vector", as_1d(self.vector))
        if self.kind not in {"clamp", "gain", "antagonist", "inject"}:
            raise ValueError(f"unknown receptor kind {self.kind!r}")
        _require_positive("gain", self.gain)

    def apply(self, h: Array) -> np.ndarray:
        if self.kind == "clamp":
            return aripiprazole(h, self.vector, self.gain, self.setpoint)
        if self.kind == "gain":
            return methylphenidate(h, self.vector, self.gain)
        if self.kind == "antagonist":
            return antagonist(h, self.vector, self.gain)
        return amphetamine(h, self.vector, self.gain)
