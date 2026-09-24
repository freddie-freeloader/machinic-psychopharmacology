"""Pharmacokinetic analogues: dose as a function of token/time.

IR methylphenidate (t½ ≈ 2.5–3.5 h) maps to a work-block pulse of
``γ`` plus a rebound undershoot on offset. OROS maps to a near-constant
low ``γ``. Depot aripiprazole maps to a fine-tuned-in direction rather
than a per-token controller — represented here as a persistent clamp
with no offset.

Time is abstract. Callers pass a scalar ``t`` in the same units they
chose when constructing the schedule (tokens, forward-passes, or
wall-clock hours).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


Formulation = Literal["ir", "oros", "depot"]


@dataclass(frozen=True)
class Pulse:
    """A rectangular work block of gain, optionally with rebound."""

    start: float
    duration: float
    peak: float
    rebound: float = 0.0
    rebound_duration: float = 0.0

    def __post_init__(self) -> None:
        if self.duration <= 0:
            raise ValueError("duration must be positive")
        if self.rebound < 0:
            raise ValueError("rebound magnitude must be non-negative")
        if self.rebound > 0 and self.rebound_duration <= 0:
            raise ValueError("rebound_duration must be positive when rebound > 0")

    def value(self, t: float) -> float:
        if self.start <= t < self.start + self.duration:
            return self.peak
        if (
            self.rebound > 0
            and self.start + self.duration
            <= t
            < self.start + self.duration + self.rebound_duration
        ):
            return -self.rebound
        return 0.0


@dataclass(frozen=True)
class Schedule:
    """A piecewise-constant PK curve over abstract time."""

    formulation: Formulation
    pulses: tuple[Pulse, ...] = ()
    baseline: float = 0.0

    def value(self, t: float) -> float:
        total = self.baseline
        for pulse in self.pulses:
            total += pulse.value(t)
        return total

    def sample(self, times: np.ndarray) -> np.ndarray:
        return np.array([self.value(float(t)) for t in np.asarray(times)], dtype=np.float64)


def ir_work_blocks(
    *,
    block_starts: list[float],
    block_duration: float,
    peak: float,
    rebound: float,
    rebound_duration: float,
) -> Schedule:
    """Immediate-release MPH: pulses of ``γ`` with post-offset dip."""
    pulses = tuple(
        Pulse(
            start=s,
            duration=block_duration,
            peak=peak,
            rebound=rebound,
            rebound_duration=rebound_duration,
        )
        for s in block_starts
    )
    return Schedule(formulation="ir", pulses=pulses)


def oros_constant(gamma: float) -> Schedule:
    """OROS analogue: constant low gain, no rebound."""
    if gamma < 0:
        raise ValueError("gamma must be non-negative")
    return Schedule(formulation="oros", baseline=gamma)


def depot_constant(kappa: float) -> Schedule:
    """Depot aripiprazole analogue: persistent clamp gain."""
    if kappa < 0:
        raise ValueError("kappa must be non-negative")
    return Schedule(formulation="depot", baseline=kappa)


def inverted_u_gain(projection: float, gamma_max: float, x0: float = 0.0, width: float = 1.0) -> float:
    """Yerkes–Dodson / inverted-U gain as a function of current projection.

    ``g(x) = γ_max · exp(−((x − x0) / width)²)``. Used when mapping
    ``γ`` against focus-direction occupancy to look for an inverted-U
    in efficacy.
    """
    if width <= 0:
        raise ValueError("width must be positive")
    if gamma_max < 0:
        raise ValueError("gamma_max must be non-negative")
    z = (projection - x0) / width
    return float(gamma_max * np.exp(-(z * z)))
