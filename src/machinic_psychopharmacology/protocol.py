"""Three-arm combination-steering protocol and toxicity probes.

The protocol is GPU-agnostic. It consumes batches of residual snapshots
and reports algebraic metrics that stand in for the experimental
readouts in the design notes:

* **efficacy**     — mean focus-direction projection (proxy for accuracy
  lift on a downstream task).
* **toxicity**     — mean absolute residual off the clamp setpoint
  (proxy for hallucination / entropy overflow).
* **rebound**      — post-offset dip of the IR schedule.
* **variance**     — run-to-run spread of the focus projection
  (proxy for reaction-time variability).
* **clamp_fail**   — fraction of tokens whose hallucination projection
  exceeds a safety threshold (rare mixed-episode analogue).

Arm 1: ``γ`` alone, IR-scheduled.
Arm 2: ``γ + κ``, raw (possibly non-orthogonal) vectors.
Arm 3: ``γ + κ``, ``v_f ⊥ v``.

These are the Pan vs Zeni contrast: arm 2 expects the clamp to blunt
the stimulant; arm 3 expects efficacy restored with the safety cap
kept.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np
from numpy.typing import NDArray

from .operators import aripiprazole, combination, methylphenidate
from .schedule import Schedule, ir_work_blocks
from .vectors import cosine_similarity, gram_schmidt, inner, unit

Array = NDArray[np.floating]
Arm = Literal["stimulant", "raw_combo", "ortho_combo"]


@dataclass(frozen=True)
class Metrics:
    efficacy: float
    toxicity: float
    rebound: float
    variance: float
    clamp_fail: float
    mean_focus: float
    mean_hallucination: float
    n_tokens: int
    arm: Arm

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "arm": self.arm,
            "efficacy": self.efficacy,
            "toxicity": self.toxicity,
            "rebound": self.rebound,
            "variance": self.variance,
            "clamp_fail": self.clamp_fail,
            "mean_focus": self.mean_focus,
            "mean_hallucination": self.mean_hallucination,
            "n_tokens": self.n_tokens,
        }


@dataclass
class Probes:
    """Linear probes on two directions of interest."""

    v_focus: np.ndarray
    v_hallucination: np.ndarray
    fail_threshold: float = 1.5

    def __post_init__(self) -> None:
        self.v_hallucination = unit(self.v_hallucination)
        # Efficacy is unique focus: the component of v_f the clamp cannot eat.
        try:
            self.v_focus = gram_schmidt(self.v_focus, self.v_hallucination)
        except ValueError:
            self.v_focus = unit(self.v_focus)

    def read(self, h: Array) -> tuple[np.ndarray, np.ndarray]:
        return inner(h, self.v_focus), inner(h, self.v_hallucination)


@dataclass
class ProtocolConfig:
    gamma: float = 0.8
    kappa: float = 1.0
    p_star: float = 0.0
    ir: Schedule = field(
        default_factory=lambda: ir_work_blocks(
            block_starts=[0.0],
            block_duration=32.0,
            peak=0.8,
            rebound=0.3,
            rebound_duration=8.0,
        )
    )
    fail_threshold: float = 1.5


SteerFn = Callable[[np.ndarray, int], np.ndarray]


def _as_batch(h: Array) -> np.ndarray:
    arr = np.asarray(h, dtype=np.float64)
    if arr.ndim == 1:
        return arr[None, :]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"expected (d,) or (n, d), got {arr.shape}")


def apply_arm(
    h: Array,
    arm: Arm,
    v_f: Array,
    v: Array,
    cfg: ProtocolConfig,
    t: float,
) -> np.ndarray:
    """Steer a residual (or batch) under one protocol arm at time ``t``."""
    gamma_t = cfg.ir.value(t) if arm == "stimulant" else cfg.gamma
    if arm == "stimulant":
        return methylphenidate(h, v_f, max(gamma_t, 0.0))
    if arm == "raw_combo":
        return combination(h, v_f, v, cfg.gamma, cfg.kappa, cfg.p_star, orthogonalize=False)
    if arm == "ortho_combo":
        return combination(h, v_f, v, cfg.gamma, cfg.kappa, cfg.p_star, orthogonalize=True)
    raise ValueError(f"unknown arm {arm!r}")


def evaluate_arm(
    residuals: Array,
    arm: Arm,
    v_f: Array,
    v: Array,
    cfg: ProtocolConfig | None = None,
    *,
    times: Array | None = None,
    baseline_efficacy: float | None = None,
) -> Metrics:
    """Run one arm over a sequence of residuals.

    ``residuals`` has shape ``(n, d)``. ``times`` defaults to
    ``0..n-1``. Efficacy is the mean post-steer focus projection,
    minus ``baseline_efficacy`` if provided (otherwise the pre-steer
    mean). Toxicity is the mean absolute hallucination-probe reading
    after steering. Rebound is the mean focus projection on tokens
    where the IR schedule is negative, or 0 when the arm is not IR.
    """
    cfg = cfg or ProtocolConfig()
    batch = _as_batch(residuals)
    n = batch.shape[0]
    if times is None:
        times = np.arange(n, dtype=np.float64)
    else:
        times = np.asarray(times, dtype=np.float64)
        if times.shape[0] != n:
            raise ValueError("times must match the residual batch length")

    probes = Probes(v_f, v, fail_threshold=cfg.fail_threshold)
    pre_focus, _ = probes.read(batch)
    steered = np.stack(
        [apply_arm(batch[i], arm, v_f, v, cfg, float(times[i])) for i in range(n)]
    )
    post_focus, post_hallu = probes.read(steered)

    base = float(pre_focus.mean()) if baseline_efficacy is None else float(baseline_efficacy)
    efficacy = float(post_focus.mean() - base)
    toxicity = float(np.abs(post_hallu).mean())
    variance = float(post_focus.var())
    clamp_fail = float((np.abs(post_hallu) > cfg.fail_threshold).mean())

    if arm == "stimulant":
        rebound_mask = np.array([cfg.ir.value(float(t)) < 0 for t in times])
        rebound = float(post_focus[rebound_mask].mean() - base) if rebound_mask.any() else 0.0
    else:
        rebound = 0.0

    return Metrics(
        efficacy=efficacy,
        toxicity=toxicity,
        rebound=rebound,
        variance=variance,
        clamp_fail=clamp_fail,
        mean_focus=float(post_focus.mean()),
        mean_hallucination=float(post_hallu.mean()),
        n_tokens=n,
        arm=arm,
    )


def run_protocol(
    residuals: Array,
    v_f: Array,
    v: Array,
    cfg: ProtocolConfig | None = None,
    *,
    times: Array | None = None,
) -> dict[Arm, Metrics]:
    """Evaluate all three arms on the same residual sequence."""
    return {
        arm: evaluate_arm(residuals, arm, v_f, v, cfg, times=times)
        for arm in ("stimulant", "raw_combo", "ortho_combo")
    }


def interaction_report(v_f: Array, v: Array) -> dict[str, float]:
    """Quantify pharmacodynamic antagonism = vector non-orthogonality."""
    cos = cosine_similarity(v_f, v)
    try:
        ortho = gram_schmidt(v_f, v)
        recovered = float(np.linalg.norm(ortho))
    except ValueError:
        recovered = 0.0
    return {
        "cosine": cos,
        "shared_energy": cos * cos,
        "orthogonalizable": float(recovered > 0),
    }
