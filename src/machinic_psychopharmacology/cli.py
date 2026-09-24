"""Command-line demo of the three-arm protocol on synthetic residuals."""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from .protocol import ProtocolConfig, interaction_report, run_protocol
from .vectors import mix_overlap, unit


def _synthetic_residuals(
    n: int,
    d: int,
    v_f: np.ndarray,
    v: np.ndarray,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Mix of on-task focus, hallucinated overflow, and isotropic noise.
    focus = rng.normal(0.6, 0.25, size=(n, 1))
    hallu = rng.normal(0.3, 0.4, size=(n, 1))
    noise = rng.normal(0.0, 0.15, size=(n, d))
    return focus * v_f + hallu * v + noise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="machinic-psychopharmacology",
        description=(
            "Run the three-arm combination-steering protocol on synthetic "
            "residual snapshots (no GPU required)."
        ),
    )
    parser.add_argument("--tokens", type=int, default=64, help="number of residual snapshots")
    parser.add_argument("--dim", type=int, default=32, help="residual dimension")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.8, help="MPH-like gain")
    parser.add_argument("--kappa", type=float, default=1.0, help="aripiprazole-like clamp gain")
    parser.add_argument("--p-star", type=float, default=0.0, dest="p_star", help="clamp setpoint")
    parser.add_argument("--overlap", type=float, default=0.4, help="cosine overlap of v_f and v")
    parser.add_argument("--json", action="store_true", help="print machine-readable output")
    args = parser.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    v = unit(rng.normal(size=args.dim))
    v_f_raw = unit(rng.normal(size=args.dim))
    # Mix a controlled overlap into the focus direction.
    rho = float(np.clip(args.overlap, -0.99, 0.99))
    v_f = mix_overlap(v, v_f_raw, rho)

    residuals = _synthetic_residuals(args.tokens, args.dim, v_f, v, args.seed)
    cfg = ProtocolConfig(gamma=args.gamma, kappa=args.kappa, p_star=args.p_star)
    results = run_protocol(residuals, v_f, v, cfg)
    interaction = interaction_report(v_f, v)

    payload = {
        "interaction": interaction,
        "arms": {arm: m.as_dict() for arm, m in results.items()},
    }
    if args.json:
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    print("Machinic psychopharmacology — three-arm protocol (synthetic residuals)")
    print(f"  n={args.tokens}  d={args.dim}  cosine(v_f, v)={interaction['cosine']:.3f}")
    print()
    print(f"{'arm':<14} {'efficacy':>10} {'toxicity':>10} {'rebound':>10} {'fail':>8}")
    for arm, m in results.items():
        print(
            f"{arm:<14} {m.efficacy:10.4f} {m.toxicity:10.4f} "
            f"{m.rebound:10.4f} {m.clamp_fail:8.3f}"
        )
    print()
    print("Claim: both combo arms clamp toxicity; ortho_combo applies gain only on the unique-focus axis.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
