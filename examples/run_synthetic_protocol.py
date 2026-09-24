#!/usr/bin/env python3
"""Sweep γ and cosine-overlap, then print the three-arm comparison.

This is the cheap, no-GPU version of the protocol in design-notes.md.
Swap the synthetic residuals for real layer-L activations (e.g. from
UK AISI llm-self-steering / vllm-lens) when running on a model.
"""

from __future__ import annotations

import argparse

import numpy as np

from machinic_psychopharmacology.protocol import ProtocolConfig, run_protocol
from machinic_psychopharmacology.vectors import mix_overlap, unit


def residuals(n: int, d: int, v_f: np.ndarray, v: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (
        rng.normal(0.6, 0.25, size=(n, 1)) * v_f
        + rng.normal(0.35, 0.4, size=(n, 1)) * v
        + rng.normal(0.0, 0.12, size=(n, d))
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    v = unit(rng.normal(size=args.dim))
    extra = unit(rng.normal(size=args.dim))

    print(f"{'overlap':>8} {'arm':<14} {'efficacy':>10} {'toxicity':>10} {'fail':>8}")
    for overlap in (0.0, 0.3, 0.6, 0.9):
        v_f = mix_overlap(v, extra, overlap if overlap < 0.999 else 0.999)
        h = residuals(args.n, args.dim, v_f, v, args.seed)
        cfg = ProtocolConfig(gamma=0.8, kappa=1.0, p_star=0.0)
        for arm, m in run_protocol(h, v_f, v, cfg).items():
            print(
                f"{overlap:8.1f} {arm:<14} {m.efficacy:10.4f} {m.toxicity:10.4f} {m.clamp_fail:8.3f}"
            )
        print()


if __name__ == "__main__":
    main()
