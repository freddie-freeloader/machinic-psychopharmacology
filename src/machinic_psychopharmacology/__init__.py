"""Pharmacology-inspired activation steering for LLMs.

Algebraic analogues of aripiprazole (setpoint clamp) and methylphenidate
(state-dependent gain), composed into a three-arm combination-steering
protocol. Numpy-only core; optional PyTorch residual hook.
"""

from .operators import (
    Receptor,
    amphetamine,
    antagonist,
    aripiprazole,
    combination,
    methylphenidate,
)
from .protocol import (
    Metrics,
    Probes,
    ProtocolConfig,
    evaluate_arm,
    interaction_report,
    run_protocol,
)
from .schedule import Pulse, Schedule, depot_constant, inverted_u_gain, ir_work_blocks, oros_constant
from .vectors import cosine_similarity, gram_schmidt, inner, l2_normalize, mix_overlap, unit

__version__ = "0.1.0"

__all__ = [
    "Receptor",
    "Metrics",
    "Probes",
    "ProtocolConfig",
    "Pulse",
    "Schedule",
    "amphetamine",
    "antagonist",
    "aripiprazole",
    "combination",
    "cosine_similarity",
    "depot_constant",
    "evaluate_arm",
    "gram_schmidt",
    "inner",
    "interaction_report",
    "inverted_u_gain",
    "ir_work_blocks",
    "l2_normalize",
    "methylphenidate",
    "mix_overlap",
    "oros_constant",
    "run_protocol",
    "unit",
    "__version__",
]
