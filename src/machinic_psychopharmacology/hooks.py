"""Optional PyTorch residual-stream hook.

This module is imported only when the ``hooks`` extra is installed
(``pip install machinic-psychopharmacology[hooks]``). The rest of the
library is numpy-only so the algebra can be tested without a GPU or a
transformer runtime.

Position-indexed steering (AISI / vllm-lens): the operator is applied
only at selected token positions. Later unsteered tokens can still
attend back to the steered KV cache.

Usage::

    import torch
    from machinic_psychopharmacology.hooks import ResidualSteerer
    from machinic_psychopharmacology.operators import combination

    v_f = torch.randn(d)
    v = torch.randn(d)
    steer = ResidualSteerer(
        lambda h: combination(h, v_f.numpy(), v.numpy(), gamma=0.8, kappa=1.0, p_star=0.0)
    )
    handle = layer.register_forward_hook(steer.hook)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover - exercised only without torch
    raise ImportError(
        "machinic_psychopharmacology.hooks requires PyTorch. "
        "Install with: pip install 'machinic-psychopharmacology[hooks]'"
    ) from exc

Array = NDArray[np.floating]
NpOp = Callable[[Array], Array]


@dataclass
class ResidualSteerer:
    """Wrap a numpy operator as a ``register_forward_hook`` callback.

    ``positions`` is a set of sequence indices to steer; ``None`` means
    all positions. When the residual has shape ``(batch, seq, d)`` the
    mask is applied on the sequence axis. When it is ``(batch, d)``
    (a single decoding step) the whole tensor is steered if
    ``current_index`` is in ``positions`` or ``positions`` is ``None``.
    """

    operator: NpOp
    positions: Sequence[int] | None = None
    current_index: int | None = None
    enabled: bool = True
    calls: int = field(default=0, init=False)

    def hook(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if not self.enabled:
            return output
        residual, rest = _unwrap(output)
        steered = self._apply(residual)
        self.calls += 1
        return _rewrap(steered, rest)

    def _apply(self, residual: torch.Tensor) -> torch.Tensor:
        arr = residual.detach().to(dtype=torch.float32).cpu().numpy()
        if arr.ndim == 3:
            # (batch, seq, d)
            out = arr.copy()
            seq = arr.shape[1]
            if self.positions is None:
                idx = range(seq)
            else:
                idx = [i for i in self.positions if 0 <= i < seq]
            for i in idx:
                out[:, i, :] = np.asarray(self.operator(arr[:, i, :]), dtype=np.float32)
            steered = out
        elif arr.ndim == 2:
            if self.positions is not None and self.current_index not in self.positions:
                return residual
            steered = np.asarray(self.operator(arr), dtype=np.float32)
        else:
            raise ValueError(f"unsupported residual shape {tuple(arr.shape)}")
        return torch.from_numpy(steered).to(device=residual.device, dtype=residual.dtype)


def _unwrap(
    output: torch.Tensor | tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...] | None]:
    if isinstance(output, tuple):
        if not output:
            raise ValueError("empty module output")
        return output[0], output[1:]
    return output, None


def _rewrap(
    residual: torch.Tensor,
    rest: tuple[torch.Tensor, ...] | None,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    if rest is None:
        return residual
    return (residual, *rest)
