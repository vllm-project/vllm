# SPDX-License-Identifier: Apache-2.0
"""ActivationCollector – accumulates per-channel activation statistics
during a warmup phase for pairwise_fp4 activation-side risk score
computation.

This is a lightweight stateful helper: call ``update(x)`` on each forward
pass during warmup, then ``finalize()`` to produce a 1-D risk-score tensor.
"""

from __future__ import annotations

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


class ActivationCollector:
    """Collects per-channel activation statistics over multiple forward passes.

    Parameters
    ----------
    num_channels : int
        Number of input channels (last dim of activation tensors).
    risk_method : str
        ``"max_abs"`` or ``"dynamic_range"``.
    warmup_samples : int
        Number of ``update()`` calls before the collector is considered ready.
    device : str or torch.device
        Device for internal accumulators.
    """

    def __init__(
        self,
        num_channels: int,
        risk_method: str = "max_abs",
        warmup_samples: int = 8,
        device: str | torch.device = "cpu",
    ) -> None:
        self.num_channels = num_channels
        self.risk_method = risk_method
        self.warmup_samples = max(1, warmup_samples)
        self._count = 0

        # Accumulators – always float32
        self._max_abs = torch.zeros(num_channels, dtype=torch.float32,
                                    device=device)
        if risk_method == "dynamic_range":
            # Track min-nonzero-abs as well
            self._min_nonzero = torch.full(
                (num_channels,), float("inf"), dtype=torch.float32,
                device=device,
            )

    # ------------------------------------------------------------------

    @property
    def ready(self) -> bool:
        """True once enough samples have been collected."""
        return self._count >= self.warmup_samples

    @property
    def count(self) -> int:
        return self._count

    # ------------------------------------------------------------------

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        """Ingest one activation tensor ``x`` of shape ``(*, C)``.

        All dimensions except the last are reduced; the last is treated
        as the channel dimension.
        """
        if self.ready:
            return  # warmup complete, ignore further calls

        # Flatten leading dims: (*, C) → (T, C)
        flat = x.reshape(-1, x.shape[-1]).float()

        # max-abs per channel
        batch_max = flat.abs().amax(dim=0)
        self._max_abs = torch.maximum(self._max_abs, batch_max)

        if self.risk_method == "dynamic_range":
            abs_flat = flat.abs()
            # Mask zeros with inf, then take per-channel min
            masked = abs_flat.where(
                abs_flat > 0,
                torch.tensor(float("inf"), device=flat.device),
            )
            batch_min = masked.amin(dim=0)
            self._min_nonzero = torch.minimum(self._min_nonzero, batch_min)

        self._count += 1

    # ------------------------------------------------------------------

    def finalize(self) -> torch.Tensor:
        """Return 1-D ``(C,)`` risk-score tensor.

        Raises ``RuntimeError`` if not enough samples have been collected.
        """
        if not self.ready:
            logger.warning(
                "ActivationCollector: finalizing with only %d/%d samples",
                self._count, self.warmup_samples,
            )

        if self.risk_method == "max_abs":
            return self._max_abs.clone()

        # dynamic_range
        eps = 1e-10
        min_nz = self._min_nonzero.clone()
        # Replace inf (all-zero channels) with eps
        min_nz = min_nz.where(min_nz.isfinite(),
                              torch.tensor(eps, dtype=torch.float32,
                                           device=min_nz.device))
        return self._max_abs / (min_nz + eps)
