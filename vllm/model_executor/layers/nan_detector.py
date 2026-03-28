# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Zero-overhead NaN/Inf detection via RMSNorm kernel instrumentation."""

from __future__ import annotations

import torch
import torch.nn as nn

from vllm.logger import init_logger

logger = init_logger(__name__)


class NaNDetector:
    """Manages per-token NaN/Inf detection flags for RMSNorm kernels.

    Singleton. Created lazily when VLLM_NAN_DETECT=1.

    The flag array has shape ``int8[num_layers, max_num_tokens]``.
    Each RMSNorm CUDA kernel block (one per token) writes
    ``flag[layer_idx][blockIdx.x] = 1`` when ``isnan(variance) ||
    isinf(variance)`` after the CUB reduction — zero cost when the
    flag pointer is NULL (the default).
    """

    _instance: NaNDetector | None = None

    def __init__(self) -> None:
        self._counter: int = 0
        self._layer_names: dict[int, str] = {}
        self._max_num_tokens: int = 0
        self._nan_flags: torch.Tensor | None = None
        self._host_flags: torch.Tensor | None = None
        self._finalized: bool = False

    @classmethod
    def get(cls) -> NaNDetector:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None

    def register(self, name: str) -> int:
        """Called from ``RMSNorm.__init__``.  Returns layer index."""
        assert not self._finalized, (
            "Cannot register new layers after NaNDetector.finalize()"
        )
        idx = self._counter
        self._layer_names[idx] = name
        self._counter += 1
        return idx

    @property
    def num_layers(self) -> int:
        return self._counter

    @property
    def nan_flags(self) -> torch.Tensor | None:
        return self._nan_flags

    @property
    def max_num_tokens(self) -> int:
        return self._max_num_tokens

    def finalize(self, device: torch.device, max_num_tokens: int) -> None:
        """Allocate ``int8[num_layers, max_num_tokens]`` flag tensors."""
        if self._finalized:
            return
        n = self._counter
        if n == 0:
            logger.warning("NaNDetector.finalize() called but no layers registered")
            return
        self._max_num_tokens = max_num_tokens
        self._nan_flags = torch.zeros(
            n, max_num_tokens, dtype=torch.int8, device=device
        )
        self._host_flags = torch.zeros(n, max_num_tokens, dtype=torch.int8).pin_memory()
        self._finalized = True
        logger.info(
            "NaN/Inf detector initialized: %d layers, max %d tokens "
            "(%.1f KB flag buffer)",
            n,
            max_num_tokens,
            n * max_num_tokens / 1024,
        )

    def update_layer_names(self, model: nn.Module) -> None:
        """Walk model modules to assign readable names."""
        from vllm.model_executor.layers.layernorm import RMSNorm

        for name, module in model.named_modules():
            if isinstance(module, RMSNorm) and hasattr(module, "_nan_detect_layer_idx"):
                idx = module._nan_detect_layer_idx
                if idx in self._layer_names:
                    self._layer_names[idx] = name

    def clear(self) -> None:
        """Zero flags before each forward pass."""
        if self._nan_flags is not None:
            self._nan_flags.zero_()

    def check(self, num_real_tokens: int) -> None:
        """D2H copy flags, scan, log results.

        Args:
            num_real_tokens: Number of real (non-padding) tokens in the
                current batch.  Flags beyond this index are padding.
        """
        if self._nan_flags is None or self._host_flags is None:
            return

        self._host_flags.copy_(self._nan_flags, non_blocking=False)

        real_flags = self._host_flags[:, :num_real_tokens]
        pad_flags = self._host_flags[:, num_real_tokens:]

        real_bad = real_flags.any(dim=1).nonzero(as_tuple=True)[0]
        if len(real_bad) > 0:
            for layer_idx in real_bad.tolist():
                token_positions = (
                    real_flags[layer_idx].nonzero(as_tuple=True)[0].tolist()
                )
                name = self._layer_names.get(layer_idx, f"layer_{layer_idx}")
                logger.error(
                    "NaN/Inf detected in real tokens at '%s' "
                    "(layer %d), token positions: %s",
                    name,
                    layer_idx,
                    token_positions,
                )

        pad_bad = pad_flags.any(dim=1).nonzero(as_tuple=True)[0]
        if len(pad_bad) > 0:
            for layer_idx in pad_bad.tolist():
                name = self._layer_names.get(layer_idx, f"layer_{layer_idx}")
                logger.warning(
                    "NaN/Inf detected in PADDING tokens at '%s' "
                    "(layer %d) — may leak into real data",
                    name,
                    layer_idx,
                )
