# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Zero-overhead NaN/Inf detection via RMSNorm kernel instrumentation
and pluggable tensor checks."""

from __future__ import annotations

import torch
import torch.nn as nn

from vllm.logger import init_logger

logger = init_logger(__name__)


def _as_fp8(data: torch.Tensor) -> torch.Tensor:
    """View uint8 KV cache data as fp8 so torch.isnan works."""
    if data.dtype == torch.uint8:
        return data.view(torch.float8_e4m3fn)
    return data


class NaNDetector:
    """Manages per-token NaN/Inf detection flags.

    Singleton. Created lazily when ``VLLM_NAN_DETECT=1``.

    The flag array has shape ``int8[num_checkpoints, max_num_tokens]``.
    Checkpoints can be:

    * **RMSNorm layers** -- the CUDA kernel writes flags via a pointer
      argument (zero-cost when disabled).
    * **Arbitrary tensors** -- call :meth:`check_tensor` which uses
      ``torch.isfinite`` to check for NaN/Inf.  CUDA-graph compatible.

    Both share the same flag array and reporting path.
    """

    _instance: NaNDetector | None = None

    def __init__(self) -> None:
        self._counter: int = 0
        self._layer_names: dict[int, str] = {}
        self._max_num_tokens: int = 0
        self._nan_flags: torch.Tensor | None = None
        self._host_flags: torch.Tensor | None = None
        self._finalized: bool = False
        self._kv_caches: list[torch.Tensor] = []

    @classmethod
    def get(cls) -> NaNDetector:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None

    # ------------------------------------------------------------------
    # Registration (before finalize)
    # ------------------------------------------------------------------

    def register(self, name: str) -> int:
        """Register a checkpoint.  Returns its index into the flag array.

        Works for both RMSNorm layers (which pass the index to the CUDA
        kernel) and arbitrary tensor checks (which pass it to
        :meth:`check_tensor`).
        """
        assert not self._finalized, (
            "Cannot register new checkpoints after NaNDetector.finalize()"
        )
        idx = self._counter
        self._layer_names[idx] = name
        self._counter += 1
        return idx

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_checkpoints(self) -> int:
        return self._counter

    @property
    def nan_flags(self) -> torch.Tensor | None:
        return self._nan_flags

    @property
    def max_num_tokens(self) -> int:
        return self._max_num_tokens

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def finalize(
        self,
        device: torch.device,
        max_num_tokens: int,
        kv_caches: list[torch.Tensor] | None = None,
    ) -> None:
        """Allocate ``int8[num_checkpoints, max_num_tokens]`` flag tensors."""
        if self._finalized:
            return
        n = self._counter
        if n == 0:
            logger.warning(
                "NaNDetector.finalize() called but nothing registered"
            )
            return
        self._max_num_tokens = max_num_tokens
        self._nan_flags = torch.zeros(
            n, max_num_tokens, dtype=torch.int8, device=device
        )
        self._host_flags = torch.zeros(
            n, max_num_tokens, dtype=torch.int8
        ).pin_memory()
        if kv_caches is not None:
            self._kv_caches = kv_caches
        self._finalized = True
        logger.info(
            "NaN/Inf detector initialized: %d checkpoints, max %d tokens "
            "(%.1f KB flag buffer)",
            n,
            max_num_tokens,
            n * max_num_tokens / 1024,
        )

    def update_layer_names(self, model: nn.Module) -> None:
        """Walk model modules to assign readable names."""
        from vllm.model_executor.layers.layernorm import RMSNorm

        for name, module in model.named_modules():
            if isinstance(module, RMSNorm) and hasattr(
                module, "_nan_detect_layer_idx"
            ):
                idx = module._nan_detect_layer_idx
                if idx in self._layer_names:
                    self._layer_names[idx] = name
            if hasattr(module, "_nan_detect_indices"):
                for attr_label, idx in module._nan_detect_indices.items():
                    if idx in self._layer_names:
                        self._layer_names[idx] = f"{name}.{attr_label}"

    # ------------------------------------------------------------------
    # Per-step operations
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Zero flags before each forward pass."""
        if self._nan_flags is not None:
            self._nan_flags.zero_()

    @torch.compiler.disable
    def check_tensor(
        self, tensor: torch.Tensor, checkpoint_idx: int
    ) -> None:
        """Check *tensor* for NaN/Inf, writing per-token flags.

        Uses ``torch.isfinite`` -- all ops stay on GPU, no D2H sync.
        CUDA-graph compatible (fixed output address).
        Decorated with ``@torch.compiler.disable`` to avoid graph breaks.

        Args:
            tensor: 2-D ``[num_tokens, hidden_size]`` tensor to check.
            checkpoint_idx: index returned by :meth:`register`.
        """
        if self._nan_flags is None:
            return
        num_tokens = tensor.shape[0]
        has_bad = (~torch.isfinite(tensor.view(num_tokens, -1))).any(dim=1)
        self._nan_flags[checkpoint_idx, :num_tokens].bitwise_or_(
            has_bad.to(torch.int8)
        )

    # ------------------------------------------------------------------
    # Post-forward checking
    # ------------------------------------------------------------------

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
                name = self._layer_names.get(
                    layer_idx, f"checkpoint_{layer_idx}"
                )
                logger.error(
                    "NaN/Inf detected in real tokens at '%s' "
                    "(checkpoint %d), token positions: %s",
                    name,
                    layer_idx,
                    token_positions,
                )

        pad_bad = pad_flags.any(dim=1).nonzero(as_tuple=True)[0]
        if len(pad_bad) > 0:
            logger.debug(
                "NaN/Inf in padding tokens at %d checkpoints",
                len(pad_bad),
            )

        if len(real_bad) > 0:
            raise RuntimeError(
                f"NaN/Inf detected at {len(real_bad)} checkpoint(s). "
                "See ERROR logs above for details."
            )

    # ------------------------------------------------------------------
    # KV cache checks (on block assignment)
    # ------------------------------------------------------------------

    def check_kv_blocks(self, block_ids: list[int]) -> None:
        """Check specific KV cache blocks for NaN/Inf.

        Called when blocks are recycled from the pool to a new request,
        to detect stale NaN left by a previous request.
        """
        if not block_ids or not self._kv_caches:
            return

        device = self._kv_caches[0].device
        indices = torch.tensor(block_ids, device=device, dtype=torch.long)

        for group_idx, kv_cache in enumerate(self._kv_caches):
            if not isinstance(kv_cache, torch.Tensor):
                continue
            if indices.max().item() >= kv_cache.shape[0]:
                continue
            blocks = _as_fp8(kv_cache[indices])
            nan_count = torch.isnan(blocks).sum().item()
            if nan_count > 0:
                logger.error(
                    "Stale NaN in recycled KV cache blocks "
                    "(group %d, block_ids=%s, nan_count=%d, "
                    "total_elements=%d)",
                    group_idx,
                    block_ids[:10],
                    nan_count,
                    blocks.numel(),
                )
