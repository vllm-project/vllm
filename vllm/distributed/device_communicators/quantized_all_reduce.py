# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Quantized all-reduce communicator using two-shot per-group quantized kernels.

Enable via environment variable:
    VLLM_ALLREDUCE_QUANTIZATION=int8   # int8 quantization
    VLLM_ALLREDUCE_QUANTIZATION=fp8    # fp8 quantization
"""

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

# Symmetric memory buffer size constants (in elements).
_MAX_BUFFER_NUMEL = 2 * (1024**3)  # absolute cap (2 GiB packed)
_DEFAULT_BUFFER_NUMEL = 16 * 1024 * 8192  # fallback when model config unavailable


def _estimate_max_numel() -> int:
    """Derive buffer size from model config, falling back to a default."""
    try:
        from vllm.config import get_current_vllm_config_or_none

        config = get_current_vllm_config_or_none()
        if config is not None:
            hidden_size = config.model_config.get_hidden_size()
            max_tokens = config.scheduler_config.max_num_batched_tokens
            numel = int(max_tokens * hidden_size * 1.05)  # 5% headroom
            numel = min(numel, _MAX_BUFFER_NUMEL)
            logger.info(
                "QuantizedAllReduce: buffer sized from model config: "
                "max_tokens=%d, hidden_size=%d, numel=%d",
                max_tokens,
                hidden_size,
                numel,
            )
            return numel
    except Exception:
        pass
    # Fallback when model config is not available
    return _DEFAULT_BUFFER_NUMEL


class QuantizedAllReduceCommunicator:
    """Two-shot quantized all-reduce using symmetric memory."""

    # Minimum numel (element count) thresholds where quantized kernels
    # beat existing backends. Determined empirically on H200.
    # {mode: {world_size: min_elements}}
    _MIN_NUMEL = {
        "int8": {8: 8_388_608, 4: 2_097_152},  # 16 MiB / 4 MiB in bf16
        "fp8": {8: 8_388_608, 4: 2_097_152},  # 16 MiB / 4 MiB in bf16
    }

    def __init__(
        self,
        group: ProcessGroup,
        device: int | str | torch.device,
        max_size_override: int | None = None,
    ):
        self.disabled = True
        self.group = group
        self.world_size = dist.get_world_size(self.group)

        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device

        mode = envs.VLLM_ALLREDUCE_QUANTIZATION.lower()
        if mode == "none":
            return

        try:
            from vllm.distributed.device_communicators.quantized_allreduce import (
                two_shot_quantized_allreduce,
            )
            from vllm.distributed.device_communicators.quantized_allreduce.two_shot_quantized_allreduce import (  # noqa: E501
                DEFAULT_GROUP_SIZE,
                MAX_BLOCK_SIZE,
                _State,
            )
        except ImportError:
            logger.warning(
                "QuantizedAllReduceCommunicator: quantized_allreduce module not found"
            )
            return

        if MAX_BLOCK_SIZE % self.world_size != 0:
            logger.info(
                "QuantizedAllReduceCommunicator: disabled for "
                "world_size=%d (not compatible with MAX_BLOCK_SIZE=%d)",
                self.world_size,
                MAX_BLOCK_SIZE,
            )
            return

        if mode in ("int8", "fp8"):
            self._kernel_fn = two_shot_quantized_allreduce
            self._use_fp8 = mode == "fp8"
            self._mode = mode
        else:
            logger.warning(
                "QuantizedAllReduceCommunicator: unknown mode '%s', disabling", mode
            )
            return

        # Pre-allocate symmetric memory buffer
        max_numel = (
            max_size_override
            if max_size_override is not None
            else _estimate_max_numel()
        )
        self._group_size = DEFAULT_GROUP_SIZE
        alignment = self._group_size * self.world_size
        max_numel = ((max_numel + alignment - 1) // alignment) * alignment
        try:
            self._state = _State(
                max_numel, self._group_size, self.device, group=self.group
            )
            logger.info(
                "QuantizedAllReduceCommunicator: enabled (%s), %d GPUs, "
                "buffer max_numel=%d (%d MiB packed)",
                self._mode,
                self.world_size,
                max_numel,
                self._state.packed_size // (1024**2),
            )
        except Exception as e:
            logger.warning(
                "QuantizedAllReduceCommunicator: failed to allocate buffer: %s", e
            )
            return

        self.disabled = False

    def should_use_quantized(self, inp: torch.Tensor) -> bool:
        if self.disabled:
            return False
        if inp.dtype != torch.bfloat16:
            return False
        numel = inp.numel()
        if numel % 8 != 0:
            return False
        if numel > self._state.max_numel:
            return False
        ws_key = 8 if self.world_size >= 8 else 4
        min_numel = self._MIN_NUMEL[self._mode][ws_key]
        return numel >= min_numel

    def all_reduce(self, inp: torch.Tensor) -> torch.Tensor | None:
        if not self.should_use_quantized(inp):
            return None
        return self._kernel_fn(inp, use_fp8=self._use_fp8, state=self._state)  # type: ignore[operator]
