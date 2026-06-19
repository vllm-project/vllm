# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Push-based custom allreduce using epoch-based 2-buffer protocol.
Ported from SGLang's CustomAllReduceV2 push allreduce.

Protocol:
  Phase 1 (push): Each rank writes its input data to ALL remote GPUs'
    push buffer regions via NVLink volatile stores. Positive zeros are
    converted to negative zeros to preserve sentinel semantics.
  Phase 2 (poll): Each rank polls its LOCAL buffer until all ranks'
    data arrives (no positive zeros remain), reduces in FP32, writes
    output, and resets buffer to positive zeros for next epoch.
"""

import logging
import os
from contextlib import contextmanager

import torch
import torch.distributed as dist

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.distributed.device_communicators.custom_all_reduce import (
    is_weak_contiguous,
)
from vllm.platforms import current_platform

logger = logging.getLogger(__name__)

_FEATURE_DESCRIPTION = "Push-based AllReduce"
_DISABLE_ENV_VAR = "VLLM_DISABLE_PUSH_ALLREDUCE"

# Push threshold maps: world_size -> buffer_bytes
# From SGLang's tuned thresholds for sm100 (B200)
PUSH_THRESHOLD_SM100 = {
    2: 4 * 1024 * 1024,  # 4 MB
    4: 2 * 1024 * 1024,  # 2 MB
    6: 1 * 1024 * 1024,  # 1 MB
    8: 720 * 1024,  # 720 KB
}

# Conservative default thresholds for architectures without tuned values
PUSH_THRESHOLD_DEFAULT = {
    2: 512 * 1024,  # 512 KB
    4: 512 * 1024,  # 512 KB
    6: 512 * 1024,  # 512 KB
    8: 512 * 1024,  # 512 KB
}

# Map GPU major compute capability -> threshold table
_THRESHOLD_BY_ARCH: dict[int, dict[int, int]] = {
    10: PUSH_THRESHOLD_SM100,  # Blackwell (sm_100)
}

# Conservative default for untuned GPUs / unrecognized world_size
DEFAULT_PUSH_BUFFER = 512 * 1024  # 512 KB


class PushAllReduce:
    """
    Push-based custom allreduce using epoch-based 2-buffer protocol.
    Ported from SGLang's CustomAllReduceV2 push allreduce.
    """

    _IS_CAPTURING = False

    def __init__(
        self,
        group: dist.ProcessGroup,
        device: torch.device,
        max_size: int | None = None,
    ):
        self.group = group
        self.device = device
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        self.disabled = False

        # Feature toggle: check env var to disable this feature
        if envs.VLLM_DISABLE_PUSH_ALLREDUCE:
            logger.info(
                "%s is DISABLED (env override)",
                _FEATURE_DESCRIPTION,
            )
            self.disabled = True
            return

        # Prerequisite checks
        if self.world_size not in (2, 4, 6, 8):
            logger.info(
                "PushAllReduce disabled: unsupported world_size=%d",
                self.world_size,
            )
            self.disabled = True
            return

        if not self._check_full_p2p():
            logger.info("PushAllReduce disabled: no full P2P connectivity")
            self.disabled = True
            return

        # Get SM count and architecture for grid size and threshold selection
        props = torch.cuda.get_device_properties(device)
        self.num_sm = props.multi_processor_count

        # Determine push buffer size from architecture-specific threshold map
        if max_size is not None:
            self.push_buffer_bytes = max_size
        else:
            arch_major = props.major
            threshold_map = _THRESHOLD_BY_ARCH.get(arch_major, PUSH_THRESHOLD_DEFAULT)
            self.push_buffer_bytes = threshold_map.get(
                self.world_size, DEFAULT_PUSH_BUFFER
            )
            if arch_major not in _THRESHOLD_BY_ARCH:
                logger.info(
                    "PushAllReduce: no tuned thresholds for sm_%d%d, "
                    "using conservative default (%d KB)",
                    arch_major,
                    props.minor,
                    self.push_buffer_bytes // 1024,
                )

        # Allow env var override (V1 fix: validate 128-byte alignment)
        env_override = os.environ.get("VLLM_PUSH_AR_BUFFER_BYTES")
        if env_override:
            val = int(env_override)
            if val == 0:
                logger.info("PushAllReduce disabled via VLLM_PUSH_AR_BUFFER_BYTES=0")
                self.disabled = True
                return
            # Round up to 128-byte alignment for kernel volatile stores
            self.push_buffer_bytes = ((val + 127) // 128) * 128

        self.max_message_bytes = self.push_buffer_bytes

        try:
            # Initialize C++ manager
            self._ptr = ops.init_push_ar(
                self.rank,
                self.world_size,
                self.push_buffer_bytes,
                self.num_sm,
            )

            # Exchange IPC handles across all ranks
            self._exchange_ipc_handles()

            logger.info(
                "PushAllReduce initialized: rank=%d, ws=%d, buffer=%d KB, sm=%d",
                self.rank,
                self.world_size,
                self.push_buffer_bytes // 1024,
                self.num_sm,
            )

            # Feature toggle: log that the feature is enabled
            logger.info(
                "%s is ENABLED",
                _FEATURE_DESCRIPTION,
            )
        except Exception as e:
            logger.warning("PushAllReduce init failed: %s", str(e))
            self.disabled = True

    def _check_full_p2p(self) -> bool:
        """Verify all GPUs have full P2P access (NVLink)."""
        num_dev = current_platform.device_count()
        for i in range(num_dev):
            for j in range(num_dev):
                if i != j and not torch.cuda.can_device_access_peer(i, j):
                    return False
        return True

    def _exchange_ipc_handles(self):
        """Exchange storage IPC handles across all ranks."""
        # Get local handle as byte tensor [sizeof(cudaIpcMemHandle_t)]
        local_handle = ops.get_push_ar_ipc_handle(self._ptr)  # shape (64,)

        # All-gather handles: each rank broadcasts its handle
        handle_list = [torch.empty_like(local_handle) for _ in range(self.world_size)]
        dist.all_gather(handle_list, local_handle, group=self.group)
        all_handles = torch.stack(handle_list)  # shape (world_size, 64)

        # Post-init opens peer IPC handles and creates PushController
        ops.post_init_push_ar(self._ptr, all_handles)

    def should_use(self, input_: torch.Tensor) -> bool:
        """Check if push allreduce should handle this input."""
        if self.disabled:
            return False
        inp_size = input_.numel() * input_.element_size()
        if inp_size == 0:
            return False
        if inp_size % 16 != 0:
            return False
        if not is_weak_contiguous(input_):
            return False
        return inp_size <= self.max_message_bytes

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor | None:
        """Perform push-based allreduce. Returns new output tensor."""
        if self._IS_CAPTURING:
            if torch.cuda.is_current_stream_capturing():
                # Actual CUDA graph capture: record kernel
                out = torch.empty_like(input_)
                ops.push_ar_all_reduce(self._ptr, input_, out)
                return out
            else:
                # Warmup before capture: mimic allocation pattern
                return torch.empty_like(input_)
        else:
            # Eager mode: launch kernel directly
            out = torch.empty_like(input_)
            ops.push_ar_all_reduce(self._ptr, input_, out)
            return out

    @contextmanager
    def capture(self):
        """Context manager for CUDA graph capture.

        SIMPLIFIED vs existing CustomAllreduce:
        The push protocol does NOT need graph buffer registration because
        it reads input LOCALLY and writes to pre-registered IPC push_buffers.
        This context just toggles the _IS_CAPTURING flag to control the
        warmup vs actual-capture behavior in all_reduce().
        """
        try:
            self._IS_CAPTURING = True
            yield
        finally:
            self._IS_CAPTURING = False

    def close(self):
        """Release C++ resources.

        V1 fix: Use hasattr(self, '_ptr') instead of 'not self.disabled'
        to handle the case where init_push_ar() succeeds but
        _exchange_ipc_handles() fails (sets disabled=True but _ptr exists).
        """
        if hasattr(self, "_ptr"):
            ops.dispose_push_ar(self._ptr)
            del self._ptr
            self.disabled = True

    def __del__(self):
        self.close()
