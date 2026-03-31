# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import atexit
import os
import random
import threading

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

import vllm.envs as envs
from vllm.config.compilation import PassConfig
from vllm.distributed.parallel_state import get_node_count
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)


fi_ar_available = False
try:
    import flashinfer.comm as flashinfer_comm  # type: ignore[no-redef]
    from flashinfer.comm.mnnvl import (
        TorchDistBackend,  # type: ignore[import-not-found, no-redef]
    )

    fi_ar_available = hasattr(flashinfer_comm, "allreduce_fusion")
except ImportError:
    pass

# Workspace for standalone allreduce and non-quant ar+rms fusion
_fi_ar_workspace = None
# Extra workspace for quant fusion patterns (only supported by trtllm backend)
_fi_ar_quant_workspace = None


def _create_workspace(
    backend: str,
    world_size: int,
    rank: int,
    max_token_num: int,
    hidden_dim: int,
    dtype: torch.dtype,
    group: ProcessGroup,
):
    """Create a flashinfer allreduce workspace, returning None on failure."""
    comm_backend = TorchDistBackend(group=group)
    rng_state = random.getstate()
    try:
        random.seed(int.from_bytes(os.urandom(16), byteorder="big"))
        workspace = flashinfer_comm.create_allreduce_fusion_workspace(
            backend=backend,
            world_size=world_size,
            rank=rank,
            max_token_num=max_token_num,
            hidden_dim=hidden_dim,
            dtype=dtype,
            comm_backend=comm_backend,
        )
    except Exception as e:
        if "multicast" in str(e).lower():
            logger.warning_once(
                "Failed to initialize FlashInfer All Reduce workspace: %s. "
                "This is expected on GPUs without NVSwitch (e.g., NVLink "
                "bridge-only or PCIe topologies).",
                e,
            )
        else:
            logger.warning_once(
                "Failed to initialize FlashInfer All Reduce workspace: %s.",
                e,
            )
        return None
    finally:
        random.setstate(rng_state)
    logger.debug(
        "Initialized FlashInfer All Reduce workspace: backend=%s, "
        "world_size=%d, rank=%d, max_token_num=%d, hidden_dim=%d, dtype=%s",
        backend,
        world_size,
        rank,
        max_token_num,
        hidden_dim,
        dtype,
    )
    return workspace


def _resolve_fi_ar_backend() -> str:
    backend = envs.VLLM_FLASHINFER_ALLREDUCE_BACKEND
    if backend != "auto":
        logger.info_once(f"Using flashinfer allreduce backend: {backend}")
        return backend

    if get_node_count() > 1:  # noqa: SIM108
        # Use mnnvl backend for multi-node setup since
        # trtllm backend does not support multi-node allreduce
        backend = "mnnvl"
    else:
        # Currently defaulting to trtllm backend for single-node
        # setup since mnnvl has issues with cudagraph:
        # https://github.com/vllm-project/vllm/issues/35772
        # Should switch back to auto when the issue is resolved.
        backend = "trtllm"

    logger.info_once(f"Auto-selected flashinfer allreduce backend: {backend}")
    return backend


def get_fi_ar_workspace(
    world_size: int,
    rank: int,
    max_token_num: int,
    hidden_dim: int,
    dtype: torch.dtype,
    group: ProcessGroup,
):
    """
    Return the allreduce workspace for non-quant patterns, initializing if needed.

    Used by AllReduceFusionPass (non-quant patterns) and FlashInferAllReduce
    for standalone allreduce. Backend is controlled by
    VLLM_FLASHINFER_ALLREDUCE_BACKEND env var.
    """
    global _fi_ar_workspace
    if _fi_ar_workspace is not None:
        return _fi_ar_workspace

    backend = _resolve_fi_ar_backend()

    if get_node_count() > 1 and backend == "trtllm":
        raise ValueError(
            "Flashinfer allreduce is not supported for multi-node allreduce with "
            "'trtllm' backend. Please use 'mnnvl' backend instead."
        )

    # Reuse the quant workspace if it was already created with the same backend
    if _fi_ar_quant_workspace is not None and _fi_ar_quant_workspace.backend == backend:
        _fi_ar_workspace = _fi_ar_quant_workspace
        return _fi_ar_workspace

    _fi_ar_workspace = _create_workspace(
        backend, world_size, rank, max_token_num, hidden_dim, dtype, group
    )
    if _fi_ar_workspace is not None:
        logger.info_once(
            "Initialized FlashInfer Allreduce norm fusion workspace "
            f"with backend={backend}"
        )
    else:
        logger.warning_once(
            "Failed to initialize FlashInfer Allreduce norm fusion workspace "
            f"with backend={backend}"
        )

    return _fi_ar_workspace


def get_fi_ar_quant_workspace(
    world_size: int,
    rank: int,
    max_token_num: int,
    hidden_dim: int,
    dtype: torch.dtype,
    group: ProcessGroup,
):
    """
    Return the allreduce workspace for quant patterns, initializing if needed.

    Always uses trtllm backend as it is the only one supporting quantization
    fusion (FP8/FP4). Returns None for multi-node setups since not supported
    by trtllm backend.
    """
    global _fi_ar_quant_workspace
    if _fi_ar_quant_workspace is not None:
        return _fi_ar_quant_workspace

    if get_node_count() > 1:
        logger.warning_once(
            "Flashinfer allreduce quantization fusion is not supported for "
            "multi-node allreduce. Disabling quant fusion."
        )
        return None

    # Reuse the non-quant workspace if it was already created with trtllm
    if _fi_ar_workspace is not None and _fi_ar_workspace.backend == "trtllm":
        _fi_ar_quant_workspace = _fi_ar_workspace
        return _fi_ar_quant_workspace

    _fi_ar_quant_workspace = _create_workspace(
        "trtllm", world_size, rank, max_token_num, hidden_dim, dtype, group
    )
    if _fi_ar_quant_workspace is not None:
        logger.info_once(
            "Initialized FlashInfer Allreduce norm quantization "
            "fusion workspace with backend=trtllm"
        )
    else:
        logger.warning_once(
            "Failed to initialize FlashInfer Allreduce norm quantization "
            "fusion workspace with backend=trtllm"
        )

    return _fi_ar_quant_workspace


_fi_ar_workspace_lock = threading.Lock()


def destroy_fi_ar_workspace():
    global _fi_ar_workspace, _fi_ar_quant_workspace
    with _fi_ar_workspace_lock:
        is_alias = _fi_ar_workspace is _fi_ar_quant_workspace

        if _fi_ar_workspace is not None:
            _fi_ar_workspace.destroy()
        if _fi_ar_quant_workspace is not None and not is_alias:
            _fi_ar_quant_workspace.destroy()

        _fi_ar_workspace = _fi_ar_quant_workspace = None


atexit.register(destroy_fi_ar_workspace)


class FlashInferAllReduce:
    def __init__(
        self,
        group: ProcessGroup,
        device: int | str | torch.device,
    ):
        self.disabled = True

        if not fi_ar_available:
            logger.info(
                "FlashInfer All Reduce is disabled because flashinfer is not available"
            )
            return

        if not current_platform.is_cuda():
            logger.info(
                "FlashInfer All Reduce is disabled because it requires CUDA platform"
            )
            return

        self.group = group
        self.world_size = dist.get_world_size(self.group)
        self.rank = dist.get_rank(self.group)
        self.device = device
        if self.world_size == 1:
            return

        # Use the same threshold as the allreduce-rms fusion pass
        # TODO: tune the threshold
        MiB = 1024 * 1024
        max_workspace_size = PassConfig.default_fi_allreduce_fusion_max_size_mb().get(
            self.world_size, None
        )
        if not max_workspace_size:
            logger.warning(
                "FlashInfer All Reduce is disabled because it "
                "is not supported for world_size=%d.",
                self.world_size,
            )
            return
        self.max_workspace_size = max_workspace_size * MiB
        self.max_num_tokens = 0
        self.disabled = False

    def _ensure_workspace(self, hidden_dim: int, dtype: torch.dtype) -> bool:
        """Ensure the all reduce workspace is initialized."""
        if self.max_num_tokens == 0:
            element_size = torch.tensor([], dtype=dtype, device="cpu").element_size()
            self.max_num_tokens = self.max_workspace_size // (hidden_dim * element_size)
        workspace = get_fi_ar_workspace(
            world_size=self.world_size,
            rank=self.rank,
            max_token_num=self.max_num_tokens,
            hidden_dim=hidden_dim,
            dtype=dtype,
            group=self.group,
        )
        if workspace is None:
            self.disabled = True
            return False
        return True

    def should_use_fi_ar(self, input_tensor: torch.Tensor) -> bool:
        if self.disabled:
            return False

        if not input_tensor.is_cuda:
            return False

        if not input_tensor.is_contiguous():
            return False

        if len(input_tensor.shape) != 2:
            return False

        num_tokens, hidden_dim = input_tensor.shape
        if not self.max_num_tokens:
            element_size = torch.tensor([], dtype=input_tensor.dtype).element_size()
            self.max_num_tokens = self.max_workspace_size // (hidden_dim * element_size)

        if num_tokens > self.max_num_tokens:
            return False

        return self._ensure_workspace(hidden_dim, input_tensor.dtype)

    def all_reduce(self, input_tensor: torch.Tensor) -> torch.Tensor:
        _, hidden_dim = input_tensor.shape
        workspace = get_fi_ar_workspace(
            world_size=self.world_size,
            rank=self.rank,
            max_token_num=self.max_num_tokens,
            hidden_dim=hidden_dim,
            dtype=input_tensor.dtype,
            group=self.group,
        )
        return flashinfer_comm.allreduce_fusion(
            input=input_tensor,
            workspace=workspace,
            pattern=flashinfer_comm.AllReduceFusionPattern.kAllReduce,
        )

    def destroy(self):
        if not self.disabled:
            destroy_fi_ar_workspace()
