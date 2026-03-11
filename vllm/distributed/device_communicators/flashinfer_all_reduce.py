# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

import vllm.envs as envs
from vllm.config.compilation import PassConfig
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

# Global workspace for standalone allreduce and non-quant ar+rms fusion
_fi_ar_workspace = None
# Extra workspace for quant fusion patterns (only supported by trtllm backend)
# Only created if primary workspace is not already trtllm
_fi_ar_quant_workspace = None


def get_fi_ar_workspace():
    return _fi_ar_workspace


def get_fi_ar_quant_workspace():
    return _fi_ar_quant_workspace


def initialize_fi_ar_workspace(
    world_size: int,
    rank: int,
    max_token_num: int,
    hidden_dim: int,
    dtype: torch.dtype,
    group: ProcessGroup,
) -> None:
    """
    Initialize the workspace if not already initialized.

    Currently, this function is called by either the AllReduceFusionPass
    or the FlashInferAllReduce backend for standalone allreduce.
    If the fusion pass is enabled via
    --compilation-config.pass_config.fuse_allreduce_rms=true,
    it will create the workspace first, and the standalone backend
    will reuse the workspace. Otherwise, the standalone backend will
    create the workspace.
    """
    global _fi_ar_workspace
    if _fi_ar_workspace is not None:
        return

    backend = envs.VLLM_FLASHINFER_ALLREDUCE_BACKEND
    comm_backend = TorchDistBackend(group=group)
    _fi_ar_workspace = flashinfer_comm.create_allreduce_fusion_workspace(
        backend=backend,
        world_size=world_size,
        rank=rank,
        max_token_num=max_token_num,
        hidden_dim=hidden_dim,
        dtype=dtype,
        comm_backend=comm_backend,
    )
    assert _fi_ar_workspace is not None
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


def initialize_fi_ar_quant_workspace(
    world_size: int,
    rank: int,
    max_token_num: int,
    hidden_dim: int,
    dtype: torch.dtype,
    group: ProcessGroup,
) -> None:
    """
    Initialize the workspace used by quantization fusion patterns.

    Currently this always creates a workspace for trtllm backend as only it
    supports quantization fusion (FP8/FP4). If the primary workspace
    is already trtllm, the quant workspace aliases to it.
    """
    global _fi_ar_quant_workspace
    if _fi_ar_quant_workspace is not None:
        return

    # If primary workspace is already trtllm, reuse it
    if _fi_ar_workspace is not None and _fi_ar_workspace.backend == "trtllm":
        _fi_ar_quant_workspace = _fi_ar_workspace
        return

    comm_backend = TorchDistBackend(group=group)
    _fi_ar_quant_workspace = flashinfer_comm.create_allreduce_fusion_workspace(
        backend="trtllm",
        world_size=world_size,
        rank=rank,
        max_token_num=max_token_num,
        hidden_dim=hidden_dim,
        dtype=dtype,
        comm_backend=comm_backend,
    )
    assert _fi_ar_quant_workspace is not None
    logger.debug(
        "Initialized FlashInfer All Reduce workspace: backend=trtllm, "
        "world_size=%d, rank=%d, max_token_num=%d, hidden_dim=%d, dtype=%s",
        world_size,
        rank,
        max_token_num,
        hidden_dim,
        dtype,
    )


def destroy_fi_ar_workspace():
    global _fi_ar_workspace
    global _fi_ar_quant_workspace
    if (
        _fi_ar_quant_workspace is not None
        and _fi_ar_quant_workspace is not _fi_ar_workspace
    ):
        _fi_ar_quant_workspace.destroy()
    _fi_ar_quant_workspace = None
    if _fi_ar_workspace is not None:
        _fi_ar_workspace.destroy()
        _fi_ar_workspace = None


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
        if get_fi_ar_workspace() is not None:
            return True
        if self.max_num_tokens == 0:
            element_size = torch.tensor([], dtype=dtype, device="cpu").element_size()
            self.max_num_tokens = self.max_workspace_size // (hidden_dim * element_size)
        try:
            initialize_fi_ar_workspace(
                world_size=self.world_size,
                rank=self.rank,
                max_token_num=self.max_num_tokens,
                hidden_dim=hidden_dim,
                dtype=dtype,
                group=self.group,
            )
            return True
        except Exception as e:
            logger.warning(
                "Failed to initialize FlashInfer All Reduce workspace: %s. "
                "FlashInfer All Reduce will be disabled.",
                e,
            )
            self.disabled = True
            return False

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
        workspace = get_fi_ar_workspace()
        return flashinfer_comm.allreduce_fusion(
            input=input_tensor,
            workspace=workspace,
            pattern=flashinfer_comm.AllReduceFusionPattern.kAllReduce,
        )

    def destroy(self):
        if not self.disabled:
            destroy_fi_ar_workspace()
