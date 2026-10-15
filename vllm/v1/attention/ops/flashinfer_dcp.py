# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adapter for FlashInfer #6483's NCCL DCP combine with mask/stride support."""

from __future__ import annotations

import functools
import importlib
import inspect
import threading
from importlib.metadata import version
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from packaging.version import Version

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator


@functools.cache
def _load_fi_dcp(device: torch.device) -> Any:
    # Update the minimum version once FlashInfer #6483 is included in a release.
    if Version(version("flashinfer-python")) < Version("0.7.1rc5"):
        raise RuntimeError(
            "fi_a2a_fused requires flashinfer-python>=0.7.1rc5 with #6483 changes"
        )
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("fi_a2a_fused requires CUDA")
    if device.index != torch.accelerator.current_device_index():
        raise RuntimeError("FI DCP workspace device must be the current CUDA device")
    if torch.cuda.get_device_capability(device)[0] < 9:
        raise RuntimeError("fi_a2a_fused requires SM90 or newer")
    if torch.cuda.nccl.version() < (2, 29, 0):
        raise RuntimeError("fi_a2a_fused requires NCCL >= 2.29")
    symm_mem = importlib.import_module("torch.distributed._symmetric_memory")
    # Check the compiled PyTorch/NCCL interface before entering FI's collective
    # allocation. An import alone does not prove these C++ APIs are available.
    module = importlib.import_module("flashinfer.comm.dcp_lse_reduce")
    if "mask" not in inspect.signature(module.decode_cp_a2a_lse_reduce).parameters:
        raise RuntimeError("fi_a2a_fused requires FlashInfer #6483 mask/stride support")
    spec = module.gen_dcp_lse_reduce_module()
    # #5726 keeps the API names but changes the transport and workspace layout.
    # #4929's smallest workspace is 32 metadata bytes + 2*2*(16 O + 4 LSE).
    if (
        "-DUSE_NCCL" not in (spec.extra_cuda_cflags or [])
        or module.decode_cp_a2a_lse_reduce_workspace_size(1, 1, 2, 8, torch.float16)
        != 112
    ):
        raise RuntimeError(
            "fi_a2a_fused requires FlashInfer's NCCL device-API DCP implementation "
            "with the #4929 workspace layout"
        )
    module.get_dcp_lse_reduce_module()
    try:
        symm_mem.set_backend("NCCL")
    except RuntimeError as error:
        raise RuntimeError(
            "fi_a2a_fused needs the NCCL torch symmetric-memory backend before "
            "any other symmetric allocation; an existing backend cannot be "
            "changed after use"
        ) from error
    return module


class FlashInferDCPWorkspace:
    """Own collective workspaces per observed stream and DBO slot.

    Each stream must execute an eager combine (or ``prepare``) on every rank
    before capture. Workspaces and stream references stay alive for graph replay.
    """

    def __init__(
        self,
        group: GroupCoordinator,
        device: torch.device,
        max_num_tokens: int,
        heads_per_rank: int,
        head_dim: int,
        dtype: torch.dtype,
        num_ubatches: int,
    ) -> None:
        self.group = group
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.accelerator.current_device_index())
        self.world_size = group.world_size
        self.rank = group.rank_in_group
        self.max_num_tokens = max_num_tokens
        self.heads_per_rank = heads_per_rank
        self.head_dim = head_dim
        self.dtype = dtype
        self.num_ubatches = num_ubatches
        self._streams: dict[int, torch.cuda.Stream] = {}
        self._workspaces: dict[int, list[torch.Tensor]] = {}
        self._prepare_lock = threading.Lock()

        error = None
        try:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("FI DCP initialization must occur outside capture")
            if not 2 <= self.world_size <= 64:
                raise ValueError("FI DCP requires 2 <= DCP <= 64")
            if min(max_num_tokens, heads_per_rank, head_dim, num_ubatches) <= 0:
                raise ValueError("FI DCP workspace geometry must be positive")
            if dtype not in (torch.float16, torch.bfloat16):
                raise ValueError("FI DCP requires float16 or bfloat16")
            if head_dim % 8:
                raise ValueError("FI DCP head rows must be 16-byte aligned")
            self._fi = _load_fi_dcp(self.device)
        except Exception as exception:
            error = str(exception)
        geometry = (max_num_tokens, heads_per_rank, head_dim, dtype, num_ubatches)
        status: list[Any] = [None] * self.world_size
        dist.all_gather_object(status, (error, geometry), group=group.cpu_group)
        if any(item[0] is not None for item in status):
            raise RuntimeError(f"FI DCP preflight failed across ranks: {status}")
        if any(item[1] != geometry for item in status):
            raise RuntimeError(
                f"FI DCP workspace geometry differs across ranks: {status}"
            )
        # NCCL symmetric rendezvous requires the DCP process group's host
        # communicator, which PyTorch creates on its first device collective.
        dist.barrier(
            group=group.device_group,
            device_ids=[torch.accelerator.current_device_index()],
        )
        self.prepare()

    def prepare(self) -> None:
        """Collectively allocate and warm every DBO slot on the current stream."""
        stream = torch.cuda.current_stream(self.device)
        stream_id = int(stream.cuda_stream)
        with self._prepare_lock:
            if stream_id in self._workspaces:
                return
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "FI DCP stream was not warmed before CUDA graph capture; "
                    "run an eager combine on every rank on this stream first"
                )
            workspaces = []
            for _ in range(self.num_ubatches):
                workspace = self._fi.decode_cp_a2a_lse_reduce_create_workspace(
                    self.max_num_tokens,
                    self.heads_per_rank,
                    self.world_size,
                    self.head_dim,
                    self.dtype,
                    self.group.device_group,
                )
                # Initialize the NCCL device communicator and exercise strided
                # source addressing and the token mask before capture.
                output = torch.full(
                    (1, self.world_size, self.heads_per_rank, self.head_dim),
                    float("nan"),
                    dtype=self.dtype,
                    device=self.device,
                ).transpose(1, 2)
                lse = torch.full(
                    (1, self.world_size, self.heads_per_rank),
                    0.0,
                    dtype=torch.float32,
                    device=self.device,
                ).transpose(1, 2)
                self._fi.decode_cp_a2a_lse_reduce(
                    output,
                    lse,
                    workspace,
                    self.rank,
                    self.world_size,
                    mask=torch.zeros(1, dtype=torch.bool, device=self.device),
                )
                stream.synchronize()
                workspaces.append(workspace)
            self._streams[stream_id] = stream
            self._workspaces[stream_id] = workspaces

    def lse_reduce(
        self,
        partial_output: torch.Tensor,
        partial_lse: torch.Tensor,
        is_lse_base_on_e: bool,
        seq_lens: torch.Tensor | None = None,
        query_start_loc: torch.Tensor | None = None,
        valid_token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Exchange destination-major vLLM heads and return the merged output."""
        from vllm.v1.attention.ops.dcp import _validate_dcp_empty_shard_args
        from vllm.v1.worker.ubatching import dbo_current_ubatch_id

        expected_heads = self.heads_per_rank * self.world_size
        if (
            partial_output.ndim != 3
            or partial_output.shape[1:] != (expected_heads, self.head_dim)
            or partial_lse.shape != partial_output.shape[:2]
        ):
            raise ValueError("FI DCP expects O [B, N*H, D] and LSE [B, N*H]")
        num_tokens = partial_output.shape[0]
        if not 0 < num_tokens <= self.max_num_tokens:
            raise ValueError("FI DCP input token count exceeds workspace capacity")
        if partial_output.dtype != self.dtype or partial_lse.dtype != torch.float32:
            raise ValueError("FI DCP requires the workspace output dtype and FP32 LSE")
        if partial_output.device != self.device or partial_lse.device != self.device:
            raise ValueError("FI DCP inputs must be on the workspace device")
        if (
            _validate_dcp_empty_shard_args(seq_lens, query_start_loc)
            and valid_token_mask is None
        ):
            raise ValueError(
                "FI DCP requires valid_token_mask prepared by the metadata builder"
            )
        if valid_token_mask is not None and (
            valid_token_mask.shape != (num_tokens,)
            or valid_token_mask.dtype != torch.bool
            or valid_token_mask.device != self.device
            or not valid_token_mask.is_contiguous()
        ):
            raise ValueError(
                "FI DCP token mask must be contiguous bool [B] on the input device"
            )
        ubatch = dbo_current_ubatch_id()
        if not 0 <= ubatch < self.num_ubatches:
            raise ValueError("FI DCP ubatch exceeds the reserved workspace slots")
        self.prepare()
        stream_id = int(torch.cuda.current_stream(self.device).cuda_stream)

        # Views only: FI masks invalid O/LSE while sending, and still publishes
        # the NCCL readiness signals for those rows.
        output = partial_output.unflatten(
            1, (self.world_size, self.heads_per_rank)
        ).transpose(1, 2)
        lse = partial_lse.unflatten(
            1, (self.world_size, self.heads_per_rank)
        ).transpose(1, 2)
        return self._fi.decode_cp_a2a_lse_reduce(
            output,
            lse,
            self._workspaces[stream_id][ubatch],
            self.rank,
            self.world_size,
            lse_mode="basee" if is_lse_base_on_e else "base2",
            mask=valid_token_mask,
        )


@functools.cache
def get_flashinfer_dcp_workspace(
    group: GroupCoordinator,
    device: torch.device,
    max_num_tokens: int,
    heads_per_rank: int,
    head_dim: int,
    dtype: torch.dtype,
    num_ubatches: int,
) -> FlashInferDCPWorkspace:
    return FlashInferDCPWorkspace(
        group, device, max_num_tokens, heads_per_rank, head_dim, dtype, num_ubatches
    )
