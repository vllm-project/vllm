# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adapter for FlashInfer #6483's destination-major NCCL DCP combine."""

from __future__ import annotations

import functools
import hashlib
import importlib
import inspect
import threading
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from packaging.version import Version

from vllm.logger import init_logger
from vllm.v1.worker.ubatching import dbo_current_ubatch_id

logger = init_logger(__name__)

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
        raise RuntimeError("fi_a2a_fused requires FlashInfer #6483 token-mask support")
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


def _fi_source_identity(module: Any) -> tuple[str, ...]:
    """Hash the loaded DCP API, binding and kernel sources.

    Ranks must load the same implementation: an older #6483 build with the same
    signature can reject a shape that the current one accepts, which would
    leave peers waiting inside the collective.
    """
    env = importlib.import_module("flashinfer.jit.env")
    paths: tuple[Path, ...] = (
        Path(module.__file__),
        Path(env.FLASHINFER_CSRC_DIR) / "dcp_lse_reduce.cu",
        Path(env.FLASHINFER_INCLUDE_DIR) / "flashinfer/comm/dcp_lse_reduce.cuh",
    )
    # Compiled artifacts mapped into this process, whether JIT-built or AOT.
    maps = Path("/proc/self/maps")
    if maps.exists():
        paths += tuple(
            sorted(
                {
                    Path(line.split()[-1])
                    for line in maps.read_text().splitlines()
                    if "dcp_lse_reduce" in line and ".so" in line
                }
            )
        )
    return tuple(hashlib.sha256(path.read_bytes()).hexdigest() for path in paths)


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
        # Validated call signature -> workspace, for the host fast path.
        self._validated: dict[tuple[Any, ...], torch.Tensor] = {}
        self._prepare_lock = threading.Lock()

        error = None
        identity = None
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
            identity = _fi_source_identity(self._fi)
        except Exception as exception:
            error = str(exception)
        geometry = (max_num_tokens, heads_per_rank, head_dim, dtype, num_ubatches)
        status: list[Any] = [None] * self.world_size
        dist.all_gather_object(
            status, (error, geometry, identity), group=group.cpu_group
        )
        if any(item[0] is not None for item in status):
            raise RuntimeError(f"FI DCP preflight failed across ranks: {status}")
        if any(item[1:] != (geometry, identity) for item in status):
            raise RuntimeError(
                f"FI DCP geometry or FlashInfer sources differ across ranks: {status}"
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
                # Initialize the NCCL device communicator before capture and
                # check the destination-major mapping. Token 0 is valid and
                # labels each (destination, head) slice; token 1 is masked and
                # must not be read. The older head-major #6483 layout either
                # rejects this shape or returns other labels.
                error = None
                try:
                    self._check_layout(workspace, stream)
                except Exception as exception:
                    error = str(exception)
                errors: list[str | None] = [None] * self.world_size
                dist.all_gather_object(errors, error, group=self.group.cpu_group)
                if any(errors):
                    raise RuntimeError(
                        f"FI DCP first call failed across ranks: {errors}"
                    )
                workspaces.append(workspace)
            self._streams[stream_id] = stream
            self._workspaces[stream_id] = workspaces

    def _check_layout(self, workspace: torch.Tensor, stream: torch.cuda.Stream) -> None:
        num_tokens = min(2, self.max_num_tokens)
        shape = (num_tokens, self.world_size, self.heads_per_rank, self.head_dim)
        output = torch.full(shape, float("nan"), dtype=self.dtype, device=self.device)
        destination = torch.arange(self.world_size, device=self.device)
        head = torch.arange(self.heads_per_rank, device=self.device) % 128
        output[0, :, :, 0] = destination[:, None].to(self.dtype)
        output[0, :, :, 1] = head[None, :].to(self.dtype)
        output[0, :, :, 2:] = 0
        lse = torch.full(
            shape[:3], float("nan"), dtype=torch.float32, device=self.device
        )
        lse[0] = 0.0
        mask = torch.arange(num_tokens, device=self.device) == 0
        result = self._fi.decode_cp_a2a_lse_reduce(
            output, lse, workspace, self.rank, self.world_size, mask=mask
        )
        stream.synchronize()
        expected = torch.zeros_like(result, dtype=torch.float32)
        expected[0, :, 0] = self.rank
        expected[0, :, 1] = head.float()
        # Every rank sends identical labels, so all ranks fail together.
        if not torch.allclose(result.float(), expected, atol=0.25, rtol=0):
            raise RuntimeError(
                "FlashInfer decode_cp_a2a_lse_reduce did not use the "
                "destination-major [B, CP, H, D] layout required by fi_a2a_fused"
            )

    @staticmethod
    def _as_fi_input(tensor: torch.Tensor, name: str) -> torch.Tensor:
        """Return a FI-compatible view, or an explicit contiguous copy.

        FI rejects strided O/LSE. Copying keeps the logical destination/head
        order (never a storage reinterpretation) and gives every rank the same
        accepted input instead of a rank-local failure inside the collective.
        """
        if tensor.is_contiguous() and tensor.data_ptr() % 16 == 0:
            return tensor
        logger.warning_once(
            "fi_a2a_fused copies %s with shape %s and strides %s to the "
            "contiguous destination-major layout FlashInfer requires.",
            name,
            tuple(tensor.shape),
            tensor.stride(),
        )
        # A fresh allocation also fixes an unaligned contiguous view.
        return tensor.clone(memory_format=torch.contiguous_format)

    def lse_reduce(
        self,
        partial_output: torch.Tensor,
        partial_lse: torch.Tensor,
        is_lse_base_on_e: bool,
        seq_lens: torch.Tensor | None = None,
        query_start_loc: torch.Tensor | None = None,
        valid_token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Exchange destination-major vLLM heads and return the merged output.

        Eager decode is host-bound, so a call whose stream, DBO slot and input
        layouts match an earlier validated call skips the checks and
        ``prepare``. Only pointer alignment can differ and is checked per call.
        """
        key = (
            torch._C._cuda_getCurrentRawStream(self.device.index),
            dbo_current_ubatch_id(),
            partial_output.shape,
            partial_output.stride(),
            partial_output.dtype,
            partial_output.get_device(),
            partial_lse.shape,
            partial_lse.stride(),
            partial_lse.dtype,
            partial_lse.get_device(),
            None
            if valid_token_mask is None
            else (
                valid_token_mask.shape,
                valid_token_mask.stride(),
                valid_token_mask.dtype,
                valid_token_mask.get_device(),
            ),
            None if seq_lens is None else seq_lens.shape,
            None if query_start_loc is None else query_start_loc.shape,
        )
        workspace = self._validated.get(key)
        if (
            workspace is None
            or (partial_output.data_ptr() | partial_lse.data_ptr()) % 16
        ):
            return self._checked_lse_reduce(
                key,
                partial_output,
                partial_lse,
                is_lse_base_on_e,
                seq_lens,
                query_start_loc,
                valid_token_mask,
            )
        return self._fi.decode_cp_a2a_lse_reduce(
            partial_output.unflatten(1, (self.world_size, self.heads_per_rank)),
            partial_lse.unflatten(1, (self.world_size, self.heads_per_rank)),
            workspace,
            self.rank,
            self.world_size,
            lse_mode="basee" if is_lse_base_on_e else "base2",
            mask=valid_token_mask,
        )

    def _checked_lse_reduce(
        self,
        key: tuple[Any, ...],
        partial_output: torch.Tensor,
        partial_lse: torch.Tensor,
        is_lse_base_on_e: bool,
        seq_lens: torch.Tensor | None,
        query_start_loc: torch.Tensor | None,
        valid_token_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        from vllm.v1.attention.ops.dcp import _validate_dcp_empty_shard_args

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

        # Total heads are destination-major, so [B, N*H] -> [B, N, H] keeps the
        # destination mapping. FI masks invalid O/LSE while sending, and still
        # publishes the NCCL readiness signals for those rows.
        output = self._as_fi_input(
            partial_output.unflatten(1, (self.world_size, self.heads_per_rank)),
            "O",
        )
        lse = self._as_fi_input(
            partial_lse.unflatten(1, (self.world_size, self.heads_per_rank)), "LSE"
        )
        workspace = self._workspaces[stream_id][ubatch]
        if (
            key[0] == stream_id
            and partial_output.is_contiguous()
            and partial_lse.is_contiguous()
        ):
            self._validated[key] = workspace
        return self._fi.decode_cp_a2a_lse_reduce(
            output,
            lse,
            workspace,
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
