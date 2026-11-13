# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Three-launch DCP A2A using mapped peer stores or NIXL GPU puts.

Setup registers persistent storage before graph capture. Each ubatch owns an
independent epoch, per-channel ready counters and two receive slots. Invocations
using the same ubatch must be serialized, with matching participation on peers.
"""

from __future__ import annotations

import socket
import uuid
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.worker.ubatching import dbo_current_ubatch_id

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator

logger = init_logger(__name__)
_COUNTER_STRIDE = 128


@triton.jit
def _nixl_wait_ready(counter, expected):
    # The zero result creates an address dependency for subsequent payload loads.
    # Keep the acquire and timeout in the same asm as the polling loop.
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b64 value, start, elapsed;
            .reg .pred ready, expired;
            mov.u64 start, %globaltimer;
        wait_ready:
            ld.acquire.sys.global.u64 value, [$1];
            setp.ge.u64 ready, value, $2;
            @ready bra ready_done;
            mov.u64 elapsed, %globaltimer;
            sub.u64 elapsed, elapsed, start;
            setp.gt.u64 expired, elapsed, 30000000000;
            @expired trap;
            nanosleep.u32 64;
            bra wait_ready;
        ready_done:
            mov.u32 $0, 0;
        }
        """,
        constraints="=r,l,l,~{memory}",
        args=[counter, expected],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _nixl_lse(recv_ptr, offset):
    value = tl.load((recv_ptr + offset).to(tl.pointer_type(tl.float32)))
    return tl.where((value != value) | (value == float("inf")), -float("inf"), value)


@triton.jit
def _nixl_wait_combine_kernel(
    recv_ptr,
    state_ptr,
    out_ptr,
    out_lse_ptr,
    num_tokens,
    active_channels,
    peer_bytes,
    counter_offset,
    tile_stride,
    TILE_TOKENS: tl.constexpr,
    N: tl.constexpr,
    H_PER_RANK: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_BASE_E: tl.constexpr,
    RETURN_LSE: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    tile = token // TILE_TOKENS
    row = token % TILE_TOKENS
    channel = tile % active_channels
    epoch = tl.load(state_ptr).to(tl.uint64)
    expected = tl.load(state_ptr + 1 + channel).to(tl.uint64)
    guard = 0
    for source in tl.static_range(N):
        counter = recv_ptr + counter_offset + source * 128 + channel * 8
        guard += _nixl_wait_ready(counter, expected)

    rows = tl.minimum(TILE_TOKENS, num_tokens - tile * TILE_TOKENS)
    slot = (epoch & 1).to(tl.int64) * N * peer_bytes
    base = slot + tile * tile_stride + guard
    lse_offset = ((rows * H_PER_RANK * HEAD_DIM * 2 + 15) // 16) * 16
    lse_offset += (row * H_PER_RANK + head) * 4

    # Keep the NCCL A2A combine's source order and exp/log formulation so that
    # changing the transport/layout does not change numerical gates.
    lse_max = -float("inf")
    for source in tl.static_range(N):
        value = _nixl_lse(recv_ptr, base + source * peer_bytes + lse_offset)
        lse_max = tl.maximum(lse_max, value)
    lse_max = tl.where(lse_max == -float("inf"), 0.0, lse_max)

    lse_sum = 0.0
    for source in tl.static_range(N):
        value = _nixl_lse(recv_ptr, base + source * peer_bytes + lse_offset)
        if IS_BASE_E:
            lse_sum += tl.exp(value - lse_max)
        else:
            lse_sum += tl.exp2(value - lse_max)
    if IS_BASE_E:  # noqa: SIM108
        global_lse = tl.log(lse_sum) + lse_max
    else:
        global_lse = tl.log2(lse_sum) + lse_max

    dims = tl.arange(0, HEAD_DIM)
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    output_offset = ((row * H_PER_RANK + head) * HEAD_DIM + dims) * 2
    for source in tl.static_range(N):
        value = _nixl_lse(recv_ptr, base + source * peer_bytes + lse_offset)
        if IS_BASE_E:
            weight = tl.exp(value - global_lse)
        else:
            weight = tl.exp2(value - global_lse)
        weight = tl.where(weight != weight, 0.0, weight)
        ptr = recv_ptr + base + source * peer_bytes + output_offset
        partial = tl.load(ptr.to(tl.pointer_type(out_ptr.dtype.element_ty))).to(
            tl.float32
        )
        partial = tl.where(weight == 0.0, 0.0, partial)
        acc += partial * weight
    tl.store(out_ptr + (token * H_PER_RANK + head) * HEAD_DIM + dims, acc)
    if RETURN_LSE:
        tl.store(out_lse_ptr + token * H_PER_RANK + head, global_lse)


class NixlDCPA2AWorkspace:
    def __init__(
        self,
        group: GroupCoordinator,
        device: torch.device,
        max_num_tokens: int,
        heads_per_rank: int,
        head_dim: int,
        dtype: torch.dtype,
        num_ubatches: int = 1,
        *,
        num_channels: int = 4,
        tile_bytes: int = 65536,
    ) -> None:
        self.group = group
        self.device = torch.device(device)
        self.world_size = group.world_size
        self.rank = group.rank_in_group
        self.max_num_tokens = max_num_tokens
        self.heads_per_rank = heads_per_rank
        self.head_dim = head_dim
        self.dtype = dtype
        self.num_ubatches = num_ubatches
        self.num_channels = num_channels
        self.agent = None
        self.registration = None
        self.local_view = 0
        self.remote_view = 0
        self.remote_agents: list[str] = []
        self.peer_transports: list[list[str]] = []
        self.peer_copy_paths: list[str] = []
        self.closed = False

        error = None
        try:
            if self.world_size <= 1:
                raise ValueError("NIXL DCP requires at least two ranks")
            if self.device.type != "cuda" or not torch.cuda.is_available():
                raise ValueError("NIXL DCP requires CUDA")
            if torch.cuda.is_current_stream_capturing():
                raise ValueError("Create the NIXL DCP workspace before graph capture")
            if dtype not in (torch.float16, torch.bfloat16):
                raise ValueError("NIXL DCP supports FP16 and BF16 output")
            if min(max_num_tokens, heads_per_rank, head_dim, num_ubatches) < 1:
                raise ValueError("NIXL DCP workspace dimensions must be positive")
            if head_dim & (head_dim - 1):
                raise ValueError("NIXL DCP requires a power-of-two head dimension")
            if num_channels not in (1, 2, 4, 8) or tile_bytes < 16:
                raise ValueError("NIXL DCP needs 1/2/4/8 channels and tile_bytes >= 16")
            if not hasattr(torch.ops._C, "nixl_dcp_a2a_dispatch"):
                raise RuntimeError(
                    "vLLM was built without the three-launch NIXL DCP operators. "
                    "Rebuild with -DVLLM_BUILD_NIXL_DCP=ON and matching "
                    "NIXL/UCX headers."
                )
            from nixl import _bindings, nixl_agent, nixl_agent_config

            if not getattr(_bindings, "HAVE_UCX_GPU_DEVICE_API", False):
                raise RuntimeError("NIXL must be built with the UCX GPU Device API")
        except Exception as exc:
            error = str(exc)
        self._check_all(error)

        self.tile_tokens = min(
            max_num_tokens,
            16384,
            max(1, tile_bytes // (heads_per_rank * (head_dim * 2 + 4))),
        )
        output_bytes = self.tile_tokens * heads_per_rank * head_dim * 2
        self.tile_stride = (
            (
                ((output_bytes + 15) // 16) * 16
                + self.tile_tokens * heads_per_rank * 4
                + 15
            )
            // 16
            * 16
        )
        tiles = (max_num_tokens + self.tile_tokens - 1) // self.tile_tokens
        self.peer_bytes = (tiles * self.tile_stride + 127) // 128 * 128
        self.counter_offset = 2 * self.world_size * self.peer_bytes
        receive_bytes = self.counter_offset + self.world_size * _COUNTER_STRIDE
        layout = (
            "mixed3-v1",
            max_num_tokens,
            heads_per_rank,
            head_dim,
            str(dtype),
            num_ubatches,
            num_channels,
            self.tile_tokens,
            self.tile_stride,
            self.peer_bytes,
        )
        layouts: list[object] = [None] * self.world_size
        dist.all_gather_object(layouts, layout, group=group.cpu_group)
        if any(peer_layout != layout for peer_layout in layouts):
            raise ValueError("NIXL DCP workspace layouts differ across ranks")

        try:
            error = None
            try:
                with torch.accelerator.device_index(self.device.index):
                    self.send = torch.empty(
                        (num_ubatches, self.world_size, self.peer_bytes),
                        device=self.device,
                        dtype=torch.uint8,
                    )
                    self.receive = torch.empty(
                        (num_ubatches, receive_bytes),
                        device=self.device,
                        dtype=torch.uint8,
                    )
                    self.receive[:, self.counter_offset :].zero_()
                    self.epoch = torch.zeros(
                        (num_ubatches, num_channels + 1),
                        device=self.device,
                        dtype=torch.int64,
                    )
                    # Same peer order as remote_view: all ranks except self.
                    self.peer_ptrs = torch.empty(
                        (self.world_size - 1,),
                        device=self.device,
                        dtype=torch.int64,
                    )
                    self.workspace_bytes = (
                        self.send.numel()
                        + self.receive.numel()
                        + self.epoch.numel() * 8
                        + self.peer_ptrs.numel() * 8
                    )
                    self.agent = nixl_agent(
                        f"vllm-dcp-{uuid.uuid4().hex}",
                        nixl_agent_config(backends=[], enable_prog_thread=True),
                    )
                    self.agent.create_backend(
                        "UCX",
                        {
                            "num_workers": "1",
                            "ucx_num_device_channels": str(num_channels),
                        },
                    )
                    self.registration = self.agent.register_memory(
                        [self.send, self.receive], backends=["UCX"]
                    )
                    self.local_view = self.agent.prep_mem_view(
                        self.agent.get_xfer_descs(self.send), backends=["UCX"]
                    )
                    torch.accelerator.synchronize(self.device)
            except Exception as exc:
                error = str(exc)
            self._check_all(error)
            assert self.agent is not None
            error = None
            try:
                metadata = (
                    self.agent.get_agent_metadata(),
                    (
                        self.receive.data_ptr(),
                        self.receive.numel(),
                        self.receive.get_device(),
                    ),
                    socket.gethostname(),
                )
            except Exception as exc:
                error = str(exc)
            self._check_all(error)
            peers: list[Any] = [None] * self.world_size
            dist.all_gather_object(peers, metadata, group=group.cpu_group)
            error = None
            try:
                descriptors = []
                for peer_rank, (peer_metadata, region, _) in enumerate(peers):
                    if peer_rank == self.rank:
                        continue
                    name = self.agent.add_remote_agent(peer_metadata)
                    self.remote_agents.append(name)
                    descriptors.append((*region, name))
                with torch.accelerator.device_index(self.device.index):
                    self.remote_view = self.agent.prep_mem_view(
                        self.agent.get_remote_descs(descriptors, "VRAM"),
                        backends=["UCX"],
                    )
                    info = torch.empty(
                        (self.world_size - 1, num_channels, 4),
                        device=self.device,
                        dtype=torch.int64,
                    )
                    torch.ops._C.nixl_dcp_peer_info(
                        self.remote_view, info, self.peer_ptrs
                    )
                    peer_info = iter(info.cpu().tolist())
                    mapped_peers = iter(self.peer_ptrs.ne(0).cpu().tolist())
                    for peer in range(self.world_size):
                        if peer == self.rank:
                            self.peer_transports.append(["self"])
                            self.peer_copy_paths.append("self")
                            continue
                        queues = set()
                        transports = []
                        for kind, endpoint, channel, depth in next(peer_info):
                            if kind not in (1, 2):
                                raise RuntimeError(
                                    f"NIXL peer {peer} lacks CUDA IPC or mlx5 GPU RDMA"
                                )
                            if kind == 2:
                                queue = (endpoint, channel)
                                if queue in queues or depth < 8:
                                    raise RuntimeError(
                                        f"NIXL peer {peer} has aliased or undersized "
                                        "GPU RDMA queues"
                                    )
                                queues.add(queue)
                            transports.append(
                                "cuda_ipc" if kind == 1 else "rc_mlx5_gda"
                            )
                        self.peer_transports.append(transports)
                        self.peer_copy_paths.append(
                            "mapped_p2p" if next(mapped_peers) else "nixl_put"
                        )
                    logger.info(
                        "NIXL DCP rank %d: hosts=%s transports=%s "
                        "copy_paths=%s workspace_bytes=%d",
                        self.rank,
                        [peer[2] for peer in peers],
                        self.peer_transports,
                        self.peer_copy_paths,
                        self.workspace_bytes,
                    )
            except Exception as exc:
                error = str(exc)
            self._check_all(error)
        except Exception:
            self._release()
            raise

    def _check_all(self, error: str | None) -> None:
        errors: list[str | None] = [None] * self.world_size
        dist.all_gather_object(errors, error, group=self.group.cpu_group)
        if any(error is not None for error in errors):
            raise RuntimeError(f"NIXL DCP setup failed across ranks: {errors}")

    def lse_reduce(
        self,
        partial_output: torch.Tensor,
        partial_lse: torch.Tensor,
        is_lse_base_on_e: bool,
        seq_lens: torch.Tensor | None = None,
        query_start_loc: torch.Tensor | None = None,
        *,
        return_lse: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        from vllm.v1.attention.ops.dcp import _validate_dcp_empty_shard_args

        if self.closed:
            raise RuntimeError("NIXL DCP workspace is closed")
        num_tokens = partial_output.shape[0]
        expected = (num_tokens, self.world_size * self.heads_per_rank, self.head_dim)
        if tuple(partial_output.shape) != expected:
            raise ValueError(f"NIXL DCP expected output shape {expected}")
        if not 0 < num_tokens <= self.max_num_tokens:
            raise ValueError("NIXL DCP token count exceeds registered capacity")
        if (
            partial_output.dtype != self.dtype
            or partial_output.device != self.send.device
        ):
            raise ValueError("NIXL DCP output dtype/device differs from the workspace")
        if (
            tuple(partial_lse.shape) != expected[:2]
            or partial_lse.device != self.send.device
            or partial_lse.dtype not in (torch.float32, torch.float16, torch.bfloat16)
        ):
            raise ValueError("NIXL DCP requires matching floating-point LSE on the GPU")
        if _validate_dcp_empty_shard_args(seq_lens, query_start_loc):
            for tensor in (seq_lens, query_start_loc):
                assert tensor is not None
                if (
                    tensor.device != self.send.device
                    or not tensor.is_contiguous()
                    or tensor.dtype != torch.int32
                    or tensor.numel() == 0
                ):
                    raise ValueError(
                        "NIXL DCP mask inputs must be nonempty contiguous CUDA int32"
                    )
        ubatch = dbo_current_ubatch_id()
        if not 0 <= ubatch < self.num_ubatches:
            raise ValueError("NIXL DCP ubatch is outside the registered workspace")
        output = torch.empty(
            (num_tokens, self.heads_per_rank, self.head_dim),
            device=self.device,
            dtype=self.dtype,
        )
        output_lse = torch.empty(
            (num_tokens, self.heads_per_rank) if return_lse else (1,),
            device=self.device,
            dtype=torch.float32,
        )
        torch.ops._C.nixl_dcp_a2a_dispatch(
            partial_output,
            partial_lse,
            seq_lens,
            query_start_loc,
            self.send,
            self.receive,
            self.epoch,
            self.peer_ptrs,
            self.local_view,
            self.remote_view,
            self.rank,
            ubatch,
            self.tile_tokens,
        )
        tiles = (num_tokens + self.tile_tokens - 1) // self.tile_tokens
        _nixl_wait_combine_kernel[(num_tokens, self.heads_per_rank)](
            self.receive[ubatch],
            self.epoch[ubatch],
            output,
            output_lse,
            num_tokens,
            min(self.num_channels, tiles),
            self.peer_bytes,
            self.counter_offset,
            self.tile_stride,
            TILE_TOKENS=self.tile_tokens,
            N=self.world_size,
            H_PER_RANK=self.heads_per_rank,
            HEAD_DIM=self.head_dim,
            IS_BASE_E=is_lse_base_on_e,
            RETURN_LSE=return_lse,
        )
        return (output, output_lse) if return_lse else output

    def _release(self) -> None:
        if self.agent is None:
            return
        with torch.accelerator.device_index(self.device.index):
            for view in (self.remote_view, self.local_view):
                if view:
                    self.agent.release_mem_view(view)
            self.remote_view = self.local_view = 0
            for name in self.remote_agents:
                self.agent.remove_remote_agent(name)
            self.remote_agents.clear()
            if self.registration is not None:
                self.agent.deregister_memory(self.registration, backends=["UCX"])
                self.registration = None
            self.agent = None

    def close(self) -> None:
        """Collective teardown; callers must discard graphs before calling this."""
        if self.closed:
            return
        torch.accelerator.synchronize(self.device)
        dist.barrier(group=self.group.cpu_group)
        self._release()
        dist.barrier(group=self.group.cpu_group)
        self.closed = True


_workspaces: dict[tuple, NixlDCPA2AWorkspace] = {}


def get_nixl_dcp_a2a_workspace(
    group: GroupCoordinator,
    device: torch.device,
    max_num_tokens: int,
    heads_per_rank: int,
    head_dim: int,
    dtype: torch.dtype,
    num_ubatches: int,
) -> NixlDCPA2AWorkspace:
    key = (group, device, max_num_tokens, heads_per_rank, head_dim, dtype, num_ubatches)
    if key not in _workspaces:
        _workspaces[key] = NixlDCPA2AWorkspace(*key)
    return _workspaces[key]


def close_nixl_dcp_a2a_workspaces() -> None:
    for workspace in _workspaces.values():
        workspace.close()
    _workspaces.clear()
