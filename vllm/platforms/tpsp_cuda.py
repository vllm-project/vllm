# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA TPSP with backend-owned P2P buffers reused by sequential ops groups."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed import distributed_c10d as c10d

from vllm.logger import init_logger
from vllm.model_executor.models.transformers.tpsp_runtime import (
    TPSPBackend,
    TPSPContext,
    TPSPNormType,
    TPSPOpsGroup,
    profile_tpsp,
)

logger = init_logger(__name__)

_O_PROJ_COMM_SM_MAX = 20
_DOWN_PROJ_COMM_SM_MAX = 10


def _compute_engine_count_min(group_name: str | None, sm_count: int) -> int:
    if group_name == "o_proj":
        reserved = min(sm_count * 50 // 100, _O_PROJ_COMM_SM_MAX)
    elif group_name == "down_proj":
        reserved = min(sm_count * 30 // 100, _DOWN_PROJ_COMM_SM_MAX)
    else:
        reserved = 0
    return sm_count - reserved


@dataclass(frozen=True)
class CudaTPSPConfig:
    microchunk_tokens_per_rank: int
    sm_count_target: int = 0


@dataclass
class CudaTPSPWorkspace:
    device: torch.device
    tp_size: int
    rank: int
    control_group: dist.ProcessGroup
    max_microchunk_tokens_per_rank: int
    p2p_handle: int = 0


@dataclass
class CudaTPSPContext:
    config: int | CudaTPSPConfig | None = None


class CudaTPSPBackend(TPSPBackend):
    """Own P2P buffers until close(); ops contexts only borrow them."""

    tpsp_max_microchunk_tokens_per_rank = 4_096
    supported_norm_types: frozenset[TPSPNormType] = frozenset({"rms_norm"})

    def __init__(self, group_name: str, device: torch.device):
        super().__init__(group_name, device)
        self._open_contexts: dict[int, CudaTPSPContext] = {}
        self._workspace: CudaTPSPWorkspace | None = None

    def open(
        self,
        *,
        dtype: torch.dtype,
        tp_size: int,
        hidden_size: int,
        max_batched_tokens: int,
        group_name: str,
        device: torch.device,
    ) -> bool:
        if self._closed:
            raise RuntimeError("TPSP backend is closed")
        if self._workspace is not None:
            raise RuntimeError("TPSP backend is already open")
        if (
            device != self.device
            or group_name != self.group_name
            or device.type != "cuda"
        ):
            return False
        if (
            dtype != torch.bfloat16
            or tp_size < 2
            or hidden_size <= 0
            or hidden_size % tp_size
            or max_batched_tokens <= 0
        ):
            logger.warning(
                "TPSP unavailable for dtype=%s tp_size=%s hidden_size=%s",
                dtype,
                tp_size,
                hidden_size,
            )
            return False
        if (
            device.index is not None
            and device.index >= torch.accelerator.device_count()
        ):
            logger.warning("CUDA TPSP device index is unavailable: %s", device.index)
            return False
        try:
            group = c10d._resolve_process_group(group_name)
        except (RuntimeError, ValueError) as exc:
            logger.warning("CUDA TPSP process group is unavailable: %s", exc)
            return False
        if dist.get_world_size(group) != tp_size:
            logger.warning("TPSP unavailable: group_name and tp_size disagree")
            return False
        if not all(
            hasattr(torch.ops._C, name)
            for name in (
                "tpsp_matmul_reduce_scatter_norm_all_gather",
                "init_tpsp_p2p",
                "get_tpsp_p2p_handles",
                "map_tpsp_p2p",
                "destroy_tpsp_p2p",
            )
        ):
            logger.warning("CUDA TPSP native operator is not built")
            return False
        from vllm.distributed.parallel_state import get_tp_group

        tp_group = get_tp_group()
        if tp_group.device_group is not group:
            logger.warning("CUDA TPSP group is not the active TP group")
            return False
        control_group = tp_group.cpu_group
        if control_group is None:
            logger.warning("CUDA TPSP CPU control group is unavailable")
            return False
        rank = dist.get_rank(group)
        max_microchunk_tokens_per_rank = min(
            math.ceil(max_batched_tokens / tp_size),
            self.tpsp_max_microchunk_tokens_per_rank,
        )
        context = CudaTPSPWorkspace(
            device,
            tp_size,
            rank,
            control_group,
            max_microchunk_tokens_per_rank,
        )
        context.p2p_handle = torch.ops._C.init_tpsp_p2p(
            device.index if device.index is not None else torch.cuda.current_device(),
            tp_size,
            rank,
            max_microchunk_tokens_per_rank,
            hidden_size,
            max_batched_tokens,
        )
        if not self._all_ranks_ready(bool(context.p2p_handle), control_group):
            if context.p2p_handle:
                torch.ops._C.destroy_tpsp_p2p(context.p2p_handle)
            dist.barrier(group=control_group)
            if rank == 0:
                logger.warning(
                    "CUDA TPSP P2P allocation is unavailable; disabling TPSP"
                )
            return False
        local_handles = torch.ops._C.get_tpsp_p2p_handles(context.p2p_handle)
        gathered_handles = [torch.empty_like(local_handles) for _ in range(tp_size)]
        dist.all_gather(gathered_handles, local_handles, group=control_group)
        peer_handles = torch.cat(gathered_handles)
        mapped = torch.ops._C.map_tpsp_p2p(context.p2p_handle, peer_handles)
        if not self._all_ranks_ready(mapped, control_group):
            torch.ops._C.destroy_tpsp_p2p(context.p2p_handle)
            dist.barrier(group=control_group)
            if rank == 0:
                logger.warning("CUDA TPSP P2P mapping is unavailable; disabling TPSP")
            return False
        if rank == 0:
            logger.info(
                "TPSP enabled with direct P2P for %s tokens per rank",
                max_microchunk_tokens_per_rank,
            )
        self._workspace = context
        return True

    def close(self, context: object | None = None) -> None:
        if context is not None:
            context = self._profile_context(context)
            del self._open_contexts[id(context)]
        else:
            workspace = self._workspace
            if workspace is not None:
                torch.cuda.synchronize(workspace.device)
                dist.barrier(group=workspace.control_group)
                torch.ops._C.destroy_tpsp_p2p(workspace.p2p_handle)
                dist.barrier(group=workspace.control_group)
                self._workspace = None
            self._open_contexts.clear()
            self._closed = True

    def profile(
        self,
        ops_groups: Sequence[TPSPOpsGroup],
        max_batched_tokens: int,
        *,
        microchunk_tokens: int | None = None,
        threshold_tokens: int | None = None,
        compute_engine_count_min: int | Mapping[str, int] | None = None,
        compute_engine_count_max: int | Mapping[str, int] | None = None,
        compute_engine_step_coarse: int = 6,
        compute_engine_step_fine: int = 1,
        microchunk_tokens_per_rank_min_max: (
            tuple[int, int] | Mapping[str, tuple[int, int]] | None
        ) = None,
        microchunk_step_coarse_per_rank: int = 256,
        microchunk_step_fine_per_rank: int = 64,
    ) -> TPSPContext | None:
        """Profile chunk/SM configurations and threshold, honoring token overrides."""
        from vllm.distributed.parallel_state import get_tp_group

        tp_size = get_tp_group().world_size
        if (
            (microchunk_tokens is None or threshold_tokens is None)
            and compute_engine_count_min is None
            and compute_engine_count_max is None
        ):
            sm_count = torch.cuda.get_device_properties(
                self.device
            ).multi_processor_count
            compute_engine_count_min = {
                entry.name: _compute_engine_count_min(entry.name, sm_count)
                for entry in ops_groups
            }
            compute_engine_count_max = {entry.name: sm_count for entry in ops_groups}
        if (
            microchunk_tokens_per_rank_min_max is None
            and microchunk_tokens is None
            and type(max_batched_tokens) is int
            and max_batched_tokens > 0
        ):
            microchunk_tokens_per_rank_min_max = self._microchunk_tokens_per_rank_range(
                max_batched_tokens, tp_size
            )
        return profile_tpsp(
            self,
            ops_groups,
            max_batched_tokens,
            microchunk_tokens=microchunk_tokens,
            threshold_tokens=threshold_tokens,
            compute_engine_count_min=compute_engine_count_min,
            compute_engine_count_max=compute_engine_count_max,
            compute_engine_step_coarse=compute_engine_step_coarse,
            compute_engine_step_fine=compute_engine_step_fine,
            microchunk_tokens_per_rank_min_max=microchunk_tokens_per_rank_min_max,
            microchunk_step_coarse_per_rank=microchunk_step_coarse_per_rank,
            microchunk_step_fine_per_rank=microchunk_step_fine_per_rank,
        )

    def tpsp_matmul_rs_norm_ag(
        self,
        projection_context: object,
        x: torch.Tensor,
        projection: torch.nn.Module,
        residual: torch.Tensor,
        norm: torch.nn.Module,
        *,
        config: object | None = None,
        norm_type: TPSPNormType = "rms_norm",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run sequentially; consume gathered output before the next TPSP call."""
        context = self._profile_context(projection_context)
        workspace = self._get_workspace()

        if norm_type != "rms_norm":
            raise ValueError("CUDA TPSP supports only RMSNorm")
        norm_kind = 0 if norm_type == "rms_norm" else 1
        a, b = x.contiguous(), projection._tpsp_transposed_weight
        norm_weight = norm.weight
        norm_bias = getattr(norm, "bias", None)
        if norm_bias is not None:
            raise ValueError("CUDA TPSP RMSNorm does not support norm bias")
        eps = norm.variance_epsilon
        projection_bias = projection.bias
        selected = self._resolve_config(
            context, context.config if config is None else config
        )
        microchunk_tokens_per_rank = selected.microchunk_tokens_per_rank
        updated_residual, gathered = (
            torch.ops._C.tpsp_matmul_reduce_scatter_norm_all_gather(
                a,
                b,
                norm_weight,
                residual,
                projection_bias,
                norm_bias,
                eps,
                norm_kind,
                microchunk_tokens_per_rank,
                workspace.tp_size,
                workspace.p2p_handle,
                selected.sm_count_target,
            )
        )
        return gathered, updated_residual

    def get_compute_engine_count(self, config: object) -> int | None:
        if isinstance(config, CudaTPSPConfig):
            return config.sm_count_target
        return None

    def get_default_compute_engine_count(self) -> int:
        return 0

    def set_config(self, handle: object, config: object) -> None:
        context = self._profile_context(handle)
        context.config = self._resolve_config(context, config)

    def prepare_weights(self, projections: Sequence[torch.nn.Module]) -> None:
        for projection in projections:
            projection._tpsp_transposed_weight = (
                projection.weight.detach().T.contiguous()
            )

    def get_tuning_configs(
        self,
        handle: object,
        microchunk_tokens_per_rank: int,
        group_name: str | None = None,
        *,
        compute_engine_counts: Sequence[int] | None = None,
        compute_engine_step: int = 6,
    ) -> tuple[CudaTPSPConfig, ...]:
        self._profile_context(handle)
        if type(compute_engine_step) is not int or compute_engine_step <= 0:
            raise ValueError("Compute-engine scan step must be positive")
        sm_count = torch.cuda.get_device_properties(self.device).multi_processor_count
        if compute_engine_counts is not None:
            targets = tuple(dict.fromkeys(compute_engine_counts))
        else:
            start = _compute_engine_count_min(group_name, sm_count)
            values = list(range(start, sm_count + 1, compute_engine_step))
            if values[-1] != sm_count:
                values.append(sm_count)
            targets = tuple(dict.fromkeys((0, *values)))
        if not targets or any(
            type(target) is not int or not 0 <= target <= sm_count for target in targets
        ):
            raise ValueError("Invalid TPSP SM count targets")
        return tuple(
            CudaTPSPConfig(microchunk_tokens_per_rank, target) for target in targets
        )

    def _microchunk_tokens_per_rank_range(
        self, max_batched_tokens: int, tp_size: int
    ) -> tuple[int, int]:
        max_tokens_per_rank = math.ceil(max_batched_tokens / tp_size)
        cap_tokens_per_rank = self.tpsp_max_microchunk_tokens_per_rank
        if cap_tokens_per_rank is not None:
            max_tokens_per_rank = min(max_tokens_per_rank, cap_tokens_per_rank)
        return min(64, max_tokens_per_rank), max_tokens_per_rank

    def _get_workspace(self) -> CudaTPSPWorkspace:
        if self._closed:
            raise RuntimeError("TPSP backend is closed")
        if self._workspace is None:
            raise RuntimeError("TPSP backend must be opened before use")
        return self._workspace

    def _create_context(self) -> CudaTPSPContext:
        if self._closed:
            raise RuntimeError("TPSP backend is closed")
        if self._workspace is None:
            raise RuntimeError("TPSP backend must be opened before use")
        context = CudaTPSPContext()
        self._open_contexts[id(context)] = context
        return context

    @staticmethod
    def _all_ranks_ready(ready: bool, group: dist.ProcessGroup) -> bool:
        supported = torch.tensor(int(ready), device="cpu")
        dist.all_reduce(supported, op=dist.ReduceOp.MIN, group=group)
        return bool(supported.item())

    def _profile_context(self, context: object) -> CudaTPSPContext:
        if self._closed:
            raise RuntimeError("TPSP backend is closed")
        if context is None:
            raise ValueError("CUDA TPSP requires a projection context")
        if (
            not isinstance(context, CudaTPSPContext)
            or self._open_contexts.get(id(context)) is not context
        ):
            raise ValueError("CUDA TPSP context belongs to another backend")
        return context

    def _resolve_config(
        self, context: CudaTPSPContext, config: object
    ) -> CudaTPSPConfig:
        if type(config) is int:
            config = CudaTPSPConfig(config)
        if (
            not isinstance(config, CudaTPSPConfig)
            or type(config.microchunk_tokens_per_rank) is not int
            or config.microchunk_tokens_per_rank <= 0
            or type(config.sm_count_target) is not int
            or config.sm_count_target < 0
        ):
            raise ValueError("Invalid CUDA TPSP tuning configuration")
        return config
