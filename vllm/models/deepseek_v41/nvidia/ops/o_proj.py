# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSV4.1 output projection and fusions."""

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from torch import nn

import vllm.envs as envs
from vllm.distributed import get_tp_group
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear.mxfp8.flashinfer import (
    FlashInferCutedslMxfp8LinearKernel,
)
from vllm.model_executor.kernels.linear.mxfp8.pytorch import TorchMxFp8LinearKernel
from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation,
    get_input_quant_key,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic
from vllm.models.common.ops.sequence_parallel import sp_reduce_scatter
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.linear import LinearBase

logger = init_logger(__name__)


def _fused_wo_a_max_tokens(layer: nn.Module) -> int:
    max_tokens = getattr(layer, "_fused_wo_a_token_limit", None)
    if max_tokens is None:
        # Additional token tiles only pay off for smaller grids.
        ctas_per_tile = layer.n_local_heads * (layer.o_lora_rank // 128)
        max_tokens = 96 if ctas_per_tile <= 128 else 32
        layer._fused_wo_a_token_limit = max_tokens
    return max_tokens


def _can_fuse_wo_a(layer: nn.Module) -> bool:
    """Whether the attention layer matches the fused WO-A kernel's layout."""
    # The fused kernel never materializes the BF16 input an adapter on wo_a
    # needs, so such a layer takes ``deep_gemm_fp8_o_proj`` instead.
    if getattr(layer.wo_a, "requires_unquantized_input", False):
        return False
    method = getattr(layer.wo_b, "scheme", layer.wo_b.quant_method)
    return (
        current_platform.is_device_capability_family(100)
        # WO-A is FP8 with per-32 scales, and WO-B takes MXFP8 input directly.
        and layer._einsum_recipe == (1, 1, 32)
        and get_input_quant_key(layer.wo_b) == kMxfp8Dynamic
        and isinstance(
            getattr(method, "kernel", None),
            (FlashInferCutedslMxfp8LinearKernel, TorchMxFp8LinearKernel),
        )
        # A group's heads form one portable (<= 8 CTA) cluster.
        and layer.n_local_heads // layer.n_local_groups <= 8
        and (layer.nope_head_dim, layer.rope_head_dim) == (448, 64)
        and layer.o_lora_rank % 128 == 0
    )


def _fused_wo_a_shape(layer: nn.Module) -> dict[str, int]:
    """The fused WO-A kernel's warmup arguments for `layer`."""
    return dict(
        max_tokens=_fused_wo_a_max_tokens(layer),
        n_groups=layer.n_local_groups,
        heads_per_group=layer.n_local_heads // layer.n_local_groups,
        o_lora_rank=layer.o_lora_rank,
    )


def precompile_dsv41_o_proj_early_trigger(layer: nn.Module, max_tokens: int) -> None:
    """Compile the fused WO-A variant `dsv41_o_proj` dispatches with
    `early_trigger`, for its token counts up to `max_tokens`."""
    if _can_fuse_wo_a(layer):
        from .fused_wo_a import _FUSED_WO_A_KERNEL

        shape = _fused_wo_a_shape(layer)
        shape["max_tokens"] = min(shape["max_tokens"], max_tokens)
        _FUSED_WO_A_KERNEL.warmup(**shape, early_trigger=True)


def register_dsv41_o_proj_warmup(layer: nn.Module) -> None:
    """Warm the fused WO-A token counts `dsv41_o_proj` can dispatch."""
    # Draft models load outside the warmup registry; their layers share the
    # target's attention shapes, so the target's registration covers them too.
    if _can_fuse_wo_a(layer):
        from .fused_wo_a import _FUSED_WO_A_KERNEL

        _FUSED_WO_A_KERNEL.register_warmup(**_fused_wo_a_shape(layer))


def dsv41_o_proj(
    layer: nn.Module,
    attn_out: torch.Tensor,
    positions: torch.Tensor,
    wo_b_proj: Callable[..., torch.Tensor] | None = None,
    early_trigger: bool = False,
) -> torch.Tensor:
    """`deep_gemm_fp8_o_proj` with WO-A fused for small SM100/SM103 batches;
    `wo_b_proj` replaces `layer._wo_b_proj`. With `early_trigger`, the
    fused WO-A launches `wo_b_proj` (PDL) right after its TMEM allocation,
    for a `wo_b_proj` that loads its weights before its grid wait."""
    wo_b_proj = wo_b_proj or layer._wo_b_proj
    o = attn_out[:, : layer.n_local_heads, :]
    cos_sin_cache = layer.rotary_emb.cos_sin_cache
    if 1 <= o.shape[0] <= _fused_wo_a_max_tokens(layer) and _can_fuse_wo_a(layer):
        from .fused_wo_a import _FUSED_WO_A_KERNEL

        q, scales = _FUSED_WO_A_KERNEL(
            x=o,
            positions=positions,
            rope=cos_sin_cache,
            weight=layer.wo_a.weight,
            weight_scale=layer.wo_a.weight_scale,
            early_trigger=early_trigger,
        )
        return wo_b_proj(
            QuantizedActivation(q, scales, o.dtype, q.shape, kMxfp8Dynamic)
        )
    z = layer.wo_a.grouped_forward(
        o,
        positions,
        cos_sin_cache,
        n_groups=layer.n_local_groups,
        heads_per_group=layer.n_local_heads // layer.n_local_groups,
        nope_dim=layer.nope_head_dim,
        rope_dim=layer.rope_head_dim,
        o_lora_rank=layer.o_lora_rank,
        einsum_recipe=layer._einsum_recipe,
        tma_aligned_scales=layer._tma_aligned_scales,
    )
    return wo_b_proj(z.flatten(1))


class HopperWoBGemmRs:
    """Pipeline the BF16 projection with the TP reduce-scatter on SM90."""

    _MIN_TOKENS = 1024

    def __init__(self, group_name: str, tp_size: int) -> None:
        self.group_name = group_name
        self.tp_size = tp_size

    @classmethod
    def try_init(cls, config: "VllmConfig") -> "HopperWoBGemmRs | None":
        from ..quantization import mxfp8_linear_as_bf16

        if (
            config.model_config.dtype != torch.bfloat16
            or config.parallel_config.use_ubatching
            or envs.VLLM_BATCH_INVARIANT
            or not envs.VLLM_MXFP8_EMULATION_DEQUANT_AT_LOAD
            or not mxfp8_linear_as_bf16()
        ):
            return None

        from torch.distributed import _symmetric_memory as symm_mem

        from vllm.distributed.device_communicators.nvlink_fabric import (
            SymmetricMemoryTopology,
            get_symmetric_memory_topology,
        )

        if not hasattr(torch.ops.symm_mem, "fused_matmul_reduce_scatter"):
            return None
        tp_group = get_tp_group()
        if (
            get_symmetric_memory_topology(tp_group.cpu_group)
            is not SymmetricMemoryTopology.INTRA_NODE
        ):
            return None
        group_name = tp_group.device_group.group_name
        try:
            # Probe the workspace collectively during model initialization.
            symm_mem.get_symm_mem_workspace(group_name, min_size=1)
        except RuntimeError as exc:
            logger.warning_once("Hopper wo_b GEMM-RS is unavailable: %s", exc)
            return None
        logger.info_once("Hopper wo_b GEMM-RS is enabled.")
        return cls(group_name, tp_group.world_size)

    def can_run(self, layer: "LinearBase") -> bool:
        from ..quantization import bf16_mxfp8_linear_method_cls

        return (
            layer.bias is None
            and type(layer.quant_method) is bf16_mxfp8_linear_method_cls()
        )

    def warn_incompatible_projection(self) -> None:
        logger.warning_once("Hopper wo_b GEMM-RS requires load-time BF16 weights.")

    def should_run(self, x: torch.Tensor) -> bool:
        return (
            x.dtype == torch.bfloat16
            and x.shape[0] >= self._MIN_TOKENS
            and x.shape[0] % self.tp_size == 0
            and not torch.cuda.is_current_stream_capturing()
        )

    def apply(self, x: torch.Tensor, layer: "LinearBase") -> torch.Tensor:
        if layer.weight.dtype != torch.bfloat16:
            return sp_reduce_scatter(layer(x))
        return torch.ops.symm_mem.fused_matmul_reduce_scatter(
            x, layer.weight.t(), "sum", 0, self.group_name
        )
