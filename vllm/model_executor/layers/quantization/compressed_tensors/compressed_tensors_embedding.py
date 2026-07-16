# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quantized embedding method for compressed-tensors.

Adds dequant-on-lookup support for a pack-quantized ``VocabParallelEmbedding``
(2-8 bit INT, channel- or group-quantized). Only the gathered token rows are
unpacked and dequantized, so the packed weight is never densified.

When the quantized embedding is reused as a tied ``lm_head`` the logits matmul
runs through vLLM's existing WNA16 Linear kernel (Marlin/Machete) -- a fused
dequant-GEMM that never materializes the dense weight -- falling back to a
full-table dequant + ``F.linear`` when no such kernel is available.
"""

import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy

from vllm.logger import init_logger
from vllm.model_executor.kernels.linear import (
    MPLinearLayerConfig,
    choose_mp_linear_kernel,
)
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.parameter import (
    BasevLLMParameter,
    ChannelQuantScaleParameter,
    GroupQuantScaleParameter,
    PackedvLLMParameter,
)
from vllm.scalar_type import scalar_types
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

__all__ = ["CompressedTensorsEmbeddingWNA16Int"]

# num_bits -> symmetric WNA16 scalar type used by the mixed-precision Linear
# kernels for the tied lm_head logits path.
_WNA16_TYPES = {
    2: scalar_types.uint2b2,
    3: scalar_types.uint3b4,
    4: scalar_types.uint4b8,
    5: scalar_types.uint5b16,
    6: scalar_types.uint6b32,
    7: scalar_types.uint7b64,
    8: scalar_types.uint8b128,
}


@triton.jit
def _dequant_kernel(
    ids_ptr,
    packed_ptr,
    scale_ptr,
    out_ptr,
    hidden,
    packed_cols,
    num_groups,
    NUM_BITS: tl.constexpr,
    PACK_FACTOR: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
    GATHER: tl.constexpr,
):
    """Unpack int32-packed INT weights and dequantize to ``out`` dtype in one
    pass (no int8 intermediate). With ``GATHER`` the source row is looked up by
    token id (embedding lookup); otherwise the full table is dequantized in
    order (``row`` is the table row), which needs no index tensor."""
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    col_mask = col < hidden
    tid = tl.load(ids_ptr + row).to(tl.int64) if GATHER else row.to(tl.int64)

    packed_idx = col // PACK_FACTOR
    shift = (col % PACK_FACTOR) * NUM_BITS
    packed = tl.load(
        packed_ptr + tid * packed_cols + packed_idx, mask=col_mask, other=0
    )
    q = ((packed >> shift) & ((1 << NUM_BITS) - 1)) - (1 << (NUM_BITS - 1))

    if GROUP_SIZE == 0:  # channel: one scale per row
        scale = tl.load(scale_ptr + tid)
    else:  # group: one scale per (row, group)
        grp = col // GROUP_SIZE
        scale = tl.load(scale_ptr + tid * num_groups + grp, mask=col_mask, other=0.0)

    out = q.to(tl.float32) * scale.to(tl.float32)
    tl.store(
        out_ptr + row * hidden + col, out.to(out_ptr.dtype.element_ty), mask=col_mask
    )


def _dequant_triton(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    hidden: int,
    num_bits: int,
    ids: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Dequantize packed rows to ``out_dtype``. With ``ids`` the named rows are
    gathered (embedding lookup); without it the full table is dequantized in
    order (logits matmul) with no index tensor allocated."""
    gather = ids is not None
    n = ids.numel() if ids is not None else weight_packed.shape[0]
    dtype = out_dtype if out_dtype is not None else weight_scale.dtype
    out = torch.empty(n, hidden, dtype=dtype, device=weight_packed.device)
    num_groups = weight_scale.shape[1]
    group_size = 0 if num_groups == 1 else hidden // num_groups
    block = min(triton.next_power_of_2(hidden), 1024)
    grid = (n, triton.cdiv(hidden, block))
    _dequant_kernel[grid](
        ids if gather else weight_packed,  # ids_ptr unused when GATHER is False
        weight_packed,
        weight_scale,
        out,
        hidden,
        weight_packed.shape[1],
        num_groups,
        NUM_BITS=num_bits,
        PACK_FACTOR=32 // num_bits,
        GROUP_SIZE=group_size,
        BLOCK=block,
        GATHER=gather,
    )
    return out


class CompressedTensorsEmbeddingWNA16Int(QuantizeMethodBase):
    def __init__(self, weight_quant: QuantizationArgs):
        self.num_bits = weight_quant.num_bits
        self.pack_factor = 32 // self.num_bits
        self.strategy = weight_quant.strategy
        self.group_size = weight_quant.group_size
        self.symmetric = weight_quant.symmetric
        self.is_group = (
            self.strategy == QuantizationStrategy.GROUP.value
            and self.group_size is not None
        )

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        weight_loader = extra_weight_attrs["weight_loader"]
        # Embedding weight is [num_embeddings(vocab), embedding_dim(hidden)];
        # vocab is the output (partitioned) dim, hidden is the input dim.
        vocab_pp = sum(output_partition_sizes)
        hidden = input_size_per_partition
        layer.hidden_size = hidden

        weight_packed = PackedvLLMParameter(
            input_dim=1,
            output_dim=0,
            packed_dim=1,
            packed_factor=self.pack_factor,
            weight_loader=weight_loader,
            data=torch.empty(vocab_pp, hidden // self.pack_factor, dtype=torch.int32),
        )

        if self.is_group:
            assert hidden % self.group_size == 0
            weight_scale = GroupQuantScaleParameter(
                output_dim=0,
                input_dim=1,
                weight_loader=weight_loader,
                data=torch.empty(
                    vocab_pp, hidden // self.group_size, dtype=params_dtype
                ),
            )
        else:
            weight_scale = ChannelQuantScaleParameter(
                output_dim=0,
                weight_loader=weight_loader,
                data=torch.empty(vocab_pp, 1, dtype=params_dtype),
            )

        weight_shape = BasevLLMParameter(
            data=torch.empty(2, dtype=torch.int64), weight_loader=weight_loader
        )

        layer.register_parameter("weight_packed", weight_packed)
        layer.register_parameter("weight_scale", weight_scale)
        layer.register_parameter("weight_shape", weight_shape)

        # Config for the fused logits kernel, used only if this embedding is
        # reused as a tied lm_head (see tie_weights/process_weights_after_loading).
        layer.logits_kernel_config = MPLinearLayerConfig(
            full_weight_shape=(input_size, output_size),
            partition_weight_shape=(hidden, vocab_pp),
            weight_type=_WNA16_TYPES[self.num_bits],
            act_type=params_dtype,
            group_size=self.group_size if self.is_group else -1,
            zero_points=not self.symmetric,
            has_g_idx=False,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Only tied lm_head embeddings run the logits matmul; input-only
        # embeddings skip the (memory-costing) fused-kernel repack.
        if not getattr(layer, "reused_as_lm_head", False):
            return
        if getattr(layer, "logits_kernel", None) is not None:
            return  # idempotent: the tied module may be visited twice

        try:
            kernel_cls = choose_mp_linear_kernel(layer.logits_kernel_config)
            kernel = kernel_cls(
                layer.logits_kernel_config,
                w_q_param_name="weight_packed",
                w_s_param_name="weight_scale",
            )
        except Exception as e:
            logger.debug(
                "No fused logits kernel for quantized tied embedding (%r); "
                "falling back to dequant + F.linear.",
                e,
            )
            return

        # The kernel repacks weight_packed/weight_scale in place; keep the
        # gather-format copies the embedding lookup needs.
        layer.gather_weight_packed = layer.weight_packed.data.clone()
        layer.gather_weight_scale = layer.weight_scale.data.clone()
        kernel.process_weights_after_loading(layer)
        layer.logits_kernel = kernel

    def _gather_weights(
        self, layer: torch.nn.Module
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # After the fused kernel repacks weight_packed, the gather copies hold
        # the compressed-tensors layout the lookup/dequant paths expect.
        return (
            getattr(layer, "gather_weight_packed", layer.weight_packed),
            getattr(layer, "gather_weight_scale", layer.weight_scale),
        )

    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        ids = input_.reshape(-1).contiguous()
        hidden = layer.hidden_size
        weight_packed, weight_scale = self._gather_weights(layer)
        deq = _dequant_triton(
            weight_packed, weight_scale, hidden, self.num_bits, ids=ids
        )
        return deq.reshape(*input_.shape, hidden)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Linear projection against the packed table (logits matmul).

        Needed when a quantized embedding is reused as a tied ``lm_head``. Prefer
        the fused WNA16 GEMM kernel set up in ``process_weights_after_loading``;
        otherwise dequantize the full table in order (no index tensor, straight
        into ``x``'s dtype) and use ``F.linear``.
        """
        kernel = getattr(layer, "logits_kernel", None)
        if kernel is not None:
            return kernel.apply_weights(layer, x, bias)

        weight_packed, weight_scale = self._gather_weights(layer)
        weight = _dequant_triton(
            weight_packed,
            weight_scale,
            layer.hidden_size,
            self.num_bits,
            out_dtype=x.dtype,
        )
        return torch.nn.functional.linear(x, weight, bias)

    def tie_weights(
        self, layer: torch.nn.Module, embed_tokens: torch.nn.Module
    ) -> torch.nn.Module:
        """Reuse the quantized embedding module as the tied ``lm_head``.

        The packed table exposes no plain ``weight`` to share, so return the
        embedding module itself; its ``apply`` serves the lm_head matmul. Flag it
        so ``process_weights_after_loading`` sets up the fused logits kernel.
        """
        embed_tokens.reused_as_lm_head = True
        return embed_tokens
