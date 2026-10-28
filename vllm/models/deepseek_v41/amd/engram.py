# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engram with a TP-sharded gate on gfx942 and gfx950."""

import torch

from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.models.deepseek_v41.common.engram import Engram
from vllm.platforms.rocm import get_cdna_version
from vllm.triton_utils import tl, triton


def _shards_engram_gate() -> bool:
    """Whether TP ranks keep a dim slice of `wkv` and exchange gate partial sums."""
    return get_cdna_version() in (3, 4)


@triton.jit
def _engram_gate_partials_kernel(
    hidden_states,
    kv,
    q_weight,
    k_weight,
    send,
    send_f32,
    hidden_stride_t,
    hidden_stride_h,
    hidden_stride_d,
    kv_stride_t,
    kv_stride_d,
    q_stride_h,
    q_stride_d,
    k_stride_h,
    k_stride_d,
    send_stride_t,
    send_f32_stride_t,
    dim_start,
    WIDTH: tl.constexpr,
    HC_MULT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Pack one token's value slice and per-hc gate partial sums into `send`.

    `kv` holds this rank's `WIDTH` columns of every key and of the value. The
    row of `send` is the value slice followed by (sum key^2, sum hidden*q*k*key)
    as FP32 pairs, one per hc copy. Each program handles one hc copy, or the
    value slice, of one token.
    """
    program_idx = tl.program_id(0)
    token_idx = (program_idx // (HC_MULT + 1)).to(tl.int64)
    hc_idx = program_idx % (HC_MULT + 1)
    offsets = tl.arange(0, BLOCK_SIZE)
    valid = offsets < WIDTH
    if hc_idx == HC_MULT:
        value = tl.load(
            kv + token_idx * kv_stride_t + (HC_MULT * WIDTH + offsets) * kv_stride_d,
            mask=valid,
        )
        tl.store(send + token_idx * send_stride_t + offsets, value, mask=valid)
    else:
        hidden = tl.load(
            hidden_states
            + token_idx * hidden_stride_t
            + hc_idx * hidden_stride_h
            + (dim_start + offsets) * hidden_stride_d,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        key = tl.load(
            kv + token_idx * kv_stride_t + (hc_idx * WIDTH + offsets) * kv_stride_d,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        q = tl.load(
            q_weight + hc_idx * q_stride_h + (dim_start + offsets) * q_stride_d,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        k = tl.load(
            k_weight + hc_idx * k_stride_h + (dim_start + offsets) * k_stride_d,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        partials = send_f32 + token_idx * send_f32_stride_t + WIDTH // 2 + 2 * hc_idx
        tl.store(partials, tl.sum(key * key, axis=0))
        tl.store(partials + 1, tl.sum(hidden * q * k * key, axis=0))


@triton.jit(do_not_specialize=["num_tokens"])
def _engram_gate_finish_kernel(
    hidden_states,
    gathered,
    gathered_f32,
    token_mask,
    output,
    num_tokens,
    hidden_stride_t,
    hidden_stride_h,
    hidden_stride_d,
    gathered_stride_t,
    gathered_f32_stride_t,
    mask_stride,
    output_stride_t,
    output_stride_h,
    output_stride_d,
    eps,
    clamp_value,
    DIM: tl.constexpr,
    WIDTH: tl.constexpr,
    TP_SIZE: tl.constexpr,
    HC_MULT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    """`_fused_engram_post_wkv_kernel` on the rank-major rows the partials
    kernel packed, all-gathered along tokens."""
    program_idx = tl.program_id(0)
    token_idx = (program_idx // HC_MULT).to(tl.int64)
    hc_idx = program_idx % HC_MULT
    dim_offsets = tl.arange(0, BLOCK_SIZE)
    dim_valid = dim_offsets < DIM
    hidden = tl.load(
        hidden_states
        + token_idx * hidden_stride_t
        + hc_idx * hidden_stride_h
        + dim_offsets * hidden_stride_d,
        mask=dim_valid,
        other=0.0,
    ).to(tl.float32)

    key_sq = 0.0
    dot = 0.0
    for rank in tl.static_range(TP_SIZE):
        partials = (
            gathered_f32
            + (rank * num_tokens + token_idx) * gathered_f32_stride_t
            + WIDTH // 2
            + 2 * hc_idx
        )
        key_sq += tl.load(partials)
        dot += tl.load(partials + 1)

    hidden_rms = tl.rsqrt(tl.sum(hidden * hidden, axis=0) / DIM + eps)
    key_rms = tl.rsqrt(key_sq / DIM + eps)
    dot *= hidden_rms * key_rms * tl.rsqrt(DIM * 1.0)
    gate_input = tl.sqrt(tl.maximum(tl.abs(dot), clamp_value))
    gate_input = tl.where(dot < 0.0, -gate_input, gate_input)
    gate = tl.sigmoid(gate_input)
    if HAS_MASK:
        active = tl.load(token_mask + token_idx * mask_stride)
        gate = tl.where(active, gate, 0.0)

    value_rank = dim_offsets // WIDTH
    value = tl.load(
        gathered
        + (value_rank * num_tokens + token_idx) * gathered_stride_t
        + dim_offsets % WIDTH,
        mask=dim_valid,
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output
        + token_idx * output_stride_t
        + hc_idx * output_stride_h
        + dim_offsets * output_stride_d,
        hidden + gate * value,
        mask=dim_valid,
    )


class RocmEngram(Engram):
    """`Engram` whose TP ranks each keep one dim slice of every key and of the
    value. The gate only needs per-token sums over dim, so the ranks exchange
    those and their value slices instead of the whole projection."""

    # Ranks whose dim slices make up one `wkv` output row.
    kv_shards: int = 1

    def _create_wkv(
        self, input_size: int, quant_config: QuantizationConfig | None, prefix: str
    ) -> ColumnParallelLinear:
        tp_size = get_tensor_model_parallel_world_size()
        if not _shards_engram_gate() or tp_size == 1 or self.use_sequence_parallel:
            return super()._create_wkv(input_size, quant_config, prefix)
        self.kv_shards = tp_size
        return MergedColumnParallelLinear(
            input_size,
            [self.dim] * (self.hc_mult + 1),
            bias=False,
            gather_output=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=prefix,
        )

    def _post_wkv(
        self,
        hidden_states: torch.Tensor,
        kv: torch.Tensor,
        token_mask: torch.Tensor | None,
        num_kv_tokens: int,
    ) -> torch.Tensor:
        if self.kv_shards == 1:
            return super()._post_wkv(hidden_states, kv, token_mask, num_kv_tokens)
        # One all-gather of each rank's value slice and gate partial sums
        # replaces the gather of the whole projection.
        num_tokens, hc_mult, dim = hidden_states.shape
        width = dim // self.kv_shards
        assert kv.shape == (num_tokens, (hc_mult + 1) * width)
        assert token_mask is None or token_mask.shape == (num_tokens,)
        output = torch.empty_like(hidden_states)
        if num_tokens == 0:
            return output

        # [value slice | FP32 (sum key^2, sum hidden*q*k*key) per hc] per token.
        send = kv.new_empty(num_tokens, width + 4 * hc_mult)
        send_f32 = send.view(torch.float32)
        _engram_gate_partials_kernel[(num_tokens * (hc_mult + 1),)](
            hidden_states,
            kv,
            self.q_weight,
            self.k_weight,
            send,
            send_f32,
            hidden_states.stride(0),
            hidden_states.stride(1),
            hidden_states.stride(2),
            kv.stride(0),
            kv.stride(1),
            self.q_weight.stride(0),
            self.q_weight.stride(1),
            self.k_weight.stride(0),
            self.k_weight.stride(1),
            send.stride(0),
            send_f32.stride(0),
            get_tensor_model_parallel_rank() * width,
            WIDTH=width,
            HC_MULT=hc_mult,
            BLOCK_SIZE=triton.next_power_of_2(width),
            num_warps=4,
        )
        gathered = tensor_model_parallel_all_gather(send, dim=0)
        gathered_f32 = gathered.view(torch.float32)

        block_size = triton.next_power_of_2(dim)
        _engram_gate_finish_kernel[(num_tokens * hc_mult,)](
            hidden_states,
            gathered,
            gathered_f32,
            token_mask if token_mask is not None else hidden_states,
            output,
            num_tokens,
            hidden_states.stride(0),
            hidden_states.stride(1),
            hidden_states.stride(2),
            gathered.stride(0),
            gathered_f32.stride(0),
            token_mask.stride(0) if token_mask is not None else 0,
            output.stride(0),
            output.stride(1),
            output.stride(2),
            self.eps,
            self.clamp_value,
            DIM=dim,
            WIDTH=width,
            TP_SIZE=self.kv_shards,
            HC_MULT=hc_mult,
            BLOCK_SIZE=block_size,
            HAS_MASK=token_mask is not None,
            num_warps=8 if block_size >= 2048 else 4,
        )
        return output
