# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F
from einops import rearrange

from vllm.config import get_current_vllm_config
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.lightning_attn import (
    lightning_attention,
    linear_decode_forward_triton,
)
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
from vllm.model_executor.layers.mamba.linear.base import LinearAttention
from vllm.model_executor.layers.minimax_rms_norm import MiniMaxText01RMSNormTP
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.attention.backends.linear_attn import LinearAttentionMetadata


def clear_linear_attention_cache_for_new_sequences(
    kv_cache: torch.Tensor,
    state_indices_tensor: torch.Tensor,
    attn_metadata: LinearAttentionMetadata,
) -> None:
    num_prefills = getattr(attn_metadata, "num_prefills", 0)
    if num_prefills <= 0:
        return

    num_decodes = getattr(attn_metadata, "num_decodes", 0)
    prefill_state_indices = getattr(attn_metadata, "state_indices_tensor_p", None)
    for prefill_idx in range(num_prefills):
        if num_decodes + prefill_idx + 1 >= len(attn_metadata.query_start_loc):
            break
        q_start = attn_metadata.query_start_loc[num_decodes + prefill_idx]
        q_end = attn_metadata.query_start_loc[num_decodes + prefill_idx + 1]
        query_len = q_end - q_start
        context_len = (
            attn_metadata.seq_lens[num_decodes + prefill_idx] - query_len
        )
        if context_len == 0:
            if prefill_state_indices is not None:
                block_to_clear = prefill_state_indices[prefill_idx]
            else:
                block_to_clear = state_indices_tensor[num_decodes + prefill_idx]
            kv_cache[block_to_clear, ...] = 0


@triton.jit
def _linear_attn_decode_spec_step_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    kv_cache_ptr,
    slope_rate_ptr,
    state_indices_ptr,
    query_start_loc_ptr,
    num_accepted_tokens_ptr,
    output_ptr,
    q_start: tl.constexpr,
    D: tl.constexpr,
    q_b_stride,
    q_h_stride,
    q_d_stride,
    k_b_stride,
    k_h_stride,
    k_d_stride,
    v_b_stride,
    v_h_stride,
    v_d_stride,
    cache_b_stride,
    cache_h_stride,
    cache_d0_stride,
    cache_d1_stride,
    state_indices_b_stride,
    state_indices_t_stride,
    output_b_stride,
    output_d_stride,
    DRAFT_IDX: tl.constexpr,
    STATE_WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    req_id = tl.program_id(0)
    head_id = tl.program_id(1)
    block_id = tl.program_id(2)

    req_start = tl.load(query_start_loc_ptr + req_id).to(tl.int64)
    req_end = tl.load(query_start_loc_ptr + req_id + 1).to(tl.int64)
    query_len = req_end - req_start
    if query_len <= DRAFT_IDX:
        return

    dst_slot = tl.load(
        state_indices_ptr
        + req_id * state_indices_b_stride
        + DRAFT_IDX * state_indices_t_stride
    ).to(tl.int64)
    if dst_slot == -1:
        return

    if DRAFT_IDX == 0:
        accepted_offset = tl.load(num_accepted_tokens_ptr + req_id).to(tl.int64) - 1
        accepted_offset = tl.maximum(accepted_offset, 0)
        accepted_offset = tl.minimum(accepted_offset, STATE_WIDTH - 1)
        src_slot = tl.load(
            state_indices_ptr
            + req_id * state_indices_b_stride
            + accepted_offset * state_indices_t_stride
        ).to(tl.int64)
    else:
        src_slot = tl.load(
            state_indices_ptr
            + req_id * state_indices_b_stride
            + (DRAFT_IDX - 1) * state_indices_t_stride
        ).to(tl.int64)
    if src_slot == -1:
        return

    token_idx = req_start - q_start + DRAFT_IDX
    qk_offsets = tl.arange(0, D)
    v_offsets = tl.arange(0, BLOCK_SIZE) + block_id * BLOCK_SIZE
    qk_mask = qk_offsets < D
    v_mask = v_offsets < D
    kv_mask = qk_mask[:, None] & v_mask[None, :]

    q = tl.load(
        q_ptr
        + token_idx * q_b_stride
        + head_id * q_h_stride
        + qk_offsets * q_d_stride,
        mask=qk_mask,
        other=0.0,
    )
    k = tl.load(
        k_ptr
        + token_idx * k_b_stride
        + head_id * k_h_stride
        + qk_offsets * k_d_stride,
        mask=qk_mask,
        other=0.0,
    )
    v = tl.load(
        v_ptr
        + token_idx * v_b_stride
        + head_id * v_h_stride
        + v_offsets * v_d_stride,
        mask=v_mask,
        other=0.0,
    )

    cache_offsets = (
        qk_offsets[:, None] * cache_d0_stride
        + v_offsets[None, :] * cache_d1_stride
    )
    src_cache_ptr = (
        kv_cache_ptr
        + src_slot * cache_b_stride
        + head_id * cache_h_stride
        + cache_offsets
    )
    dst_cache_ptr = (
        kv_cache_ptr
        + dst_slot * cache_b_stride
        + head_id * cache_h_stride
        + cache_offsets
    )

    slope = tl.load(slope_rate_ptr + head_id)
    decay = tl.exp(-slope)
    kv_old = tl.load(src_cache_ptr, mask=kv_mask, other=0.0)
    kv_new = k[:, None] * v[None, :] + decay * kv_old

    output = tl.sum(q[:, None].to(tl.float32) * kv_new, axis=0)
    tl.store(dst_cache_ptr, kv_new, mask=kv_mask)
    tl.store(
        output_ptr
        + token_idx * output_b_stride
        + (head_id * D + v_offsets) * output_d_stride,
        output,
        mask=v_mask,
    )


def _linear_attention_decode_spec(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_cache: torch.Tensor,
    slope_rate: torch.Tensor,
    state_indices_tensor: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    q_start: int,
    q_end: int | None,
    slot_start: int,
    slot_end: int | None,
    block_size: int,
) -> torch.Tensor:
    q_decode = q[q_start:q_end]
    k_decode = k[q_start:q_end]
    v_decode = v[q_start:q_end]
    hidden = torch.empty(
        (q_decode.shape[0], q.shape[1] * q.shape[2]),
        device=q.device,
        dtype=q.dtype,
    )
    hidden.zero_()

    state_indices_tensor = state_indices_tensor[slot_start:slot_end]
    query_start_loc = query_start_loc.to(device=q.device)

    batch_size = state_indices_tensor.shape[0]
    num_heads = q_decode.shape[1]
    head_dim = q_decode.shape[2]
    assert k_decode.shape == (q_decode.shape[0], num_heads, head_dim)
    assert v_decode.shape == (q_decode.shape[0], num_heads, head_dim)
    state_width = state_indices_tensor.shape[1]

    grid = (batch_size, num_heads, triton.cdiv(head_dim, block_size))
    for draft_idx in range(state_width):
        _linear_attn_decode_spec_step_kernel[grid](
            q_decode,
            k_decode,
            v_decode,
            kv_cache,
            slope_rate,
            state_indices_tensor,
            query_start_loc,
            num_accepted_tokens[:batch_size],
            hidden,
            q_start,
            head_dim,
            q_decode.stride(0),
            q_decode.stride(1),
            q_decode.stride(2),
            k_decode.stride(0),
            k_decode.stride(1),
            k_decode.stride(2),
            v_decode.stride(0),
            v_decode.stride(1),
            v_decode.stride(2),
            kv_cache.stride(0),
            kv_cache.stride(1),
            kv_cache.stride(2),
            kv_cache.stride(3),
            state_indices_tensor.stride(0),
            state_indices_tensor.stride(1),
            hidden.stride(0),
            hidden.stride(1),
            DRAFT_IDX=draft_idx,
            STATE_WIDTH=state_width,
            BLOCK_SIZE=block_size,
        )

    return hidden


def linear_attention_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_cache: torch.Tensor,
    slope_rate: torch.Tensor,
    state_indices_tensor: torch.Tensor,
    q_start: int = 0,
    q_end: int | None = None,
    slot_start: int = 0,
    slot_end: int | None = None,
    block_size: int = 32,
    num_accepted_tokens: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
) -> torch.Tensor:
    if (
        state_indices_tensor.dim() > 1
        and num_accepted_tokens is not None
        and query_start_loc is not None
    ):
        return _linear_attention_decode_spec(
            q,
            k,
            v,
            kv_cache,
            slope_rate,
            state_indices_tensor,
            query_start_loc,
            num_accepted_tokens,
            q_start,
            q_end,
            slot_start,
            slot_end,
            block_size,
        )

    q = q[q_start:q_end].unsqueeze(2).contiguous()
    k = k[q_start:q_end].unsqueeze(2).contiguous()
    v = v[q_start:q_end].unsqueeze(2).contiguous()
    slot_id = state_indices_tensor[slot_start:slot_end]
    return linear_decode_forward_triton(
        q, k, v, kv_cache, slope_rate, slot_id, block_size
    )


def linear_attention_prefill_and_mix(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_cache: torch.Tensor,
    state_indices_tensor: torch.Tensor,
    attn_metadata: LinearAttentionMetadata,
    slope_rate: torch.Tensor,
    block_size: int,
    decode_fn: Callable[..., torch.Tensor],
    prefix_fn: Callable[..., torch.Tensor],
    layer_idx: int | None = None,
) -> torch.Tensor:
    hidden = []
    req_offset = getattr(attn_metadata, "num_decodes", 0)
    prefill_state_indices = getattr(attn_metadata, "state_indices_tensor_p", None)
    for _prefill_idx in range(getattr(attn_metadata, "num_prefills", 0)):
        if req_offset + _prefill_idx + 1 >= len(attn_metadata.query_start_loc):
            break
        if prefill_state_indices is not None and _prefill_idx >= len(
            prefill_state_indices
        ):
            break
        if prefill_state_indices is None and _prefill_idx >= len(state_indices_tensor):
            break
        _start = attn_metadata.query_start_loc[req_offset + _prefill_idx]
        _end = attn_metadata.query_start_loc[req_offset + _prefill_idx + 1]
        if prefill_state_indices is not None:
            slot_id = prefill_state_indices[_prefill_idx]
        else:
            slot_id = state_indices_tensor[req_offset + _prefill_idx]
        qs = q[_start:_end].transpose(0, 1).contiguous()
        ks = k[_start:_end].transpose(0, 1).contiguous()
        vs = v[_start:_end].transpose(0, 1).contiguous()
        slice_layer_cache = kv_cache[slot_id, ...]
        out_slice = prefix_fn(
            qs,
            ks,
            vs,
            slice_layer_cache,
            slope_rate,
            block_size,
            layer_idx=layer_idx,
        )
        hidden.append(out_slice.contiguous())

    if attn_metadata.num_decode_tokens > 0:
        hidden_decode = decode_fn(
            q, k, v, kv_cache, state_indices_tensor, attn_metadata
        )
        hidden.insert(0, hidden_decode)

    if not hidden:
        return torch.empty(
            (0, q.size(1) * q.size(2)), device=q.device, dtype=q.dtype
        )

    hidden = torch.concat(hidden, dim=0).contiguous()
    return hidden


class MiniMaxText01LinearKernel:
    @staticmethod
    def jit_linear_forward_prefix(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_caches: torch.Tensor,
        slope_rate: torch.Tensor,
        block_size: int,
        layer_idx: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        slope_rate = slope_rate.to(torch.float32)
        should_pad_dim = q.dim() == 3
        if should_pad_dim:
            q = q.unsqueeze(0)
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)
        b, h, n, d = q.shape
        e = d
        kv_history = kv_caches.reshape(1, h, d, e).contiguous()
        output, kv_history = lightning_attention(
            q, k, v, slope_rate, block_size=block_size, kv_history=kv_history
        )
        kv_caches.copy_(kv_history[:, :, -1, :, :].reshape(h, d, e))
        assert output.shape[0] == 1, "batch size must be 1"
        return rearrange(output.squeeze(0), "h n d -> n (h d)")


@PluggableLayer.register("minimax_text_01_attention")
class MiniMaxText01LinearAttention(LinearAttention):
    def __init__(
        self,
        config,
        vllm_config,
        prefix: str = "linear_attn",
    ) -> None:
        super().__init__(config, vllm_config, prefix)

        self.tp_heads = self.num_heads // self.tp_size
        self.qkv_size = self.num_heads * self.head_dim
        self.tp_hidden = self.head_dim * self.tp_heads

        self.qkv_proj = ColumnParallelLinear(
            self.hidden_size,
            self.hidden_inner_size * 3,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.output_gate = ColumnParallelLinear(
            self.hidden_size,
            self.hidden_inner_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.output_gate",
        )
        self.out_proj = RowParallelLinear(
            self.hidden_inner_size,
            self.hidden_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.out_proj",
        )
        self.norm = MiniMaxText01RMSNormTP(
            self.hidden_inner_size,
            eps=1e-5,
        )

        slope_rate = MiniMaxText01LinearAttention._build_slope_tensor(self.num_heads)
        if self.num_hidden_layers <= 1:
            self.slope_rate = slope_rate * (1 + 1e-5)
        else:
            self.slope_rate = slope_rate * (
                1 - self.layer_idx / (self.num_hidden_layers - 1) + 1e-5
            )
        self.tp_slope = self.slope_rate[
            self.tp_rank * self.tp_heads : (self.tp_rank + 1) * self.tp_heads
        ].contiguous()

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    @staticmethod
    def weight_direct_load(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        assert param.size() == loaded_weight.size()
        param.data.copy_(loaded_weight)
        return

    @staticmethod
    def _build_slope_tensor(n_attention_heads: int):
        def get_slopes(n):
            def get_slopes_power_of_2(n):
                start = 2 ** (-(2 ** -(math.log2(n) - 3)))
                ratio = start
                return [start * ratio**i for i in range(n)]

            if math.log2(n).is_integer():
                return get_slopes_power_of_2(n)
            else:
                closest_power_of_2 = 2 ** math.floor(math.log2(n))
                return (
                    get_slopes_power_of_2(closest_power_of_2)
                    + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
                )

        slopes = torch.tensor(
            get_slopes(n_attention_heads), dtype=torch.float32
        ).reshape(n_attention_heads, 1, 1)
        return slopes

    def _prefill_and_mix_infer(
        self, q, k, v, kv_cache, state_indices_tensor, attn_metadata
    ):
        return linear_attention_prefill_and_mix(
            q=q,
            k=k,
            v=v,
            kv_cache=kv_cache,
            state_indices_tensor=state_indices_tensor,
            attn_metadata=attn_metadata,
            slope_rate=self.tp_slope,
            block_size=self.BLOCK,
            decode_fn=self._decode_infer,
            prefix_fn=MiniMaxText01LinearKernel.jit_linear_forward_prefix,
            layer_idx=self.layer_idx,
        )

    def _decode_infer(self, q, k, v, kv_cache, state_indices_tensor, attn_metadata):
        decode_state_indices = getattr(attn_metadata, "state_indices_tensor_d", None)
        if decode_state_indices is None:
            decode_state_indices = state_indices_tensor
        hidden = linear_attention_decode(
            q,
            k,
            v,
            kv_cache,
            self.tp_slope,
            decode_state_indices,
            q_start=0,
            q_end=attn_metadata.num_decode_tokens,
            slot_start=0,
            slot_end=attn_metadata.num_decodes,
            block_size=32,
            num_accepted_tokens=getattr(attn_metadata, "num_accepted_tokens", None),
            query_start_loc=getattr(attn_metadata, "query_start_loc_d", None),
        )
        return hidden

    def forward(
        self, hidden_states: torch.Tensor, output: torch.Tensor, positions: torch.Tensor
    ) -> None:
        torch.ops.vllm.linear_attention(
            hidden_states,
            output,
            positions,
            self.prefix,
        )

    def _forward(
        self, hidden_states: torch.Tensor, output: torch.Tensor, positions: torch.Tensor
    ) -> None:
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata
        attn_metadata: AttentionMetadata | None = None
        if attn_metadata_raw is not None:
            assert isinstance(attn_metadata_raw, dict)
            attn_metadata = attn_metadata_raw[self.prefix]
            assert isinstance(attn_metadata, LinearAttentionMetadata)
            num_actual_tokens = (
                attn_metadata.num_prefill_tokens + attn_metadata.num_decode_tokens
            )
        else:
            num_actual_tokens = hidden_states.shape[0]

        qkv, _ = self.qkv_proj(hidden_states[:num_actual_tokens])
        qkv32 = qkv.to(torch.float32)
        qkvact = torch.nn.functional.silu(qkv32)
        qkvact = qkvact.view((qkv.shape[0], self.tp_heads, -1))
        q, k, v = torch.split(qkvact, [self.head_dim] * 3, dim=-1)
        if attn_metadata is not None:
            kv_cache = self.kv_cache[0]
            state_indices_tensor = attn_metadata.state_indices_tensor
            clear_linear_attention_cache_for_new_sequences(
                kv_cache, state_indices_tensor, attn_metadata
            )

        decode_only = getattr(attn_metadata, "num_prefills", 0) == 0
        if attn_metadata is None:
            hidden = torch.empty(
                (q.shape[0], q.shape[1] * q.shape[2]), device=q.device, dtype=q.dtype
            )
        else:
            if not decode_only:
                hidden = self._prefill_and_mix_infer(
                    q, k, v, kv_cache, state_indices_tensor, attn_metadata
                )
            else:
                hidden = self._decode_infer(
                    q, k, v, kv_cache, state_indices_tensor, attn_metadata
                )
        hidden = self.norm._forward(hidden)
        gate, _ = self.output_gate(hidden_states[:num_actual_tokens])
        hidden = F.sigmoid(gate) * hidden
        hidden = hidden.to(hidden_states.dtype)

        output[:num_actual_tokens], _ = self.out_proj(hidden)


def linear_attention(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    positions: torch.Tensor,
    layer_name: str,
) -> None:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    self._forward(hidden_states=hidden_states, output=output, positions=positions)


def linear_attention_fake(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    positions: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="linear_attention",
    op_func=linear_attention,
    mutates_args=["output"],
    fake_impl=linear_attention_fake,
)
