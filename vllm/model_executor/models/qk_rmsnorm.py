import torch
import torch_npu

import triton
import triton.language as tl
import triton.runtime.driver as driver

import torch_npu._inductor

from vllm.forward_context import get_forward_context


@triton.jit
def split_qkv_rmsnorm_rope_kernel(
    input_ptr,
    sin_ptr,
    cos_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    q_weight_ptr,
    q_bias_ptr,
    k_weight_ptr,
    k_bias_ptr,
    batch_size,
    q_hidden_size: tl.constexpr,
    kv_hidden_size: tl.constexpr,
    total_hidden_size: tl.constexpr,
    eps: tl.constexpr,
    Q_BLOCK_SIZE: tl.constexpr,
    KV_BLOCK_SIZE: tl.constexpr,
    BIAS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HALF_HEAD_DIM: tl.constexpr,
):
    row_pid = tl.program_id(0)
    col_pid = tl.program_id(1)
    row_step = tl.num_programs(0)
    # q
    weight_values = tl.load(q_weight_ptr + tl.arange(0, HEAD_DIM))
    if BIAS:
        bias_values = tl.load(q_bias_ptr + tl.arange(0, HEAD_DIM))
    input_offset = row_pid * total_hidden_size
    output_offset = row_pid * q_hidden_size
    input_offset_step = row_step * total_hidden_size
    output_offset_step = row_step * q_hidden_size
    for row_idx in tl.range(row_pid, batch_size, row_step):
        col_indices = col_pid * Q_BLOCK_SIZE + tl.arange(0, Q_BLOCK_SIZE)
        valid_mask = col_indices < q_hidden_size
        input_values = (
            tl.load(input_ptr + input_offset + col_indices, mask=valid_mask, other=0.0)
            .to(tl.float32)
            .reshape(Q_BLOCK_SIZE // HEAD_DIM, HEAD_DIM)
        )
        squares = input_values * input_values
        variances = tl.sum(squares, axis=1) / HEAD_DIM
        reciprocal_std = (1 / tl.sqrt(variances + eps)).reshape(
            Q_BLOCK_SIZE // HEAD_DIM, 1
        )
        normalized_values = (
            input_values * reciprocal_std
        )  # (Q_BLOCK_SIZE//HEAD_DIM, HEAD_DIM)
        if BIAS:
            normalized_values = (normalized_values * weight_values + bias_values).to(
                tl.bfloat16
            )
        else:
            normalized_values = (normalized_values * weight_values).to(tl.bfloat16)
        # rope
        sc_offsets = row_idx * HEAD_DIM + tl.arange(0, HEAD_DIM)
        sin = (tl.load(sin_ptr + sc_offsets)).reshape(1, HEAD_DIM)
        cos = (tl.load(cos_ptr + sc_offsets)).reshape(1, HEAD_DIM)
        x1 = tl.extract_slice(
            normalized_values,
            offsets=(0, 0),
            sizes=(Q_BLOCK_SIZE // HEAD_DIM, HALF_HEAD_DIM),
            strides=(1, 1),
        )
        x2 = tl.extract_slice(
            normalized_values,
            offsets=(0, HALF_HEAD_DIM),
            sizes=(Q_BLOCK_SIZE // HEAD_DIM, HALF_HEAD_DIM),
            strides=(1, 1),
        )
        cat_x = tl.zeros((Q_BLOCK_SIZE // HEAD_DIM, HEAD_DIM), dtype=tl.bfloat16)
        cat_x = tl.insert_slice(
            cat_x,
            -x2,
            offsets=(0, 0),
            sizes=(Q_BLOCK_SIZE // HEAD_DIM, HALF_HEAD_DIM),
            strides=(1, 1),
        )
        cat_x = tl.insert_slice(
            cat_x,
            x1,
            offsets=(0, HALF_HEAD_DIM),
            sizes=(Q_BLOCK_SIZE // HEAD_DIM, HALF_HEAD_DIM),
            strides=(1, 1),
        )
        roped_q = cat_x * sin + normalized_values * cos
        # store
        tl.store(
            q_ptr + output_offset + col_indices,
            roped_q.reshape(Q_BLOCK_SIZE).to(q_ptr.dtype.element_ty),
            mask=valid_mask,
        )
        input_offset += input_offset_step
        output_offset += output_offset_step

    # k
    weight_values = tl.load(k_weight_ptr + tl.arange(0, HEAD_DIM))
    if BIAS:
        bias_values = tl.load(k_bias_ptr + tl.arange(0, HEAD_DIM))
    input_offset = row_pid * total_hidden_size + q_hidden_size
    output_offset = row_pid * kv_hidden_size
    output_offset_step = row_step * kv_hidden_size
    for row_idx in tl.range(row_pid, batch_size, row_step):
        col_indices = col_pid * KV_BLOCK_SIZE + tl.arange(0, KV_BLOCK_SIZE)
        valid_mask = col_indices < kv_hidden_size
        input_values = (
            tl.load(input_ptr + input_offset + col_indices, mask=valid_mask, other=0.0)
            .to(tl.float32)
            .reshape(KV_BLOCK_SIZE // HEAD_DIM, HEAD_DIM)
        )
        squares = input_values * input_values
        variances = tl.sum(squares, axis=1) / HEAD_DIM
        reciprocal_std = (1 / tl.sqrt(variances + eps)).reshape(
            KV_BLOCK_SIZE // HEAD_DIM, 1
        )
        normalized_values = (
            input_values * reciprocal_std
        )  # (KV_BLOCK_SIZE/HEAD_DIM, HEAD_DIM)
        if BIAS:
            normalized_values = (normalized_values * weight_values + bias_values).to(
                tl.bfloat16
            )
        else:
            normalized_values = (normalized_values * weight_values).to(tl.bfloat16)
        # # rope
        sc_offsets = row_idx * HEAD_DIM + tl.arange(0, HEAD_DIM)
        sin = (tl.load(sin_ptr + sc_offsets)).reshape(1, HEAD_DIM)
        cos = (tl.load(cos_ptr + sc_offsets)).reshape(1, HEAD_DIM)
        x1 = tl.extract_slice(
            normalized_values,
            offsets=(0, 0),
            sizes=(KV_BLOCK_SIZE // HEAD_DIM, HALF_HEAD_DIM),
            strides=(1, 1),
        )
        x2 = tl.extract_slice(
            normalized_values,
            offsets=(0, HALF_HEAD_DIM),
            sizes=(KV_BLOCK_SIZE // HEAD_DIM, HALF_HEAD_DIM),
            strides=(1, 1),
        )
        cat_x = tl.zeros((KV_BLOCK_SIZE // HEAD_DIM, HEAD_DIM), dtype=tl.bfloat16)
        cat_x = tl.insert_slice(
            cat_x,
            -x2,
            offsets=(0, 0),
            sizes=(KV_BLOCK_SIZE // HEAD_DIM, HALF_HEAD_DIM),
            strides=(1, 1),
        )
        cat_x = tl.insert_slice(
            cat_x,
            x1,
            offsets=(0, HALF_HEAD_DIM),
            sizes=(KV_BLOCK_SIZE // HEAD_DIM, HALF_HEAD_DIM),
            strides=(1, 1),
        )
        roped_k = cat_x * sin + normalized_values * cos
        # store
        tl.store(
            k_ptr + output_offset + col_indices,
            roped_k.to(tl.bfloat16).reshape(KV_BLOCK_SIZE),
            mask=valid_mask,
        )
        input_offset += input_offset_step
        output_offset += output_offset_step

    # v
    input_offset = row_pid * total_hidden_size + q_hidden_size + kv_hidden_size
    output_offset = row_pid * kv_hidden_size
    for _ in tl.range(row_pid, batch_size, row_step):
        col_indices = col_pid * KV_BLOCK_SIZE + tl.arange(0, KV_BLOCK_SIZE)
        valid_mask = col_indices < kv_hidden_size
        input_values = tl.load(
            input_ptr + input_offset + col_indices, mask=valid_mask, other=0.0
        )
        tl.store(v_ptr + output_offset + col_indices, input_values, mask=valid_mask)
        input_offset += input_offset_step
        output_offset += output_offset_step


kernels = {}

def get_npu_properties():
    device = torch.npu.current_device()
    return driver.active.utils.get_device_properties(device)

num_vectorcore = get_npu_properties()["num_vectorcore"]

def split_qkv_rmsnorm_rope(
    input,
    sin,
    cos,
    q_weight,
    k_weight,
    q_hidden_size,
    kv_hidden_size,
    head_dim,
    eps,
    q_bias,
    k_bias,
):

    KV_BLOCK_SIZE = triton.next_power_of_2(head_dim)
    assert KV_BLOCK_SIZE == head_dim
    assert q_hidden_size % kv_hidden_size == 0
    Q_BLOCK_SIZE = q_hidden_size // kv_hidden_size * head_dim
    batch_size = input.shape[0]
    total_hidden_size = q_hidden_size + kv_hidden_size * 2
    q_output = torch.empty(
        batch_size, q_hidden_size, device=input.device, dtype=input.dtype
    )
    k_output = torch.empty(
        batch_size, kv_hidden_size, device=input.device, dtype=input.dtype
    )
    v_output = torch.empty(
        batch_size, kv_hidden_size, device=input.device, dtype=input.dtype
    )
    n_cols = kv_hidden_size // KV_BLOCK_SIZE
    assert num_vectorcore % n_cols == 0
    n_rows = num_vectorcore // n_cols
    BIAS = q_bias is not None

    split_qkv_rmsnorm_rope_kernel[(n_rows, n_cols, 1)](
            input,
            sin,
            cos,
            q_output,
            k_output,
            v_output,
            q_weight,
            q_bias,
            k_weight,
            k_bias,
            batch_size,
            q_hidden_size,
            kv_hidden_size,
            total_hidden_size,
            eps,
            Q_BLOCK_SIZE,
            KV_BLOCK_SIZE,
            BIAS,
            head_dim,
            head_dim // 2,
        )
    return q_output, k_output, v_output


# example code to replace Qwen3MoeAttention.forward
def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        forward_context = get_forward_context()
        is_first_layer = forward_context.is_first_layer
        if is_first_layer:
            cos_sin = self.rotary_emb.cos_sin_cache.index_select(0, positions)
            last_dim = cos_sin.size()[-1]
            cos, sin = cos_sin.reshape(-1, 2, last_dim // 2).repeat(
                1, 1, 2).chunk(2, dim=-2)
            # BSNH
            self.rotary_emb.cos = cos.view(1, -1, 1, last_dim).contiguous()
            self.rotary_emb.sin = sin.view(1, -1, 1, last_dim).contiguous()
            forward_context.is_first_layer = False
        q, k, v = split_qkv_rmsnorm_rope(
            qkv,
            self.rotary_emb.sin,
            self.rotary_emb.cos,
            self.q_norm.weight,
            self.k_norm.weight,
            self.q_size,
            self.kv_size,
            self.head_dim,
            self.q_norm.variance_epsilon,
            None,
            None
        )
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output
