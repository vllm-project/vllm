"""Helion ReplaySSM decode for Kimi Delta Attention.

The public :func:`helion_fused_recurrent_kda_replayssm_decode` mirrors
``fused_recurrent_linear_replayssm_decode(..., is_kda=True)`` from
``sglang.kernels.ops.attention.fla.fused_recurrent_linear_replayssm``.  That
Triton kernel is gate-generic (GDN scalar gate / KDA per-K gate); this module
implements the KDA path only, and additionally supports the bounded gate,
which the Triton ReplaySSM kernel does not expose.
"""

from __future__ import annotations

import helion
import helion.language as hl
import torch

from vllm.kernels.helion.ops.kda import KDA_SMALL_VALUE_HEAD_THRESHOLD
from vllm.kernels.helion.ops.kda.kda_decode import (
    validate_packed_decode_inputs,
)

# SGLang initializes torch.distributed, but this kernel has no collectives.
_IGNORED_WARNINGS = [helion.exc.ProcessGroupNameNotFound]
_LOG2_E = 1.4426950408889634

# Indexing and eviction-policy lists are positional in the traced load order.
# Retune them if the ReplaySSM body gains, loses, or reorders loads.
_KDA_REPLAYSSM_FP32_CONFIG = helion.Config(
    atomic_indexing=[],
    block_sizes=[32, 128],
    indexing=[
        "tensor_descriptor",
        "tensor_descriptor",
        "tensor_descriptor",
        "tensor_descriptor",
        "tensor_descriptor",
        "pointer",
        "pointer",
        "tensor_descriptor",
        "pointer",
        "tensor_descriptor",
        "tensor_descriptor",
        "tensor_descriptor",
        "pointer",
        "tensor_descriptor",
        "pointer",
        "tensor_descriptor",
        "pointer",
        "pointer",
        "tensor_descriptor",
        "tensor_descriptor",
        "pointer",
        "pointer",
        "tensor_descriptor",
        "tensor_descriptor",
        "tensor_descriptor",
        "pointer",
        "pointer",
        "tensor_descriptor",
        "tensor_descriptor",
        "tensor_descriptor",
        "tensor_descriptor",
    ],
    l2_groupings=[4],
    load_eviction_policies=[
        "last",
        "last",
        "first",
        "first",
        "",
        "first",
        "",
        "first",
        "first",
        "",
        "last",
        "first",
        "last",
        "first",
        "last",
        "first",
        "",
        "first",
        "last",
        "",
        "last",
        "first",
        "last",
        "first",
        "first",
    ],
    loop_orders=[[1, 2, 0]],
    num_warps=1,
    num_stages=1,
    pid_type="flat",
    range_flattens=[None, None],
    range_multi_buffers=[None, False],
    range_num_stages=[0, 3],
    range_unroll_factors=[0, 0],
)

_KDA_REPLAYSSM_BF16_CONFIG = helion.Config(
    atomic_indexing=[],
    block_sizes=[64, 64],
    indexing=[
        "pointer",
        "pointer",
        "tensor_descriptor",
        "pointer",
        "tensor_descriptor",
        "pointer",
        "tensor_descriptor",
        "pointer",
        "pointer",
        "tensor_descriptor",
        "tensor_descriptor",
        "pointer",
        "tensor_descriptor",
        "tensor_descriptor",
        "pointer",
        "pointer",
        "tensor_descriptor",
        "tensor_descriptor",
        "pointer",
        "tensor_descriptor",
        "tensor_descriptor",
        "pointer",
        "tensor_descriptor",
        "pointer",
        "pointer",
        "pointer",
        "pointer",
        "pointer",
        "pointer",
        "pointer",
        "pointer",
    ],
    l2_groupings=[2],
    load_eviction_policies=[
        "first",
        "last",
        "last",
        "first",
        "",
        "",
        "last",
        "last",
        "first",
        "first",
        "first",
        "",
        "last",
        "first",
        "last",
        "first",
        "",
        "first",
        "",
        "first",
        "first",
        "",
        "",
        "first",
        "last",
    ],
    loop_orders=[[1, 2, 0]],
    num_stages=1,
    num_warps=1,
    pid_type="flat",
    range_flattens=[None, False],
    range_multi_buffers=[None, False],
    range_num_stages=[0, 0],
    range_unroll_factors=[0, 4],
)

# Small-head BF16 ReplaySSM uses the FP32 tile schedule with direct PID order.
_KDA_REPLAYSSM_BF16_SMALL_HEAD_CONFIG = helion.Config.from_dict(
    {**_KDA_REPLAYSSM_FP32_CONFIG, "l2_groupings": [1]}
)


def _log_decay(
    *,
    raw_gate: torch.Tensor,
    decay_rate: torch.Tensor,
    lower_bound: float,
    use_lower_bound: hl.constexpr,
) -> torch.Tensor:
    if use_lower_bound:
        return lower_bound * torch.sigmoid(decay_rate * raw_gate)
    gate_exp = torch.exp2(raw_gate * _LOG2_E)
    softplus = torch.where(
        raw_gate <= 20.0,
        torch.log(1.0 + gate_exp),
        raw_gate,
    )
    return -decay_rate * softplus


def _load_log_decay(
    *,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    batch_index: torch.Tensor,
    value_head: torch.Tensor,
    key_indices: torch.Tensor,
    key_valid: torch.Tensor,
    key_dim: int,
    decay_rate: torch.Tensor,
    lower_bound: float,
    use_lower_bound: hl.constexpr,
) -> torch.Tensor:
    """Load the current raw gate and apply the selected KDA gate contract."""
    raw_gate = hl.load(
        a,
        [batch_index, value_head * key_dim + key_indices],
        extra_mask=key_valid,
    ).float()
    raw_gate = (
        raw_gate
        + hl.load(
            dt_bias,
            [value_head * key_dim + key_indices],
            extra_mask=key_valid,
        ).float()
    )
    return _log_decay(
        raw_gate=raw_gate,
        decay_rate=decay_rate,
        lower_bound=lower_bound,
        use_lower_bound=use_lower_bound,
    )


def _helion_fused_recurrent_kda_replayssm_decode_body(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    d_cache: torch.Tensor,
    k_cache: torch.Tensor,
    g_cache: torch.Tensor,
    out: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    write_pos: torch.Tensor,
    force_flush: torch.Tensor | None,
    lower_bound: float,
    cache_block: hl.constexpr,  # pyrefly: ignore[bad-function-definition]
    use_qk_l2norm_in_kernel: hl.constexpr,  # pyrefly: ignore[bad-function-definition]
    use_lower_bound: hl.constexpr,  # pyrefly: ignore[bad-function-definition]
) -> torch.Tensor:
    """Reconstruct the buffered state, emit one token, and flush if needed."""
    B = mixed_qkv.size(0)
    HV = hl.specialize(initial_state.size(1))
    V = hl.specialize(initial_state.size(2))
    K = hl.specialize(initial_state.size(3))
    H = hl.specialize(k_cache.size(1))
    cache_length = hl.specialize(d_cache.size(2))
    heads_per_q = HV // H

    # Keep storage setup inline so the generated host wrapper does not call
    # Python helpers on every eager invocation.
    state_strides = hl.specialize(
        (
            initial_state.stride(0),
            initial_state.stride(1),
            initial_state.stride(2),
            initial_state.stride(3),
        )
    )
    d_strides = hl.specialize(
        (
            d_cache.stride(0),
            d_cache.stride(1),
            d_cache.stride(2),
            d_cache.stride(3),
        )
    )
    k_strides = hl.specialize(
        (
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(2),
            k_cache.stride(3),
        )
    )
    g_strides = hl.specialize(
        (
            g_cache.stride(0),
            g_cache.stride(1),
            g_cache.stride(2),
            g_cache.stride(3),
        )
    )
    state_storage_size = (
        (initial_state.size(0) - 1) * state_strides[0]
        + (HV - 1) * state_strides[1]
        + (V - 1) * state_strides[2]
        + (K - 1) * state_strides[3]
        + 1
    )
    d_storage_size = (
        (d_cache.size(0) - 1) * d_strides[0]
        + (HV - 1) * d_strides[1]
        + (cache_length - 1) * d_strides[2]
        + (V - 1) * d_strides[3]
        + 1
    )
    k_storage_size = (
        (k_cache.size(0) - 1) * k_strides[0]
        + (H - 1) * k_strides[1]
        + (cache_length - 1) * k_strides[2]
        + (K - 1) * k_strides[3]
        + 1
    )
    g_storage_size = (
        (g_cache.size(0) - 1) * g_strides[0]
        + (HV - 1) * g_strides[1]
        + (cache_length - 1) * g_strides[2]
        + (K - 1) * g_strides[3]
        + 1
    )
    # Omitting storage_offset preserves each source view's existing offset,
    # which is required for envelope and page-major cache views.
    state_storage = initial_state.as_strided([state_storage_size], [1])
    d_storage = d_cache.as_strided([d_storage_size], [1])
    k_storage = k_cache.as_strided([k_storage_size], [1])
    g_storage = g_cache.as_strided([g_storage_size], [1])

    hl.specialize(
        (
            mixed_qkv.stride(0),
            mixed_qkv.stride(1),
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            A_log.stride(0),
            dt_bias.stride(0),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            ssm_state_indices.stride(0),
            write_pos.stride(0),
        )
    )

    block_v = hl.register_block_size(16, V)
    block_k = hl.register_block_size(16, K)

    for tile_b, tile_hv, tile_v in hl.tile([B, HV, V], block_size=[1, 1, block_v]):
        i_b = tile_b.id
        i_hv = tile_hv.id
        i_h = i_hv // heads_per_q
        state_index = ssm_state_indices[i_b].long()
        v_valid = tile_v.index < V

        if state_index < 0:
            hl.store(
                out,
                [i_b, 0, i_hv, tile_v.index],
                hl.zeros([tile_v], dtype=out.dtype),
                extra_mask=v_valid,
            )
        else:
            cursor = write_pos[i_b].long()
            is_flush = cursor == cache_length - 1
            if force_flush is not None:
                is_flush = is_flush | (force_flush[i_b] != 0)
            should_append = is_flush == 0

            cache_positions = hl.arange(cache_block)
            cache_valid = cache_positions < cursor
            d_offsets = (
                state_index * d_strides[0]
                + i_hv * d_strides[1]
                + cache_positions[:, None] * d_strides[2]
                + tile_v.index[None, :] * d_strides[3]
            )
            d_values = hl.load(
                d_storage,
                [d_offsets],
                extra_mask=cache_valid[:, None] & v_valid[None, :],
            ).float()
            # Advanced loads preserve tensor-axis order: d_cache contributes
            # [L, V], while the reconstruction dot consumes [V, L].
            d_dot = d_values.T.to(out.dtype)

            full_k = hl.arange(K)
            q_offsets = i_h * K + full_k
            full_k_offsets = H * K + i_h * K + full_k
            q_full = mixed_qkv[i_b, q_offsets].float()
            k_full = mixed_qkv[i_b, full_k_offsets].float()
            if use_qk_l2norm_in_kernel:
                q_rnorm = 1.0 / torch.sqrt((q_full * q_full).sum() + 1e-6)
                k_rnorm = 1.0 / torch.sqrt((k_full * k_full).sum() + 1e-6)
            else:
                q_rnorm = 1.0
                k_rnorm = 1.0

            value_offsets = 2 * H * K + i_hv * V + tile_v.index
            value = hl.load(
                mixed_qkv,
                [i_b, value_offsets],
                extra_mask=v_valid,
            ).float()
            # ReplaySSM Triton rounds beta through the input dtype before FP32
            # accumulation; packed KDA intentionally keeps beta in FP32.
            beta = torch.sigmoid(b[i_b, i_hv].float()).to(b.dtype).float()
            A = torch.exp2(A_log[i_hv].float() * _LOG2_E)

            state_q = hl.zeros([tile_v], dtype=torch.float32)
            state_k = hl.zeros([tile_v], dtype=torch.float32)
            current_kq = hl.zeros([], dtype=torch.float32)

            # Reconstruct each K tile directly from the checkpoint and ring.
            # The full state is never materialized outside registers.
            for tile_k in hl.tile(K, block_size=block_k):
                k_valid = tile_k.index < K
                q_value = hl.load(
                    mixed_qkv,
                    [i_b, i_h * K + tile_k.index],
                    extra_mask=k_valid,
                ).float()
                k_value = hl.load(
                    mixed_qkv,
                    [i_b, H * K + i_h * K + tile_k.index],
                    extra_mask=k_valid,
                ).float()
                q_value = q_value * q_rnorm
                k_value = k_value * k_rnorm
                q_scaled = q_value * scale
                current_kq = current_kq + (k_value * q_scaled).sum()

                cache_mask = cache_valid[:, None] & k_valid[None, :]
                g_offsets = (
                    state_index * g_strides[0]
                    + i_hv * g_strides[1]
                    + cache_positions[:, None] * g_strides[2]
                    + tile_k.index[None, :] * g_strides[3]
                )
                cached_k_offsets = (
                    state_index * k_strides[0]
                    + i_h * k_strides[1]
                    + cache_positions[:, None] * k_strides[2]
                    + tile_k.index[None, :] * k_strides[3]
                )
                state_offsets = (
                    state_index * state_strides[0]
                    + i_hv * state_strides[1]
                    + tile_v.index[:, None] * state_strides[2]
                    + tile_k.index[None, :] * state_strides[3]
                )
                state_mask = v_valid[:, None] & k_valid[None, :]
                cached_g = hl.load(
                    g_storage,
                    [g_offsets],
                    extra_mask=cache_mask,
                ).float()
                gate_prefix = torch.cumsum(cached_g, dim=0)
                gate_total = cached_g.sum(0)
                replay_decay = torch.where(
                    cache_mask,
                    torch.exp2((gate_total[None, :] - gate_prefix) * _LOG2_E),
                    0.0,
                )
                total_decay = torch.exp2(gate_total * _LOG2_E)
                cached_k = hl.load(
                    k_storage,
                    [cached_k_offsets],
                    extra_mask=cache_mask,
                ).float()
                cached_k = (cached_k * replay_decay).to(out.dtype)
                state = hl.load(
                    state_storage,
                    [state_offsets],
                    extra_mask=state_mask,
                ).float()
                state = state * total_decay[None, :]
                state = state + hl.dot(
                    d_dot,
                    cached_k,
                    out_dtype=torch.float32,
                )

                current_gate = _load_log_decay(
                    a=a,
                    dt_bias=dt_bias,
                    batch_index=i_b,
                    value_head=i_hv,
                    key_indices=tile_k.index,
                    key_valid=k_valid,
                    key_dim=K,
                    decay_rate=A,
                    lower_bound=lower_bound,
                    use_lower_bound=use_lower_bound,
                )
                current_decay = torch.exp2(current_gate * _LOG2_E)
                q_effective = q_scaled * current_decay
                k_effective = k_value * current_decay
                state_q = state_q + (state * q_effective[None, :]).sum(-1)
                state_k = state_k + (state * k_effective[None, :]).sum(-1)

                if should_append:
                    if tile_v.id == 0:
                        if i_hv == i_h * heads_per_q:
                            current_k_offsets = (
                                state_index * k_strides[0]
                                + i_h * k_strides[1]
                                + cursor * k_strides[2]
                                + tile_k.index * k_strides[3]
                            )
                            hl.store(
                                k_storage,
                                [current_k_offsets],
                                k_value.to(k_cache.dtype),
                                extra_mask=(cursor < cache_length) & k_valid,
                            )
                        current_g_offsets = (
                            state_index * g_strides[0]
                            + i_hv * g_strides[1]
                            + cursor * g_strides[2]
                            + tile_k.index * g_strides[3]
                        )
                        hl.store(
                            g_storage,
                            [current_g_offsets],
                            current_gate,
                            extra_mask=(cursor < cache_length) & k_valid,
                        )

            delta = beta * (value - state_k)
            output = state_q + delta * current_kq
            hl.store(
                out,
                [i_b, 0, i_hv, tile_v.index],
                output.to(out.dtype),
                extra_mask=v_valid,
            )

            if is_flush:
                # Reconstruct again now that the current delta is available,
                # then fold the current rank-one update into the checkpoint.
                # Helion cannot lower the cumsum through a device helper without
                # creating a separate unsupported scan subgraph.
                for tile_k in hl.tile(K, block_size=block_k):
                    k_valid = tile_k.index < K
                    k_value = hl.load(
                        mixed_qkv,
                        [i_b, H * K + i_h * K + tile_k.index],
                        extra_mask=k_valid,
                    ).float()
                    k_value = k_value * k_rnorm

                    cache_mask = cache_valid[:, None] & k_valid[None, :]
                    g_offsets = (
                        state_index * g_strides[0]
                        + i_hv * g_strides[1]
                        + cache_positions[:, None] * g_strides[2]
                        + tile_k.index[None, :] * g_strides[3]
                    )
                    cached_k_offsets = (
                        state_index * k_strides[0]
                        + i_h * k_strides[1]
                        + cache_positions[:, None] * k_strides[2]
                        + tile_k.index[None, :] * k_strides[3]
                    )
                    state_offsets = (
                        state_index * state_strides[0]
                        + i_hv * state_strides[1]
                        + tile_v.index[:, None] * state_strides[2]
                        + tile_k.index[None, :] * state_strides[3]
                    )
                    state_mask = v_valid[:, None] & k_valid[None, :]
                    cached_g = hl.load(
                        g_storage,
                        [g_offsets],
                        extra_mask=cache_mask,
                    ).float()
                    gate_prefix = torch.cumsum(cached_g, dim=0)
                    gate_total = cached_g.sum(0)
                    replay_decay = torch.where(
                        cache_mask,
                        torch.exp2((gate_total[None, :] - gate_prefix) * _LOG2_E),
                        0.0,
                    )
                    total_decay = torch.exp2(gate_total * _LOG2_E)
                    cached_k = hl.load(
                        k_storage,
                        [cached_k_offsets],
                        extra_mask=cache_mask,
                    ).float()
                    cached_k = (cached_k * replay_decay).to(out.dtype)
                    state = hl.load(
                        state_storage,
                        [state_offsets],
                        extra_mask=state_mask,
                    ).float()
                    state = state * total_decay[None, :]
                    state = state + hl.dot(
                        d_dot,
                        cached_k,
                        out_dtype=torch.float32,
                    )
                    current_gate = _load_log_decay(
                        a=a,
                        dt_bias=dt_bias,
                        batch_index=i_b,
                        value_head=i_hv,
                        key_indices=tile_k.index,
                        key_valid=k_valid,
                        key_dim=K,
                        decay_rate=A,
                        lower_bound=lower_bound,
                        use_lower_bound=use_lower_bound,
                    )
                    current_decay = torch.exp2(current_gate * _LOG2_E)
                    state = state * current_decay[None, :]
                    state = state + delta[:, None] * k_value[None, :]
                    hl.store(
                        state_storage,
                        [state_offsets],
                        state.to(initial_state.dtype),
                        extra_mask=v_valid[:, None] & k_valid[None, :],
                    )
            else:
                current_d_offsets = (
                    state_index * d_strides[0]
                    + i_hv * d_strides[1]
                    + cursor * d_strides[2]
                    + tile_v.index * d_strides[3]
                )
                hl.store(
                    d_storage,
                    [current_d_offsets],
                    delta.to(d_cache.dtype),
                    extra_mask=(cursor < cache_length) & v_valid,
                )

    return out


_helion_fused_recurrent_kda_replayssm_decode_fp32 = helion.kernel(
    _helion_fused_recurrent_kda_replayssm_decode_body,
    static_shapes=False,
    config=_KDA_REPLAYSSM_FP32_CONFIG,
    ignore_warnings=_IGNORED_WARNINGS,
)
_helion_fused_recurrent_kda_replayssm_decode_bf16 = helion.kernel(
    _helion_fused_recurrent_kda_replayssm_decode_body,
    static_shapes=False,
    config=_KDA_REPLAYSSM_BF16_CONFIG,
    ignore_warnings=_IGNORED_WARNINGS,
)
_helion_fused_recurrent_kda_replayssm_decode_bf16_small_head = helion.kernel(
    _helion_fused_recurrent_kda_replayssm_decode_body,
    static_shapes=False,
    config=_KDA_REPLAYSSM_BF16_SMALL_HEAD_CONFIG,
    ignore_warnings=_IGNORED_WARNINGS,
)


def _select_replayssm_decode_kernel(
    *,
    is_bf16_state: bool,
    num_v_heads: int,
) -> helion.Kernel:
    if is_bf16_state and num_v_heads <= KDA_SMALL_VALUE_HEAD_THRESHOLD:
        return _helion_fused_recurrent_kda_replayssm_decode_bf16_small_head
    if is_bf16_state:
        return _helion_fused_recurrent_kda_replayssm_decode_bf16
    return _helion_fused_recurrent_kda_replayssm_decode_fp32


def helion_fused_recurrent_kda_replayssm_decode(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    d_cache: torch.Tensor,
    k_cache: torch.Tensor,
    g_cache: torch.Tensor,
    out: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    write_pos: torch.Tensor,
    force_flush: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    lower_bound: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one buffered KDA decode step using caller-owned ReplaySSM state.

    Allocates nothing persistent: the caller owns ``d_cache`` / ``k_cache`` /
    ``g_cache`` and is responsible for advancing ``write_pos`` modulo the ring
    length after a non-flush step and resetting it to zero after a natural or
    forced flush. ``initial_state`` is both the checkpoint read (h0) and the
    flush-only checkpoint write (ht), in place.
    """
    batch = mixed_qkv.size(0)
    if a.ndim not in (2, 3) or not a.is_contiguous():
        raise ValueError("KDA `a` must be a contiguous 2D or 3D tensor.")
    if dt_bias.ndim not in (1, 2) or not dt_bias.is_contiguous():
        raise ValueError("KDA `dt_bias` must be a contiguous 1D or 2D tensor.")
    flat_a = a.view(batch, -1)
    flat_dt_bias = dt_bias.view(-1)
    _, num_q_heads, num_v_heads, key_dim, value_dim = validate_packed_decode_inputs(
        mixed_qkv,
        flat_a,
        b,
        A_log,
        flat_dt_bias,
        initial_state,
        out,
        ssm_state_indices,
    )

    if write_pos.ndim != 1 or write_pos.dtype is not torch.int32:
        raise ValueError("`write_pos` must be a 1D int32 tensor.")
    if write_pos.shape != (batch,):
        raise ValueError(f"`write_pos` must have shape {(batch,)}.")
    if force_flush is not None and (
        force_flush.ndim != 1
        or force_flush.dtype is not torch.int32
        or force_flush.shape != (batch,)
    ):
        raise ValueError("`force_flush` must be a length-B int32 tensor or None.")

    cache_length = d_cache.size(2)
    if cache_length < 1:
        raise ValueError("ReplaySSM cache length must be at least 1.")
    if d_cache.shape[1:] != (num_v_heads, cache_length, value_dim):
        raise ValueError("`d_cache` must have shape [slots, HV, L, V].")
    if k_cache.shape[1:] != (num_q_heads, cache_length, key_dim):
        raise ValueError("`k_cache` must have shape [slots, H, L, K].")
    if g_cache.shape[1:] != (num_v_heads, cache_length, key_dim):
        raise ValueError("`g_cache` must have shape [slots, HV, L, K].")
    if g_cache.dtype is not torch.float32:
        raise ValueError("`g_cache` must have dtype torch.float32.")

    device = mixed_qkv.device
    if any(
        tensor.device != device for tensor in (d_cache, k_cache, g_cache, write_pos)
    ):
        raise ValueError("ReplaySSM inputs must be on the same device.")
    if force_flush is not None and force_flush.device != device:
        raise ValueError("`force_flush` must be on the same device as the inputs.")

    cache_block = helion.next_power_of_2(max(16, cache_length))
    use_lower_bound = lower_bound is not None
    kernel = _select_replayssm_decode_kernel(
        is_bf16_state=initial_state.dtype is torch.bfloat16,
        num_v_heads=num_v_heads,
    )
    result = kernel(
        mixed_qkv,
        flat_a,
        b,
        A_log,
        flat_dt_bias,
        scale,
        initial_state,
        d_cache,
        k_cache,
        g_cache,
        out,
        ssm_state_indices,
        write_pos,
        force_flush,
        0.0 if lower_bound is None else lower_bound,
        cache_block,
        use_qk_l2norm_in_kernel,
        use_lower_bound,
    )
    return result, initial_state
