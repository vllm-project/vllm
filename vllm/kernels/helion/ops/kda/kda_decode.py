"""Helion implementation of SGLang's packed KDA decode contract."""

from __future__ import annotations

import helion
import helion.language as hl
import torch

from vllm.kernels.helion.ops.kda import KDA_SMALL_VALUE_HEAD_THRESHOLD

# SGLang initializes torch.distributed, but this kernel has no collectives.
_IGNORED_WARNINGS = [helion.exc.ProcessGroupNameNotFound]
_LOG2_E = 1.4426950408889634

# Tile V on the CUDA x axis so tensor-parallel head counts do not change the
# grid width.
_KDA_CONFIG = helion.Config(
    block_sizes=[8],
    loop_orders=[[2, 1, 0]],
    num_warps=1,
    num_stages=1,
    indexing="pointer",
    pid_type="xyz",
)

_KDA_BF16_CONFIG = helion.Config(
    atomic_indexing=[],
    block_sizes=[16],
    indexing="pointer",
    l2_groupings=[16],
    # Policies are positional in the traced load order. Retune them if the
    # decode body gains, loses, or reorders loads.
    load_eviction_policies=[
        "",
        "last",
        "first",
        "last",
        "first",
        "first",
        "last",
        "",
        "last",
    ],
    loop_orders=[[1, 2, 0]],
    num_stages=1,
    num_warps=1,
    pid_type="flat",
    range_flattens=[None],
    range_multi_buffers=[None],
    range_num_stages=[],
    range_unroll_factors=[0],
)

# The bounded sigmoid gate has lower ALU and register pressure than the
# unbounded softplus gate, allowing small-head BF16 decode to use a wider V
# tile. The same tile regresses the unbounded path.
_KDA_BF16_SMALL_HEAD_CONFIG = helion.Config(
    block_sizes=[32],
    loop_orders=[[2, 1, 0]],
    num_warps=1,
    num_stages=1,
    indexing="pointer",
    pid_type="xyz",
)


def _helion_fused_recurrent_kda_packed_decode_body(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    lower_bound: float,
    initial_state: torch.Tensor,
    out: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    use_qk_l2norm_in_kernel: hl.constexpr = False,  # pyrefly: ignore[bad-function-definition]
    use_fast_rsqrt: hl.constexpr = False,  # pyrefly: ignore[bad-function-definition]
    use_lower_bound: hl.constexpr = False,  # pyrefly: ignore[bad-function-definition]
) -> torch.Tensor:
    """Fused packed KDA decode body; mutates ``initial_state`` and ``out``."""
    B = mixed_qkv.size(0)
    HV = hl.specialize(initial_state.size(-3))
    V = hl.specialize(initial_state.size(-2))
    K = hl.specialize(initial_state.size(-1))
    H = hl.specialize((mixed_qkv.size(1) - HV * V) // (2 * K))
    heads_per_q = HV // H

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
            initial_state.stride(0),
            initial_state.stride(1),
            initial_state.stride(2),
            initial_state.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            ssm_state_indices.stride(0),
        )
    )

    block_v = hl.register_block_size(1, V)

    for tile_b, tile_hv, tile_v in hl.tile([B, HV, V], block_size=[1, 1, block_v]):
        k_offsets = hl.arange(K)
        i_b = tile_b.id
        i_hv = tile_hv.id
        i_h = i_hv // heads_per_q

        state_index = ssm_state_indices[i_b].long()
        if state_index < 0:
            out[i_b, 0, i_hv, tile_v] = 0.0
        else:
            q_offsets = i_h * K + k_offsets
            k_input_offsets = H * K + i_h * K + k_offsets
            v_offsets = 2 * H * K + i_hv * V + tile_v.index

            raw_gate = a[i_b, i_hv * K + k_offsets].float()
            raw_gate = raw_gate + dt_bias[i_hv * K + k_offsets].float()
            A_log_value = A_log[i_hv].float()
            A = torch.exp2(A_log_value * _LOG2_E)
            if use_lower_bound:
                log_decay = lower_bound * torch.sigmoid(A * raw_gate)
            else:
                gate_exp = torch.exp2(raw_gate * _LOG2_E)
                softplus = torch.where(
                    raw_gate <= 20.0,
                    torch.log(1.0 + gate_exp),
                    raw_gate,
                )
                log_decay = -A * softplus
            beta = torch.sigmoid(b[i_b, i_hv].float())

            state = initial_state[state_index, i_hv, tile_v.index, k_offsets].float()
            decay = torch.exp2(log_decay * _LOG2_E)
            state = state * decay[None, :]

            k = mixed_qkv[i_b, k_input_offsets].float()
            if use_qk_l2norm_in_kernel:
                k_norm = (k * k).sum() + 1e-6
                if use_fast_rsqrt:
                    k = k * torch.rsqrt(k_norm)
                else:
                    k = k / torch.sqrt(k_norm)
            v = mixed_qkv[i_b, v_offsets].float()
            value_residual = v - (state * k[None, :]).sum(-1)
            value_residual = value_residual * beta
            state = state + value_residual[:, None] * k[None, :]

            q = mixed_qkv[i_b, q_offsets].float()
            if use_qk_l2norm_in_kernel:
                q_norm = (q * q).sum() + 1e-6
                if use_fast_rsqrt:
                    q = q * torch.rsqrt(q_norm)
                else:
                    q = q / torch.sqrt(q_norm)
            q = q * scale
            output = (state * q[None, :]).sum(-1)

            out[i_b, 0, i_hv, tile_v] = output.to(out.dtype)
            initial_state[state_index, i_hv, tile_v.index, k_offsets] = state

    return out


_helion_fused_recurrent_kda_packed_decode = helion.kernel(
    _helion_fused_recurrent_kda_packed_decode_body,
    static_shapes=False,
    config=_KDA_CONFIG,
    ignore_warnings=_IGNORED_WARNINGS,
)
_helion_fused_recurrent_kda_packed_decode_bf16 = helion.kernel(
    _helion_fused_recurrent_kda_packed_decode_body,
    static_shapes=False,
    config=_KDA_BF16_CONFIG,
    ignore_warnings=_IGNORED_WARNINGS,
)
_helion_fused_recurrent_kda_packed_decode_bf16_small_head = helion.kernel(
    _helion_fused_recurrent_kda_packed_decode_body,
    static_shapes=False,
    config=_KDA_BF16_SMALL_HEAD_CONFIG,
    ignore_warnings=_IGNORED_WARNINGS,
)


def _select_decode_kernel(
    *,
    is_bf16_state: bool,
    num_v_heads: int,
    use_lower_bound: bool,
) -> helion.Kernel:
    if (
        is_bf16_state
        and use_lower_bound
        and num_v_heads <= KDA_SMALL_VALUE_HEAD_THRESHOLD
    ):
        return _helion_fused_recurrent_kda_packed_decode_bf16_small_head
    if is_bf16_state:
        return _helion_fused_recurrent_kda_packed_decode_bf16
    return _helion_fused_recurrent_kda_packed_decode


def validate_packed_decode_inputs(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_state: torch.Tensor,
    out: torch.Tensor,
    ssm_state_indices: torch.Tensor,
) -> tuple[int, int, int, int, int]:
    """Apply the shape and layout checks from SGLang's packed wrapper."""
    if mixed_qkv.ndim != 2:
        raise ValueError(
            f"`mixed_qkv` must be a 2D tensor (got ndim={mixed_qkv.ndim})."
        )
    if mixed_qkv.stride(-1) != 1:
        raise ValueError("`mixed_qkv` must be contiguous in the last dim.")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(
            f"`a` and `b` must be 2D tensors (got a.ndim={a.ndim}, b.ndim={b.ndim})."
        )
    if a.stride(-1) != 1 or b.stride(-1) != 1:
        raise ValueError("`a`/`b` must be contiguous in the last dim.")
    if A_log.ndim != 1 or dt_bias.ndim != 1:
        raise ValueError("`A_log`/`dt_bias` must be 1D tensors.")
    if A_log.stride(0) != 1 or dt_bias.stride(0) != 1:
        raise ValueError("`A_log`/`dt_bias` must be contiguous.")
    if ssm_state_indices.ndim != 1:
        raise ValueError(
            "`ssm_state_indices` must be 1D for packed decode "
            f"(got ndim={ssm_state_indices.ndim})."
        )
    if not out.is_contiguous():
        raise ValueError("`out` must be contiguous.")

    device = mixed_qkv.device
    if any(
        tensor.device != device
        for tensor in (
            a,
            b,
            A_log,
            dt_bias,
            initial_state,
            out,
            ssm_state_indices,
        )
    ):
        raise ValueError("All inputs must be on the same device.")

    B = mixed_qkv.shape[0]
    if a.shape[0] != B or b.shape[0] != B:
        raise ValueError(
            "Mismatched batch sizes: "
            f"mixed_qkv.shape[0]={B}, a.shape[0]={a.shape[0]}, "
            f"b.shape[0]={b.shape[0]}."
        )
    if ssm_state_indices.shape[0] != B:
        raise ValueError(
            f"`ssm_state_indices` must have shape [B] "
            f"(got {tuple(ssm_state_indices.shape)}; expected ({B},))."
        )

    if initial_state.ndim != 4:
        raise ValueError(
            f"`initial_state` must be a 4D tensor (got ndim={initial_state.ndim})."
        )
    if initial_state.stride(-1) != 1:
        raise ValueError("`initial_state` must be contiguous in the last dim.")
    HV, V, K = initial_state.shape[-3:]
    if a.shape[1] != HV * K:
        raise ValueError(
            f"`a` must have shape [B, HV*K] with HV={HV}, K={K} "
            f"(got a.shape={tuple(a.shape)})."
        )
    if b.shape[1] != HV:
        raise ValueError(
            f"`b` must have shape [B, HV] with HV={HV} (got b.shape={tuple(b.shape)})."
        )
    if A_log.numel() != HV:
        raise ValueError(f"`A_log` must have {HV} elements (got {A_log.numel()}).")
    if dt_bias.numel() != HV * K:
        raise ValueError(
            f"`dt_bias` must have {HV * K} elements (got {dt_bias.numel()})."
        )
    if out.shape != (B, 1, HV, V):
        raise ValueError(
            f"`out` must have shape {(B, 1, HV, V)} (got out.shape={tuple(out.shape)})."
        )

    qkv_dim = mixed_qkv.shape[1]
    qk_dim = qkv_dim - HV * V
    if qk_dim <= 0 or qk_dim % 2 != 0:
        raise ValueError(
            f"Invalid packed `mixed_qkv` last dim={qkv_dim} for HV={HV}, V={V}."
        )
    q_dim = qk_dim // 2
    if q_dim % K != 0:
        raise ValueError(
            f"Invalid packed Q size {q_dim}: must be divisible by K={K}. "
            "KDA packed decode requires num_q_heads == num_k_heads and "
            "head_q_dim == head_k_dim."
        )
    H = q_dim // K
    if H <= 0 or HV % H != 0:
        raise ValueError(
            f"Invalid head config inferred from mixed_qkv: H={H}, HV={HV}."
        )
    return B, H, HV, K, V


def helion_fused_recurrent_kda_packed_decode(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    out: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    use_qk_l2norm_in_kernel: bool = False,
    lower_bound: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Helion implementation of SGLang's packed KDA decode contract.

    Inputs, mutations, padding semantics, and outputs match
    ``fused_recurrent_kda_packed_decode``:

    * ``mixed_qkv`` is ``[B, 2*H*K + HV*V]`` after the short convolution.
    * ``a`` and ``b`` are raw forget-gate and beta logits.
    * ``lower_bound`` selects the bounded sigmoid decay used by safe-gate KDA.
    * ``initial_state`` is ``[num_slots, HV, V, K]`` and is updated in place.
    * ``ssm_state_indices == -1`` writes a zero output and leaves state untouched.
    * ``out`` is ``[B, 1, HV, V]`` and is written in place.
    * The return is the same ``(out, initial_state)`` object pair supplied by the
      caller.
    """
    _, _, num_v_heads, _, _ = validate_packed_decode_inputs(
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        initial_state,
        out,
        ssm_state_indices,
    )
    use_lower_bound = lower_bound is not None
    is_bf16_state = initial_state.dtype is torch.bfloat16
    kernel = _select_decode_kernel(
        is_bf16_state=is_bf16_state,
        num_v_heads=num_v_heads,
        use_lower_bound=use_lower_bound,
    )
    lower_bound_value = 0.0 if lower_bound is None else lower_bound
    result = kernel(
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        scale,
        lower_bound_value,
        initial_state,
        out,
        ssm_state_indices,
        use_qk_l2norm_in_kernel,
        is_bf16_state,
        use_lower_bound,
    )
    return result, initial_state
