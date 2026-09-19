# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU backend for the GLM5Next KDA (Kimi Delta Attention) operator.

This module is the CPU counterpart of
``vllm.models.glm5next.nvidia.ops.third_party.kda`` and exposes the same two
entry points with the same signatures, so ``common/kda.py`` can dispatch to it
without any call-site change.

The AVX-512 operator handles its supported dense-state contracts. The explicit
fp32 PyTorch recurrence remains as a fallback while native packed/state/spec
coverage is added. The independent mathematical oracle lives in
``tests/models/glm5next/test_kda_cpu.py`` and does not call backend helpers.

Contract notes (verified against the Triton kernels and the upstream reference
test ``tests/models/glm5next/test_kda_recurrent.py``):

- The recurrent state is **V-major** ``[..., V, K]``. The real chunk path stores
  per-sequence states as ``[N, H, V, K]``; the kernel addresses the state block
  as ``o_v[:, None] * K + o_k[None, :]``, which is V-major. The V/K axes cannot
  be inferred from the shape alone because KDA is square, so the axis order is
  load-bearing and is asserted by the tests through asymmetric values.
- ``ssm_state_indices`` may be 1-D ``[num_seqs]`` (plain decode) or 2-D
  ``[num_seqs, num_spec]`` (speculative decode). For the 2-D case the initial
  state is read from column ``num_accepted_tokens - 1`` and each token writes
  back to its own column.
- A state index ``<= 0`` is ``NULL_BLOCK_ID`` (a padded / invalid slot): such
  sequences are skipped entirely and their state is neither read nor written.
- ``inplace_final_state=True`` (the default) returns the *same* state object it
  was given and mutates it in place; ``False`` allocates per-token states.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from vllm.model_executor.layers.mamba.ops.cpu.causal_conv1d import (
    causal_conv1d_update_cpu as _causal_conv1d_update_cpu,
)

__all__ = [
    "chunk_kda_with_fused_gate",
    "fused_recurrent_kda",
    "gather_initial_states_cpu",
    "scatter_states_cpu",
    "causal_conv1d_update_cpu",
]

# A padded or otherwise invalid state slot. Matches NULL_BLOCK_ID / PAD_SLOT_ID
# used by the v1 metadata builders.
_NULL_BLOCK_ID = 0

_QK_L2NORM_EPS = 1e-6


def causal_conv1d_update_cpu(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: bool | str | None = None,
    conv_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    max_query_len: int = -1,
    pad_slot_id: int = 0,
    **kwargs,
) -> torch.Tensor:
    """CPU short-convolution update with the GLM speculative contract."""
    if num_accepted_tokens is None:
        return _causal_conv1d_update_cpu(
            x,
            conv_state,
            weight,
            bias,
            activation,
            conv_state_indices,
            query_start_loc,
            pad_slot_id,
            **kwargs,
        )
    if query_start_loc is None or conv_state_indices is None:
        raise ValueError(
            "speculative CPU convolution requires query_start_loc and "
            "conv_state_indices"
        )
    if x.ndim != 2 or conv_state.ndim != 3 or weight.ndim != 2:
        raise ValueError("invalid speculative CPU convolution shapes")
    if max_query_len < 1:
        raise ValueError("max_query_len must be positive for speculative convolution")
    state_len_full = conv_state.shape[-1]
    width = weight.shape[-1]
    if state_len_full < width - 1:
        raise ValueError("conv_state is shorter than the convolution window")
    if conv_state_indices.ndim != 1:
        raise ValueError("CPU speculative convolution expects 1-D cache slots")

    original_dtype = x.dtype
    x_work = x.to(conv_state.dtype)
    weight_work = weight.to(conv_state.dtype)
    bias_work = None if bias is None else bias.to(conv_state.dtype)
    output = torch.zeros_like(x_work)
    batch = conv_state_indices.numel()
    if query_start_loc.numel() != batch + 1:
        raise ValueError("query_start_loc must have one more entry than cache slots")
    if num_accepted_tokens.numel() != batch:
        raise ValueError("num_accepted_tokens must match cache slots")

    for seq in range(batch):
        bos = int(query_start_loc[seq].item())
        eos = int(query_start_loc[seq + 1].item())
        slot = int(conv_state_indices[seq].item())
        if slot <= pad_slot_id or eos <= bos:
            continue
        offset = int(num_accepted_tokens[seq].item()) - 1
        seq_len = eos - bos
        state_len = state_len_full - (max_query_len - seq_len)
        if offset < 0 or state_len < width - 1:
            raise ValueError("invalid speculative convolution state window")

        window = conv_state[slot, :, offset : offset + width - 1].clone()
        for token in range(seq_len):
            value = x_work[bos + token]
            acc = (window * weight_work[:, :-1]).sum(-1)
            acc = acc + value * weight_work[:, -1]
            if bias_work is not None:
                acc = acc + bias_work
            if activation in (True, "silu", "swish"):
                acc = torch.nn.functional.silu(acc)
            elif activation not in (None, False):
                raise ValueError(f"unsupported activation: {activation}")
            output[bos + token] = acc
            window = torch.cat((window[:, 1:], value[:, None]), dim=-1)

        updated = conv_state[slot].clone()
        for idx in range(state_len):
            if idx + seq_len < state_len:
                updated[:, idx] = conv_state[slot, :, offset + idx + 1]
            else:
                updated[:, idx] = x_work[bos + idx - (state_len - seq_len)]
        conv_state[slot, :, :state_len].copy_(updated[:, :state_len])

    return output.to(original_dtype)


def _native_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    scale: float,
    sigmoid_beta: bool,
    a_log: torch.Tensor,
    g_bias: torch.Tensor,
    lower_bound: float,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not hasattr(torch.ops._C, "glm5next_kda_recurrent"):
        return None
    if beta.shape[-1] != v.shape[-2]:
        return None
    return torch.ops._C.glm5next_kda_recurrent(
        q,
        k,
        v,
        g,
        beta,
        state,
        scale,
        sigmoid_beta,
        a_log,
        g_bias,
        True,
        lower_bound,
    )


def _validate_state(state: torch.Tensor, name: str) -> None:
    if state.dim() != 4:
        raise ValueError(
            f"`{name}` must be a 4-D tensor of shape [num_slots, H, V, K] "
            f"(got ndim={state.dim()})."
        )
    # Each state row must be dense; only stride(0) may be padded by the
    # mamba page pool.  An empty pool is invalid because state slots are
    # addressed by integer indices below.
    if state.shape[0] == 0:
        raise ValueError(f"`{name}` must contain at least one state slot.")
    if not state[0].is_contiguous():
        raise ValueError(f"`{name}` rows must be contiguous.")


def gather_initial_states_cpu(
    state: torch.Tensor,
    indices: torch.Tensor,
    has_initial_state: torch.Tensor,
) -> torch.Tensor:
    """Gather state rows and zero rows without an initialized cache entry."""
    if state.ndim < 2 or indices.ndim != 1 or has_initial_state.ndim != 1:
        raise ValueError("invalid state gather shapes")
    if indices.device != state.device or has_initial_state.device != state.device:
        raise ValueError("state, indices, and has_initial_state must share a device")
    if indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("indices must have int32 or int64 dtype")
    if has_initial_state.dtype is not torch.bool:
        raise ValueError("has_initial_state must have bool dtype")
    if indices.shape != has_initial_state.shape:
        raise ValueError("indices and has_initial_state must have the same shape")
    if state.shape[0] == 0:
        raise ValueError("state must contain at least one slot")
    if not state[0].is_contiguous():
        raise ValueError("state rows must be contiguous")
    # The CUDA helper substitutes slot zero for rows without an initial state
    # before loading.  Do the same so a sentinel index in an uninitialized row
    # cannot trigger an out-of-range gather on CPU.
    valid = has_initial_state & (indices > _NULL_BLOCK_ID)
    safe_indices = torch.where(valid, indices, torch.zeros_like(indices))
    gathered = state[safe_indices].clone()
    return torch.where(
        valid.reshape(-1, *([1] * (state.ndim - 1))),
        gathered,
        torch.zeros_like(gathered),
    )


def scatter_states_cpu(
    state: torch.Tensor,
    source: torch.Tensor,
    indices: torch.Tensor,
) -> None:
    """Scatter dense state rows into cache slots in place."""
    if state.ndim < 2 or source.ndim != state.ndim or indices.ndim != 1:
        raise ValueError("invalid state scatter shapes")
    if source.device != state.device or indices.device != state.device:
        raise ValueError("state, source, and indices must share a device")
    if indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("indices must have int32 or int64 dtype")
    if source.shape[0] != indices.shape[0] or source.shape[1:] != state.shape[1:]:
        raise ValueError("source and indices do not match the state pool")
    if state.shape[0] == 0 or source.shape[0] == 0:
        raise ValueError("state and source must contain at least one row")
    if not state[0].is_contiguous() or not source[0].is_contiguous():
        raise ValueError("state rows must be contiguous")
    valid = indices > _NULL_BLOCK_ID
    state[indices[valid]] = source[valid]


def _state_slot(
    indices: torch.Tensor | None,
    sequence: int,
    token: int | None,
    accepted: torch.Tensor | None,
    default_slot: int | None = None,
) -> int:
    """Return the cache slot addressed by one recurrent state access."""
    if indices is None:
        if default_slot is None:
            raise ValueError("default_slot is required without state indices")
        return default_slot + (0 if token is None else token)
    if indices.dim() == 1:
        if token not in (None, 0):
            raise ValueError(
                "1-D `ssm_state_indices` is only valid for one token per sequence."
            )
        return int(indices[sequence].item())
    if accepted is None:
        raise ValueError("2-D `ssm_state_indices` requires `num_accepted_tokens`.")
    if token is None:
        return int(indices[sequence, int(accepted[sequence].item()) - 1].item())
    return int(indices[sequence, token].item())


def _resolve_seqs(
    cu_seqlens: torch.Tensor | None,
    total_tokens: int,
    batch: int,
    tokens_per_batch: int,
) -> tuple[int, Callable[[int], tuple[int, int]]]:
    """Return ``N`` and a callable mapping sequence index -> (bos, eos)."""
    if cu_seqlens is None:
        return batch, lambda n: (
            n * tokens_per_batch,
            (n + 1) * tokens_per_batch,
        )

    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError("`cu_seqlens` must be a 1-D tensor with two entries.")
    n_seqs = int(cu_seqlens.numel()) - 1
    if int(cu_seqlens[0].item()) != 0 or int(cu_seqlens[-1].item()) != total_tokens:
        raise ValueError("`cu_seqlens` must start at 0 and end at token count.")

    def bounds(n: int) -> tuple[int, int]:
        return int(cu_seqlens[n].item()), int(cu_seqlens[n + 1].item())

    return n_seqs, bounds


def _kda_gate(
    raw_g: torch.Tensor,
    a_log: torch.Tensor,
    g_bias: torch.Tensor | None,
    lower_bound: float,
) -> torch.Tensor:
    """Bounded ("safe") KDA gate in fp32.

    ``lower_bound / (1 + exp(-exp(A_log) * (raw_g + g_bias)))``. ``A_log`` is a
    per-head scalar, so it broadcasts along the head axis only; ``g_bias`` is
    laid out per (head, key-dim) as a flat ``[H * K]`` buffer, matching the
    kernel's ``g_bias + i_h * K + o_k`` addressing.

    Args:
        raw_g: ``[H, K]`` raw gate logits for one token.
        a_log: ``[1, 1, H, 1]`` (or anything flattenable to ``[H]``).
        g_bias: flat ``[H * K]`` bias, or already ``[H, K]``.
        lower_bound: Negative bound applied to the sigmoid gate.

    Returns:
        ``[H, K]`` fp32 gate values.

    """
    h, k = raw_g.shape[-2], raw_g.shape[-1]
    # A_log arrives as [1, 1, H, 1] from the checkpoint; flatten to [H] and
    # broadcast along the key dim.
    a = a_log.reshape(-1).to(torch.float32).exp().unsqueeze(-1)
    if a.shape[0] != h:
        raise ValueError(f"a_log has {a.shape[0]} heads, expected {h}")
    x = raw_g.to(torch.float32)
    if g_bias is not None:
        bias = g_bias.to(torch.float32)
        if bias.dim() == 1:
            if bias.numel() != h * k:
                raise ValueError("g_bias must have H * K elements")
            bias = bias.reshape(h, k)
        elif bias.shape != (h, k):
            raise ValueError(f"g_bias must have shape {(h, k)}")
        x = x + bias
    return lower_bound / (1.0 + torch.exp(-(a * x)))


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    return x / torch.sqrt(x.square().sum(-1, keepdim=True) + _QK_L2NORM_EPS)


def _token_view(
    x: torch.Tensor,
    name: str,
) -> torch.Tensor:
    """Flatten the leading ``[B, T]`` token dimensions.

    The common layer token-flattens every input to a batch of 1, while the
    public NVIDIA wrapper also permits an ordinary dense batch when
    ``cu_seqlens`` is absent.
    """
    if x.dim() < 3:
        raise ValueError(
            f"`{name}` must be at least 3-D with leading batch/token dims "
            f"(got ndim={x.dim()})."
        )
    return x.reshape(-1, *x.shape[2:])


def _validate_state_shape(
    state: torch.Tensor, name: str, value_heads: int, value_dim: int, key_dim: int
) -> None:
    _validate_state(state, name)
    expected = (value_heads, value_dim, key_dim)
    if tuple(state.shape[1:]) != expected:
        raise ValueError(
            f"`{name}` rows must have shape {expected}, got {tuple(state.shape[1:])}."
        )


def _expand_heads(x: torch.Tensor, value_heads: int, name: str) -> torch.Tensor:
    """Expand query-head tensors to the value-head layout used by the state."""
    if x.ndim == 0:
        return x.expand(value_heads)
    heads = x.shape[0]
    if heads == value_heads:
        return x
    if value_heads % heads:
        raise ValueError(
            f"{name} has {heads} heads, which cannot map to {value_heads} value heads"
        )
    return x.repeat_interleave(value_heads // heads, dim=0)


def fused_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    inplace_final_state: bool = True,
    use_qk_l2norm_in_kernel: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    sigmoid_beta: bool = False,
    a_log: torch.Tensor | None = None,
    g_bias: torch.Tensor | None = None,
    compute_gate: bool = False,
    lower_bound: float | None = -5.0,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """CPU implementation of the GLM5Next KDA recurrent (decode) kernel.

    Mirrors ``fused_recurrent_gated_delta_rule_fwd_kernel`` for the KDA
    configuration (``IS_KDA=True``, ``SAFE_GATE=True``).

    Args:
        q: Queries of shape ``[B, T, H, K]``.
        k: Keys of shape ``[B, T, H, K]``.
        v: Values of shape ``[B, T, HV, V]``.
        g: raw gate logits, ``[1, T, HV, K]``.
        beta: raw beta logits, ``[1, T, HV]`` (per head) or ``[1, T, HV, V]``.
        scale: Query scaling factor. Defaults to ``1 / sqrt(K)``.
        initial_state: state pool ``[num_slots, HV, V, K]``.
        inplace_final_state: Mutate and return ``initial_state`` when true.
        use_qk_l2norm_in_kernel: Apply the KDA q/k L2 normalization.
        cu_seqlens: Packed sequence boundaries for variable-length inputs.
        ssm_state_indices: 1-D ``[num_seqs]`` slot indices for ordinary decode,
            or 2-D ``[num_seqs, query_len]`` slots for speculative decode.
        num_accepted_tokens: Accepted-token offsets for speculative decoding.
        out: optional output buffer of shape ``k.shape``.
        sigmoid_beta: Apply sigmoid to beta logits before the update.
        a_log: Per-query-head gate parameter used when ``compute_gate`` is true.
        g_bias: Per-query-head gate bias used when ``compute_gate`` is true.
        compute_gate: Compute the bounded KDA gate from ``g`` and gate weights.
        lower_bound: Negative bound for the bounded KDA gate.

    Returns:
        ``(o, final_state)`` where ``o`` is ``[1, T, HV, V]`` and
        ``final_state`` is ``initial_state`` when ``inplace_final_state``.

    Note:
        When ``ssm_state_indices`` is given, ``final_state`` *is* the state pool
        and is addressed by **slot**, not by sequence: read back the state of
        sequence ``n`` as ``final_state[ssm_state_indices[n]]``. This mirrors
        the kernel, which stores to ``ht + final_state_idx * stride``.

    """
    if beta is None:
        raise ValueError("`beta` is required for KDA.")
    if compute_gate and (a_log is None or g_bias is None):
        raise ValueError("`compute_gate` requires `a_log` and `g_bias`.")
    if compute_gate and lower_bound is None:
        raise ValueError(
            "`compute_gate` implements the bounded (safe gate) branch only."
        )

    if ssm_state_indices is not None and ssm_state_indices.dim() not in (1, 2):
        raise ValueError("`ssm_state_indices` must be 1-D or 2-D.")
    if ssm_state_indices is not None and ssm_state_indices.dim() == 2:
        if num_accepted_tokens is None:
            raise ValueError("2-D `ssm_state_indices` requires `num_accepted_tokens`.")
        if num_accepted_tokens.numel() != ssm_state_indices.shape[0]:
            raise ValueError("`num_accepted_tokens` must have one value per sequence.")
    elif num_accepted_tokens is not None:
        raise ValueError("`num_accepted_tokens` requires 2-D `ssm_state_indices`.")

    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError("The batch size must be 1 when `cu_seqlens` is provided.")
    if not (q.shape[:2] == k.shape[:2] == v.shape[:2] == g.shape[:2]):
        raise ValueError("q, k, v and g must have matching batch/token shapes.")
    if beta.shape[:2] != q.shape[:2]:
        raise ValueError("beta must have matching batch/token dimensions.")

    k_flat = _token_view(k, "k")
    T, _, K = k_flat.shape
    v_flat = _token_view(v, "v")
    HV, V = v_flat.shape[1:]
    q_flat = _token_view(q, "q")
    g_flat = _token_view(g, "g")
    beta_flat = _token_view(beta, "beta")

    if q_flat.shape[0] != T or g_flat.shape[0] != T or beta_flat.shape[0] != T:
        raise ValueError("`q`, `k`, `g` and `beta` must have the same token count.")
    if scale is None:
        scale = K**-0.5

    n_seqs, bounds = _resolve_seqs(cu_seqlens, T, q.shape[0], q.shape[1])
    if ssm_state_indices is not None and ssm_state_indices.shape[0] != n_seqs:
        raise ValueError(
            "`ssm_state_indices` must have one entry per sequence "
            f"(got {ssm_state_indices.shape[0]}, expected {n_seqs})."
        )
    if (
        ssm_state_indices is not None
        and ssm_state_indices.dim() == 1
        and any(bounds(n)[1] - bounds(n)[0] != 1 for n in range(n_seqs))
    ):
        raise ValueError("1-D `ssm_state_indices` requires one token per sequence.")
    if ssm_state_indices is not None and ssm_state_indices.dim() == 2:
        assert num_accepted_tokens is not None
        if any(
            bounds(n)[1] - bounds(n)[0] > ssm_state_indices.shape[1]
            for n in range(n_seqs)
        ):
            raise ValueError(
                "2-D `ssm_state_indices` has fewer token slots than a sequence."
            )
        if bool((num_accepted_tokens < 1).any()) or bool(
            (num_accepted_tokens > ssm_state_indices.shape[1]).any()
        ):
            raise ValueError("num_accepted_tokens is outside the state index window.")

    if out is None:
        o = torch.empty_like(v)
    else:
        if out.shape != v.shape or out.dtype != v.dtype or not out.is_contiguous():
            raise ValueError("`out` must be contiguous and match v shape/dtype.")
        o = out
    o_flat = o.reshape(-1, HV, v_flat.shape[-1])

    if initial_state is None:
        raise ValueError("`initial_state` is required for the recurrent path.")
    _validate_state_shape(initial_state, "initial_state", HV, V, K)

    if inplace_final_state:
        final_state = initial_state
    else:
        final_state = torch.empty(
            (T, HV, v_flat.shape[-1], K),
            dtype=initial_state.dtype,
            device=initial_state.device,
        )

    for n in range(n_seqs):
        bos, eos = bounds(n)

        # This sequence's state slot. A non-positive slot is NULL_BLOCK_ID (a
        # padded / invalid slot) and the sequence is skipped entirely, matching
        # the kernel's early `return`: its state is neither read nor written.
        # Without slot indices the sequence index doubles as the slot.
        slot = _state_slot(
            ssm_state_indices,
            n,
            None,
            num_accepted_tokens,
            default_slot=bos,
        )
        if ssm_state_indices is not None and slot <= _NULL_BLOCK_ID:
            continue

        state = initial_state[slot].clone()

        if (
            ssm_state_indices is not None
            and ssm_state_indices.dim() == 2
            and inplace_final_state
            and use_qk_l2norm_in_kernel
            and compute_gate
            and a_log is not None
            and g_bias is not None
            and lower_bound is not None
            and num_accepted_tokens is not None
            and hasattr(torch.ops._C, "glm5next_kda_recurrent")
        ):
            for local_t, t in enumerate(range(bos, eos)):
                native = _native_recurrent_kda(
                    q_flat[t : t + 1].unsqueeze(0),
                    k_flat[t : t + 1].unsqueeze(0),
                    v_flat[t : t + 1].unsqueeze(0),
                    g_flat[t : t + 1].unsqueeze(0),
                    beta_flat[t : t + 1].unsqueeze(0),
                    state,
                    scale,
                    sigmoid_beta,
                    a_log,
                    g_bias,
                    float(lower_bound),
                )
                if native is None:
                    raise RuntimeError("native KDA dispatch became unavailable")
                native_out, state = native
                o_flat[t] = native_out[0, 0]
                write_slot = _state_slot(
                    ssm_state_indices,
                    n,
                    local_t,
                    num_accepted_tokens,
                )
                if write_slot > _NULL_BLOCK_ID:
                    final_state[write_slot] = state.to(final_state.dtype)
            continue

        if (
            ssm_state_indices is not None
            and ssm_state_indices.dim() == 1
            and inplace_final_state
            and use_qk_l2norm_in_kernel
            and compute_gate
            and a_log is not None
            and g_bias is not None
            and lower_bound is not None
        ):
            native = _native_recurrent_kda(
                q_flat[bos:eos].unsqueeze(0),
                k_flat[bos:eos].unsqueeze(0),
                v_flat[bos:eos].unsqueeze(0),
                g_flat[bos:eos].unsqueeze(0),
                beta_flat[bos:eos].unsqueeze(0),
                state,
                scale,
                sigmoid_beta,
                a_log,
                g_bias,
                float(lower_bound),
            )
            if native is not None:
                native_out, native_state = native
                o_flat[bos:eos] = native_out[0]
                final_state[slot] = native_state.to(final_state.dtype)
                continue

        # An empty segment (bos == eos) makes the loop below a no-op, which is
        # the desired behaviour: it consumes nothing and, on the inplace path,
        # `final_state` already holds this sequence's initial state.
        s = state
        for local_t, t in enumerate(range(bos, eos)):
            b_q = _expand_heads(q_flat[t], HV, "q").to(torch.float32)
            b_k = _expand_heads(k_flat[t], HV, "k").to(torch.float32)
            b_v = v_flat[t].to(torch.float32)

            if use_qk_l2norm_in_kernel:
                b_q = _l2norm(b_q)
                b_k = _l2norm(b_k)
            b_q = b_q * scale

            if compute_gate:
                assert a_log is not None
                assert g_bias is not None
                assert lower_bound is not None
                b_g = _kda_gate(g_flat[t], a_log, g_bias, float(lower_bound))
            else:
                b_g = g_flat[t].to(torch.float32)
            b_g = _expand_heads(b_g, HV, "g")
            # Decay is per (head, key-dim); broadcast along the V axis of the
            # [HV, V, K] state.
            s = s * b_g.exp().unsqueeze(1)

            b_v = b_v - torch.einsum("hvk,hk->hv", s, b_k)

            b_beta = beta_flat[t].to(torch.float32)
            if sigmoid_beta:
                b_beta = torch.sigmoid(b_beta)
            b_beta = _expand_heads(b_beta, HV, "beta")
            if b_beta.dim() == 0:
                b_v = b_v * b_beta
            elif b_beta.dim() == 1:
                b_v = b_v * b_beta.unsqueeze(-1)
            else:
                b_v = b_v * b_beta

            s = s + b_v[:, :, None] * b_k[:, None, :]
            o_flat[t] = torch.einsum("hvk,hk->hv", s, b_q).to(o.dtype)

            if inplace_final_state:
                write_slot = _state_slot(
                    ssm_state_indices,
                    n,
                    local_t,
                    num_accepted_tokens,
                    default_slot=bos,
                )
                if ssm_state_indices is None or write_slot > _NULL_BLOCK_ID:
                    final_state[write_slot] = s.to(final_state.dtype)
            else:
                final_state[t] = s.to(final_state.dtype)

    return o, final_state


def chunk_kda_with_fused_gate(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    g_bias: torch.Tensor | None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    safe_gate: bool = False,
    lower_bound: float = -5.0,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """CPU implementation of the GLM5Next chunked-prefill KDA operator.

    The chunk path is evaluated as a sequential token recurrence over each
    ``cu_seqlens`` segment. That is mathematically equivalent to the chunked
    formulation (chunking is only a parallelization strategy) and is the
    intended bring-up behaviour; it is deliberately not optimized.

    Unlike the recurrent path, the chunk path returns a *fresh* per-sequence
    final state of shape ``[N, H, V, K]`` when ``output_final_state`` is set,
    and it consumes ``beta`` already sigmoided in fp32.

    Args:
        q: Queries of shape ``[B, T, H, K]``.
        k: Keys of shape ``[B, T, H, K]``.
        v: Values of shape ``[B, T, HV, V]``.
        raw_g: raw gate logits ``[1, T, H, K]``.
        beta: pre-sigmoided fp32 beta ``[1, T, H]`` (or head-broadcastable).
        A_log: per-head ``[1, 1, H, 1]`` (or flattenable to ``[H]``).
        g_bias: per (head, key-dim) bias ``[H * K]``.
        scale: Query scaling factor. Defaults to ``1 / sqrt(K)``.
        initial_state: ``[N, H, V, K]`` per-sequence initial states.
        output_final_state: return the per-sequence final states.
        use_qk_l2norm_in_kernel: Apply the KDA q/k L2 normalization.
        cu_seqlens: Packed sequence boundaries for variable-length inputs.
        safe_gate: Require the bounded KDA gate variant.
        lower_bound: Negative bound for the bounded KDA gate.

    Returns:
        ``(o, final_state)`` with ``o`` shaped like ``v`` and ``final_state``
        ``[N, H, V, K]`` or ``None``.

    """
    if not safe_gate:
        raise NotImplementedError(
            "CPU GLM5Next KDA only implements the bounded (safe) gate; "
            "the unbounded GDN-family gate is unreachable for GLM-5.3-Flash."
        )

    k_flat = _token_view(k, "k")
    T, _, K = k_flat.shape
    v_flat = _token_view(v, "v")
    HV, V = v_flat.shape[1:]
    q_flat = _token_view(q, "q")
    raw_g_flat = _token_view(raw_g, "raw_g")
    beta_flat = _token_view(beta, "beta")

    if q_flat.shape[0] != T or raw_g_flat.shape[0] != T or beta_flat.shape[0] != T:
        raise ValueError("`q`, `k`, `raw_g` and `beta` must have the same token count.")
    if scale is None:
        scale = K**-0.5

    n_seqs, bounds = _resolve_seqs(cu_seqlens, T, q.shape[0], q.shape[1])

    if initial_state is not None:
        _validate_state_shape(initial_state, "initial_state", HV, V, K)
        if initial_state.shape[0] < n_seqs:
            raise ValueError("initial_state has fewer rows than packed sequences")

    o = torch.empty_like(v)
    o_flat = o.reshape(-1, HV, V)

    final_state = (
        torch.empty(
            (n_seqs, HV, v_flat.shape[-1], K),
            dtype=torch.float32,
            device=v.device,
        )
        if output_final_state
        else None
    )

    for n in range(n_seqs):
        bos, eos = bounds(n)
        if initial_state is not None:
            s = initial_state[n].to(torch.float32).clone()
        else:
            s = torch.zeros(
                (HV, v_flat.shape[-1], K), dtype=torch.float32, device=v.device
            )
        if eos <= bos:
            if final_state is not None:
                final_state[n] = s.to(final_state.dtype)
            continue

        if use_qk_l2norm_in_kernel and g_bias is not None:
            native = _native_recurrent_kda(
                q_flat[bos:eos].unsqueeze(0),
                k_flat[bos:eos].unsqueeze(0),
                v_flat[bos:eos].unsqueeze(0),
                raw_g_flat[bos:eos].unsqueeze(0),
                beta_flat[bos:eos].unsqueeze(0),
                s,
                scale,
                False,
                A_log,
                g_bias,
                float(lower_bound),
            )
            if native is not None:
                native_out, s = native
                o_flat[bos:eos] = native_out[0]
                if final_state is not None:
                    final_state[n] = s.to(final_state.dtype)
                continue

        for t in range(bos, eos):
            b_q = _expand_heads(q_flat[t], HV, "q").to(torch.float32)
            b_k = _expand_heads(k_flat[t], HV, "k").to(torch.float32)
            b_v = v_flat[t].to(torch.float32)

            if use_qk_l2norm_in_kernel:
                b_q = _l2norm(b_q)
                b_k = _l2norm(b_k)
            b_q = b_q * scale

            b_g = _kda_gate(raw_g_flat[t], A_log, g_bias, float(lower_bound))
            b_g = _expand_heads(b_g, HV, "raw_g")
            # Decay is per (head, key-dim); broadcast along the V axis.
            s = s * b_g.exp().unsqueeze(1)

            b_v = b_v - torch.einsum("hvk,hk->hv", s, b_k)

            # beta is already sigmoided on the chunk path.
            b_beta = beta_flat[t].to(torch.float32)
            b_beta = _expand_heads(b_beta, HV, "beta")
            b_v = b_v * b_beta if b_beta.dim() == 0 else b_v * b_beta.unsqueeze(-1)

            # Outer product along the V axis: [HV, V, 1] * [H, 1, K].
            s = s + b_v[:, :, None] * b_k[:, None, :]
            o_flat[t] = torch.einsum("hvk,hk->hv", s, b_q).to(o.dtype)

        if final_state is not None:
            final_state[n] = s.to(final_state.dtype)

    return o, final_state
