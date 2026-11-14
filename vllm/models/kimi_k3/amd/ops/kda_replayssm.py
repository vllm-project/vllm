# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 KDA ReplaySSM on ROCm.

The speculative verify runs through AITER's FlyDSL kernel
`flydsl_kda_verify_with_f_b` (ROCm/aiter#7529) in ring mode. It reads the
checkpoint `state` and `conv_state` and writes neither. Instead, for each step
it records the step's inputs in the request's ring, which sits in the KDA page:
the convolved V and K before the L2 norm (BF16), and the log decay g and beta
(FP32). It also writes the step's conv window to the layer's
`intermediate_conv_window` line. Once sampling knows that `n` tokens were
accepted, `KDAReplaySSMCommit` replays steps `0..n-1` of the ring onto the
checkpoint and copies window `n-1` into `conv_state`. It does this for every
KDA layer of a KV cache group in one launch.

The fold follows the FlyDSL recurrence: K is scaled by
`rsqrt(sum(k^2) + 1e-6)`, the state decays by `exp2(g * log2(e))`, and the
delta rule runs on that decayed state. The state is rounded to its dtype only
once, after the last step.
"""

import functools
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import torch

from vllm.config import VllmConfig
from vllm.distributed import divide
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import PIN_MEMORY

if TYPE_CHECKING:
    from vllm.models.kimi_k3.amd.kda import KimiK3DeltaAttention

# The only shapes the FlyDSL kernel is built for: Kimi-K3 at TP8.
KDA_REPLAYSSM_HEADS = 12
KDA_REPLAYSSM_HEAD_DIM = 128
KDA_REPLAYSSM_CONV_SIZE = 4
KDA_REPLAYSSM_L2NORM_EPS = 1e-6
# (raw_v, raw_k, g, beta) of the ring, after (conv_state, recurrent_state).
KDA_REPLAYSSM_RING_DTYPES = (
    torch.bfloat16,
    torch.bfloat16,
    torch.float32,
    torch.float32,
)
# Per-layer base addresses in the commit's table: state, raw_v, raw_k, g, beta,
# conv_state and intermediate_conv_window.
_NUM_ADDRS = 7
# V rows of one commit program, out of HEAD_DIM.
_COMMIT_BLOCK_V = 16


def use_kda_replayssm(vllm_config: VllmConfig) -> bool:
    """Whether Kimi-K3's KDA layers verify through the FlyDSL ReplaySSM ring.

    `validate_mamba_cached_kernel` sets `use_kda_recoverssm` to the same value
    for Kimi-K3. That validator runs after the platform sizes the Mamba page,
    though, and the page needs the answer first.
    """
    return (
        vllm_config.cache_config.use_replayssm
        and vllm_config.num_speculative_tokens > 0
    )


def kda_replayssm_state_shape(
    tp_size: int,
    num_heads: int,
    head_dim: int,
    conv_kernel_size: int,
    num_spec: int,
) -> tuple[tuple[int, ...], ...]:
    """The KDA page with a ring: the checkpoint conv state (no per-draft
    columns), the recurrent state, then the ring `(raw_v, raw_k, g, beta)` with
    one entry per verified token."""
    conv_state_shape, recurrent_state_shape = MambaStateShapeCalculator.kda_state_shape(
        tp_size, num_heads, head_dim, conv_kernel_size=conv_kernel_size, num_spec=0
    )
    local_heads = divide(num_heads, tp_size)
    ring_len = num_spec + 1
    return (
        conv_state_shape,
        recurrent_state_shape,
        (local_heads, ring_len, head_dim),
        (local_heads, ring_len, head_dim),
        (local_heads, ring_len, head_dim),
        (local_heads, ring_len),
    )


def kda_replayssm_state_dtype(
    state_dtypes: tuple[torch.dtype, torch.dtype],
) -> tuple[torch.dtype, ...]:
    return tuple(state_dtypes) + KDA_REPLAYSSM_RING_DTYPES


@functools.cache
def get_flydsl_kda_verify() -> Callable | None:
    """AITER's FlyDSL KDA verify, or None when AITER lacks it or the GPU is not
    gfx942 / gfx950."""
    try:
        from aiter.ops.flydsl import (
            flydsl_kda_verify_with_f_b,
            is_flydsl_kda_decode_supported,
        )
    except ImportError:
        return None
    if not is_flydsl_kda_decode_supported():
        return None
    return flydsl_kda_verify_with_f_b


_PRECOMPILED: set[tuple] = set()


def precompile_flydsl_kda_verify(
    verify: Callable,
    *,
    num_tokens: int,
    state_dtype: torch.dtype,
    norm_eps: float,
    lower_bound: float,
    device: torch.device,
) -> None:
    """Build every wave-count specialization with a batch-1 padding call, so
    CUDA graph capture compiles nothing. The kernel is shared by every layer."""
    key = (num_tokens, state_dtype, float(norm_eps), float(lower_bound), device)
    if key in _PRECOMPILED:
        return
    heads, dim = KDA_REPLAYSSM_HEADS, KDA_REPLAYSSM_HEAD_DIM
    channels = 3 * heads * dim
    width = KDA_REPLAYSSM_CONV_SIZE - 1
    bf16 = dict(dtype=torch.bfloat16, device=device)
    f32 = dict(dtype=torch.float32, device=device)
    # A negative cache index makes the row padding: nothing is read from or
    # written to a cache.
    padding = torch.full((1,), -1, dtype=torch.int32, device=device)
    args = dict(
        f_a=torch.zeros(num_tokens, dim, **bf16),
        f_b_weight=torch.zeros(heads, dim, dim, **bf16),
        x=torch.zeros(num_tokens, channels, **bf16),
        conv_weight=torch.zeros(channels, KDA_REPLAYSSM_CONV_SIZE, **f32),
        conv_bias=None,
        conv_state=torch.zeros(1, channels, width, **bf16),
        raw_beta=torch.zeros(1, num_tokens, heads, **bf16),
        A_log=torch.zeros(heads, **f32),
        dt_bias=torch.zeros(heads * dim, **f32),
        lower_bound=lower_bound,
        state=torch.zeros(1, heads, dim, dim, dtype=state_dtype, device=device),
        cache_indices=padding,
        intermediate_indices=padding,
        intermediate_conv_window=torch.zeros(1, num_tokens, channels, width, **bf16),
        output_gate=torch.zeros(num_tokens, heads, dim, **bf16),
        norm_weight=torch.ones(dim, **bf16),
        norm_eps=norm_eps,
        num_tokens=num_tokens,
        replay_ring=(
            torch.zeros(1, heads, num_tokens, dim, **bf16),
            torch.zeros(1, heads, num_tokens, dim, **bf16),
            torch.zeros(1, heads, num_tokens, dim, **f32),
            torch.zeros(1, heads, num_tokens, **f32),
        ),
    )
    for num_warps in verify.num_warps:
        verify(**args, num_warps=num_warps)
    _PRECOMPILED.add(key)


@triton.jit
def _kda_replayssm_commit_kernel(
    state_ref_ptr,
    raw_v_ref_ptr,
    raw_k_ref_ptr,
    g_ref_ptr,
    beta_ref_ptr,
    conv_ref_ptr,
    window_ref_ptr,
    base_addrs_ptr,
    cache_indices_ptr,
    line_indices_ptr,
    request_rows_ptr,
    num_accepted_ptr,
    stride_state_slot,
    stride_raw_v_slot,
    stride_raw_k_slot,
    stride_g_slot,
    stride_beta_slot,
    stride_conv_slot,
    stride_conv_dim,
    stride_conv_width,
    stride_window_line,
    stride_window_step,
    stride_window_dim,
    stride_window_width,
    num_slots,
    num_lines,
    eps,
    MAX_STEPS: tl.constexpr,
    RING_LEN: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CONV_STATE_LEN: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BV: tl.constexpr,
    NUM_ADDRS: tl.constexpr,
    HAS_REQUEST_ROWS: tl.constexpr,
):
    pid_v = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_lh = tl.program_id(2)
    layer = pid_lh // NUM_HEADS
    head = pid_lh % NUM_HEADS

    # Same padding rule as the verify, which wrote nothing for such a row.
    slot = tl.load(cache_indices_ptr + pid_b)
    line = tl.load(line_indices_ptr + pid_b)
    if (slot < 0) | (slot >= num_slots) | (line < 0) | (line >= num_lines):
        return
    row = pid_b
    if HAS_REQUEST_ROWS:
        row = tl.load(request_rows_ptr + pid_b)
    n = tl.minimum(tl.load(num_accepted_ptr + row), MAX_STEPS)
    if n <= 0:
        return
    slot = slot.to(tl.int64)
    line = line.to(tl.int64)

    addrs = base_addrs_ptr + layer * NUM_ADDRS
    state_ptr = tl.load(addrs + 0).to(tl.pointer_type(state_ref_ptr.dtype.element_ty))
    raw_v_ptr = tl.load(addrs + 1).to(tl.pointer_type(raw_v_ref_ptr.dtype.element_ty))
    raw_k_ptr = tl.load(addrs + 2).to(tl.pointer_type(raw_k_ref_ptr.dtype.element_ty))
    g_ptr = tl.load(addrs + 3).to(tl.pointer_type(g_ref_ptr.dtype.element_ty))
    beta_ptr = tl.load(addrs + 4).to(tl.pointer_type(beta_ref_ptr.dtype.element_ty))
    conv_ptr = tl.load(addrs + 5).to(tl.pointer_type(conv_ref_ptr.dtype.element_ty))
    window_ptr = tl.load(addrs + 6).to(tl.pointer_type(window_ref_ptr.dtype.element_ty))

    head_ring = head * RING_LEN * HEAD_DIM
    state_ptr += slot * stride_state_slot + head * HEAD_DIM * HEAD_DIM
    raw_v_ptr += slot * stride_raw_v_slot + head_ring
    raw_k_ptr += slot * stride_raw_k_slot + head_ring
    g_ptr += slot * stride_g_slot + head_ring
    beta_ptr += slot * stride_beta_slot + head * RING_LEN

    offs_v = pid_v * BV + tl.arange(0, BV)
    offs_k = tl.arange(0, HEAD_DIM)
    # [H, V, K] state: V rows are independent, so each program owns BV of them.
    state_ptrs = state_ptr + offs_v[:, None] * HEAD_DIM + offs_k[None, :]
    h = tl.load(state_ptrs).to(tl.float32)
    for t in range(n):
        k = tl.load(raw_k_ptr + t * HEAD_DIM + offs_k).to(tl.float32)
        k = k * tl.math.rsqrt(tl.sum(k * k, axis=0) + eps)
        decay = tl.exp2(tl.load(g_ptr + t * HEAD_DIM + offs_k) * 1.4426950408889634)
        v = tl.load(raw_v_ptr + t * HEAD_DIM + offs_v).to(tl.float32)
        beta = tl.load(beta_ptr + t)
        h = h * decay[None, :]
        v_new = (v - tl.sum(h * k[None, :], axis=1)) * beta
        h = h + v_new[:, None] * k[None, :]
    tl.store(state_ptrs, h.to(state_ptr.dtype.element_ty))

    # The conv window after token n-1 becomes the conv state. This program
    # copies the head's BV channels of each of Q, K and V.
    conv_ptr += slot * stride_conv_slot
    window_ptr += line * stride_window_line + (n - 1) * stride_window_step
    offs_w = tl.arange(0, BLOCK_W)
    mask = (offs_v[:, None] < HEAD_DIM) & (offs_w[None, :] < CONV_STATE_LEN)
    for part in tl.static_range(3):
        channels = (part * NUM_HEADS + head) * HEAD_DIM + offs_v
        window = tl.load(
            window_ptr
            + channels[:, None] * stride_window_dim
            + offs_w[None, :] * stride_window_width,
            mask=mask,
        )
        tl.store(
            conv_ptr
            + channels[:, None] * stride_conv_dim
            + offs_w[None, :] * stride_conv_width,
            window,
            mask=mask,
        )


def conv_state_dim_major(conv_state: torch.Tensor) -> torch.Tensor:
    """`conv_state` as `[slots, dim, width - 1]` in either conv layout."""
    return conv_state if is_conv_state_dim_first() else conv_state.transpose(-1, -2)


class KDAReplaySSMCommit:
    """Folds the accepted ring steps of one KV cache group's KDA layers into
    their checkpoints, in one launch for all of them.

    The layers' caches are addressed through a table of per-layer base
    addresses, rebuilt whenever a KV cache is rebound (the profiling cache,
    then the real one). Every layer of a group has the same page layout, so
    the strides are shared.
    """

    def __init__(self, layers: Sequence["KimiK3DeltaAttention"]) -> None:
        assert layers
        self.layers = tuple(layers)
        self._bound_key: tuple[int, ...] | None = None
        self._base_addrs: torch.Tensor | None = None
        self._refs: tuple[torch.Tensor, ...] = ()
        self._strides: tuple[int, ...] = ()

    @staticmethod
    def _tensors(layer: "KimiK3DeltaAttention") -> tuple[torch.Tensor, ...]:
        conv_state, state, raw_v, raw_k, g, beta = layer.kv_cache
        window = layer.kda_conv_window
        assert window is not None
        return (state, raw_v, raw_k, g, beta, conv_state_dim_major(conv_state), window)

    @staticmethod
    def _layout(tensors: tuple[torch.Tensor, ...]) -> tuple[int, ...]:
        state, raw_v, raw_k, g, beta, conv, window = tensors
        return (
            state.stride(0),
            raw_v.stride(0),
            raw_k.stride(0),
            g.stride(0),
            beta.stride(0),
            *conv.stride(),
            *window.stride(),
            min(t.shape[0] for t in (state, raw_v, raw_k, g, beta, conv)),
            window.shape[0],
        )

    def _bind(self) -> None:
        key = tuple(layer.kv_cache[1].data_ptr() for layer in self.layers)
        if key == self._bound_key:
            return
        tensors = [self._tensors(layer) for layer in self.layers]
        layout = self._layout(tensors[0])
        for layer, layer_tensors in zip(self.layers, tensors):
            if self._layout(layer_tensors) != layout:
                raise ValueError(
                    f"{layer.prefix}: the KDA ReplaySSM commit needs one page "
                    "layout for every layer of a KV cache group"
                )
        # This runs inside a serving step, which must not block the host.
        self._base_addrs = torch.tensor(
            [[t.data_ptr() for t in layer_tensors] for layer_tensors in tensors],
            dtype=torch.int64,
            pin_memory=PIN_MEMORY,
        ).to(tensors[0][0].device, non_blocking=True)
        self._refs = tensors[0]
        self._strides = layout
        self._bound_key = key

    def __call__(
        self,
        cache_indices: torch.Tensor,
        line_indices: torch.Tensor,
        request_rows: torch.Tensor | None,
        num_accepted_tokens: torch.Tensor,
        num_rows: int,
        num_tokens: int,
    ) -> None:
        """Commit spec rows `0..num_rows-1`; `num_accepted_tokens[row]` counts
        the tokens accepted, bonus included, of batch row `row`
        (`request_rows[b]` of spec row `b`, or `b`)."""
        if num_rows == 0:
            return
        self._bind()
        assert self._base_addrs is not None
        (
            stride_state,
            stride_raw_v,
            stride_raw_k,
            stride_g,
            stride_beta,
            stride_conv_slot,
            stride_conv_dim,
            stride_conv_width,
            stride_window_line,
            stride_window_step,
            stride_window_dim,
            stride_window_width,
            num_slots,
            num_lines,
        ) = self._strides
        state, raw_v, raw_k, g, beta, conv, window = self._refs
        heads, ring_len, head_dim = raw_v.shape[1:]
        grid = (head_dim // _COMMIT_BLOCK_V, num_rows, len(self.layers) * heads)
        _kda_replayssm_commit_kernel[grid](
            state,
            raw_v,
            raw_k,
            g,
            beta,
            conv,
            window,
            self._base_addrs,
            cache_indices,
            line_indices,
            request_rows,
            num_accepted_tokens,
            stride_state,
            stride_raw_v,
            stride_raw_k,
            stride_g,
            stride_beta,
            stride_conv_slot,
            stride_conv_dim,
            stride_conv_width,
            stride_window_line,
            stride_window_step,
            stride_window_dim,
            stride_window_width,
            num_slots,
            num_lines,
            KDA_REPLAYSSM_L2NORM_EPS,
            MAX_STEPS=num_tokens,
            RING_LEN=ring_len,
            NUM_HEADS=heads,
            HEAD_DIM=head_dim,
            CONV_STATE_LEN=conv.shape[-1],
            BLOCK_W=triton.next_power_of_2(conv.shape[-1]),
            BV=_COMMIT_BLOCK_V,
            NUM_ADDRS=_NUM_ADDRS,
            HAS_REQUEST_ROWS=request_rows is not None,
            num_warps=4,
        )
