# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deferred RecoverSSM verify for KDA layers with an FP32 state (SM10x CuTe).

The RecoverSSM verify of a speculative step reads each request's checkpoint
without writing it, and the commit after sampling reads it again to fold the
accepted tokens in: two reads and one write of the state per step. Deferred, the
commit only records each request's accepted count (`pending`, per request
slot, whose verify records stay in the same slot); the next verify replays those
tokens into the checkpoint from the records, writes it back and verifies the new
window from it: one read and one write of the state per step.

One kernel runs a layer's verify step: the short convolution, the q / k
normalization, the F_B gate projection and decays, the replay, the chunked verify
and the gated RMSNorm (`cute_dsl/fused_tc.py`: persistent, the dot products and
the gate on the tensor cores; where it does not fit shared memory,
`cute_dsl/fused_ws.py`, persistent and warp specialized, for large steps, else
`cute_dsl/fused.py`, one item per CTA or cluster). `fused_ws` and `fused` are
copied from the Kimi-K3 specialized decode.
"""

from collections.abc import Callable
from functools import partial

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_stream

from vllm.triton_utils import tl, triton

from .cute_dsl import fused_tc, fused_ws
from .cute_dsl.fused import KdaFused
from .cute_dsl.fused_tc import KdaFusedTC
from .cute_dsl.fused_ws import KdaFusedWS
from .cute_dsl.host import fake, layout

K = 128
# (request, head) items from which KdaFusedWS takes the step where KdaFusedTC does not
# fit (B200, 64 heads, q 6, 4 replayed, ms per step KdaFused / KdaFusedWS: 512 items
# 1.43 / 1.53, 768 items 2.19 / 1.98, 1024 items 2.92 / 2.28).
WS_MIN_ITEMS = 768
# CTAs per item of KdaFused by item count: clusters fill the GPU at small steps.
FUSED_SPLIT = ((48, 8), (96, 4), (192, 2), (1 << 30, 1))


def decode_conv_weight(weight: torch.Tensor) -> torch.Tensor:
    """The [3, width, H * K] (q, k, v) layout of a [3 * H * K, 1, width] conv
    weight that the kernels read."""
    width = weight.shape[-1]
    return weight.detach().view(3, -1, width).transpose(1, 2).contiguous()


def tensor_cores_fit(heads, tokens, conv_dim_first, conv_size, max_requests, device):
    """Whether KdaFusedTC runs a layer of `heads` x 128 at q `tokens` (conv
    window q + 2 of `conv_size`-byte elements) for steps of up to
    `max_requests` requests: its operands fit shared memory."""
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    items = fused_tc.cta_items(heads, max_requests, sms)
    smem = fused_tc.shared_bytes(
        tokens, tokens, conv_dim_first, tokens + 2, conv_size, items
    )
    return tokens >= 3 and 2 * tokens <= fused_tc.PROBES and smem <= fused_tc.SMEM_LIMIT


def _kernel(heads, max_requests, requests, tokens, conv_state, qkv):
    """KdaFusedTC where its operands stream in (16-byte aligned rows, a conv window
    of at least q + 2) and fit shared memory at `max_requests` requests (any
    batch up to that), else KdaFusedWS from WS_MIN_ITEMS items, else KdaFused."""
    items = heads * requests
    shape, stride = conv_state.shape, conv_state.stride()
    window = shape[-1]
    dim_first = stride[-1] == 1
    size = conv_state.element_size()
    aligned = (stride[0] * size) % 16 == 0 and (qkv.stride(0) * 2) % 16 == 0
    if not dim_first:
        aligned = aligned and (stride[-1] * size) % 16 == 0
    streams = aligned and window >= tokens + 2 and tokens >= 3
    sms = torch.cuda.get_device_properties(qkv.device).multi_processor_count
    if (
        streams
        and window == tokens + 2
        and tensor_cores_fit(heads, tokens, dim_first, size, max_requests, qkv.device)
    ):
        return KdaFusedTC(heads, tokens, 0, False, dim_first, window, sms, max_requests)
    ctas = -(-items // max(2, -(-items // sms)))
    if streams and items >= WS_MIN_ITEMS:
        smem = partial(fused_ws.shared_bytes, tokens, tokens, dim_first, window, size)
        stage_fb = smem(True) <= fused_ws.SMEM_LIMIT
        if stage_fb or smem() <= fused_ws.SMEM_LIMIT:
            return KdaFusedWS(
                heads, tokens, 0, False, dim_first, window, ctas, stage_fb
            )
    split = next(s for limit, s in FUSED_SPLIT if items < limit)
    return KdaFused(heads, tokens, 0, False, split)


# The operands whose leading dimension follows the batch (or the cache's slots).
_BATCH_OPERANDS = frozenset((0, 3, 4, 5, 8, 9, 10, 11, 14, 15, 16, 17, 18))


class KdaDeferredVerify:
    """The deferred RecoverSSM verify of one KDA layer shape (heads, window) for
    steps of up to `max_requests` requests. KdaFusedTC compiles once per operand
    layout with the batch read at run time; the fallbacks compile per batch."""

    def __init__(self, num_heads: int, window: int, max_requests: int) -> None:
        self.num_heads, self.window = num_heads, window
        self.max_requests = max_requests
        self._compiled: dict[tuple, Callable[..., None]] = {}

    def __call__(
        self,
        *,
        qkv: torch.Tensor,
        gate: torch.Tensor,
        f_a: torch.Tensor,
        beta: torch.Tensor,
        conv_weight: torch.Tensor,
        conv_state: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        f_b_weight: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_eps: float,
        lower_bound: float,
        state: torch.Tensor,
        state_indices: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        correction: torch.Tensor,
        key_gate: torch.Tensor,
        record_rows: torch.Tensor,
        pending: torch.Tensor,
        out: torch.Tensor,
    ) -> torch.Tensor:
        """One verify step of `B = state_indices.shape[0]` requests of `q`
        tokens each. `qkv` [B q, 3 H K] is the projection before the conv,
        `gate` [B q, H K] the output gate, `f_a` [B q, K], `beta` [B q, H]
        (raw), `conv_weight` [3, width, H K] (`decode_conv_weight`),
        `conv_state` [slots, 3 H K, window] (a transposed view for the
        state-len-first layout), `f_b_weight` [H K, K], `norm_weight` [K]
        FP32, `state` [slots, H, K, K] FP32 and `state_indices` [B, n]
        (column 0: the checkpoint slot). `correction` / `key_gate` are the
        verify records by record slot, `record_rows` [B] each request's record
        slot and `pending` [record slots] the accepted tokens of its last
        verify not yet in its checkpoint. Writes `out` [B q, H K], the replayed
        checkpoints, the rolled conv window and the new records."""
        requests = state_indices.shape[0]
        operands = (
            qkv, conv_weight, dt_bias, conv_state, f_a, beta, A_log, dt_bias,
            state_indices, state, out, gate, norm_weight,
            f_b_weight, num_accepted_tokens[:requests], correction, key_gate,
            record_rows[:requests], pending,
        )  # fmt: skip
        scalars = tuple(map(cutlass.Float32, (lower_bound, K**-0.5, norm_eps)))
        assert requests <= self.max_requests
        kernel = _kernel(
            self.num_heads, self.max_requests, requests, self.window, conv_state, qkv
        )
        specs = list(map(layout, operands))
        if isinstance(kernel, KdaFusedTC):
            for i in _BATCH_OPERANDS:
                shape, stride, dtype = specs[i]
                specs[i] = ((None, *shape[1:]), stride, dtype)
        key = (type(kernel), *specs)
        compiled = self._compiled.get(key)
        if compiled is None:
            fakes = []
            for shape, stride, dtype in specs:
                if shape[0] is None:
                    shape = (cute.sym_int(), *shape[1:])
                fakes.append(fake(shape, stride, dtype))
            with torch.accelerator.device_index(qkv.device.index):
                compiled = cute.compile(
                    kernel,
                    *fakes[:13],
                    *scalars,
                    *fakes[13:],
                    make_fake_stream(use_tvm_ffi_env_stream=True),
                    options="--enable-tvm-ffi",
                )
            self._compiled[key] = compiled
        # TVM-FFI launch on the current stream, straight from the torch tensors.
        compiled(*operands[:13], lower_bound, K**-0.5, norm_eps, *operands[13:])
        return out


@triton.jit
def _record_pending_kernel(
    state_indices_ptr,
    commit_lens_ptr,
    record_rows_ptr,
    pending_ptr,
    null_block_id,
    stride_state_indices,
):
    """After sampling: request b's commit length becomes the pending replay of
    its record slot, and no commit runs for it."""
    b = tl.program_id(0)
    record_row = tl.load(record_rows_ptr + b)
    slot = tl.load(state_indices_ptr + b * stride_state_indices)
    if (record_row < 0) | (slot <= null_block_id):
        return
    tl.store(pending_ptr + record_row, tl.load(commit_lens_ptr + b))
    tl.store(commit_lens_ptr + b, 0)


def record_pending(
    state_indices: torch.Tensor,
    commit_lens: torch.Tensor,
    record_rows: torch.Tensor,
    pending: torch.Tensor,
    null_block_id: int,
) -> None:
    """The deferred commit of a prepared commit plan (`commit_lens`)."""
    _record_pending_kernel[(state_indices.shape[0],)](
        state_indices, commit_lens, record_rows, pending, null_block_id,
        state_indices.stride(0), num_warps=1,
    )  # fmt: skip
