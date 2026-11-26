# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash KDA ReplaySSM verify step (CUDA, sm_90a/100a/103a).

One launch per layer covers the `f_b` / `g_b` projections, the short conv, the
replay of the committed ring records onto the checkpoint, the new tokens and the
gated RMSNorm. A verify step (T = 6, 9) flushes when `c + q > L` and stores the
committed state. See ops/kda_replayssm.cu.
"""

import functools
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import NamedTuple

import torch

from vllm.config import VllmConfig
from vllm.config.mamba import MambaBackendEnum
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionBackend,
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

RING_LEN = 32
WINDOW = 16
RING_DTYPES = (torch.float32, torch.bfloat16, torch.float32)
# Layers per materialize program: few programs to skip, enough to publish.
MATERIALIZE_LAYERS = 8
_OPS = Path(__file__).parent / "ops"


def ring_shapes(num_heads: int, head_dim: int) -> tuple[tuple[int, int, int], ...]:
    """The u, k and g rings in their own block: `[H, R, D]` each."""
    return ((num_heads, RING_LEN, head_dim),) * 3


def ring_views(
    pages: torch.Tensor, num_heads: int, head_dim: int
) -> tuple[torch.Tensor, ...]:
    """The u, k and g rings laid out from the start of each `[S, C]` int8 page."""
    views, offset = [], 0
    for shape, dtype in zip(ring_shapes(num_heads, head_dim), RING_DTYPES):
        nbytes = math.prod(shape) * dtype.itemsize
        views.append(pages[:, offset : offset + nbytes].view(dtype).view(-1, *shape))
        offset += nbytes
    assert offset <= pages.size(1), "the rings must fit in one Mamba page"
    return tuple(views)


class _VerifyMetadataBuilder(GDNAttentionMetadataBuilder):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # A verify row's block table holds its state block and its ring block.
        self.spec_state_indices_tensor = self.spec_state_indices_tensor[:, :2]


class KDAReplaySSMVerifyBackend(GDNAttentionBackend):
    @staticmethod
    def get_builder_cls() -> type[GDNAttentionMetadataBuilder]:
        return _VerifyMetadataBuilder


def check_supported(
    vllm_config: VllmConfig,
    num_heads: int,
    head_dim: int,
    conv_size: int,
    state_dtypes: tuple[torch.dtype, torch.dtype],
    lower_bound: float | None,
    gate_projs: tuple[LinearBase, ...],
) -> None:
    """Reject at startup every configuration the kernel was not built for."""
    cache_config = vllm_config.cache_config
    query_len = 1 + vllm_config.num_speculative_tokens
    # The step's fp32 exp of the summed gates (each >= lower_bound) of up to
    # `query_len` tokens must not flush to zero.
    min_lower_bound = math.log(torch.finfo(torch.float32).tiny) / query_len
    unsupported = {
        f"{query_len - 1} speculative tokens (needs 5 or 8)": query_len not in (6, 9),
        f"{num_heads} local heads (needs 8 or 16: TP8 or TP4)": num_heads
        not in (8, 16),
        "head_dim other than 128 or conv width other than 4": (head_dim, conv_size)
        != (128, 4),
        f"--replayssm-buffer-len other than {WINDOW}": (
            cache_config.replayssm_buffer_len != WINDOW
        ),
        "--mamba-backend other than triton": vllm_config.mamba_config.backend
        != MambaBackendEnum.TRITON,
        "the DS conv-state layout": is_conv_state_dim_first(),
        "a model dtype or cache dtypes other than bf16 / bf16 conv / fp32 state": (
            vllm_config.model_config.dtype,
            *state_dtypes,
        )
        != (torch.bfloat16, torch.bfloat16, torch.float32),
        "GPUs other than SM90 / SM100 / SM103": current_platform.get_device_capability()
        not in ((9, 0), (10, 0), (10, 3)),
        f"a gate lower bound outside ({min_lower_bound:.2f}, 0)": lower_bound is None
        or not min_lower_bound < lower_bound < 0,
        "quantized f_b / g_b projections": not all(
            isinstance(p.quant_method, UnquantizedLinearMethod) for p in gate_projs
        ),
        "LoRA": vllm_config.lora_config is not None,
    }
    reasons = [reason for reason, bad in unsupported.items() if bad]
    if reasons:
        raise ValueError("GLM-5.3 KDA ReplaySSM does not support " + ", ".join(reasons))
    _module()


@functools.cache
def _module():
    from flashinfer.jit.core import gen_jit_spec

    spec = gen_jit_spec(
        "glm5next_kda_replayssm",
        [_OPS / "kda_replayssm.cu", _OPS / "kda_replayssm_binding.cu"],
        extra_cuda_cflags=["-lineinfo"]
        + [f"-gencode=arch=compute_{a},code=sm_{a}" for a in ("90a", "100a", "103a")],
        use_fast_math=False,
    )
    return spec.build_and_load()


def kda_replayssm_step(
    x: torch.Tensor,
    beta: torch.Tensor,
    f_a: torch.Tensor,
    g_a: torch.Tensor,
    conv_weight: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    f_b_weight: torch.Tensor,
    g_b_weight: torch.Tensor,
    norm_weight: torch.Tensor,
    query_start_loc: torch.Tensor,
    slot: torch.Tensor,
    ring_slot: torch.Tensor,
    num_accepted: torch.Tensor,
    ring_start: torch.Tensor,
    num_committed: torch.Tensor,
    conv_state: torch.Tensor,
    checkpoint: torch.Tensor,
    u_ring: torch.Tensor,
    k_ring: torch.Tensor,
    g_ring: torch.Tensor,
    out: torch.Tensor,
    lower_bound: float,
    scale: float,
    norm_eps: float,
) -> None:
    """One step of N requests with up to T tokens each (`conv_state` holds T + 2
    rows), on the packed token rows `[M, ...]`; out rows past the batch total
    are zeroed.

    The shapes and dtypes are those of `step` in ops/kda_replayssm_binding.cu.
    Request n's rings are at slot `ring_slot[n]`, its other state at `slot[n]`.
    """
    module = _module()
    M, _, H, D = x.shape
    scratch_bytes = module.scratch_bytes(M, H, D)
    scratch = torch.empty(scratch_bytes, dtype=torch.uint8, device=x.device)
    module.step(
        x, beta, f_a, g_a, conv_weight, A_log, dt_bias, f_b_weight, g_b_weight,
        norm_weight, query_start_loc, slot, ring_slot, num_accepted, ring_start,
        num_committed,
        conv_state, checkpoint, u_ring, k_ring, g_ring, out, lower_bound, scale,
        norm_eps, scratch,
    )  # fmt: skip


class StepCursors(NamedTuple):
    query_start_loc: torch.Tensor
    slot: torch.Tensor
    ring_slot: torch.Tensor
    num_accepted: torch.Tensor
    ring_start: torch.Tensor
    num_committed: torch.Tensor


def step_cursors(layer, metadata: GDNAttentionMetadata) -> StepCursors:
    """The step's verify rows and cursors, built once per step and metadata
    and held in the forward context. The cursors are the ReplaySSM tracker's,
    committed after acceptance by the runner."""
    held = get_forward_context().additional_kwargs.setdefault(
        "kda_replayssm_cursors", {}
    )
    if (cached := held.get(id(metadata))) is not None:
        return cached
    assert metadata.num_spec_decodes and not metadata.num_decodes
    assert metadata.spec_query_start_loc is not None
    assert metadata.spec_state_indices_tensor is not None
    assert metadata.num_accepted_tokens is not None
    slot = metadata.spec_state_indices_tensor[:, 0].contiguous()
    cursors = StepCursors(
        query_start_loc=metadata.spec_query_start_loc,
        slot=slot,
        # The rings fill the request's speculative block.
        ring_slot=metadata.spec_state_indices_tensor[:, 1].contiguous(),
        num_accepted=metadata.num_accepted_tokens,
        ring_start=layer._replayssm_ring_start.index_select(0, slot),
        num_committed=layer._replayssm_prev_num_accepted.index_select(0, slot),
    )
    held[id(metadata)] = cursors
    return cursors


@triton.jit
def _slot_base(ptrs, strides, i, slot, DTYPE: tl.constexpr):
    return tl.load(ptrs + i).to(tl.pointer_type(DTYPE)) + slot * tl.load(strides + i)


@triton.jit
def _materialize_kernel(
    ptrs,
    strides,
    src_slots,
    ring_slots,
    dst_slots,
    ring_start,
    flush_count,
    active_request_indices,
    num_layers,
    PAD_SLOT_ID: tl.constexpr,
    MAX_NUM_REQS: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    LAYERS: tl.constexpr,
):
    req = tl.load(active_request_indices + tl.program_id(0))
    if req < 0:
        return
    head = tl.program_id(1) % NUM_HEADS
    first_layer = tl.program_id(1) // NUM_HEADS * LAYERS
    rows = tl.arange(0, R)
    offs = tl.arange(0, D)
    records = rows[:, None] < tl.load(flush_count + req)
    ring_rows = (tl.load(ring_start + req) + rows) % R
    ring = head * R * D + ring_rows[:, None] * D + offs[None, :]
    state = head * D * D + offs[:, None] * D + offs[None, :]
    for layer in range(first_layer, min(first_layer + LAYERS, num_layers)):
        slot_offset = layer * MAX_NUM_REQS + req
        src = tl.load(src_slots + slot_offset).to(tl.int64)
        dst = tl.load(dst_slots + slot_offset).to(tl.int64)
        if (src != PAD_SLOT_ID) & (dst != PAD_SLOT_ID):
            # `[layers, 4]` tables: checkpoint, u, k, g.
            i = 4 * layer
            ring_slot = tl.load(ring_slots + slot_offset).to(tl.int64)
            u = _slot_base(ptrs, strides, i + 1, ring_slot, tl.float32)
            k = _slot_base(ptrs, strides, i + 2, ring_slot, tl.bfloat16)
            g = _slot_base(ptrs, strides, i + 3, ring_slot, tl.float32)
            u = tl.load(u + ring, mask=records, other=0.0)
            k = tl.load(k + ring, mask=records, other=0.0).to(tl.float32)
            g = tl.load(g + ring, mask=records, other=0.0)
            # S_n = S_0 e^{G} + sum_j u_j (k_j e^{gates after j})^T, G the gate sum.
            after = tl.cumsum(g, axis=0, reverse=True) - g
            S = tl.load(_slot_base(ptrs, strides, i, src, tl.float32) + state)
            S = S * tl.exp(tl.sum(g, axis=0))[None, :]
            S = tl.dot(tl.trans(u), k * tl.exp(after), S, input_precision="tf32x3")
            tl.store(_slot_base(ptrs, strides, i, dst, tl.float32) + state, S)


def replayssm_materializer(mixers: Sequence) -> Callable[..., None]:
    """The prefix-cache snapshot writer of these KDA layers: it stores each
    `src` checkpoint `[H, V, K]`, advanced by the first `flush_count` records
    of the rings at `ring_slots`, at `dst` (a `[layers, max_num_reqs]` slot
    table each), for the requests in `active_request_indices`."""
    caches = [mixer.kv_cache[1:] for mixer in mixers]
    device = caches[0][0].device
    ptrs, strides = (
        torch.tensor(
            [[f(state) for state in cache] for cache in caches],
            dtype=torch.int64,
            device=device,
        )
        for f in (torch.Tensor.data_ptr, lambda t: t.stride(0))
    )
    num_heads, head_dim = caches[0][0].shape[1:3]

    def materialize(
        src_slots: torch.Tensor,
        ring_slots: torch.Tensor,
        dst_slots: torch.Tensor,
        ring_start: torch.Tensor,
        flush_count: torch.Tensor,
        active_request_indices: torch.Tensor,
        num_reqs: int,
    ) -> None:
        num_layers = len(caches)
        _materialize_kernel[
            (num_reqs, num_heads * triton.cdiv(num_layers, MATERIALIZE_LAYERS))
        ](
            ptrs,
            strides,
            src_slots,
            ring_slots,
            dst_slots,
            ring_start,
            flush_count,
            active_request_indices,
            num_layers,
            PAD_SLOT_ID=NULL_BLOCK_ID,
            MAX_NUM_REQS=src_slots.size(1),
            NUM_HEADS=num_heads,
            D=head_dim,
            R=RING_LEN,
            LAYERS=MATERIALIZE_LAYERS,
            num_warps=8,
        )

    return materialize
