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
from pathlib import Path
from typing import NamedTuple

import torch

from vllm.config import VllmConfig
from vllm.config.mamba import MambaBackendEnum
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.platforms import current_platform
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionBackend,
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)

RING_LEN = 32
WINDOW = 16
RING_DTYPES = (torch.float32, torch.bfloat16, torch.float32)
_OPS = Path(__file__).parent / "ops"


def ring_shapes(num_heads: int, head_dim: int) -> tuple[tuple[int, int, int], ...]:
    """The u, k and g rings appended to the KDA state: `[H, R, D]` each."""
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
    state_dtypes: tuple[torch.dtype, ...],
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
        "prefix caching": cache_config.enable_prefix_caching,
        "--mamba-backend other than triton": vllm_config.mamba_config.backend
        != MambaBackendEnum.TRITON,
        "more than 256 sequences": vllm_config.scheduler_config.max_num_seqs > 256,
        "the DS conv-state layout": is_conv_state_dim_first(),
        "a model dtype or cache dtypes other than bf16 / bf16 conv / fp32 state": (
            vllm_config.model_config.dtype,
            *state_dtypes[:2],
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
