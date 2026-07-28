# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pre-JIT the FlashInfer ReplaySSM speculative SSU before CUDA-graph capture.

``checkpointing_ssu`` JIT-compiles one module per specialisation key. Left to
the first real request, that compile lands inside graph capture (and, on the V2
runner, after it -- ``warmup_kernels`` runs post-capture, so only the shared
pre-capture ``kernel_warmup`` hook can cover both runners).

Warms against isolated dummy tensors rather than the live KV cache: the kernel
mutates the state and the ring in place.
"""

from typing import TYPE_CHECKING, Any

import torch

from vllm.config.mamba import MambaBackendEnum
from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2
from vllm.model_executor.layers.mamba.ops.replayssm_spec_flashinfer import (
    get_replayssm_spec_flashinfer_backend,
)
from vllm.tracing import instrument
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)

# pad_slot_id is NULL_BLOCK_ID (0), so a one-row cache indexed at 0 would only
# exercise the padded-row early exit. Reserve row 0 and warm on row 1.
_WARMUP_CACHE_ROWS = 2
_WARMUP_SLOT = 1
_MMA_WARP_SIZE = 32
_MMA_FRAG_SIZE = 8
_MMA_M_TILE = 16


def _jit_key(mixer: MambaMixer2) -> tuple[Any, ...]:
    """The tuple FlashInfer's _get_module specialises on.

    Kept explicit even where components coincide in this configuration, so a
    future dtype or sharding change forces a new warmup entry instead of
    silently compiling at serving time. ``algorithm`` and
    ``precompute_heads_per_cta`` are runtime choices, not part of the key.
    """
    dtypes = mixer.get_state_dtype()
    state_dtype = dtypes[1]
    activation_dtype = dtypes[2]
    _, ssm_shape, *_ = mixer.get_state_shape()
    local_nheads, head_dim, dstate = ssm_shape
    local_ngroups = mixer.n_groups // mixer.tp_size
    return (
        state_dtype,
        activation_dtype,  # input_dtype
        activation_dtype,  # dt_dtype
        mixer.D.dtype,  # weight_dtype (D and dt_bias share it)
        torch.float32,  # matrixA_dtype
        torch.int32,  # stateIndex_dtype
        None,  # state_scale_dtype: fp8/int8 state is out of scope
        head_dim,
        dstate,
        mixer.max_spec_len,
        mixer.replayssm_buffer_len,
        local_nheads // local_ngroups,  # heads_per_group
        local_ngroups,
        0,  # philox_rounds: stochastic rounding is rejected in config
        False,  # enable_pdl
    )


def _warm_one(key: tuple[Any, ...], device: torch.device, algorithm: str) -> None:
    (
        state_dtype,
        activation_dtype,
        _dt_dtype,
        weight_dtype,
        _matrix_a_dtype,
        _state_index_dtype,
        _state_scale_dtype,
        head_dim,
        dstate,
        max_spec_len,
        buffer_len,
        heads_per_group,
        ngroups,
        _philox,
        _pdl,
    ) = key
    nheads = heads_per_group * ngroups
    ring_len = buffer_len + max_spec_len
    rows = 1
    num_tokens = max_spec_len

    state = torch.zeros(
        _WARMUP_CACHE_ROWS, nheads, head_dim, dstate, dtype=state_dtype, device=device
    )
    x_cache = torch.zeros(
        _WARMUP_CACHE_ROWS,
        nheads,
        ring_len,
        head_dim,
        dtype=activation_dtype,
        device=device,
    )
    b_cache = torch.zeros(
        _WARMUP_CACHE_ROWS,
        ngroups,
        ring_len,
        dstate,
        dtype=activation_dtype,
        device=device,
    )
    # Positive so the replayed decays stay bounded, as in production softplus.
    dt_cache = torch.ones(
        _WARMUP_CACHE_ROWS, nheads, ring_len, dtype=torch.float32, device=device
    )
    ring_start = torch.zeros(_WARMUP_CACHE_ROWS, dtype=torch.int32, device=device)
    history_len = torch.zeros(_WARMUP_CACHE_ROWS, dtype=torch.int32, device=device)

    x = torch.zeros(
        1, num_tokens, nheads, head_dim, dtype=activation_dtype, device=device
    )
    out = torch.zeros_like(x)
    b = torch.zeros(
        1, num_tokens, ngroups, dstate, dtype=activation_dtype, device=device
    )
    c = torch.zeros_like(b)
    dt = (
        torch.zeros(1, num_tokens, nheads, dtype=activation_dtype, device=device)
        .unsqueeze(-1)
        .expand(-1, -1, -1, head_dim)
    )
    a = torch.full((nheads,), -1.0, dtype=torch.float32, device=device)[
        :, None, None
    ].expand(-1, head_dim, dstate)
    d = torch.zeros(nheads, dtype=weight_dtype, device=device)[:, None].expand(
        -1, head_dim
    )
    dt_bias = torch.zeros(nheads, dtype=weight_dtype, device=device)[:, None].expand(
        -1, head_dim
    )
    state_batch_indices = torch.full(
        (rows,), _WARMUP_SLOT, dtype=torch.int32, device=device
    )
    cu_seqlens = torch.tensor([0, num_tokens], dtype=torch.int32, device=device)

    scratch: dict[str, torch.Tensor] = {}
    if algorithm != "monolith":
        k_old = ((buffer_len + 7) // 8) * 8
        scratch = {
            "cb_scaled": torch.zeros(
                rows,
                nheads,
                _MMA_WARP_SIZE,
                _MMA_FRAG_SIZE,
                dtype=activation_dtype,
                device=device,
            ),
            "cumAdt_vec": torch.zeros(
                rows, nheads, _MMA_M_TILE, dtype=torch.float32, device=device
            ),
            "cb_old": torch.zeros(
                rows,
                nheads,
                _MMA_WARP_SIZE,
                k_old // 2,
                dtype=activation_dtype,
                device=device,
            ),
        }

    assert state_batch_indices[0].item() != NULL_BLOCK_ID
    get_replayssm_spec_flashinfer_backend()(
        state,
        x_cache,
        b_cache,
        dt_cache,
        ring_start,
        history_len,
        x,
        dt,
        a,
        b,
        c,
        out,
        D=d,
        dt_bias=dt_bias,
        dt_softplus=True,
        state_batch_indices=state_batch_indices,
        query_start_loc=cu_seqlens,
        max_spec_len=max_spec_len,
        replayssm_buffer_len=buffer_len,
        **scratch,
    )


@instrument(span_name="FlashInfer ReplaySSM spec warmup")
def replayssm_spec_flashinfer_warmup(worker: "Worker") -> None:
    """Compile every distinct FlashInfer specialisation this model will use.

    Runs on every TP rank: each rank JITs against its own TP-local head and
    group counts.
    """
    vllm_config = worker.vllm_config
    if not vllm_config.cache_config.use_replayssm_spec:
        return
    if vllm_config.mamba_config.backend != MambaBackendEnum.FLASHINFER:
        return

    algorithm = vllm_config.mamba_config.replayssm_spec_algorithm
    # A uniform Mamba2 stack collapses to one key, but mixed head/group
    # sharding across layers would not, so deduplicate rather than assume.
    keys = {
        _jit_key(module)
        for module in worker.get_model().modules()
        if isinstance(module, MambaMixer2)
    }
    if not keys:
        return

    device = worker.device
    logger.info(
        "Warming up %d FlashInfer ReplaySSM spec specialisation(s) (algorithm=%s).",
        len(keys),
        algorithm,
    )
    for key in keys:
        _warm_one(key, device, algorithm)
    torch.accelerator.synchronize()
