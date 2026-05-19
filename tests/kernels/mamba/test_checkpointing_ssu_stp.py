# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""STP parity tests for FlashInfer checkpointing SSU.

These tests compare the new FlashInfer ``checkpointing_ssu`` path against the
old FlashInfer ``selective_state_update`` path in the single-token decode flow
that vLLM uses without speculative decoding.
"""

from __future__ import annotations

import pytest
import torch

from vllm.config.mamba import MambaBackendEnum, MambaConfig
from vllm.model_executor.layers.mamba.ops.ssu_dispatch import FlashInferSSUBackend
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

try:
    import flashinfer.mamba  # noqa: F401

    HAS_FLASHINFER = True
except Exception:  # pragma: no cover - environment-dependent
    HAS_FLASHINFER = False


requires_flashinfer = pytest.mark.skipif(
    not HAS_FLASHINFER or not torch.cuda.is_available(),
    reason="FlashInfer Mamba kernels require CUDA and flashinfer",
)


def _make_backend() -> FlashInferSSUBackend:
    return FlashInferSSUBackend(
        MambaConfig(
            backend=MambaBackendEnum.FLASHINFER,
            checkpoint_interval=1,
        )
    )


def _make_weights(
    *,
    nheads: int,
    head_dim: int,
    dstate: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # A is per-head and broadcast over the head_dim/dstate axes.
    A = -torch.rand(nheads, device=device, dtype=torch.float32)
    A = A[:, None, None].expand(nheads, head_dim, dstate)
    D = torch.zeros(nheads, device=device, dtype=dtype)[:, None].expand(
        nheads, head_dim
    )
    dt_bias = torch.zeros(nheads, device=device, dtype=dtype)[:, None].expand(
        nheads, head_dim
    )
    return A, D, dt_bias


def _make_checkpointing_cache(
    *,
    cache_size: int,
    nheads: int,
    head_dim: int,
    dstate: int,
    ngroups: int,
    max_window: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    return {
        "old_x": torch.zeros(
            cache_size,
            max_window,
            nheads,
            head_dim,
            device=device,
            dtype=dtype,
        ),
        "old_B": torch.zeros(
            cache_size,
            2,
            max_window,
            ngroups,
            dstate,
            device=device,
            dtype=dtype,
        ),
        "old_dt": torch.zeros(
            cache_size,
            2,
            nheads,
            max_window,
            device=device,
            dtype=torch.float32,
        ),
        "old_cumAdt": torch.zeros(
            cache_size,
            2,
            nheads,
            max_window,
            device=device,
            dtype=torch.float32,
        ),
        "cache_buf_idx": torch.zeros(cache_size, device=device, dtype=torch.int32),
        "prev_num_accepted_tokens": torch.zeros(
            cache_size, device=device, dtype=torch.int32
        ),
    }


def _make_decode_inputs(
    *,
    batch_size: int,
    nheads: int,
    head_dim: int,
    dstate: int,
    ngroups: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(seed)
    x = 0.1 * torch.randn(
        batch_size,
        nheads,
        head_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    dt_base = 0.1 * torch.randn(
        batch_size,
        nheads,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    dt = dt_base.unsqueeze(-1).expand(batch_size, nheads, head_dim)
    B = 0.1 * torch.randn(
        batch_size,
        ngroups,
        dstate,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    C = 0.1 * torch.randn(
        batch_size,
        ngroups,
        dstate,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    return x, dt, B, C


def _call_backend(
    backend: FlashInferSSUBackend,
    *,
    state: torch.Tensor,
    cache: dict[str, torch.Tensor] | None,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    dt_bias: torch.Tensor,
    state_batch_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int = 1,
) -> torch.Tensor:
    out = torch.empty_like(x)
    kwargs = {}
    if cache is not None:
        kwargs = {
            "old_x": cache["old_x"],
            "old_B": cache["old_B"],
            "old_dt": cache["old_dt"],
            "old_cumAdt": cache["old_cumAdt"],
            "cache_buf_idx": cache["cache_buf_idx"],
            "prev_num_accepted_tokens": cache["prev_num_accepted_tokens"],
        }
    backend(
        state,
        x,
        dt,
        A,
        B,
        C,
        D,
        dt_bias,
        dt_softplus=True,
        state_batch_indices=state_batch_indices,
        null_block_id=NULL_BLOCK_ID,
        out=out,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        **kwargs,
    )
    return out


@requires_flashinfer
@pytest.mark.parametrize("metadata_max_seqlen", [1, 8])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_checkpointing_ssu_stp_outputs_match_old_flashinfer(
    batch_size: int,
    metadata_max_seqlen: int,
    dtype: torch.dtype,
) -> None:
    """The new STP path should produce the same token outputs as old FI SSU.

    With ``checkpoint_interval=1``, checkpointing SSU still keeps one replay
    token between calls, so the HBM state is not expected to equal the old
    materialized state at every step. The observable decode outputs should
    match, and this test exercises several consecutive STP decode iterations
    so the replay tracker is used.
    """

    torch.manual_seed(0)
    device = torch.device("cuda")
    cache_size = 8
    nheads = 4
    head_dim = 64
    dstate = 128
    ngroups = 1
    max_window = 1

    old_backend = _make_backend()
    new_backend = _make_backend()

    initial_state = 0.1 * torch.randn(
        cache_size,
        nheads,
        head_dim,
        dstate,
        device=device,
        dtype=torch.float16,
    )
    old_state = initial_state.clone()
    new_state = initial_state.clone()
    cache = _make_checkpointing_cache(
        cache_size=cache_size,
        nheads=nheads,
        head_dim=head_dim,
        dstate=dstate,
        ngroups=ngroups,
        max_window=max_window,
        device=device,
        dtype=dtype,
    )
    A, D, dt_bias = _make_weights(
        nheads=nheads,
        head_dim=head_dim,
        dstate=dstate,
        device=device,
        dtype=dtype,
    )

    slots = torch.arange(batch_size, device=device, dtype=torch.int32) * 2 + 1
    state_batch_indices = slots[:, None]
    cu_seqlens = torch.arange(batch_size + 1, device=device, dtype=torch.int32)

    for step in range(5):
        x, dt, B, C = _make_decode_inputs(
            batch_size=batch_size,
            nheads=nheads,
            head_dim=head_dim,
            dstate=dstate,
            ngroups=ngroups,
            device=device,
            dtype=dtype,
            seed=100 + step,
        )
        old_out = _call_backend(
            old_backend,
            state=old_state,
            cache=None,
            x=x,
            dt=dt,
            A=A,
            B=B,
            C=C,
            D=D,
            dt_bias=dt_bias,
            state_batch_indices=state_batch_indices,
            cu_seqlens=cu_seqlens,
            max_seqlen=metadata_max_seqlen,
        )
        new_out = _call_backend(
            new_backend,
            state=new_state,
            cache=cache,
            x=x,
            dt=dt,
            A=A,
            B=B,
            C=C,
            D=D,
            dt_bias=dt_bias,
            state_batch_indices=state_batch_indices,
            cu_seqlens=cu_seqlens,
            max_seqlen=metadata_max_seqlen,
        )

        torch.testing.assert_close(
            new_out.float(),
            old_out.float(),
            atol=3e-2,
            rtol=3e-2,
            msg=f"STP output mismatch at decode step {step}",
        )
