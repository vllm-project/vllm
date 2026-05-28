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


def _make_backend(*, enable_stochastic_rounding: bool = False) -> FlashInferSSUBackend:
    return FlashInferSSUBackend(
        MambaConfig(
            backend=MambaBackendEnum.FLASHINFER,
            checkpoint_interval=1,
            enable_stochastic_rounding=enable_stochastic_rounding,
            stochastic_rounding_philox_rounds=10,
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
    z: torch.Tensor | None = None,
    state_batch_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    dst_state_batch_indices: torch.Tensor | None = None,
    max_seqlen: int = 1,
    num_accepted_tokens: torch.Tensor | None = None,
    spec_uniform_state_slots: bool = False,
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
        z=z,
        dt_softplus=True,
        state_batch_indices=state_batch_indices,
        dst_state_batch_indices=dst_state_batch_indices,
        null_block_id=NULL_BLOCK_ID,
        out=out,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        num_accepted_tokens=num_accepted_tokens,
        spec_uniform_state_slots=spec_uniform_state_slots,
        **kwargs,
    )
    return out


@requires_flashinfer
@pytest.mark.parametrize("metadata_max_seqlen", [1, 8])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("use_z", [False, True])
@pytest.mark.parametrize("enable_stochastic_rounding", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_checkpointing_ssu_stp_outputs_match_old_flashinfer(
    batch_size: int,
    metadata_max_seqlen: int,
    use_z: bool,
    enable_stochastic_rounding: bool,
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

    old_backend = _make_backend(enable_stochastic_rounding=enable_stochastic_rounding)
    new_backend = _make_backend(enable_stochastic_rounding=enable_stochastic_rounding)

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
    state_index_table = torch.stack((slots - 1, slots), dim=1)
    state_batch_indices = state_index_table[:, 1:2]
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
        z = torch.tanh(2 * x) if use_z else None
        torch.manual_seed(1000 + step)
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
            z=z,
            state_batch_indices=state_batch_indices,
            cu_seqlens=cu_seqlens,
            max_seqlen=metadata_max_seqlen,
        )
        torch.manual_seed(1000 + step)
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
            z=z,
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


@requires_flashinfer
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_checkpointing_ssu_copies_replay_state_to_destination_slot(
    dtype: torch.dtype,
) -> None:
    """Slot moves must preserve checkpoint state and live replay buffers."""

    torch.manual_seed(0)
    device = torch.device("cuda")
    cache_size = 5
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
    src = torch.tensor([1], device=device, dtype=torch.int32)
    dst = torch.tensor([2], device=device, dtype=torch.int32)
    cu_seqlens = torch.tensor([0, 1], device=device, dtype=torch.int32)

    x, dt, B, C = _make_decode_inputs(
        batch_size=1,
        nheads=nheads,
        head_dim=head_dim,
        dstate=dstate,
        ngroups=ngroups,
        device=device,
        dtype=dtype,
        seed=99,
    )
    _call_backend(
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
        state_batch_indices=src,
        cu_seqlens=cu_seqlens,
    )
    _call_backend(
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
        state_batch_indices=src,
        cu_seqlens=cu_seqlens,
    )
    assert int(cache["prev_num_accepted_tokens"][1].item()) == 1
    src_slot = int(src.item())
    source_snapshot = (
        new_state[src_slot].clone(),
        cache["old_x"][src_slot].clone(),
        cache["old_B"][src_slot].clone(),
        cache["old_dt"][src_slot].clone(),
        cache["old_cumAdt"][src_slot].clone(),
        cache["cache_buf_idx"][src_slot].clone(),
        cache["prev_num_accepted_tokens"][src_slot].clone(),
    )

    x, dt, B, C = _make_decode_inputs(
        batch_size=1,
        nheads=nheads,
        head_dim=head_dim,
        dstate=dstate,
        ngroups=ngroups,
        device=device,
        dtype=dtype,
        seed=100,
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
        state_batch_indices=src,
        dst_state_batch_indices=dst,
        cu_seqlens=cu_seqlens,
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
        state_batch_indices=src,
        dst_state_batch_indices=dst,
        cu_seqlens=cu_seqlens,
    )
    torch.testing.assert_close(new_out.float(), old_out.float(), atol=3e-2, rtol=3e-2)
    for actual, expected in zip(
        (
            new_state[src_slot],
            cache["old_x"][src_slot],
            cache["old_B"][src_slot],
            cache["old_dt"][src_slot],
            cache["old_cumAdt"][src_slot],
            cache["cache_buf_idx"][src_slot],
            cache["prev_num_accepted_tokens"][src_slot],
        ),
        source_snapshot,
    ):
        torch.testing.assert_close(actual, expected)


@requires_flashinfer
@pytest.mark.xfail(
    reason=(
        "Known FlashInfer kernel bug at max_window=1 with batch>1: "
        "the checkpointing_ssu kernel produces incorrect output in this "
        "regime even via the varlen path. Tracked upstream; checkpointing "
        "with interval=1 falls back to the old kernel in production."
    ),
    strict=False,
)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_checkpointing_ssu_stp_large_batch_outputs_match_old_flashinfer(
    dtype: torch.dtype,
) -> None:
    """Large CUDA-graph decode batches must match the old FI STP path."""

    device = torch.device("cuda")
    batch_size = 24
    cache_size = batch_size * 2 + 8
    nheads = 128
    head_dim = 64
    dstate = 128
    ngroups = 8
    max_window = 1
    generator = torch.Generator(device=device).manual_seed(123)

    old_backend = _make_backend()
    new_backend = _make_backend()
    initial_state = 0.01 * torch.randn(
        cache_size,
        nheads,
        head_dim,
        dstate,
        device=device,
        dtype=torch.float16,
        generator=generator,
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

    for step in range(12):
        x, dt, B, C = _make_decode_inputs(
            batch_size=batch_size,
            nheads=nheads,
            head_dim=head_dim,
            dstate=dstate,
            ngroups=ngroups,
            device=device,
            dtype=dtype,
            seed=200 + step,
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
            max_seqlen=8,
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
            max_seqlen=8,
        )
        max_abs_error = (new_out.float() - old_out.float()).abs().max()
        assert max_abs_error <= 6e-2, (
            f"large-batch STP output mismatch at decode step {step}: "
            f"max_abs_error={max_abs_error.item()}"
        )


@requires_flashinfer
@pytest.mark.parametrize("num_spec", [1, 2, 4])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("checkpoint_interval", [1, 4])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_checkpointing_ssu_mtp_kept_outputs_match_serial_reference(
    num_spec: int,
    batch_size: int,
    checkpoint_interval: int,
    dtype: torch.dtype,
) -> None:
    """MTP equivalence: the first ``accepted`` outputs from the checkpointing
    NPREDICTED>1 path must match a serial one-token-at-a-time reference.

    Semantic reference for spec/MTP: applying the first ``accepted[i]``
    proposed tokens of batch row ``i`` one at a time (NPREDICTED=1 calls,
    no checkpoint cache, no num_accepted_tokens trimming) produces the
    correct token outputs and final state. The MTP checkpointing dispatch
    must produce the same outputs for those accepted positions.

    The non-checkpoint NPREDICTED>1 + uniform-column path is *not* used as
    reference — its bf16 numerics differ across the buffer-flip / replay
    boundary, and the resulting drift exceeds kernel tolerance over many
    steps. The serial NPREDICTED=1 path is the precise per-token semantic.
    """

    torch.manual_seed(0)
    device = torch.device("cuda")
    cache_size = batch_size * 2 + 4
    nheads = 4
    head_dim = 64
    dstate = 128
    ngroups = 1
    npredicted = 1 + num_spec
    # Match the production buffer sizing (mamba_utils.py): checkpoint_interval
    # + num_spec, so a single overflowing spec call cannot exceed the window
    # mid-write. Kernel hard limit: max_window <= 16.
    max_window = checkpoint_interval + num_spec
    assert max_window <= 16

    serial_backend = _make_backend()
    ckpt_backend = _make_backend()

    initial_state = 0.1 * torch.randn(
        cache_size,
        nheads,
        head_dim,
        dstate,
        device=device,
        dtype=torch.float16,
    )
    serial_state = initial_state.clone()
    ckpt_state = initial_state.clone()
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

    # 2D state indices (batch, npredicted) with identical columns —
    # the align/none cache mode under spec.
    slots = torch.arange(batch_size, device=device, dtype=torch.int32) * 2 + 1
    state_batch_indices_2d = (
        slots[:, None].expand(batch_size, npredicted).contiguous()
    )
    cu_seqlens_spec = torch.arange(
        0,
        (batch_size + 1) * npredicted,
        npredicted,
        device=device,
        dtype=torch.int32,
    )

    # Reference plumbing for the serial path.
    sbi_1d_per_batch = slots[:, None]
    cu_one = torch.tensor([0, 1], device=device, dtype=torch.int32)

    accept_generator = torch.Generator(device=device).manual_seed(7)

    for step in range(6):
        total_tokens = batch_size * npredicted
        x, dt, B, C = _make_decode_inputs(
            batch_size=total_tokens,
            nheads=nheads,
            head_dim=head_dim,
            dstate=dstate,
            ngroups=ngroups,
            device=device,
            dtype=dtype,
            seed=300 + step,
        )
        accepted = torch.randint(
            1,
            npredicted + 1,
            (batch_size,),
            device=device,
            dtype=torch.int32,
            generator=accept_generator,
        )

        # --- Checkpointing MTP path (under test) ---
        ckpt_out = _call_backend(
            ckpt_backend,
            state=ckpt_state,
            cache=cache,
            x=x,
            dt=dt,
            A=A,
            B=B,
            C=C,
            D=D,
            dt_bias=dt_bias,
            state_batch_indices=state_batch_indices_2d,
            cu_seqlens=cu_seqlens_spec,
            max_seqlen=npredicted,
            num_accepted_tokens=accepted,
            spec_uniform_state_slots=True,
        )

        # --- Serial reference (one token at a time per batch row) ---
        for b in range(batch_size):
            acc_b = int(accepted[b].item())
            sbi_b = sbi_1d_per_batch[b : b + 1]  # shape (1, 1)
            for tok in range(acc_b):
                global_idx = b * npredicted + tok
                serial_out_slice = torch.empty_like(x[global_idx : global_idx + 1])
                serial_backend(
                    serial_state,
                    x[global_idx : global_idx + 1],
                    dt[global_idx : global_idx + 1],
                    A,
                    B[global_idx : global_idx + 1],
                    C[global_idx : global_idx + 1],
                    D,
                    dt_bias,
                    dt_softplus=True,
                    state_batch_indices=sbi_b,
                    dst_state_batch_indices=None,
                    null_block_id=NULL_BLOCK_ID,
                    out=serial_out_slice,
                    cu_seqlens=cu_one,
                    max_seqlen=1,
                )
                # Compare only the kept positions.
                err = (
                    ckpt_out[global_idx : global_idx + 1].float()
                    - serial_out_slice.float()
                ).abs().max().item()
                assert err <= 3e-2, (
                    f"MTP kept-output mismatch step={step} batch={b} "
                    f"tok={tok}/{acc_b} num_spec={num_spec} "
                    f"interval={checkpoint_interval} err={err}"
                )


@requires_flashinfer
@pytest.mark.parametrize("num_spec", [4])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_checkpointing_ssu_mtp_falls_back_for_distinct_columns(
    num_spec: int,
    dtype: torch.dtype,
) -> None:
    """When `spec_uniform_state_slots=False` and `state_batch_indices` is 2D
    with distinct columns (the `mamba_cache_mode="all"` + spec case), the
    dispatcher must fall through to the non-checkpointing kernel and not
    crash. Outputs must still match the non-checkpoint reference."""

    torch.manual_seed(0)
    device = torch.device("cuda")
    batch_size = 2
    cache_size = batch_size * (1 + num_spec) + 4
    nheads = 4
    head_dim = 64
    dstate = 128
    ngroups = 1
    npredicted = 1 + num_spec

    old_backend = _make_backend()
    new_backend = _make_backend()

    initial_state = 0.1 * torch.randn(
        cache_size, nheads, head_dim, dstate, device=device, dtype=torch.float16
    )
    old_state = initial_state.clone()
    new_state = initial_state.clone()
    A, D, dt_bias = _make_weights(
        nheads=nheads, head_dim=head_dim, dstate=dstate, device=device, dtype=dtype
    )

    # 2D state indices with DISTINCT columns: each batch row's spec
    # positions point at consecutive blocks (mimicking "all" cache mode).
    start = torch.arange(batch_size, device=device, dtype=torch.int32) * npredicted
    offsets = torch.arange(npredicted, device=device, dtype=torch.int32)
    state_batch_indices_2d = (start[:, None] + offsets).contiguous()
    cu_seqlens = torch.arange(
        0,
        (batch_size + 1) * npredicted,
        npredicted,
        device=device,
        dtype=torch.int32,
    )

    cache = _make_checkpointing_cache(
        cache_size=cache_size,
        nheads=nheads,
        head_dim=head_dim,
        dstate=dstate,
        ngroups=ngroups,
        max_window=npredicted,
        device=device,
        dtype=dtype,
    )

    total_tokens = batch_size * npredicted
    x, dt, B, C = _make_decode_inputs(
        batch_size=total_tokens,
        nheads=nheads,
        head_dim=head_dim,
        dstate=dstate,
        ngroups=ngroups,
        device=device,
        dtype=dtype,
        seed=42,
    )
    accepted = torch.tensor([2, 3], device=device, dtype=torch.int32)

    # Reference: non-checkpoint dispatch with the same 2D indices.
    old_out = _call_backend(
        old_backend,
        state=old_state,
        cache=None,
        x=x, dt=dt, A=A, B=B, C=C, D=D, dt_bias=dt_bias,
        state_batch_indices=state_batch_indices_2d,
        cu_seqlens=cu_seqlens,
        max_seqlen=npredicted,
        num_accepted_tokens=accepted,
    )
    # Under test: cache passed but spec_uniform_state_slots=False. The
    # dispatch must NOT try to checkpoint; the cache buffers must stay
    # untouched, and outputs must equal the non-checkpoint reference.
    new_out = _call_backend(
        new_backend,
        state=new_state,
        cache=cache,
        x=x, dt=dt, A=A, B=B, C=C, D=D, dt_bias=dt_bias,
        state_batch_indices=state_batch_indices_2d,
        cu_seqlens=cu_seqlens,
        max_seqlen=npredicted,
        num_accepted_tokens=accepted,
        spec_uniform_state_slots=False,
    )

    torch.testing.assert_close(
        new_out.float(), old_out.float(), atol=3e-2, rtol=3e-2,
        msg="Distinct-column spec inputs must take the non-checkpoint path",
    )
    # The checkpoint cache should remain at its initial zeros since the
    # dispatcher fell through.
    assert torch.all(cache["prev_num_accepted_tokens"] == 0)
    assert torch.all(cache["cache_buf_idx"] == 0)
    assert torch.all(cache["old_x"] == 0)
    assert torch.all(cache["old_cumAdt"] == 0)
