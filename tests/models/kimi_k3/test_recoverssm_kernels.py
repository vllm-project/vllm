# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The CuTe KDA RecoverSSM verify and state commit against the Triton reference
(``causal_conv1d_update`` + ``kda_recoverssm_verify``, and the Triton commit), on
Mamba page views as vLLM binds them: one padded page per slot holding the conv
state, the SSM state and the verify records."""

from math import prod
from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

if not (
    current_platform.is_cuda() and current_platform.is_device_capability_family(100)
):
    pytest.skip("The CuTe RecoverSSM kernels need SM10x", allow_module_level=True)
pytest.importorskip("cutlass")

from vllm.model_executor.layers.mamba.ops.causal_conv1d import (  # noqa: E402
    causal_conv1d_update,
)
from vllm.models.kimi_k3.nvidia.ops.recoverssm import (  # noqa: E402
    KDARecoverSSMCommitContext,
    kda_recoverssm_verify,
)
from vllm.models.kimi_k3.nvidia.recoverssm.kernels import (  # noqa: E402
    CuteRecoverSSMVerify,
    compile_state_commit,
)
from vllm.models.kimi_k3.nvidia.recoverssm.layout import (  # noqa: E402
    RecoverSSMRecords,
)

D = 128
LOWER_BOUND = -5.0
DEVICE = torch.device("cuda")
# Extra bytes per page: vLLM pads Mamba pages to the attention page size.
PAGE_PADDING = 4096


def _pages(slots: int, heads: int, window: int, dim_first: bool):
    """Page views of (conv state, SSM state, corrections, key / gate) with a
    padded slot stride, filled with random values."""
    conv_dim, conv_len = 3 * heads * D, window + 2
    conv_shape = (
        (slots, conv_dim, conv_len) if dim_first else (slots, conv_len, conv_dim)
    )
    specs = [
        (torch.bfloat16, conv_shape),
        (torch.bfloat16, (slots, heads, D, D)),
        (torch.float32, (slots, heads, window, D)),
        (torch.bfloat16, (slots, heads, window, 2 * D)),
    ]
    sizes = [dtype.itemsize * prod(shape[1:]) for dtype, shape in specs]
    page = sum(sizes) + PAGE_PADDING
    raw = torch.randn(slots, page // 4, device=DEVICE).view(torch.uint8)
    views = []
    for i, (dtype, shape) in enumerate(specs):
        inner = torch.empty(shape[1:], device="meta").stride()
        views.append(
            raw.view(dtype)
            .view(-1)
            .as_strided(
                shape,
                (page // dtype.itemsize, *inner),
                sum(sizes[:i]) // dtype.itemsize,
            )
        )
    conv, state, correction, key_gate = views
    # Bounded values: random bits are not finite floats.
    conv.copy_(torch.randn_like(conv))
    state.copy_(torch.randn_like(state) * 0.1)
    correction.copy_(torch.randn_like(correction))
    key_gate.copy_(torch.randn_like(key_gate))
    return (raw, *views)


def _clone_pages(pages):
    raw = pages[0].clone()
    base = pages[0].data_ptr()
    return (
        raw,
        *(
            raw.view(t.dtype)
            .view(-1)
            .as_strided(t.shape, t.stride(), (t.data_ptr() - base) // t.element_size())
            for t in pages[1:]
        ),
    )


def _inputs(batch: int, heads: int, window: int, varlen: bool, row_pad: int):
    """Raw projections laid out as the KDA in-projection output: q/k/v, then the
    output gate, f_a, beta and the alignment padding in one row."""
    lens = torch.full((batch,), window, dtype=torch.int32)
    if varlen and batch > 2:
        lens[0] = window - 2
        lens[1] = 0
    query_start = torch.zeros(batch + 1, dtype=torch.int32)
    query_start[1:] = torch.cumsum(lens, 0)
    tokens = int(query_start[-1])
    width = 4 * heads * D + D + heads + row_pad
    projected = torch.randn(tokens, width, device=DEVICE, dtype=torch.bfloat16)
    mixed_qkv = projected[:, : 3 * heads * D]
    raw_beta = projected[:, 4 * heads * D + D : 4 * heads * D + D + heads]
    raw_gate = torch.randn(tokens, heads * D, device=DEVICE, dtype=torch.bfloat16)
    return lens, query_start.to(DEVICE), mixed_qkv, raw_gate, raw_beta


@pytest.mark.parametrize("heads", [96, 12])
@pytest.mark.parametrize("dim_first", [True, False])
@pytest.mark.parametrize("conv_weight_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("batch,varlen", [(1, False), (8, False), (13, True)])
@pytest.mark.parametrize("row_pad", [0, 3])
@pytest.mark.parametrize("recoverssm_accepted", [True, False])
@torch.inference_mode()
def test_cute_verify_matches_triton(
    heads, dim_first, conv_weight_dtype, batch, varlen, row_pad, recoverssm_accepted
):
    """``row_pad`` 3 misaligns the projection rows, so the dim-first layout falls
    back from the warp-specialized verify to the plain tcgen05 one.
    ``recoverssm_accepted`` feeds RecoverSSM's accepted counts (always 1: the
    commit compacts the conv state); otherwise the stock spec-decode counts."""
    torch.manual_seed(batch * 7 + heads + row_pad)
    window = 5
    slots = 3 * batch + 1
    lens, query_start, mixed_qkv, raw_gate, raw_beta = _inputs(
        batch, heads, window, varlen, row_pad
    )
    tokens = mixed_qkv.shape[0]
    A_log = torch.randn(heads, device=DEVICE) * 0.5
    dt_bias = torch.randn(heads * D, device=DEVICE) * 0.5
    conv_weight = (torch.randn(3 * heads * D, 4, device=DEVICE) * 0.5).to(
        conv_weight_dtype
    )
    accepted = (
        torch.ones(batch, dtype=torch.int32, device=DEVICE)
        if recoverssm_accepted
        else torch.randint(1, window + 1, (batch,), dtype=torch.int32, device=DEVICE)
    )
    state_indices = torch.randperm(slots - 1, device=DEVICE)[:batch].to(torch.int32) + 1
    if varlen and batch > 4:
        # Padded requests: CUDA graph batches pad with the null block.
        state_indices[3:5] = 0

    pages = _pages(slots, heads, window, dim_first)
    ref = _clone_pages(pages)
    initial = _clone_pages(pages)
    _, conv, state, correction, key_gate = pages
    _, ref_conv, ref_state, ref_correction, ref_key_gate = ref

    # Reference: the stock convolution, then the Triton verify.
    ref_out = torch.full(
        (1, tokens, heads, D), 7.0, device=DEVICE, dtype=torch.bfloat16
    )
    conv_out = torch.empty_like(mixed_qkv)
    causal_conv1d_update(
        mixed_qkv,
        ref_conv if dim_first else ref_conv.transpose(-1, -2),
        conv_weight,
        None,
        activation="silu",
        conv_state_indices=state_indices,
        num_accepted_tokens=accepted,
        query_start_loc=query_start,
        max_query_len=window,
        validate_data=False,
        out=conv_out,
    )
    q, k, v = (
        x.unflatten(-1, (heads, D)).unsqueeze(0) for x in conv_out.split(heads * D, -1)
    )
    kda_recoverssm_verify(
        q=q,
        k=k,
        v=v,
        raw_g=raw_gate.view(1, tokens, heads, D),
        raw_beta=raw_beta.unsqueeze(0),
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=LOWER_BOUND,
        checkpoint_state=ref_state,
        correction_cache=ref_correction,
        kg_cache=ref_key_gate,
        query_start_loc=query_start,
        state_indices=state_indices,
        spec_query_len=window,
        out=ref_out,
    )

    verify = CuteRecoverSSMVerify(
        num_heads=heads,
        window=window,
        conv_dim_first=dim_first,
        conv_weight_dtype=conv_weight_dtype,
        device=DEVICE,
    )
    verify.compile()
    out = torch.full((tokens, heads * D), 7.0, device=DEVICE, dtype=torch.bfloat16)
    verify(
        mixed_qkv=mixed_qkv,
        raw_gate=raw_gate,
        raw_beta=raw_beta,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=LOWER_BOUND,
        state=state,
        correction=correction,
        key_gate=key_gate,
        query_start_loc=query_start,
        state_indices=state_indices,
        num_accepted_tokens=accepted,
        conv_state=conv,
        conv_weight=conv_weight,
        out=out,
    )
    torch.accelerator.synchronize()
    assert int(verify.counter) == 0  # left zeroed for the next launch

    live = [b for b in range(batch) if int(state_indices[b]) > 0 and int(lens[b]) > 0]
    bounds = query_start.tolist()
    live_tokens = [t for b in live for t in range(bounds[b], bounds[b + 1])]
    dead_tokens = [
        t
        for b in range(batch)
        if int(state_indices[b]) <= 0
        for t in range(bounds[b], bounds[b + 1])
    ]
    ref_out = ref_out.view(tokens, heads * D)
    torch.testing.assert_close(
        out[live_tokens].float(), ref_out[live_tokens].float(), atol=3e-2, rtol=3e-2
    )
    if dead_tokens:
        assert (out[dead_tokens] == 0).all()
    for b in live:
        slot, n = int(state_indices[b]), int(lens[b])
        torch.testing.assert_close(
            correction[slot, :, :n], ref_correction[slot, :, :n], atol=3e-2, rtol=3e-2
        )
        torch.testing.assert_close(key_gate[slot, :, :n], ref_key_gate[slot, :, :n])
        # The rolled conv state: two history rows, then the raw tokens.
        if dim_first:
            torch.testing.assert_close(
                conv[slot, :, : n + 2], ref_conv[slot, :, : n + 2]
            )
        else:
            torch.testing.assert_close(conv[slot, : n + 2], ref_conv[slot, : n + 2])
    # The checkpoint is never written; other slots' pages are untouched.
    torch.testing.assert_close(state, ref_state, atol=0, rtol=0)
    untouched = sorted(set(range(slots)) - {int(state_indices[b]) for b in live})
    for page, before in zip(pages[1:], initial[1:]):
        assert torch.equal(page[untouched], before[untouched])


def _layer(pages, heads: int) -> SimpleNamespace:
    _, conv, state, correction, key_gate = pages
    return SimpleNamespace(
        pages=pages,
        kv_cache=(conv, state, correction, key_gate),
        recoverssm_records=RecoverSSMRecords(correction, key_gate),
        A_log=torch.randn(heads, device=DEVICE) * 0.5,
        dt_bias=torch.randn(heads * D, device=DEVICE) * 0.5,
        local_num_heads=heads,
        head_dim=D,
        gate_lower_bound=LOWER_BOUND,
    )


@pytest.mark.parametrize("heads", [96, 16])
@pytest.mark.parametrize("batch", [1, 8, 13])
@torch.inference_mode()
def test_cute_state_commit_matches_triton(heads, batch, monkeypatch):
    """The CuTe fold of every layer's accepted tokens equals the Triton fold,
    with varlen windows, rejected-everything (0 accepted) and dead requests."""
    monkeypatch.setenv("VLLM_SSM_CONV_STATE_LAYOUT", "DS")
    torch.manual_seed(heads + batch)
    window, num_layers = 5, 3
    slots = 2 * batch + 1
    layers = []
    for _ in range(num_layers):
        layers.append(_layer(_pages(slots, heads, window, dim_first=True), heads))
    lens = torch.full((batch,), window, dtype=torch.int32)
    if batch > 2:
        lens[0] = 2
    query_start = torch.zeros(batch + 1, dtype=torch.int32)
    query_start[1:] = torch.cumsum(lens, 0)
    query_start = query_start.to(DEVICE)
    accepted = torch.randint(0, window + 1, (batch,), dtype=torch.int32, device=DEVICE)
    state_indices = torch.randperm(slots - 1, device=DEVICE)[:batch].to(torch.int32) + 1
    if batch > 3:
        state_indices[3] = 0

    compile_state_commit(heads, window, DEVICE)
    initial = [_clone_pages(layer.pages) for layer in layers]
    results = {}
    for cute in (False, True):
        for layer, before in zip(layers, initial):
            layer.pages[0].copy_(before[0])
        context = KDARecoverSSMCommitContext.create(
            layers, spec_query_len=window, max_num_reqs=batch, cute_state_commit=cute
        )
        assert (context.state_commit is not None) == cute
        context.commit(accepted, state_indices, query_start)
        torch.accelerator.synchronize()
        results[cute] = [layer.kv_cache[1].clone() for layer in layers]
    states_before = [before[2] for before in initial]

    for triton_state, cute_state, before in zip(
        results[False], results[True], states_before
    ):
        torch.testing.assert_close(
            cute_state.float(), triton_state.float(), atol=2e-2, rtol=2e-2
        )
        # Requests with no accepted tokens keep their state, as do unused slots.
        for b in range(batch):
            slot = int(state_indices[b])
            if slot > 0 and int(accepted[b]) == 0:
                assert torch.equal(cute_state[slot], before[slot])
        unused = sorted(set(range(slots)) - set(state_indices.tolist()))
        assert torch.equal(cute_state[unused], before[unused])
