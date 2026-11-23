# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import math
import os
import sys
from types import ModuleType

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.mamba.gdn.replayssm import (
    load_gdn_replayssm_kernel,
    load_gdn_replayssm_materializer,
    pack_replayssm_rows,
    plan_gdn_replayssm_cursors,
    run_gdn_replayssm,
)

# FlashInfer's HMMA kernel reads Q/K/V rows from block-strided pools only in
# this mode, which it selects when it is first imported.
os.environ.setdefault("SGLANG_GDN_WY_STRIDED_QKV", "1")

_HMMA_MODULE = "flashinfer.gdn_kernels.gdn_decode_bf16_wy_ucache_flush"
_STATE_ENV = {torch.float16: "fp16", torch.bfloat16: "bf16"}


@pytest.mark.parametrize("state_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("loaded_dtype", [torch.float16, torch.bfloat16])
def test_materializer_requires_configured_state_dtype(
    monkeypatch, state_dtype, loaded_dtype
):
    """A FlashInfer imported before vLLM set its state dtype must be rejected."""
    provider = ModuleType("provider")
    materialize = lambda **kwargs: None
    monkeypatch.setattr(provider, "gdn_prefix_materialize", materialize, raising=False)
    monkeypatch.setattr(provider, "state_ty_torch", lambda: loaded_dtype, raising=False)
    monkeypatch.setitem(
        sys.modules, "flashinfer.gdn_kernels.gdn_prefix_materialize", provider
    )
    load_gdn_replayssm_materializer.cache_clear()
    try:
        if loaded_dtype == state_dtype:
            assert load_gdn_replayssm_materializer(state_dtype) is materialize
        else:
            match = f"GDN_UCACHE_STATE_DTYPE={_STATE_ENV[state_dtype]}"
            with pytest.raises(ValueError, match=match):
                load_gdn_replayssm_materializer(state_dtype)
    finally:
        load_gdn_replayssm_materializer.cache_clear()


@pytest.mark.parametrize("state_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("loaded_dtype", [torch.float16, torch.bfloat16])
def test_hmma_kernel_requires_configured_state_dtype(
    monkeypatch, state_dtype, loaded_dtype
):
    """The HMMA kernel runs with its backend pinned and the configured state."""
    calls: list[dict] = []
    provider = ModuleType("provider")
    provider.ST_TORCH = loaded_dtype  # type: ignore[attr-defined]
    provider.gated_delta_rule_mtp_ucache_flush = (  # type: ignore[attr-defined]
        lambda **kwargs: calls.append(kwargs)
    )
    monkeypatch.setitem(sys.modules, _HMMA_MODULE, provider)
    load_gdn_replayssm_kernel.cache_clear()
    try:
        if loaded_dtype != state_dtype:
            match = f"GDN_UCACHE_STATE_DTYPE={_STATE_ENV[state_dtype]}"
            with pytest.raises(ValueError, match=match):
                load_gdn_replayssm_kernel("hmma", state_dtype)
            return
        load_gdn_replayssm_kernel("hmma", state_dtype)(q=None)
        assert calls == [{"q": None, "backend": "hmma"}]
    finally:
        load_gdn_replayssm_kernel.cache_clear()


def test_pack_replayssm_rows_preserves_tokens_and_masks_padding():
    values = torch.tensor([[1.0], [2.0], [3.0]])
    packed = pack_replayssm_rows(
        values, torch.tensor([0, 1, 3]), 4, padding_value=-torch.inf
    )
    torch.testing.assert_close(
        packed.squeeze(-1),
        torch.tensor(
            [
                [1.0, -torch.inf, -torch.inf, -torch.inf],
                [2.0, 3.0, -torch.inf, -torch.inf],
            ]
        ),
    )


def test_uniform_replayssm_packing_preserves_alignment_and_padding():
    width = 4
    values = torch.arange(2 * width * 4, dtype=torch.bfloat16).view(2 * width, 4)
    starts = torch.tensor([0, width, 2 * width])
    packed = pack_replayssm_rows(values, starts, width, uniform=True)
    assert packed.data_ptr() == values.data_ptr()
    torch.testing.assert_close(packed, pack_replayssm_rows(values, starts, width))
    # A gate view can be contiguous but start at an unaligned byte offset.
    unaligned = torch.zeros(values.numel() + 1, dtype=values.dtype)[1:].view_as(values)
    unaligned.copy_(values)
    aligned = pack_replayssm_rows(unaligned, starts, width, uniform=True)
    assert aligned.data_ptr() % 16 == 0
    torch.testing.assert_close(aligned, packed)
    # A column slice of a wider projection is copied to a contiguous buffer.
    projection = torch.zeros(2 * width, 8, dtype=values.dtype)[:, :4]
    projection.copy_(values)
    sliced = pack_replayssm_rows(projection, starts, width, uniform=True)
    assert sliced.is_contiguous() and sliced.data_ptr() % 16 == 0
    torch.testing.assert_close(sliced, packed)
    # Eager staging can include a zero-token graph-padding row.
    padded_starts = torch.tensor([0, width, width])
    padded = pack_replayssm_rows(values[:width], padded_starts, width, uniform=True)
    torch.testing.assert_close(padded[0], values[:width])
    assert padded[1].eq(0).all()


def test_uniform_replayssm_passes_qkv_views_and_aligned_gates():
    width = 4
    tokens, dim = 2 * width, 128
    projection = torch.arange(tokens * 4 * dim).to(torch.bfloat16).view(tokens, -1)
    mixed_qkv = projection[:, : 3 * dim]
    gates = torch.ones(tokens, 2, dtype=torch.bfloat16)
    a, b = gates[:, :1], gates[:, 1:]

    def kernel(**kwargs):
        # UMMA makes its own contiguous Q/K/V copies. Keep the packed view.
        assert kwargs["q"].data_ptr() == mixed_qkv.data_ptr()
        for name, expected in zip(("q", "k", "v"), mixed_qkv.split(dim, -1)):
            torch.testing.assert_close(
                kwargs[name].flatten(0, 2), expected, rtol=0, atol=0
            )
        for name, expected in (("a", a), ("b", b)):
            assert kwargs[name].is_contiguous()
            assert kwargs[name].data_ptr() % 16 == 0
            torch.testing.assert_close(kwargs[name].flatten(0, 1), expected)
        kwargs["output"].copy_(kwargs["q"])

    output = run_gdn_replayssm(
        kernel=kernel,
        executed_width=width,
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=torch.zeros(1),
        dt_bias=torch.zeros(1),
        checkpoint_state=torch.zeros(3, 1, dim, dim, dtype=torch.float16),
        replayssm_cache=(
            torch.zeros(3, 1, 32, dim, dtype=torch.bfloat16),
            torch.zeros(3, 1, 32, dim, dtype=torch.bfloat16),
            torch.zeros(3, 1, 32),
        ),
        cursors=torch.tensor([[1, 2], [0, 0], [0, 0]], dtype=torch.int32),
        query_start_loc=torch.tensor([0, width, tokens], dtype=torch.int32),
        output_indices=torch.arange(tokens),
        num_k_heads=1,
        num_v_heads=1,
        head_k_dim=dim,
        head_v_dim=dim,
        uniform_query_length=width,
    )
    torch.testing.assert_close(output[:, 0], mixed_qkv[:, :dim], rtol=0, atol=0)


def _reference_gdn_step(
    state,
    q,
    k,
    v,
    a,
    b,
    A_log,
    dt_bias,
):
    """FP32 recurrence using FlashInfer's effective BF16 gate constants."""
    state = state.float().clone()
    A_log = A_log.to(torch.bfloat16).float()
    dt_bias = dt_bias.to(torch.bfloat16).float()
    group_size = v.shape[1] // k.shape[1]
    outputs = []
    for token in range(q.shape[0]):
        k_hv = F.normalize(k[token].float(), dim=-1).repeat_interleave(
            group_size, dim=0
        )
        q_hv = F.normalize(q[token].float(), dim=-1).repeat_interleave(
            group_size, dim=0
        )
        q_hv *= 1.0 / math.sqrt(q.shape[-1])
        decay = -torch.exp(A_log) * F.softplus(a[token].float() + dt_bias)
        beta = torch.sigmoid(b[token].float())
        state *= torch.exp(decay)[:, None, None]
        prediction = torch.einsum("hvk,hk->hv", state, k_hv)
        update = (v[token].float() - prediction) * beta[:, None]
        state += update[:, :, None] * k_hv[:, None, :]
        outputs.append(torch.einsum("hvk,hk->hv", state, q_hv))
    return torch.stack(outputs), state


def _reconstruct_replayssm_state(
    checkpoint,
    k_cache,
    u_cache,
    g_cache,
    committed,
    ring_start,
):
    state = checkpoint.float().clone()
    if committed == 0:
        return state
    rows = torch.tensor(
        [(ring_start + offset) % k_cache.size(1) for offset in range(committed)],
        dtype=torch.long,
        device=checkpoint.device,
    )
    k_logical = k_cache.index_select(1, rows).float().repeat_interleave(2, dim=0)
    u_logical = u_cache.index_select(1, rows).float()
    g_logical = g_cache.index_select(1, rows).float()
    final_g = g_logical[:, -1]
    weights = torch.exp(final_g[:, None] - g_logical)
    return torch.exp(final_g)[:, None, None] * state + torch.einsum(
        "hpv,hpk->hvk", weights[:, :, None] * u_logical, k_logical
    )


_UMMA_CASES = [
    (1, 1, 1),
    (2, 2, 1),
    (3, 3, 2),
    (4, 3, 2),
    (5, 4, 3),
    (6, 6, 4),
    (7, 7, 5),
    (8, 6, 4),
    (8, 8, 4),
]
# The HMMA kernel computes 4 or 8 tokens per request.
_HMMA_CASES = [(4, 3, 2), (4, 4, 3), (8, 6, 4), (8, 8, 4)]


def _skip_unless_backend_runs(backend: str, state_dtype: torch.dtype) -> None:
    if not torch.cuda.is_available():
        pytest.skip("FlashInfer GDN ReplaySSM requires a CUDA device")
    major = torch.cuda.get_device_capability()[0]
    if backend == "umma":
        if major != 10:
            pytest.skip("FlashInfer's UMMA kernel requires an SM100-class GPU")
        return
    if major < 9:
        pytest.skip("FlashInfer's HMMA kernel requires an SM90 or newer GPU")
    # FlashInfer fixes the HMMA state dtype and Q/K/V mode when first imported.
    provider = importlib.import_module(_HMMA_MODULE)
    if state_dtype != provider.ST_TORCH or not getattr(provider, "_STRIDED_QKV", True):
        pytest.skip(
            "FlashInfer was imported before this module set its GDN settings; set "
            f"GDN_UCACHE_STATE_DTYPE={_STATE_ENV[state_dtype]} and "
            "SGLANG_GDN_WY_STRIDED_QKV=1 to run this case"
        )


@pytest.mark.parametrize(
    "backend,state_dtype,width,query_len,accepted",
    [("umma", torch.float16, *case) for case in _UMMA_CASES]
    + [
        ("hmma", state_dtype, *case)
        for state_dtype in (torch.bfloat16, torch.float16)
        for case in _HMMA_CASES
    ],
)
@torch.inference_mode()
def test_gdn_replayssm_output_and_accepted_state(
    backend, state_dtype, width, query_len, accepted
):
    """Check nonzero outputs and accepted state through replay, wrap, and flush."""
    _skip_unless_backend_runs(backend, state_dtype)
    device, dtype = torch.device("cuda"), torch.bfloat16
    generator = torch.Generator(device=device).manual_seed(20260909 + width)
    ring_slots = 32

    def random_tensor(*shape):
        return torch.randn(*shape, generator=generator, device=device).to(dtype)

    # Use one page stride for every state, as MambaBase.bind_kv_cache does.
    shapes = [
        (2, 128, 128),
        (2, ring_slots, 128),
        (1, ring_slots, 128),
        (2, ring_slots),
    ]
    dtypes = [state_dtype, dtype, dtype, torch.float32]
    sizes = [math.prod(shape) * dt.itemsize for shape, dt in zip(shapes, dtypes)]
    pages = torch.zeros(2, sum(sizes) + 16, dtype=torch.uint8, device=device)
    states = []
    offset = 0
    for shape, dt, size in zip(shapes, dtypes, sizes):
        states.append(pages[:, offset : offset + size].view(dt).view(2, *shape))
        offset += size
    checkpoint, u_cache, k_cache, g_cache = states
    pages[0].fill_(17)  # Nonzero canaries catch writes to the padding page.
    checkpoint.copy_(random_tensor(2, 2, 128, 128) * 0.03)
    ring_start = torch.zeros(2, dtype=torch.int32, device=device)
    if width == 1:
        ring_start[1] = 24  # Cross the ring boundary before the first flush.
    committed = torch.zeros_like(ring_start)
    A_log = torch.nn.Parameter(torch.full((2,), -3.0, device=device))
    dt_bias = torch.nn.Parameter(random_tensor(2) * 0.2)
    accepted_state = checkpoint[1].float().clone()
    padding_page = pages[0].clone()
    capture = width in (1, 4)
    graph = None
    state_indices = torch.tensor(
        [1, 0] if capture else [1], dtype=torch.int32, device=device
    )
    # As in the metadata builder: rows padded to 16 bytes, sliced to the batch.
    num_rows = state_indices.numel()
    cursors = torch.empty(3, 4 * -(-num_rows // 4), dtype=torch.int32, device=device)
    cursors = cursors[:, :num_rows]

    def replay_step(**inputs):
        # The model plans once per cache group, inside the captured graph.
        plan_gdn_replayssm_cursors(state_indices, ring_start, committed, cursors)
        return run_gdn_replayssm(**inputs)

    # Eighteen steps cross a flush and the ring boundary.
    for step in range(18):
        if capture and step == 9:
            # Captured execution must observe an in-place weight reload.
            A_log.add_(2.0)
            dt_bias.add_(1.0)
        q, k = random_tensor(query_len, 1, 128), random_tensor(query_len, 1, 128)
        v = random_tensor(query_len, 2, 128) * 0.1
        a, b = random_tensor(query_len, 2) * 0.2, random_tensor(query_len, 2)
        mixed_qkv = torch.cat((q.flatten(1), k.flatten(1), v.flatten(1)), dim=-1)
        if width == query_len > 1:
            projection = mixed_qkv.new_zeros(query_len, mixed_qkv.shape[1] + 128)
            projection[:, : mixed_qkv.shape[1]].copy_(mixed_qkv)
            mixed_qkv = projection[:, : mixed_qkv.shape[1]]
            gates = torch.stack((a, b), dim=1)
            a, b = gates[:, 0], gates[:, 1]
        if capture:
            # Captured padding has input storage but no token or live state.
            mixed_qkv, a, b = (
                torch.cat((values, torch.ones_like(values[:1]).expand(width, -1)))
                for values in (mixed_qkv, a, b)
            )
        inputs = dict(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=A_log,
            dt_bias=dt_bias,
            checkpoint_state=checkpoint,
            replayssm_cache=(u_cache, k_cache, g_cache),
            cursors=cursors,
            query_start_loc=torch.tensor(
                [0, query_len, query_len] if capture else [0, query_len],
                dtype=torch.int32,
                device=device,
            ),
            output_indices=torch.tensor(
                list(range(query_len)) + ([width] if capture else []),
                dtype=torch.int32,
                device=device,
            ),
            kernel=load_gdn_replayssm_kernel(backend, state_dtype),
            executed_width=width,
            uniform_query_length=query_len,
            num_k_heads=1,
            num_v_heads=2,
            head_k_dim=128,
            head_v_dim=128,
        )
        if capture:
            if graph is None:
                graph_inputs = inputs
                saved_pages = pages.clone()
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    replay_step(**graph_inputs)
                torch.cuda.current_stream().wait_stream(stream)
                pages.copy_(saved_pages)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    graph_output = replay_step(**graph_inputs)
                pages.copy_(saved_pages)
            else:
                for name in ("mixed_qkv", "a", "b"):
                    graph_inputs[name].copy_(inputs[name])
            graph.replay()
            output = graph_output[:query_len]
            assert graph_output[query_len:].eq(0).all()
            torch.testing.assert_close(pages[0], padding_page, rtol=0, atol=0)
            a, b = a[:query_len], b[:query_len]
        else:
            output = replay_step(**inputs)
        expected_output, _ = _reference_gdn_step(
            accepted_state, q, k, v, a, b, A_log, dt_bias
        )
        torch.testing.assert_close(output.float(), expected_output, atol=8e-3, rtol=0)
        _, accepted_state = _reference_gdn_step(
            accepted_state,
            q[:accepted],
            k[:accepted],
            v[:accepted],
            a[:accepted],
            b[:accepted],
            A_log,
            dt_bias,
        )
        old_committed = int(committed[1].item())
        if old_committed + width > 16:
            ring_start[1] = (ring_start[1] + old_committed) % ring_slots
            committed[1] = accepted
        else:
            committed[1] = old_committed + accepted
        reconstructed = _reconstruct_replayssm_state(
            checkpoint[1],
            k_cache[1],
            u_cache[1],
            g_cache[1],
            int(committed[1].item()),
            int(ring_start[1].item()),
        )
        torch.testing.assert_close(reconstructed, accepted_state, atol=2e-2, rtol=0)
