# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import builtins
import math
from types import ModuleType

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.mamba.gdn.replayssm import (
    check_gdn_replayssm_dependencies,
    pack_replayssm_rows,
    run_gdn_replayssm,
)


@pytest.mark.parametrize("width,mode", [(1, "stp"), (4, "mtp"), (8, "mtp")])
def test_dependencies_import_only_selected_decode_api(monkeypatch, width, mode):
    """An MTP-only FlashInfer installation must not need the STP module."""
    module = ModuleType("provider")
    kernel = lambda **kwargs: None
    setattr(module, f"gated_delta_rule_{mode}_ucache_flush", kernel)
    suffix = "stp" if mode == "stp" else "flush"
    expected = f"flashinfer.gdn_kernels.gdn_decode_bf16_wy_ucache_{suffix}"
    original_import = builtins.__import__

    def import_provider(name, *args, **kwargs):
        if name.startswith("flashinfer"):
            assert name == expected, f"Unselected dependency: {name}"
            return module
        return original_import(name, *args, **kwargs)

    monkeypatch.setenv("SGLANG_GDN_WY_STRIDED_QKV", "1")
    monkeypatch.setattr(builtins, "__import__", import_provider)
    check_gdn_replayssm_dependencies.cache_clear()
    try:
        assert check_gdn_replayssm_dependencies(width, False) is kernel
    finally:
        check_gdn_replayssm_dependencies.cache_clear()


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


@pytest.mark.parametrize("width", [4, 8])
def test_uniform_replayssm_packing_preserves_alignment_and_padding(width):
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
    strided = torch.zeros(2 * width, 8, dtype=values.dtype)[:, ::2]
    strided.copy_(values)
    repacked = pack_replayssm_rows(
        strided, starts, width, uniform=True, allow_strided_rows=True
    )
    assert repacked.is_contiguous()
    torch.testing.assert_close(repacked, packed)
    projection = torch.zeros(2 * width, 8, dtype=values.dtype)[:, :4]
    projection.copy_(values)
    viewed = pack_replayssm_rows(
        projection, starts, width, uniform=True, allow_strided_rows=True
    )
    assert viewed.data_ptr() == projection.data_ptr()
    assert viewed.stride() == (width * 8, 8, 1)
    torch.testing.assert_close(viewed, packed)
    gate = pack_replayssm_rows(projection, starts, width, uniform=True)
    assert gate.is_contiguous() and gate.data_ptr() % 16 == 0
    torch.testing.assert_close(gate, packed)
    # Eager staging can include a zero-token graph-padding row.
    padded_starts = torch.tensor([0, width, width])
    padded = pack_replayssm_rows(values[:width], padded_starts, width, uniform=True)
    torch.testing.assert_close(padded[0], values[:width])
    assert padded[1].eq(0).all()


@pytest.mark.parametrize("query_lens", [[1, 1], [1, 0]])
def test_single_token_packing_matches_ragged_path(query_lens):
    """STP copies strided inputs and preserves zero-token padding rows."""
    num_tokens = sum(query_lens)
    values = torch.arange(num_tokens * 8, dtype=torch.bfloat16).view(num_tokens, 8)
    values = values[:, ::2]
    starts = torch.tensor([0, *torch.tensor(query_lens).cumsum(0).tolist()])
    packed = pack_replayssm_rows(values, starts, 4, single_token=True)
    expected = pack_replayssm_rows(values, starts, 4)
    torch.testing.assert_close(packed, expected, rtol=0, atol=0)
    assert packed.data_ptr() % 16 == 0
    assert packed[:, 1:].eq(0).all()


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


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0),
    reason="FlashInfer GDN ReplaySSM requires an SM90+ CUDA device",
)
@pytest.mark.parametrize(
    "width,query_len,accepted",
    [(1, 1, 1), (4, 3, 2), (4, 4, 2), (8, 6, 4), (8, 8, 4)],
)
@torch.inference_mode()
def test_gdn_replayssm_output_and_accepted_state(
    monkeypatch, width, query_len, accepted
):
    """Check nonzero outputs and accepted state through replay, wrap, and flush."""
    monkeypatch.setenv("SGLANG_GDN_WY_STRIDED_QKV", "1")
    module = (
        "gdn_decode_bf16_wy_ucache_stp"
        if width == 1
        else "gdn_decode_bf16_wy_ucache_flush"
    )
    pytest.importorskip(f"flashinfer.gdn_kernels.{module}")
    device, dtype = torch.device("cuda"), torch.bfloat16
    generator = torch.Generator(device=device).manual_seed(20260909 + width)
    ring_slots = 16 if width == 1 else 32

    def random_tensor(*shape):
        return torch.randn(*shape, generator=generator, device=device).to(dtype)

    # Use one page stride for every state, as MambaBase.bind_kv_cache does.
    shapes = [
        (2, 128, 128),
        (2, ring_slots, 128),
        (1, ring_slots, 128),
        (2, ring_slots),
    ]
    dtypes = [dtype, dtype, dtype, torch.float32]
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
    committed = torch.zeros_like(ring_start)
    A_log = torch.nn.Parameter(torch.full((2,), -3.0, device=device))
    dt_bias = torch.nn.Parameter(random_tensor(2) * 0.2)
    accepted_state = checkpoint[1].float().clone()
    padding_page = pages[0].clone()
    graph = None

    # Eighteen steps cross the native STP flush and the MTP ring boundary.
    for step in range(18):
        if width == 1 and step == 9:
            # Captured execution must observe an in-place weight reload.
            A_log.add_(2.0)
        q, k = random_tensor(query_len, 1, 128), random_tensor(query_len, 1, 128)
        v = random_tensor(query_len, 2, 128) * 0.1
        a, b = random_tensor(query_len, 2) * 0.2, random_tensor(query_len, 2)
        mixed_qkv = torch.cat((q.flatten(1), k.flatten(1), v.flatten(1)), dim=-1)
        if width == query_len and width in (4, 8):
            projection = mixed_qkv.new_zeros(query_len, mixed_qkv.shape[1] + 128)
            projection[:, : mixed_qkv.shape[1]].copy_(mixed_qkv)
            mixed_qkv = projection[:, : mixed_qkv.shape[1]]
            gates = torch.stack((a, b), dim=1)
            a, b = gates[:, 0], gates[:, 1]
        if width == 1:
            # Captured padding has input storage but no token or live state.
            mixed_qkv = torch.cat((mixed_qkv, torch.ones_like(mixed_qkv)))
            a, b = (
                torch.cat((a, torch.ones_like(a))),
                torch.cat((b, torch.ones_like(b))),
            )
        inputs = dict(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=A_log,
            dt_bias=dt_bias,
            checkpoint_state=checkpoint,
            replayssm_cache=(u_cache, k_cache, g_cache),
            ring_start=ring_start,
            num_committed=committed,
            query_start_loc=torch.tensor(
                [0, 1, 1] if width == 1 else [0, query_len],
                dtype=torch.int32,
                device=device,
            ),
            state_indices=torch.tensor(
                [1, 0] if width == 1 else [1], dtype=torch.int32, device=device
            ),
            output_indices=torch.arange(
                2 if width == 1 else query_len, dtype=torch.int32, device=device
            ),
            executed_query_width=width,
            uniform_query_length=query_len,
            num_k_heads=1,
            num_v_heads=2,
            head_k_dim=128,
            head_v_dim=128,
        )
        if width == 1:
            if graph is None:
                graph_inputs = inputs
                saved_pages = pages.clone()
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    run_gdn_replayssm(**graph_inputs)
                torch.cuda.current_stream().wait_stream(stream)
                pages.copy_(saved_pages)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    graph_output = run_gdn_replayssm(**graph_inputs)
                pages.copy_(saved_pages)
            else:
                for name in ("mixed_qkv", "a", "b"):
                    graph_inputs[name].copy_(inputs[name])
            graph.replay()
            output = graph_output[:1]
            assert graph_output[1].eq(0).all()
            torch.testing.assert_close(pages[0], padding_page, rtol=0, atol=0)
            a, b = a[:1], b[:1]
        else:
            output = run_gdn_replayssm(**inputs)
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
        if width == 1:
            committed[1] = 0 if old_committed >= 15 else old_committed + 1
        elif old_committed + width > 16:
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
