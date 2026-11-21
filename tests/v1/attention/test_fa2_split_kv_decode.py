# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FA2 split-KV decode must match the paged varlen call it replaces."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.fa_utils import is_fa_version_supported

if not current_platform.is_cuda() or not is_fa_version_supported(2):
    pytest.skip("FA2 on CUDA only", allow_module_level=True)

from vllm.v1.attention.backends import flash_attn
from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionImpl,
    FlashAttentionMetadataBuilder,
    _fa2_split_kv_decode,
)

BLOCK_SIZE = 16
HEAD_SIZE = 128
NUM_SMS = current_platform.num_compute_units(torch.accelerator.current_device_index())


def _make_builder(num_q_heads, windows, query_len, spec_config=None, **attrs):
    """A builder stub set up by the real `_init_fa2_split_kv`."""
    defaults = dict(
        use_full_cuda_graph=True,
        dcp_world_size=1,
        num_heads_q=num_q_heads,
        block_size=BLOCK_SIZE,
        device=torch.device("cuda", torch.accelerator.current_device_index()),
    )
    builder = SimpleNamespace(**(defaults | attrs))
    layers = {
        f"layer{i}": SimpleNamespace(impl=SimpleNamespace(sliding_window=window))
        for i, window in enumerate(windows)
    }
    vllm_config = SimpleNamespace(
        speculative_config=spec_config, num_speculative_tokens=query_len - 1
    )
    with mock.patch.object(
        flash_attn, "get_layers_from_vllm_config", return_value=layers
    ):
        FlashAttentionMetadataBuilder._init_fa2_split_kv(
            builder, 2, vllm_config, list(layers)
        )
    return builder


def _window_inputs(builder, query_start_loc, seq_lens, block_table, query_len):
    metadata = SimpleNamespace(
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        block_table=block_table,
        fa2_split_kv_query_len=query_len,
    )
    FlashAttentionMetadataBuilder._fill_fa2_window(builder, metadata)
    return metadata


def _check_matches_varlen(
    num_heads: tuple[int, int],
    seq_lens: list[int],
    query_len: int,
    window: int | None,
    softcap: float = 0.0,
    use_alibi: bool = False,
) -> None:
    torch.manual_seed(0)
    num_q_heads, num_kv_heads = num_heads
    num_reqs = len(seq_lens)
    max_blocks = cdiv(max(seq_lens), BLOCK_SIZE) + 2
    num_blocks = num_reqs * max_blocks
    cache_shape = (num_blocks, BLOCK_SIZE, num_kv_heads, HEAD_SIZE)
    key_cache = torch.randn(cache_shape, dtype=torch.bfloat16, device="cuda")
    value_cache = torch.randn(cache_shape, dtype=torch.bfloat16, device="cuda")
    block_table = (
        torch.randperm(num_blocks, device="cuda")
        .view(num_reqs, max_blocks)
        .to(torch.int32)
    )
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
    query = torch.randn(
        num_reqs * query_len,
        num_q_heads,
        HEAD_SIZE,
        dtype=torch.bfloat16,
        device="cuda",
    )
    query_start_loc = torch.arange(
        0, num_reqs * query_len + 1, query_len, dtype=torch.int32, device="cuda"
    )
    window_left = -1 if window is None else window - 1
    scale = HEAD_SIZE**-0.5
    alibi_slopes = (
        torch.rand(num_q_heads, dtype=torch.float32, device="cuda")
        if use_alibi
        else None
    )

    expected = flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        max_seqlen_q=query_len,
        cu_seqlens_q=query_start_loc,
        max_seqlen_k=max(seq_lens),
        seqused_k=seq_lens_t,
        softmax_scale=scale,
        causal=True,
        window_size=None if window is None else [window_left, 0],
        block_table=block_table,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        fa_version=2,
    )

    split_inputs = (seq_lens_t, block_table)
    if window is not None:
        builder = _make_builder(num_q_heads, [(window_left, 0)], query_len)
        split_inputs = _window_inputs(
            builder, query_start_loc, seq_lens_t, block_table, query_len
        ).fa2_window_inputs

    output = torch.empty_like(query)
    _fa2_split_kv_decode(
        query,
        key_cache,
        value_cache,
        output,
        *split_inputs,
        query_len=query_len,
        softmax_scale=scale,
        window_left=window_left,
        alibi_slopes=alibi_slopes,
        softcap=softcap,
    )
    torch.testing.assert_close(output, expected, atol=2e-3, rtol=1e-2)


QUERY_LEN_WINDOW = [(4, None), (1, 256), (4, 256)]


@pytest.mark.parametrize("num_heads", [(8, 2), (10, 2), (20, 4), (8, 8)])
@pytest.mark.parametrize("seq_lens", [[37], [5, 300, 1029], [2000, 64, 4100]])
@pytest.mark.parametrize("query_len,window", QUERY_LEN_WINDOW)
def test_split_kv_decode_matches_varlen(num_heads, seq_lens, query_len, window):
    _check_matches_varlen(num_heads, seq_lens, query_len, window)


@pytest.mark.parametrize("query_len,window", QUERY_LEN_WINDOW)
@pytest.mark.parametrize("softcap,use_alibi", [(30.0, False), (0.0, True)])
def test_split_kv_decode_softcap_alibi(query_len, window, softcap, use_alibi):
    _check_matches_varlen(
        (10, 2), [5, 300, 1029], query_len, window, softcap, use_alibi
    )


@pytest.mark.parametrize("num_heads,use_alibi", [((8, 8), False), ((10, 2), True)])
def test_split_kv_decode_single_query_global(num_heads, use_alibi):
    """q_len 1 on a global layer, where varlen cannot swap a GQA group in."""
    _check_matches_varlen(num_heads, [5, 300, 1029], 1, None, use_alibi=use_alibi)


@pytest.mark.parametrize(
    "windows,window_left,width",
    [
        ([(255, 0)], 255, 18),  # cdiv(255 + 4 + 15, 16)
        ([(4095, 0), (4095, 0)], 4095, 258),
        ([(255, 0), (-1, -1)], -1, None),  # mixed windows: no window table
        ([(-1, -1)], -1, None),  # global group
        ([(255, 255)], -1, None),  # bidirectional window
    ],
)
def test_init_window_table(windows, window_left, width):
    builder = _make_builder(10, windows, 4)
    assert builder.fa2_split_kv_max_reqs == (NUM_SMS - 1) // 10
    assert builder.fa2_decode_query_len == 4
    assert builder.fa2_window_left == window_left
    if width is not None:
        shape = (builder.fa2_split_kv_max_reqs, width)
        assert builder._fa2_window_block_table.shape == shape


@pytest.mark.parametrize(
    "case", ["batch_invariant", "adaptive", "piecewise", "dcp", "too_many_heads"]
)
def test_init_disabled(monkeypatch, case):
    spec_config, attrs, num_q_heads = None, {}, 10
    if case == "batch_invariant":
        monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    elif case == "adaptive":
        spec_config = SimpleNamespace(enable_adaptive_verification=True)
    elif case == "piecewise":
        attrs["use_full_cuda_graph"] = False
    elif case == "dcp":
        attrs["dcp_world_size"] = 2
    else:
        num_q_heads = NUM_SMS
    builder = _make_builder(num_q_heads, [(255, 0)], 4, spec_config, **attrs)
    assert builder.fa2_split_kv_max_reqs == 0
    assert builder.fa2_window_left == -1


@pytest.mark.parametrize(
    "num_reqs,num_tokens,max_query_len,causal,expected",
    [
        (1, 4, 4, True, 4),  # spec-decode verify
        (1, 1, 1, True, 1),  # decode; the impl skips GQA global layers
        (8, 32, 4, True, 4),  # at the request limit
        (9, 36, 4, True, 0),  # too many requests to leave SMs idle
        (2, 5, 4, True, 0),  # mixed query lengths
        (1, 8, 8, True, 0),  # longer than a decode / verify step
        (0, 0, 0, True, 0),  # empty batch
        (1, 4, 4, False, 0),  # non-causal
        (1, 4, 4, "tensor", 0),  # per-request causal
    ],
)
def test_split_kv_query_len(num_reqs, num_tokens, max_query_len, causal, expected):
    if causal == "tensor":
        causal = torch.ones(num_reqs, dtype=torch.int32)
    builder = SimpleNamespace(fa2_split_kv_max_reqs=8, fa2_decode_query_len=4)
    query_len = FlashAttentionMetadataBuilder._fa2_split_kv_query_len(
        builder, num_reqs, num_tokens, max_query_len, causal
    )
    assert query_len == expected


@pytest.mark.parametrize(
    "window,fill_in_graph", [(None, False), (256, False), (256, True)]
)
def test_split_kv_decode_cuda_graph_replay(window, fill_in_graph):
    """Replay one graph with new lengths, seq_len-0 padding and window refills."""
    # fill_in_graph refills inside the graph, like update_draft_decode_metadata().
    torch.manual_seed(0)
    num_q_heads, num_kv_heads, query_len, max_reqs = 10, 2, 4, 4
    max_blocks = 128
    num_blocks = max_reqs * max_blocks
    cache_shape = (num_blocks, BLOCK_SIZE, num_kv_heads, HEAD_SIZE)
    key_cache = torch.randn(cache_shape, dtype=torch.bfloat16, device="cuda")
    value_cache = torch.randn(cache_shape, dtype=torch.bfloat16, device="cuda")
    block_table = torch.zeros((max_reqs, max_blocks), dtype=torch.int32, device="cuda")
    seq_lens = torch.zeros(max_reqs, dtype=torch.int32, device="cuda")
    query = torch.empty(
        max_reqs * query_len,
        num_q_heads,
        HEAD_SIZE,
        dtype=torch.bfloat16,
        device="cuda",
    )
    output = torch.empty_like(query)
    query_start_loc = torch.arange(
        0, max_reqs * query_len + 1, query_len, dtype=torch.int32, device="cuda"
    )
    window_left = -1 if window is None else window - 1
    scale = HEAD_SIZE**-0.5
    split_inputs = (seq_lens, block_table)
    if window is not None:
        builder = _make_builder(num_q_heads, [(window_left, 0)], query_len)
        metadata = _window_inputs(
            builder, query_start_loc, seq_lens, block_table, query_len
        )
        split_inputs = metadata.fa2_window_inputs

    def run():
        if fill_in_graph:
            FlashAttentionMetadataBuilder._fill_fa2_window(builder, metadata)
        _fa2_split_kv_decode(
            query,
            key_cache,
            value_cache,
            output,
            *split_inputs,
            query_len=query_len,
            softmax_scale=scale,
            window_left=window_left,
            alibi_slopes=None,
            softcap=0.0,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()

    for real_seq_lens in ([2000], [5, 1500], [37, 2048, 700], [2040, 4, 300, 1100]):
        num_reqs = len(real_seq_lens)
        seq_lens.zero_()
        seq_lens[:num_reqs] = torch.tensor(real_seq_lens, dtype=torch.int32)
        block_table.copy_(
            torch.randperm(num_blocks, device="cuda").view(max_reqs, max_blocks)
        )
        query.normal_()
        if window is not None and not fill_in_graph:
            FlashAttentionMetadataBuilder._fill_fa2_window(builder, metadata)
        graph.replay()

        num_tokens = num_reqs * query_len
        expected = flash_attn_varlen_func(
            q=query[:num_tokens],
            k=key_cache,
            v=value_cache,
            max_seqlen_q=query_len,
            cu_seqlens_q=query_start_loc[: num_reqs + 1],
            max_seqlen_k=max(real_seq_lens),
            seqused_k=seq_lens[:num_reqs],
            softmax_scale=scale,
            causal=True,
            window_size=None if window is None else [window_left, 0],
            block_table=block_table[:num_reqs],
            fa_version=2,
        )
        torch.testing.assert_close(output[:num_tokens], expected, atol=2e-3, rtol=1e-2)
        assert torch.isfinite(output).all()


@pytest.mark.parametrize(
    "query_len,window,num_heads,alibi,version,has_table,capturing,expected",
    [
        (4, -1, (10, 2), False, 2, False, True, "full"),  # verify, global layer
        (1, -1, (10, 2), False, 2, False, True, None),  # varlen swaps GQA group
        (1, -1, (8, 8), False, 2, False, True, "full"),  # MHA cannot swap
        (1, -1, (10, 2), True, 2, False, True, "full"),  # ALiBi cannot swap
        (1, 255, (10, 2), False, 2, True, True, "window"),  # sliding window
        (1, 255, (10, 2), False, 2, False, True, None),  # group, mixed windows
        (4, -1, (10, 2), False, 3, False, True, None),  # not FA2
        (4, -1, (10, 2), False, 2, False, False, None),  # eager or piecewise
        (0, -1, (10, 2), False, 2, False, True, None),  # batch not eligible
    ],
)
def test_split_kv_inputs_gate(
    monkeypatch,
    query_len,
    window,
    num_heads,
    alibi,
    version,
    has_table,
    capturing,
    expected,
):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    seq_lens, block_table = torch.ones(3), torch.ones(3, 2)
    window_inputs = (torch.ones(2), torch.ones(2, 2))
    impl = SimpleNamespace(
        vllm_flash_attn_version=version,
        sliding_window=(window, 0 if window >= 0 else -1),
        num_heads=num_heads[0],
        num_kv_heads=num_heads[1],
        alibi_slopes=torch.ones(num_heads[0]) if alibi else None,
    )
    metadata = SimpleNamespace(
        fa2_split_kv_query_len=query_len,
        fa2_window_inputs=window_inputs if has_table else None,
        query_start_loc=torch.arange(3),
        seq_lens=seq_lens,
        block_table=block_table,
    )
    inputs = FlashAttentionImpl._fa2_split_kv_inputs(impl, metadata)
    if expected is None:
        assert inputs is None
    elif expected == "window":
        assert inputs is window_inputs
    else:
        assert inputs is not None
        assert inputs[0]._base is seq_lens and inputs[1]._base is block_table
        assert inputs[0].shape == (2,) and inputs[1].shape == (2, 2)
