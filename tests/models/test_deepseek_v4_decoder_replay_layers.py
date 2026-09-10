# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decoder-side SWA bounded replay: the replay layers see exactly each request's
replay window, with metadata that matches what the full batch would have given
those rows, except that a trimmed request's window stops at the replay window's
start."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
from vllm.forward_context import (
    ForwardContext,
    get_forward_context,
    override_forward_context,
)
from vllm.models.deepseek_v41.decoder_replay_layers import (
    DecoderReplayLayers,
    ReplayBatchBuilder,
    ReplayMetadataBuilder,
)
from vllm.models.deepseek_v41.sparse_mla import DeepseekV4SparseMLAMetadataBuilder
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadataBuilder
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadataBuilder
from vllm.v1.kv_cache_interface import MLAAttentionSpec, SlidingWindowMLASpec

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="metadata builders need CUDA"
)

WINDOW = 128
DEVICE = torch.device("cuda")


def _vllm_config():
    cfg = MagicMock()
    cfg.model_config.max_model_len = 4096
    cfg.model_config.hf_config = SimpleNamespace(sliding_window=WINDOW, index_topk=512)
    cfg.scheduler_config.max_num_batched_tokens = 2048
    cfg.scheduler_config.max_num_seqs = 16
    cfg.speculative_config = None
    cfg.parallel_config.decode_context_parallel_size = 1
    cfg.parallel_config.prefill_context_parallel_size = 1
    cfg.parallel_config.cp_kv_cache_interleave_size = 1
    cfg.attention_config.resolve_indexer_kv_dtype.return_value = "fp8"
    return cfg


def _common(
    query_lens: list[int],
    seq_lens: list[int],
    block_size: int,
    device_query_lens: list[int] | None = None,
):
    num_reqs = len(query_lens)
    qsl_cpu = torch.tensor([0, *torch.tensor(query_lens).cumsum(0).tolist()]).int()
    qsl = torch.tensor(
        [0, *torch.tensor(device_query_lens or query_lens).cumsum(0).tolist()]
    ).int()
    num_tokens = int(qsl_cpu[-1])
    positions = torch.cat(
        [torch.arange(s - q, s) for q, s in zip(query_lens, seq_lens)]
    ).to(DEVICE)
    block_table = (
        torch.arange(num_reqs, device=DEVICE)[:, None] * 64
        + torch.arange(64, device=DEVICE)[None, :]
    ).int()
    req = torch.repeat_interleave(
        torch.arange(num_reqs, device=DEVICE), torch.tensor(query_lens, device=DEVICE)
    )
    slot_mapping = (
        block_table[req, positions // block_size].long() * block_size
        + positions % block_size
    )
    return CommonAttentionMetadata(
        query_start_loc=qsl.to(DEVICE),
        query_start_loc_cpu=qsl_cpu,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=DEVICE),
        seq_lens_cpu_upper_bound=torch.tensor(seq_lens, dtype=torch.int32),
        num_reqs=num_reqs,
        num_actual_tokens=num_tokens,
        max_query_len=max(query_lens),
        max_seq_len=max(seq_lens),
        block_table_tensor=block_table,
        slot_mapping=slot_mapping,
        positions=positions,
    )


def _builders(cfg):
    swa_spec = SlidingWindowMLASpec(
        block_size=32,
        num_kv_heads=1,
        head_size=584,
        dtype=torch.uint8,
        sliding_window=WINDOW,
        alignment=576,
        cache_dtype_str="fp8_ds_mla",
        model_version="deepseek_v4",
    )
    mla_spec = MLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=584,
        dtype=torch.uint8,
        tokens_per_state=1,
        alignment=576,
    )
    idx_spec = MLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=132,
        dtype=torch.uint8,
        tokens_per_state=1,
        alignment=128,
    )
    return (
        DeepseekSparseSWAMetadataBuilder(swa_spec, ["swa"], cfg, DEVICE),
        DeepseekV4SparseMLAMetadataBuilder(mla_spec, ["mla"], cfg, DEVICE),
        DeepseekV32IndexerMetadataBuilder(
            idx_spec, ["idx"], cfg, DEVICE, block_table_width=64
        ),
    )


def _attn(topk_buffer=None, candidate_buffer=None):
    source_attn = SimpleNamespace(
        prefix="mla",
        indexer=SimpleNamespace(k_cache=SimpleNamespace(prefix="idx")),
        topk_indices_buffer=topk_buffer,
        candidate_block_buffer=candidate_buffer,
    )
    replay_attn = [SimpleNamespace(swa_cache_layer=SimpleNamespace(prefix="swa"))]
    return source_attn, replay_attn


def _make_batch(common, num_batch_tokens):
    builder = ReplayBatchBuilder(WINDOW, max_num_tokens=2048, max_num_reqs=16)
    return builder.build(common, num_batch_tokens)


def _replay_layers(run_layers, topk_buffer, candidate_buffer, graph_sizes=()):
    source_attn, replay_attn = _attn(topk_buffer, candidate_buffer)
    return DecoderReplayLayers(
        _vllm_config(),
        WINDOW,
        source_attn,
        replay_attn,
        run_layers,
        list(graph_sizes),
    )


# decode (1 token), trimmed prefill (300 of 300), untrimmed prefill (100 of 300)
QUERY_LENS = [1, 300, 100]
SEQ_LENS = [500, 300, 300]
REPLAY_ROWS = [0, *range(301 - WINDOW, 301), *range(301, 401)]


def _full_metadata(cfg):
    swa_b, mla_b, idx_b = _builders(cfg)
    common_swa = _common(QUERY_LENS, SEQ_LENS, block_size=32)
    common_src = _common(QUERY_LENS, SEQ_LENS, block_size=128)
    return {
        "swa": swa_b.build(0, common_swa),
        "mla": mla_b.build(0, common_src),
        "idx": idx_b.build(0, common_src),
    }


def test_batch_keeps_each_request_replay_window():
    batch = _make_batch(_common(QUERY_LENS, SEQ_LENS, block_size=32), 401)
    assert batch.trims
    assert batch.rows.tolist() == REPLAY_ROWS
    assert batch.query_start_loc_cpu.tolist() == [0, 1, 129, 229]
    assert batch.query_start_loc.tolist() == [0, 1, 129, 229]
    assert batch.num_tokens == 229
    assert batch.max_query_len == WINDOW
    # Only the trimmed request's window is floored, at its replay window start.
    assert batch.replay_start.tolist() == [0, 300 - WINDOW, 0]

    short = _make_batch(_common([1, 100], [500, 300], block_size=32), 101)
    assert not short.trims
    assert short.rows.tolist() == list(range(101))

    # Adaptive verification splits the leading verification requests on the GPU
    # (CPU [2, 2] vs device [1, 3]): never trimmed, their rows stay put, and the
    # device boundaries only move by the rows trimmed off the prefill.
    adaptive = _make_batch(
        _common([2, 2, 300], [500, 500, 300], 32, device_query_lens=[1, 3, 300]), 304
    )
    assert adaptive.rows.tolist() == [*range(4), *range(304 - WINDOW, 304)]
    assert adaptive.query_start_loc.tolist() == [0, 1, 4, 4 + WINDOW]
    assert adaptive.query_start_loc_cpu.tolist() == [0, 2, 4, 4 + WINDOW]


def test_replay_metadata_matches_full_batch_rows():
    cfg = _vllm_config()
    full = _full_metadata(cfg)
    batch = _make_batch(full["swa"].common, 401)
    replay = ReplayMetadataBuilder(*_attn(), max_num_tokens=2048).build(
        full, batch, batch.num_tokens
    )
    rows = batch.rows

    swa, swa_full = replay["swa"], full["swa"]
    assert swa.num_decode_tokens == 1 and swa.num_prefill_tokens == 228
    assert torch.equal(swa.slot_mapping, swa_full.slot_mapping[rows])
    assert torch.equal(swa.token_to_req_indices, swa_full.token_to_req_indices[rows])
    assert torch.equal(swa.decode_swa_indices, swa_full.decode_swa_indices)
    assert torch.equal(swa.decode_swa_lens, swa_full.decode_swa_lens)
    # Prefill rows are indexed past the decode token in both layouts.
    prefill_rows = rows[1:] - 1
    lens, lens_full = swa.prefill_swa_lens, swa_full.prefill_swa_lens[prefill_rows]
    idx = swa.prefill_swa_indices[:, 0]
    idx_full = swa_full.prefill_swa_indices[prefill_rows, 0]
    # Untrimmed request: identical to the full batch.
    assert torch.equal(lens[WINDOW:], lens_full[WINDOW:])
    assert torch.equal(idx[WINDOW:], idx_full[WINDOW:])
    # Trimmed request: the window grows from the replay window's start, so its
    # first token sees only itself and the last sees the full window as before.
    assert lens[:WINDOW].tolist() == list(range(1, WINDOW + 1))
    assert torch.equal(idx[WINDOW - 1], idx_full[WINDOW - 1])
    assert idx[0, 0] == idx_full[0, WINDOW - 1] and idx[0, 1] == -1

    mla, mla_full = replay["mla"], full["mla"]
    assert torch.equal(mla.req_id_per_token, mla_full.req_id_per_token[rows])
    assert mla.query_start_loc.tolist() == [0, 1, 129, 229]

    idx_md, idx_full_md = replay["idx"].prefill.chunks[0], full["idx"].prefill.chunks[0]
    assert torch.equal(idx_md.cu_seqlen_ks, idx_full_md.cu_seqlen_ks[prefill_rows])
    assert torch.equal(idx_md.cu_seqlen_ke, idx_full_md.cu_seqlen_ke[prefill_rows])
    assert torch.equal(idx_md.token_to_seq, idx_full_md.token_to_seq)


def test_run_gathers_states_and_realigns_shared_indexer_buffers():
    cfg = _vllm_config()
    full = _full_metadata(cfg)
    topk = torch.arange(401 * 4, device=DEVICE).view(401, 4).int()
    candidates = torch.arange(401 * 3, device=DEVICE).view(401, 3).int()
    topk_before, candidates_before = topk.clone(), candidates.clone()
    seen = {}

    def run_layers(hidden_states, *rest):
        seen["hidden_states"] = hidden_states.clone()
        seen["replay_metadata"] = get_forward_context().attn_metadata
        return (hidden_states, rest[2])  # pre_mix

    layers = _replay_layers(run_layers, topk, candidates)
    hidden = torch.arange(401, device=DEVICE, dtype=torch.float32)[:, None]
    states = (hidden, hidden.long(), None, hidden, hidden, hidden, hidden)
    context = ForwardContext(no_compile_layers={}, attn_metadata=full, slot_mapping={})
    full_swa_indices = full["swa"].prefill_swa_indices.clone()
    with override_forward_context(context):
        outputs = layers(*states)

    rows = torch.tensor(REPLAY_ROWS, device=DEVICE)
    assert torch.equal(seen["hidden_states"], hidden[rows])
    # The replay was built by a private builder: the runner's stays untouched.
    assert seen["replay_metadata"]["swa"].num_prefill_tokens == 228
    assert torch.equal(full["swa"].prefill_swa_indices, full_swa_indices)
    assert context.attn_metadata is full
    assert torch.equal(topk[: len(REPLAY_ROWS)], topk_before[rows])
    assert torch.equal(candidates[: len(REPLAY_ROWS)], candidates_before[rows])
    # Results come back on full-batch rows; dropped rows are zero.
    assert outputs[0].shape[0] == 401
    assert torch.equal(outputs[0][rows], hidden[rows])
    assert outputs[0][1:173].abs().sum() == 0


def _fake_attention(x: torch.Tensor, out: torch.Tensor) -> None:
    """Stands in for the eager attention break: reads the metadata in the
    forward context at run time, like the real kernels do."""

    def run() -> None:
        swa = get_forward_context().attn_metadata["swa"]
        # Real kernels cover the metadata's token count; padding rows stay.
        n = swa.num_decode_tokens + swa.num_prefill_tokens
        req = swa.token_to_req_indices[:n].to(x.dtype)
        out[:n] = x[:n] + req[:, None]

    capture = BreakableCUDAGraphCapture.current()
    if capture is not None and capture.capturing:
        capture.add_eager(run)
    else:
        run()


def _fake_replay_layers(hidden, positions, _, pre_mix, post_mix, res_mix, residual):
    # Like the window KV insert, this captured op takes the metadata's slot
    # mapping by address: it must live in a buffer that is refilled per step.
    slots = get_forward_context().attn_metadata["swa"].slot_mapping
    x = hidden * 2 + slots[: hidden.shape[0], None].to(hidden.dtype)
    out = torch.empty_like(x)
    _fake_attention(x, out)
    return (out + positions[:, None].to(out.dtype), pre_mix + residual)


def _states(seed: int):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    hidden = torch.randn(401, 3, device=DEVICE, generator=g)
    return (
        hidden,
        torch.arange(401, device=DEVICE),
        None,
        hidden + 1,
        hidden + 2,
        hidden + 3,
        hidden + 4,
    )


def _context(metadata):
    return ForwardContext(
        no_compile_layers={},
        attn_metadata=metadata,
        slot_mapping={},
        is_padding=torch.zeros(401, dtype=torch.bool, device=DEVICE),
    )


def test_late_layer_graphs_match_eager():
    """The replay runs as an eager break of the model graph and replays its own
    graph; both must match the eager path on a batch the graphs never saw."""
    from vllm.platforms import current_platform
    from vllm.utils.torch_utils import _current_stream_tls

    cfg = _vllm_config()
    swa_b, mla_b, idx_b = _builders(cfg)
    # Capture-time batch: nothing to trim, like the runner's dummy batches.
    dummy = {
        "swa": swa_b.build(0, _common([101, 100, 100, 100], [101, 100, 100, 100], 32)),
        "mla": mla_b.build(0, _common([101, 100, 100, 100], [101, 100, 100, 100], 128)),
        "idx": idx_b.build(0, _common([101, 100, 100, 100], [101, 100, 100, 100], 128)),
    }
    real = _full_metadata(cfg)
    eager = _replay_layers(_fake_replay_layers, None, None)
    graphed = _replay_layers(_fake_replay_layers, None, None, graph_sizes=[512])
    states = _states(0)

    prev_stream = getattr(_current_stream_tls, "value", None)
    stream = torch.cuda.Stream()
    try:
        with torch.cuda.stream(stream):
            with override_forward_context(_context(dummy)):
                graphed(*states)  # profile run: allocates the static buffers
                outer = BreakableCUDAGraphCapture(
                    current_platform.get_global_graph_pool()
                )
                with outer:
                    hidden_out, pre_mix_out = graphed(*states)
            # Replay on a different, trimmed batch.
            new_states = _states(1)
            for dst, src in zip(states, new_states):
                if dst is not None:
                    dst.copy_(src)
            with override_forward_context(_context(real)):
                outer.replay()
                torch.accelerator.synchronize()
                expected = eager(*states)
                assert torch.equal(hidden_out, expected[0])
                assert torch.equal(pre_mix_out, expected[1])
                # Eager model forward, graphed replay.
                outputs = graphed(*states)
                assert torch.equal(outputs[0], expected[0])
                assert torch.equal(outputs[1], expected[1])
    finally:
        torch.cuda.current_stream().wait_stream(stream)
        _current_stream_tls.value = prev_stream
