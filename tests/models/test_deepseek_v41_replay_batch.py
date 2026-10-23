# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The decoder-side SWA bounded replay batch DeepseekV41ModelState prepares."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from vllm.config import CUDAGraphMode
from vllm.forward_context import (
    BatchDescriptor,
    ForwardContext,
    override_forward_context,
)
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsCapturer,
)
from vllm.models.deepseek_v41.decoder_replay_layers import DecoderReplayLayers
from vllm.models.deepseek_v41.nvidia.model_state import DeepseekV41ModelState
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.default import DefaultModelState

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the replay buffers live on the GPU"
)

WINDOW = 128
DEVICE = torch.device("cuda")
GROUPS = SimpleNamespace(
    kv_cache_groups=[
        SimpleNamespace(
            layer_names=["swa"],
            kv_cache_spec=SimpleNamespace(
                prefix_cacheable=False, prefix_replay_tokens=WINDOW
            ),
        ),
        SimpleNamespace(
            layer_names=["mla"],
            kv_cache_spec=SimpleNamespace(
                prefix_cacheable=True, prefix_replay_tokens=0
            ),
        ),
    ]
)
BLOCK_TABLES = (torch.zeros(4, 4, device=DEVICE),) * 2

# decode (1 token), trimmed prefill (300 of 300), untrimmed prefill (100 of 300)
QUERY_LENS = [1, 300, 100]
SEQ_LENS = [500, 300, 300]
PREFILLING = [False, True, True]
REPLAY_ROWS = [0, *range(301 - WINDOW, 301), *range(301, 401)]


@pytest.fixture
def state(monkeypatch):
    """A DeepseekV41ModelState whose attention builds are recorded, not run."""
    cfg = MagicMock()
    cfg.model_config.enable_prompt_embeds = False
    cfg.model_config.uses_mrope = False
    cfg.model_config.is_multimodal_model = False
    cfg.scheduler_config.max_num_seqs = 16
    cfg.scheduler_config.max_num_batched_tokens = 1024
    cfg.parallel_config.data_parallel_size = 1
    cfg.parallel_config.prefill_context_parallel_size = 1
    cfg.aux_output_config.enable_return_routed_experts = False
    cfg.compilation_config.fast_moe_cold_start = False
    cfg.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    layers = DecoderReplayLayers(WINDOW, MagicMock(), MagicMock(), set(), "swa_first")
    model = SimpleNamespace(token_lookback_depth=0, decoder_replay_layers=layers)
    builds: list = []

    def prepare_attn(self, input_batch, cg_mode, block_tables, slot_mappings, *a, **kw):
        build = SimpleNamespace(
            batch=input_batch,
            cg_mode=cg_mode,
            slot_mappings=slot_mappings.clone(),
            replay_start=kw["model_specific_attn_metadata"].replay_start,
            attn_metadata=object(),
        )
        builds.append(build)
        return {"swa": build.attn_metadata, "swa_first": build.attn_metadata}

    monkeypatch.setattr(DefaultModelState, "prepare_attn", prepare_attn)
    state = DeepseekV41ModelState(cfg, model, None, DEVICE)
    state.builds = builds  # type: ignore[attr-defined]
    return state


def _input_batch(
    query_lens: list[int],
    seq_lens: list[int],
    is_prefilling: list[bool],
    device_query_lens: list[int] | None = None,
    num_tokens_after_padding: int | None = None,
    max_query_len: int | None = None,
    prefill_len: list[int] | None = None,
    computed: list[int] | None = None,
) -> InputBatch:
    num_reqs, num_tokens = len(query_lens), sum(query_lens)
    num_padded = num_tokens_after_padding or num_tokens
    batch = InputBatch.make_dummy(num_reqs, num_tokens, InputBuffers(16, 1024, DEVICE))
    batch = replace(
        batch,
        num_tokens=num_tokens,
        num_tokens_after_padding=num_padded,
        query_start_loc=torch.tensor(
            [0, *np.cumsum(device_query_lens or query_lens)],
            dtype=torch.int32,
            device=DEVICE,
        ),
        query_start_loc_np=np.array([0, *np.cumsum(query_lens)], dtype=np.int32),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=DEVICE),
        seq_lens_cpu_upper_bound=torch.tensor(seq_lens, dtype=torch.int32),
        is_prefilling_np=np.array(is_prefilling, dtype=np.bool_),
        positions=torch.cat(
            [torch.arange(s - q, s) for q, s in zip(query_lens, seq_lens)]
            + [torch.zeros(num_padded - num_tokens, dtype=torch.int64)]
        ).to(DEVICE),
        is_padding=torch.arange(num_padded, device=DEVICE) >= num_tokens,
        max_query_len=max_query_len,
    )
    if prefill_len is not None:
        batch = replace(
            batch,
            prefill_len_np=np.array(prefill_len, dtype=np.int32),
            num_computed_prefill_tokens_np=np.array(computed, dtype=np.int32),
        )
    return batch


def _slot_mappings(num_tokens: int) -> torch.Tensor:
    return torch.stack([torch.arange(num_tokens), torch.arange(num_tokens) * 10]).to(
        DEVICE
    )


def _prepare(state, batch, cg_mode):
    """Runs prepare_attn; returns the replay batch and the replay build's inputs.
    The first replay layer's sliding-window build follows the replay build."""
    state.prepare_attn(
        batch,
        cg_mode,
        BLOCK_TABLES,
        _slot_mappings(batch.num_tokens_after_padding),
        [],
        GROUPS,
    )
    replay = state.decoder_replay_layers.replay_batch
    return (None, None) if replay is None else (replay, state.builds[-2])


def test_replay_batch_keeps_each_request_window(state):
    state._replay_start_np[2] = 50  # the encoder-side replay start of request 2
    batch = _input_batch(QUERY_LENS, SEQ_LENS, PREFILLING)
    replay, build = _prepare(state, batch, CUDAGraphMode.NONE)
    assert replay is not None
    rows = torch.tensor(REPLAY_ROWS, device=DEVICE)
    assert torch.equal(replay.rows, rows)
    sub = build.batch
    assert build.cg_mode == CUDAGraphMode.NONE
    assert sub.query_start_loc.tolist() == [0, 1, 129, 229]
    assert sub.query_start_loc_np.tolist() == [0, 1, 129, 229]
    assert sub.num_tokens == sub.num_tokens_after_padding == 229
    assert sub.max_query_len == WINDOW
    # The trimmed request's window starts at its replay window; a higher
    # encoder-side replay start stands.
    assert build.replay_start.tolist() == [0, 300 - WINDOW, 50]
    assert torch.equal(sub.positions, batch.positions[rows])
    assert torch.equal(build.slot_mappings, _slot_mappings(401)[:, rows])
    # The first replay layer keeps the encoder-side window starts: it wrote
    # every row's window KV.
    first = state.builds[-1]
    assert first.batch is sub and first.cg_mode == CUDAGraphMode.NONE
    assert first.replay_start.tolist() == [0, 0, 50]
    context = replay.forward_context
    assert context.attn_metadata == {
        "swa": build.attn_metadata,
        "swa_first": first.attn_metadata,
    }
    assert torch.equal(context.slot_mapping["mla"], _slot_mappings(401)[1, rows])
    assert context.dp_metadata is None and context.is_padding is None


def test_replay_context_restores_tp_routing_rows(state):
    """Capture uses the compact replay batch's size, not the outer batch's."""
    batch = _input_batch(QUERY_LENS, SEQ_LENS, PREFILLING)
    replay, _ = _prepare(state, batch, CUDAGraphMode.NONE)
    context = replay.forward_context
    assert context.is_padding is None and context.dp_metadata is None
    count = len(REPLAY_ROWS)
    assert context.batch_descriptor.num_tokens == count
    capturer = RoutedExpertsCapturer.__new__(RoutedExpertsCapturer)
    capturer.dp_rank, capturer.tp_size = 0, 2
    capturer.device_buffer = torch.full((401, 1, 2), -1, device=DEVICE)
    expected = torch.tensor(REPLAY_ROWS, device=DEVICE)[:, None].expand(-1, 2)
    gathered = torch.cat((expected, torch.full((1, 2), -1, device=DEVICE)))
    with (
        override_forward_context(context),
        patch(
            "vllm.model_executor.layers.fused_moe.routed_experts_capturer.get_tp_group"
        ) as group,
    ):
        group.return_value.all_gather.return_value = gathered
        capturer.capture(0, gathered[: (count + 1) // 2])
    torch.testing.assert_close(capturer.device_buffer[:count, 0], expected)
    assert (capturer.device_buffer[count:] == -1).all()


def test_replay_batch_keeps_device_decode_boundaries(state):
    """Adaptive verification resizes decodes on the GPU (CPU [2, 2], device
    [1, 3]): their rows stay and the boundaries shift only by trimmed rows."""
    batch = _input_batch(
        [2, 2, 300], [10, 10, 300], PREFILLING, device_query_lens=[1, 3, 300]
    )
    replay, build = _prepare(state, batch, CUDAGraphMode.NONE)
    assert replay is not None
    assert replay.rows.tolist() == [0, 1, 2, 3, *range(304 - WINDOW, 304)]
    assert build.batch.query_start_loc.tolist() == [0, 1, 4, 4 + WINDOW]


def test_prompt_logprobs_requests_keep_their_rows(state):
    """Keep the whole scoring chunk, even before its scoring offset, while
    generation still trims and can reuse the scoring request's slot."""
    layers = state.decoder_replay_layers
    layers.trim_threshold = WINDOW + 1
    layers.run_layers = lambda hidden: (hidden + 1,)
    for prompt_logprobs, start in (
        (0, None),
        (-1, None),
        (None, 0),
        (None, 200),
        (0, 0),
    ):
        params = SamplingParams(
            prompt_logprobs=prompt_logprobs,
            prompt_logprob_start=start or 0,
            prompt_logprob_token_ids=(
                np.ones((299 - start, 1), dtype=np.int32) if start is not None else None
            ),
        )
        for cg_mode in (CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE):
            for chunk_size, offset in ((300, 0), (150, 0), (150, 150)):
                # Request slots deliberately differ from their batch positions.
                batch = replace(
                    _input_batch(
                        [1, chunk_size, 300],
                        [500, offset + chunk_size, 300],
                        PREFILLING,
                    ),
                    idx_mapping_np=np.array([3, 7, 9], dtype=np.int32),
                    idx_mapping=torch.tensor(
                        [3, 7, 9], dtype=torch.int32, device=DEVICE
                    ),
                )
                state.add_request(
                    7, SimpleNamespace(replay_start=0, sampling_params=params)
                )
                replay, build = _prepare(state, batch, cg_mode)
                assert replay is not None
                end = 1 + chunk_size
                rows = [*range(end), *range(end + 300 - WINDOW, end + 300)]
                assert replay.rows.tolist() == rows
                assert build.batch.query_start_loc.tolist() == [0, 1, end, end + WINDOW]
                assert build.batch.max_query_len == chunk_size
                torch.testing.assert_close(build.batch.positions, batch.positions[rows])

                # Check values, not just shapes: discarded rows are zero-filled.
                hidden = torch.arange(
                    1, batch.num_tokens * 2 + 1, device=DEVICE
                ).reshape(-1, 2)
                (actual,) = layers._run(hidden)
                expected = torch.zeros_like(hidden)
                expected[rows] = hidden[rows] + 1
                torch.testing.assert_close(actual, expected)

                state.add_request(
                    7, SimpleNamespace(replay_start=0, sampling_params=SamplingParams())
                )
                replay, _ = _prepare(state, batch, cg_mode)
                assert replay is not None
                assert replay.rows.tolist() == [
                    0,
                    *range(end - WINDOW, end),
                    *range(end + 300 - WINDOW, end + 300),
                ]


@pytest.mark.parametrize("cg_mode", [CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE])
def test_aux_output_requests_keep_every_routing_row(state, cg_mode):
    """R3 needs every prefill row even when final logits only need the window."""
    state.vllm_config.aux_output_config.enable_return_routed_experts = True
    for index in range(3):
        state.add_request(
            index,
            SimpleNamespace(
                replay_start=0,
                sampling_params=SamplingParams(),
            ),
        )
    batch = _input_batch(QUERY_LENS, SEQ_LENS, PREFILLING)
    state.decoder_replay_layers.trim_threshold = WINDOW + 1
    replay, _ = _prepare(state, batch, cg_mode)
    if cg_mode == CUDAGraphMode.NONE:
        assert replay is None
    else:
        torch.testing.assert_close(replay.rows, torch.arange(401, device=DEVICE))


def test_replay_batch_keeps_adaptive_verification_query_bound(state):
    """Adaptive verification bounds the decodes' device-side query lengths
    above their CPU lengths; the replay batch keeps that bound."""
    batch = _input_batch(QUERY_LENS, SEQ_LENS, PREFILLING, max_query_len=200)
    replay, build = _prepare(state, batch, CUDAGraphMode.NONE)
    assert replay is not None and build.batch.max_query_len == 200


def test_graph_steps_trim_only_piecewise_at_threshold(state):
    """A CUDA graph keeps the layers on its whole batch, unless it is a PIECEWISE
    one at the trim threshold or above, as its capture decided; an eager step
    trims a batch to its rows."""
    batch = _input_batch(QUERY_LENS, SEQ_LENS, PREFILLING, num_tokens_after_padding=512)
    for cg_mode in (CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL):
        assert _prepare(state, batch, cg_mode) == (None, None)
    replay, build = _prepare(state, batch, CUDAGraphMode.NONE)
    assert replay is not None
    assert replay.rows.tolist() == REPLAY_ROWS
    assert build.cg_mode == CUDAGraphMode.NONE
    assert build.batch.num_tokens_after_padding == len(REPLAY_ROWS)
    layers = state.decoder_replay_layers
    graph = ForwardContext(
        no_compile_layers={},
        attn_metadata={},
        slot_mapping={},
        cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
        batch_descriptor=BatchDescriptor(num_tokens=512),
    )
    for threshold in (513, 512):
        layers.trim_threshold = threshold
        replay, _ = _prepare(state, batch, CUDAGraphMode.PIECEWISE)
        with override_forward_context(graph):  # the capture's decision
            assert (
                layers.uses_replay_batch() == (replay is not None) == (threshold == 512)
            )
    assert replay is not None and replay.rows.tolist() == REPLAY_ROWS
    assert _prepare(state, batch, CUDAGraphMode.FULL) == (None, None)


def test_whole_batch_forwards_get_no_replay_batch(state):
    short = _input_batch([100, 100], [100, 100], [True, True])
    for cg_mode in (CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL):
        assert _prepare(state, short, cg_mode) == (None, None)
    assert len(state.builds) == 3  # the batch's own metadata only
    # Dummy batches, captures included (prepared as eager), are not prefills and
    # keep their rows.
    dummy = _input_batch(
        QUERY_LENS, SEQ_LENS, [False] * 3, num_tokens_after_padding=512
    )
    assert _prepare(state, dummy, CUDAGraphMode.NONE) == (None, None)


@pytest.fixture
def dp_state(state, monkeypatch):
    """`state` on DP rank 0 of 2; `state.other` sets what rank 1 reports
    (trims, replay tokens)."""
    import vllm.models.deepseek_v41.nvidia.model_state as module

    state.vllm_config.parallel_config.data_parallel_size = 2
    state.vllm_config.parallel_config.data_parallel_rank = 0
    state.vllm_config.parallel_config.is_moe_model = True
    state.other = (False, 0)  # type: ignore[attr-defined]

    def all_reduce(tensor, group):
        tensor[:, 1] = torch.tensor(state.other, dtype=torch.int32)

    monkeypatch.setattr(module.dist, "all_reduce", all_reduce)
    monkeypatch.setattr(module, "get_dp_group", lambda: SimpleNamespace(cpu_group=None))
    return state


def test_dp_ranks_replay_together(dp_state):
    """A trimming rank makes the others replay too, on every rank's count."""
    short = _input_batch([100, 100], [100, 100], [True, True])
    dp_state.other = (False, 200)
    assert _prepare(dp_state, short, CUDAGraphMode.NONE) == (None, None)

    dp_state.other = (True, 300)  # rank 1 trims to 300 rows
    replay, build = _prepare(dp_state, short, CUDAGraphMode.NONE)
    assert replay is not None
    assert replay.rows.tolist() == list(range(200))  # this rank keeps every row
    dp_metadata = replay.forward_context.dp_metadata
    assert dp_metadata.num_tokens_across_dp_cpu.tolist() == [200, 300]


def test_idle_dp_rank_dummy_trims_with_its_peers(dp_state):
    """An idle rank's dummy keeps one row, so peers do not wait on its replay."""
    dummy = _input_batch([256, 256], [256, 256], [False, False])
    dp_state.other = (True, 129)
    replay, _ = _prepare(dp_state, dummy, CUDAGraphMode.NONE)
    assert replay is not None and replay.rows.shape[0] == 1
    dp_metadata = replay.forward_context.dp_metadata
    assert dp_metadata.num_tokens_across_dp_cpu.tolist() == [1, 129]


def test_zero_prefix_window_warms_gather_and_skips_pad(monkeypatch):
    """ROCm leaves the prefix-cache window at 0. Replay still needs replay_start
    and a warmed gather kernel. The pad kernel stays on the prefix-cache path."""
    import vllm.models.deepseek_v41.nvidia.model_state as model_state

    calls = {"pad_warmup": 0, "gather_warmup": 0, "pad_launch": 0}

    def pad_warmup(*args, **kwargs):
        calls["pad_warmup"] += 1

    def gather_warmup(*args, **kwargs):
        calls["gather_warmup"] += 1

    class _Pad:
        warmup = staticmethod(pad_warmup)

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                calls["pad_launch"] += 1

            return launch

    monkeypatch.setattr(model_state, "_pad_replayed_slots_kernel", _Pad())
    monkeypatch.setattr(
        model_state._gather_replay_batch_kernel, "warmup", gather_warmup
    )

    cfg = MagicMock()
    cfg.model_config.enable_prompt_embeds = False
    cfg.model_config.uses_mrope = False
    cfg.model_config.is_multimodal_model = False
    cfg.scheduler_config.max_num_seqs = 16
    cfg.scheduler_config.max_num_batched_tokens = 1024
    cfg.parallel_config.data_parallel_size = 1
    cfg.parallel_config.prefill_context_parallel_size = 1
    cfg.compilation_config.fast_moe_cold_start = False
    cfg.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    layers = SimpleNamespace(
        window=WINDOW, replay_batch=None, metadata_prefixes=set(), trim_threshold=None
    )
    model = SimpleNamespace(token_lookback_depth=0, decoder_replay_layers=layers)
    groups = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(
                layer_names=["swa"],
                kv_cache_spec=SimpleNamespace(
                    prefix_cacheable=False, prefix_replay_tokens=0
                ),
            ),
            SimpleNamespace(
                layer_names=["mla"],
                kv_cache_spec=SimpleNamespace(
                    prefix_cacheable=True, prefix_replay_tokens=0
                ),
            ),
        ]
    )

    def prepare_attn(self, input_batch, cg_mode, block_tables, slot_mappings, *a, **kw):
        return {"swa": kw["model_specific_attn_metadata"]}

    monkeypatch.setattr(DefaultModelState, "prepare_attn", prepare_attn)
    state = DeepseekV41ModelState(cfg, model, None, DEVICE)
    state._replay_start_np[0] = 16
    batch = _input_batch([8], [24], [True])
    state.prepare_attn(
        batch,
        CUDAGraphMode.NONE,
        BLOCK_TABLES,
        _slot_mappings(batch.num_tokens_after_padding),
        [],
        groups,
    )
    assert state._replay[0] == 0
    assert calls == {"pad_warmup": 0, "gather_warmup": 1, "pad_launch": 0}
    assert state._replay_start[:1].tolist() == [16]


def test_dp_ranks_pad_to_one_replay_graph(dp_state):
    """Every rank pads its replay batch to the graph fitting the largest rank's."""
    dp_state.replay_cudagraphs = MagicMock()
    dp_state.replay_cudagraphs.dispatch.return_value = BatchExecutionDescriptor(
        CUDAGraphMode.PIECEWISE, 512, None
    )
    dp_state.other = (True, 300)
    batch = _input_batch(QUERY_LENS, SEQ_LENS, PREFILLING)
    replay, _ = _prepare(dp_state, batch, CUDAGraphMode.NONE)
    assert dp_state.replay_cudagraphs.dispatch.call_args.args[:2] == (3, 300)
    assert replay.forward_context.batch_descriptor.num_tokens == 512
    dp_metadata = replay.forward_context.dp_metadata
    assert dp_metadata.num_tokens_across_dp_cpu.tolist() == [512, 512]


# The fixture has no replay graphs, so the reach is max_num_batched_tokens (1024)
# plus the window: a chunk ending 1152 or more tokens before its prompt's end is far.
FAR_PROMPT, NEAR_PROMPT = 2000, 1200


def test_far_prefill_chunk_replays_one_row(state):
    """A chunk far from its prompt's end replays its last row, which reads only
    itself; a chunk within reach of the end keeps its window."""
    batch = _input_batch(
        [200, 200, 1],
        [200, 200, 500],
        [True, True, False],
        prefill_len=[FAR_PROMPT, NEAR_PROMPT, 0],
        computed=[0, 0, 0],
    )
    replay, build = _prepare(state, batch, CUDAGraphMode.NONE)
    assert replay is not None
    assert replay.rows.tolist() == [199, *range(400 - WINDOW, 400), 400]
    assert build.replay_start.tolist() == [199, 200 - WINDOW, 0]
    assert state._skipped_tail_np[:2].tolist() == [True, False]


def test_chunk_after_one_row_chunk_reads_only_itself(state):
    """The rows a one-row chunk skipped hold no replay-layer KV: the next chunk's
    window starts at its own first row, and it takes the replay batch even when
    nothing trims."""
    prompt = 1400  # 200 + 100 + 1100: the first chunk is far, the second is not
    first = _input_batch([200], [200], [True], prefill_len=[prompt], computed=[0])
    replay, _ = _prepare(state, first, CUDAGraphMode.NONE)
    assert replay.rows.tolist() == [199]
    second = _input_batch([100], [300], [True], prefill_len=[prompt], computed=[200])
    replay, build = _prepare(state, second, CUDAGraphMode.NONE)
    assert replay is not None and replay.rows.tolist() == list(range(100))
    assert build.replay_start.tolist() == [200]
    assert not state._skipped_tail_np[0]


def test_eager_forward_only_after_a_one_row_chunk(state):
    """A step that would keep the whole batch runs eager when a chunk follows a
    one-row chunk; at the trim threshold it takes the replay batch anyway."""
    state.decoder_replay_layers.trim_threshold = 768
    state._skipped_tail_np[0] = True

    def needs_eager(num_scheduled, skipped=True):
        state._skipped_tail_np[0] = skipped
        return state.needs_eager_forward(
            num_tokens=int(sum(num_scheduled)),
            idx_mapping=np.arange(len(num_scheduled)),
            is_prefilling=np.array([True] + [False] * (len(num_scheduled) - 1)),
            num_computed_prefill_tokens=np.zeros(len(num_scheduled), np.int32),
            num_scheduled_tokens=np.array(num_scheduled),
            prefill_len=np.full(len(num_scheduled), 5000),
        )

    assert needs_eager([300, 1, 1])
    assert not needs_eager([900, 1, 1])
    assert needs_eager([400, 400])  # uniform: may take a FULL graph
    assert not needs_eager([300, 1, 1], skipped=False)
