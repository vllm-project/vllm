# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The decoder-side SWA bounded replay batch DeepseekV41ModelState prepares."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from vllm.config import CUDAGraphMode
from vllm.models.deepseek_v41.nvidia.model_state import DeepseekV41ModelState
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
    cfg.compilation_config.fast_moe_cold_start = False
    cfg.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    layers = SimpleNamespace(window=WINDOW, replay_batch=None, metadata_prefixes=set())
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
        return {"swa": build.attn_metadata}

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
) -> InputBatch:
    num_reqs, num_tokens = len(query_lens), sum(query_lens)
    num_padded = num_tokens_after_padding or num_tokens
    batch = InputBatch.make_dummy(
        num_reqs, max(num_reqs, num_tokens), InputBuffers(16, 1024, DEVICE)
    )
    return replace(
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


def _slot_mappings(num_tokens: int) -> torch.Tensor:
    return torch.stack([torch.arange(num_tokens), torch.arange(num_tokens) * 10]).to(
        DEVICE
    )


def _prepare(state, batch, cg_mode):
    """Runs prepare_attn; returns the replay batch and the replay build's inputs."""
    state.prepare_attn(
        batch,
        cg_mode,
        BLOCK_TABLES,
        _slot_mappings(batch.num_tokens_after_padding),
        [],
        GROUPS,
    )
    replay = state.decoder_replay_layers.replay_batch
    return (None, None) if replay is None else (replay, state.builds[-1])


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
    context = replay.forward_context
    assert context.attn_metadata == {"swa": build.attn_metadata}
    assert torch.equal(context.slot_mapping["mla"], _slot_mappings(401)[1, rows])
    assert context.dp_metadata is None and context.is_padding is None


@pytest.mark.parametrize(
    "query_len,seq_len,global_start,global_end,keep_all,prefilling",
    [
        (300, 300, 0, 300, False, True),
        (0, 0, 0, 300, False, False),
        (100, 100, 0, 600, False, True),
        (100, 500, 0, 600, False, True),
        (100, 600, 0, 600, False, True),
        (100, 1100, 1000, 1100, False, True),
        (100, 500, 0, 600, True, True),
        (6, 606, 600, 606, False, False),
    ],
)
def test_pcp_replay_keeps_local_slots_and_pads_to_peer(
    state,
    monkeypatch,
    query_len,
    seq_len,
    global_start,
    global_end,
    keep_all,
    prefilling,
):
    """Gathered cache slots must follow the new rank-major padded KV payload."""
    from vllm.models.deepseek_v41.nvidia import model_state

    state.pcp_size = 2
    padding = max(WINDOW + 3, query_len if keep_all else 0)
    state._keeps_rows_np[0] = keep_all
    floor = (
        global_end - WINDOW
        if prefilling and not keep_all and global_end - global_start > WINDOW
        else 0
    )
    kept = min(query_len, max(seq_len - floor, 0))
    gathers = []

    def all_gather(local, dim):
        assert dim == 1 and local.shape == (2, padding)
        gathers.append(local.clone())
        return torch.cat((local, local), dim=1)

    group = SimpleNamespace(rank_in_group=1, cpu_group=None, all_gather=all_gather)
    monkeypatch.setattr(model_state, "get_pcp_group", lambda: group)

    def agree(tensor, **kwargs):
        # The other rank trims and needs WINDOW+3 payload rows; the empty
        # peer must follow the same collectives without inventing a real row.
        if tensor.ndim == 2:
            tensor[0, 0] = global_end
            tensor[1, 0] = -global_start
        else:
            tensor.copy_(torch.tensor([1, padding], dtype=tensor.dtype))

    monkeypatch.setattr(model_state.dist, "all_reduce", agree)
    batch = _input_batch(
        [query_len],
        [seq_len],
        [prefilling],
        num_tokens_after_padding=max(1, query_len),
    )
    local_slots = _slot_mappings(batch.num_tokens_after_padding)
    gathered_slots = torch.cat((local_slots, local_slots + 10000), dim=1)
    state._replay = (WINDOW, torch.tensor([1], dtype=torch.int32, device=DEVICE))
    state._kept_slot_mappings = torch.empty((2, 1024), dtype=torch.int64, device=DEVICE)
    state._prepare_replay_batch(
        batch,
        CUDAGraphMode.NONE,
        BLOCK_TABLES,
        gathered_slots,
        [],
        GROUPS,
        torch.zeros(1, dtype=torch.int32, device=DEVICE),
    )
    replay = state.decoder_replay_layers.replay_batch
    rows = torch.arange(query_len - kept, query_len, device=DEVICE)
    assert replay is not None and torch.equal(replay.rows, rows)
    assert torch.equal(gathers[0][:, :kept], (local_slots + 10000)[:, rows])
    assert torch.all(gathers[0][:, kept:] == -1)
    assert replay.forward_context.batch_descriptor.num_tokens == padding
    assert replay.forward_context.is_padding.tolist() == [False] * kept + [True] * (
        padding - kept
    )
    assert state.builds[-1].batch.positions[kept:].count_nonzero() == 0
    if kept:
        assert state.builds[-1].replay_start.tolist() == [floor]


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
    """Every prompt row of a prompt-logprobs request is read, so it never
    trims; the other prefills still do."""
    state._keeps_rows_np[1] = True
    batch = _input_batch([1, 300, 300], [500, 300, 300], PREFILLING)
    replay, build = _prepare(state, batch, CUDAGraphMode.NONE)
    assert replay is not None
    assert replay.rows.tolist() == [0, *range(1, 301), *range(601 - WINDOW, 601)]
    assert build.batch.max_query_len == 300


def test_replay_batch_keeps_adaptive_verification_query_bound(state):
    """Adaptive verification bounds the decodes' device-side query lengths
    above their CPU lengths; the replay batch keeps that bound."""
    batch = _input_batch(QUERY_LENS, SEQ_LENS, PREFILLING, max_query_len=200)
    replay, build = _prepare(state, batch, CUDAGraphMode.NONE)
    assert replay is not None and build.batch.max_query_len == 200


def test_graph_steps_trim_only_piecewise_at_threshold(state):
    """A CUDA graph keeps the layers on its whole batch, unless it is a PIECEWISE
    one at the trim threshold or above; an eager step trims a batch to its rows."""
    batch = _input_batch(QUERY_LENS, SEQ_LENS, PREFILLING, num_tokens_after_padding=512)
    for cg_mode in (CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL):
        assert _prepare(state, batch, cg_mode) == (None, None)
    replay, build = _prepare(state, batch, CUDAGraphMode.NONE)
    assert replay is not None
    assert replay.rows.tolist() == REPLAY_ROWS
    assert build.cg_mode == CUDAGraphMode.NONE
    assert build.batch.num_tokens_after_padding == len(REPLAY_ROWS)
    state._trim_threshold = 256
    replay, _ = _prepare(state, batch, CUDAGraphMode.PIECEWISE)
    assert replay is not None and replay.rows.tolist() == REPLAY_ROWS
    assert _prepare(state, batch, CUDAGraphMode.FULL) == (None, None)


def test_whole_batch_forwards_get_no_replay_batch(state):
    short = _input_batch([100, 100], [100, 100], [True, True])
    for cg_mode in (CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL):
        assert _prepare(state, short, cg_mode) == (None, None)
    assert len(state.builds) == 3  # the batch's own metadata only
    # Startup prepares builders even before a real request first trims.
    assert state._replay_attn_groups is not None
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
