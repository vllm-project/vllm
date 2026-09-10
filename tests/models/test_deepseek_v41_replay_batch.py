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
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
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
    cfg.compilation_config.cudagraph_capture_sizes = [256, 512]
    cfg.parallel_config.data_parallel_size = 1
    layers = SimpleNamespace(window=WINDOW, replay_batch=None)
    model = SimpleNamespace(token_lookback_depth=0, decoder_replay_layers=layers)
    builds: list = []

    def prepare_attn(self, input_batch, cg_mode, block_tables, slot_mappings, *a, **kw):
        builds.append(
            SimpleNamespace(
                batch=input_batch,
                slot_mappings=slot_mappings.clone(),
                replay_start=kw["model_specific_attn_metadata"].replay_start,
            )
        )
        return {"swa": object()}

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
    batch = InputBatch.make_dummy(num_reqs, num_tokens, InputBuffers(16, 1024, DEVICE))
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
    replay = state.decoder_replay.replay_batch
    return replay, (state.builds[-1] if replay is not None else None)


def test_replay_batch_keeps_each_request_window(state):
    state._req_replay_start[2] = 50  # the encoder-side replay start of request 2
    batch = _input_batch(QUERY_LENS, SEQ_LENS, PREFILLING)
    replay, build = _prepare(state, batch, CUDAGraphMode.NONE)
    assert replay is not None and replay.trims and replay.graph_size is None
    rows = torch.tensor(REPLAY_ROWS, device=DEVICE)
    assert torch.equal(replay.rows, rows)
    sub = build.batch
    assert sub.query_start_loc.tolist() == [0, 1, 129, 229]
    assert sub.query_start_loc_np.tolist() == [0, 1, 129, 229]
    assert sub.num_tokens == sub.num_tokens_after_padding == 229
    assert sub.max_query_len == WINDOW
    # The trimmed request's window starts at its replay window; a higher
    # encoder-side replay start stands.
    assert build.replay_start.tolist() == [0, 300 - WINDOW, 50]
    assert torch.equal(sub.positions, batch.positions[rows])
    assert torch.equal(build.slot_mappings, _slot_mappings(401)[:, rows])
    assert torch.equal(replay.slot_mapping["mla"], _slot_mappings(401)[1, rows])
    assert not replay.is_padding.any() and replay.dp_metadata is None


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
    state._req_keeps_rows[1] = True
    batch = _input_batch([1, 300, 300], [500, 300, 300], PREFILLING)
    replay, build = _prepare(state, batch, CUDAGraphMode.NONE)
    assert replay is not None and replay.trims
    assert replay.rows.tolist() == [0, *range(1, 301), *range(601 - WINDOW, 601)]
    assert build.batch.max_query_len == 300


def test_replay_batch_keeps_adaptive_verification_query_bound(state):
    """Adaptive verification bounds the decodes' device-side query lengths
    above their CPU lengths; the replay batch keeps that bound."""
    batch = _input_batch(QUERY_LENS, SEQ_LENS, PREFILLING, max_query_len=200)
    replay, build = _prepare(state, batch, CUDAGraphMode.NONE)
    assert replay is not None and build.batch.max_query_len == 200


def test_whole_batch_forwards_get_no_replay_batch(state):
    short = _input_batch([100, 100], [100, 100], [True, True])
    assert _prepare(state, short, CUDAGraphMode.NONE) == (None, None)
    assert _prepare(state, short, CUDAGraphMode.FULL) == (None, None)
    assert len(state.builds) == 2  # the batch's own metadata only


@pytest.mark.parametrize(
    ("prefilling", "graph_size", "num_tokens"),
    [
        (PREFILLING, 256, 229),  # the smallest graph fitting the trimmed rows
        ([False] * 3, 512, 401),  # a dummy (capture) batch keeps its rows
    ],
)
def test_replay_graph_size(state, prefilling, graph_size, num_tokens):
    batch = _input_batch(QUERY_LENS, SEQ_LENS, prefilling, num_tokens_after_padding=512)
    replay, build = _prepare(state, batch, CUDAGraphMode.PIECEWISE)
    assert replay is not None
    assert replay.graph_size == graph_size and replay.rows.shape[0] == num_tokens
    assert replay.trims == (num_tokens < 401)
    assert build.batch.num_tokens_after_padding == graph_size
    assert (
        replay.is_padding[num_tokens:].all()
        and not replay.is_padding[:num_tokens].any()
    )
    assert (build.slot_mappings[:, num_tokens:] == PAD_SLOT_ID).all()
