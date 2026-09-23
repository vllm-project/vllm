# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SWA bounded replay in DeepseekV41ModelState: the batch's replay starts reach
the sliding-window builders, and the replayed tokens' slots are padded in the
prefix-cacheable groups only."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from vllm.config import CUDAGraphMode
from vllm.models.deepseek_v41.nvidia.model_state import DeepseekV41ModelState
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadataBuilder
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.default import DefaultModelState

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

DEVICE = torch.device("cuda")
WINDOW = 4
# Group 0 is prefix-cacheable (compressed KV), group 1 is the replayed window.
KV_CACHE_CONFIG = SimpleNamespace(
    kv_cache_groups=[
        SimpleNamespace(
            kv_cache_spec=SimpleNamespace(prefix_cacheable=True, prefix_replay_tokens=0)
        ),
        SimpleNamespace(
            kv_cache_spec=SimpleNamespace(
                prefix_cacheable=False, prefix_replay_tokens=WINDOW
            )
        ),
    ]
)


@pytest.fixture
def state(monkeypatch):
    cfg = MagicMock()
    cfg.model_config.enable_prompt_embeds = False
    cfg.model_config.uses_mrope = False
    cfg.model_config.is_multimodal_model = False
    cfg.scheduler_config.max_num_seqs = 8
    cfg.scheduler_config.max_num_batched_tokens = 64
    seen: dict = {}

    def prepare_attn(self, input_batch, cg_mode, block_tables, slot_mappings, *a, **kw):
        seen["slot_mappings"] = slot_mappings.clone()
        seen["extra"] = kw["model_specific_attn_metadata"]
        return {}

    monkeypatch.setattr(DefaultModelState, "prepare_attn", prepare_attn)
    state = DeepseekV41ModelState(
        cfg, SimpleNamespace(token_lookback_depth=0), None, DEVICE
    )
    state.seen = seen  # type: ignore[attr-defined]
    return state


def _batch(query_lens, seq_lens, idx_mapping, is_prefilling):
    num_reqs, num_tokens = len(query_lens), sum(query_lens)
    batch = InputBatch.make_dummy(num_reqs, num_tokens, InputBuffers(8, 64, DEVICE))
    return replace(
        batch,
        query_start_loc=torch.tensor(
            [0, *np.cumsum(query_lens)], dtype=torch.int32, device=DEVICE
        ),
        idx_mapping_np=np.array(idx_mapping, dtype=np.intp),
        is_prefilling_np=np.array(is_prefilling, dtype=np.bool_),
        positions=torch.cat(
            [torch.arange(s - q, s) for q, s in zip(query_lens, seq_lens)]
        ).to(DEVICE),
    )


def _prepare(state, batch):
    slot_mappings = torch.arange(2 * batch.num_tokens, device=DEVICE).view(2, -1)
    state.prepare_attn(
        batch, CUDAGraphMode.NONE, (), slot_mappings, [], KV_CACHE_CONFIG
    )
    swa_builder = MagicMock(spec=DeepseekSparseSWAMetadataBuilder)
    extra = state.seen["extra"]
    replay_start = extra.get_extra_attn_kwargs(swa_builder, batch.num_reqs)[
        "replay_start"
    ]
    assert extra.get_extra_attn_kwargs(MagicMock(), batch.num_reqs) == {}
    return replay_start.tolist(), state.seen["slot_mappings"]


def test_replayed_tokens_write_only_the_window_group(state):
    # Request state 3 resumes a hit at 16: positions 16..19 are replayed.
    state.add_request(3, SimpleNamespace(replay_start=16))
    state.add_request(1, SimpleNamespace(replay_start=0))
    batch = _batch([6, 2], [22, 2], idx_mapping=[3, 1], is_prefilling=[True, True])
    replay_start, slots = _prepare(state, batch)
    assert replay_start == [16, 0]
    padded = [PAD_SLOT_ID] * WINDOW + [4, 5, 6, 7]
    assert slots[0].tolist() == padded  # compressed KV keeps its cached rows
    assert slots[1].tolist() == list(range(8, 16))  # the window is rebuilt


def test_only_prefills_carry_a_replay_start(state):
    state.add_request(0, SimpleNamespace(replay_start=16))
    # The same request, decoding: its rows sit above the hit.
    batch = _batch([1], [40], idx_mapping=[0], is_prefilling=[False])
    replay_start, slots = _prepare(state, batch)
    assert replay_start == [0]
    assert slots.tolist() == [[0], [1]]
