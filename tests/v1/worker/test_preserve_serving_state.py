# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.serving_state import preserve_serving_state
from vllm.v1.worker.gpu.states import RequestState

BLOCK_SIZE = 16
MAX_NUM_REQS = 8
MAX_MODEL_LEN = 256

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="RequestState allocates on the accelerator"
)


class _Runner:
    def __init__(self) -> None:
        self.req_states = RequestState(
            max_num_reqs=MAX_NUM_REQS,
            max_model_len=MAX_MODEL_LEN,
            max_num_batched_tokens=1024,
            num_speculative_steps=0,
            vocab_size=128,
            device=torch.device("cuda"),
        )
        self.block_tables = BlockTables(
            block_sizes=[BLOCK_SIZE],
            max_num_reqs=MAX_NUM_REQS,
            max_num_batched_tokens=1024,
            max_num_blocks_per_group=[MAX_MODEL_LEN // BLOCK_SIZE],
            device=torch.device("cuda"),
            kernel_block_sizes=[BLOCK_SIZE],
        )
        self.eep_eplb_suppressed = False
        self.zeroed: list[list[int]] = []
        self.kv_block_zeroer = SimpleNamespace(zero_block_ids=self.zeroed.append)
        self.removed: list[str] = []

    def _remove_request(self, req_id: str) -> None:
        self.removed.append(req_id)
        self.req_states.remove_request(req_id)


def _add_request(runner: _Runner, req_id: str) -> int:
    runner.req_states.add_request(
        req_id=req_id,
        prompt_len=4,
        all_token_ids=list(range(4)),
        num_computed_tokens=0,
        max_tokens=4,
    )
    return runner.req_states.req_id_to_index[req_id]


def _pool_state(runner: _Runner):
    return (
        list(runner.req_states.free_indices),
        dict(runner.req_states.req_id_to_index),
        dict(runner.req_states.index_to_req_id),
    )


def test_warmup_uses_full_pool_and_null_blocks():
    runner = _Runner()
    pool_before = _pool_state(runner)
    blocks_before = runner.block_tables.block_tables[0].gpu.clone()

    with preserve_serving_state(runner):
        assert len(runner.req_states.free_indices) == MAX_NUM_REQS
        idx = _add_request(runner, "_warmup")
        runner.block_tables.append_block_ids(idx, ([5, 6, 7],), overwrite=True)
        runner.block_tables.apply_staged_writes()
        written = runner.block_tables.block_tables[0].gpu[idx, :3]
        assert torch.equal(written, torch.zeros_like(written))

    assert _pool_state(runner) == pool_before
    assert torch.equal(runner.block_tables.block_tables[0].gpu, blocks_before)
    assert runner.removed == ["_warmup"]
    assert not runner.eep_eplb_suppressed
    assert not runner.block_tables.redirect_writes_to_null_block
    assert runner.zeroed == [[0]]


def test_warmup_restores_state_on_error():
    runner = _Runner()
    pool_before = _pool_state(runner)

    with pytest.raises(RuntimeError), preserve_serving_state(runner):
        _add_request(runner, "_warmup")
        raise RuntimeError

    assert _pool_state(runner) == pool_before
    assert runner.removed == ["_warmup"]
    assert not runner.block_tables.redirect_writes_to_null_block


def test_warmup_rejects_live_requests():
    runner = _Runner()
    _add_request(runner, "live")

    with (
        pytest.raises(AssertionError, match="empty request pool"),
        preserve_serving_state(runner),
    ):
        pass
