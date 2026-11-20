# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Data-path checks for the NIXL SSM boundary-state transfer. Position i of
the Mamba ids P sends is read by the worker into position i of the ids D
receives into, so:

- P: the boundary id P sends holds P's cached state at exactly
  `remote_ssm_boundary_tokens` (hash and token count), and the last id is P's
  running-state slot `row[(n - 1) // bs]`.
- D: the first received id is D's row slot `boundary // bs - 1` (the slot
  cache_blocks hashes for the boundary), the last is the running-state slot the
  worker reads from (`row[(n - 1) // bs]`, mamba_utils), and the block that
  serves D's next-turn hit is a received block whose P source held the state at
  exactly the hit length.
"""

import dataclasses
from unittest.mock import patch

import pytest

from vllm.platforms import current_platform
from vllm.v1.core.kv_cache_utils import get_block_hash

from .test_nixl_hybrid_boundary_state import (
    BLOCK_SIZE,
    MAMBA_GROUP,
    _make_request,
    _run_prefill,
    _turn1_prompt,
)
from .utils import (
    EOS_TOKEN_ID,
    create_model_runner_output,
    create_scheduler,
    create_vllm_config,
    make_kv_cache_config,
    set_mtp_config,
)

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _cpu_device_type():
    with patch.object(current_platform, "device_type", "cpu"):
        yield


def _make_sched(kv_role: str, xfer: bool, mtp: bool, spec_blocks: int):
    vllm_config = create_vllm_config(
        block_size=BLOCK_SIZE,
        max_num_batched_tokens=256,
        kv_role=kv_role,
        kv_connector_extra_config={"ssm_boundary_state_xfer": xfer},
    )
    if mtp:
        set_mtp_config(vllm_config, disable_lookahead_block_hashes=True)
        vllm_config.speculative_config.num_speculative_tokens = max(1, spec_blocks)
        vllm_config.cache_config.prefix_cache_retention_interval = 0
    vllm_config.scheduler_config.disable_hybrid_kv_cache_manager = False
    vllm_config.scheduler_config.async_scheduling = False
    vllm_config.cache_config.mamba_cache_mode = "align"
    kv_cache_config = make_kv_cache_config(
        block_size=BLOCK_SIZE,
        mamba_enabled=True,
        mamba_cache_mode="align",
        num_blocks=200,
    )
    kv_cache_config.hash_block_size = BLOCK_SIZE
    if spec_blocks:
        group = kv_cache_config.kv_cache_groups[MAMBA_GROUP]
        kv_cache_config.kv_cache_groups[MAMBA_GROUP] = dataclasses.replace(
            group,
            kv_cache_spec=dataclasses.replace(
                group.kv_cache_spec, num_speculative_blocks=spec_blocks
            ),
        )
    return create_scheduler(
        vllm_config, kv_cache_config=kv_cache_config, hash_block_size=BLOCK_SIZE
    )


def _mgr(sched):
    return sched.kv_cache_manager.coordinator.single_type_managers[MAMBA_GROUP]


def _row(sched, req_id: str) -> list[int]:
    return [b.block_id for b in _mgr(sched).req_to_blocks.get(req_id, [])]


def _state_tokens(pool, block_id: int):
    block = pool.blocks[block_id]
    return block.block_hash_num_tokens if block.block_hash is not None else None


CASES = [
    # mtp, spec_blocks, prompt_len, expected boundary (None: running state only)
    (False, 0, 58, 48),
    (False, 0, 49, None),
    (False, 0, 64, 48),
    (True, 0, 58, 32),
    (True, 0, 49, 32),
    (True, 0, 64, 32),
    (True, 1, 58, 32),
    (True, 1, 49, 32),
    (True, 1, 64, 32),
    (True, 3, 58, 32),
    (True, 3, 49, 32),
    (True, 3, 64, 32),
]


@pytest.mark.parametrize("xfer", [True, False])
@pytest.mark.parametrize("mtp,spec_blocks,prompt_len,boundary", CASES)
def test_boundary_state_data_path(xfer, mtp, spec_blocks, prompt_len, boundary):
    bs = BLOCK_SIZE
    n = prompt_len - 1  # P computes the prompt minus its last token
    prefiller = _make_sched("kv_producer", xfer, mtp, spec_blocks)
    decoder = _make_sched("kv_consumer", xfer, mtp, spec_blocks)
    p_pool = prefiller.kv_cache_manager.block_pool
    d_pool = decoder.kv_cache_manager.block_pool
    prompt = _turn1_prompt(prompt_len)

    p_req, params = _run_prefill(prefiller, prompt)
    sent = list(params["remote_block_ids"][MAMBA_GROUP])
    sent_boundary = params.get("remote_ssm_boundary_tokens")
    p_row = _row(prefiller, p_req.request_id)

    # D: receive, then decode three tokens and finish.
    d_req = _make_request("d-1", prompt, max_tokens=3)
    d_req.kv_transfer_params = dict(
        params, do_remote_prefill=True, do_remote_decode=False
    )
    decoder.add_request(d_req)
    so = decoder.schedule()
    recv = so.kv_connector_metadata.reqs_to_recv[d_req.request_id]
    loc = list(recv.local_block_ids[MAMBA_GROUP])
    d_row = _row(decoder, d_req.request_id)
    decoder.update_from_output(
        so, create_model_runner_output([], finished_recving={d_req.request_id})
    )
    out_tokens: list[int] = []
    for i in range(3):
        so = decoder.schedule()
        if i == 0:
            d_num_computed_step1 = d_req.num_computed_tokens
        last = i == 2
        decoder.update_from_output(
            so, create_model_runner_output([d_req], use_eos=last, token_id=1000 + i)
        )
        out_tokens.append(EOS_TOKEN_ID if last else 1000 + i)
    decoder.schedule()
    assert d_req.is_finished()

    turn2 = prompt + out_tokens + [500 + i for i in range(10)]
    r2 = _make_request("d-next", turn2, max_tokens=1)
    computed, hit, _ = decoder.kv_cache_manager.get_computed_blocks(r2)
    mamba_hit = [b.block_id for b in computed.blocks[MAMBA_GROUP] if not b.is_null]

    # Running state: last sent id is P's running slot, last received id is D's.
    running_idx = (n - 1) // bs
    if p_row:
        assert sent[-1] == p_row[running_idx], "P running slot"
    assert loc[-1] == d_row[running_idx], "D running slot"
    assert len(loc) == len(sent)

    if xfer and boundary is not None:
        b_idx = boundary // bs - 1
        assert sent_boundary == boundary
        assert len(sent) == 2 and sent[0] != sent[-1]
        # P: the sent boundary block holds P's cached state at the boundary.
        p_block = p_pool.blocks[sent[0]]
        assert p_block.block_hash_num_tokens == boundary
        assert get_block_hash(p_block.block_hash) == p_req.block_hashes[b_idx]
        # D: received into the slot that gets hashed for the boundary.
        assert loc[0] == d_row[b_idx], "D boundary slot"
        d_block = d_pool.blocks[loc[0]]
        assert d_block.block_hash_num_tokens == boundary
        assert get_block_hash(d_block.block_hash) == d_req.block_hashes[b_idx]
        if prompt_len % bs:
            assert hit == boundary
            assert mamba_hit and mamba_hit[-1] == loc[0]
        else:
            # D computes the last prompt token itself and checkpoints a
            # block-aligned prompt locally, above the received boundary.
            assert hit >= boundary
    else:
        assert sent_boundary is None
        assert len(sent) == 1

    # General invariant: a D hit is served by a block cached at exactly the hit
    # length. If that block was received and the hit is within P's computed
    # tokens, its P source held the state at the hit length. (A hit past n is
    # D's own checkpoint: D computed the last prompt token on top of the
    # received running state, e.g. prompt_len % bs == 0.)
    if hit:
        assert mamba_hit and _state_tokens(d_pool, mamba_hit[-1]) == hit
        if hit <= n:
            src_tokens = ([sent_boundary] if len(sent) == 2 else []) + [n]
            assert mamba_hit[-1] in loc, "hit block was not received"
            assert src_tokens[loc.index(mamba_hit[-1])] == hit
        else:
            assert hit <= d_num_computed_step1
