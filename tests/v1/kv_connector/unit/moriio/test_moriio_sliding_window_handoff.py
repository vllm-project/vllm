# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MoRIIO READ P/D handoff of sliding-window KV, on CPU schedulers.

The decoder recomputes the last prompt token (position N - 1), whose window
starts at N - WINDOW. These tests drive real `Scheduler`s and
`KVCacheManager`s on both legs (only the transport is absent) and check that
the block holding that position survives on the prefiller, is allocated on the
decoder, and is paired with the right prefill block for the READ (#65550).
"""

import uuid

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
    MoRIIOConnectorMetadata,
    get_role,
    set_role,
)
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig
from vllm.v1.request import Request, RequestStatus

from ..utils import (
    create_model_runner_output,
    create_request,
    create_scheduler,
    create_vllm_config,
    make_kv_cache_config,
    make_kv_cache_group,
)

BLOCK_SIZE = 16
WINDOW = 128
PROMPT_LENGTHS = range(WINDOW + 1, WINDOW + 3 * BLOCK_SIZE + 1)
_ZMQ_ADDR = "host:127.0.0.1,handshake:4789,notify:4789"


@pytest.fixture(autouse=True)
def restore_moriio_role():
    role = get_role()
    yield
    set_role(role)


def _sliding_window_only_kv_cache_config() -> KVCacheConfig:
    spec = AttentionSpec(
        block_size=BLOCK_SIZE, num_kv_heads=4, head_size=16, dtype=torch.float16
    )
    return KVCacheConfig(
        num_blocks=1000,
        kv_cache_tensors=[],
        kv_cache_groups=[
            make_kv_cache_group(
                ["layer0", "layer1"],
                spec,
                SlidingWindowManager,
                SlidingWindowManager.config(sliding_window=WINDOW),
            )
        ],
    )


def _scheduler(kv_role: str, sliding_window_only: bool = False):
    """A READ scheduler over Full + SWA groups under HMA, or, for a uniform
    sliding-window model, a lone SWA group with HMA disabled (no window clip,
    but out-of-window blocks are still freed)."""
    vllm_config = create_vllm_config(
        block_size=BLOCK_SIZE,
        max_num_batched_tokens=64,
        kv_connector="MoRIIOConnector",
        kv_role=kv_role,
        kv_connector_extra_config={"read_mode": True, "backend": "xgmi"},
        disable_hybrid_kv_cache_manager=sliding_window_only,
    )
    kv_cache_config = (
        _sliding_window_only_kv_cache_config()
        if sliding_window_only
        else make_kv_cache_config(
            BLOCK_SIZE, swa_enabled=True, sw_size=WINDOW, num_blocks=1000
        )
    )
    kv_cache_config.hash_block_size = BLOCK_SIZE
    return create_scheduler(vllm_config, kv_cache_config=kv_cache_config)


def _sliding_window_group(scheduler) -> int:
    (group_id,) = [
        i
        for i, g in enumerate(scheduler.kv_cache_config.kv_cache_groups)
        if issubclass(g.manager_cls, SlidingWindowManager)
    ]
    assert list(scheduler.kv_cache_config.transfer_groups) == list(
        scheduler.kv_cache_config.kv_cache_groups
    )
    return group_id


def _block_table(scheduler, request_id: str, group_id: int) -> list[int]:
    manager = scheduler.kv_cache_manager.coordinator.single_type_managers[group_id]
    return [b.block_id for b in manager.req_to_blocks[request_id]]


def _null_block(scheduler) -> int:
    return scheduler.kv_cache_manager.block_pool.null_block.block_id


def _window_start_block(num_tokens: int) -> int:
    """Logical block holding the oldest position the recomputed token reads."""
    return max(0, num_tokens - WINDOW) // BLOCK_SIZE


def _make_request(num_tokens: int, request_id: str, **kwargs) -> Request:
    request = create_request(num_tokens=num_tokens, block_size=BLOCK_SIZE, **kwargs)
    request.request_id = request_id
    return request


def _new_request_id() -> str:
    return f"___prefill_addr_{_ZMQ_ADDR}___decode_addr_{_ZMQ_ADDR}_{uuid.uuid4().hex}"


def _run_prefill(num_tokens: int, request_id: str, sliding_window_only: bool = False):
    """Prefill the prompt on a READ producer until the connector publishes it.

    Returns the scheduler, the block table handed to `request_finished`, and
    the kv_transfer_params it returned for the decoder.
    """
    scheduler = _scheduler("kv_producer", sliding_window_only)
    request = _make_request(num_tokens, request_id, do_remote_decode=True)
    request.kv_transfer_params["transfer_id"] = "xfer"
    scheduler.add_request(request)

    finished: dict = {}
    connector = scheduler.connector
    request_finished = connector.request_finished_all_groups

    def spy(req, block_ids):
        finished["block_ids"] = [list(group) for group in block_ids]
        finished["result"] = request_finished(req, block_ids)
        return finished["result"]

    connector.request_finished_all_groups = spy
    while "result" not in finished:
        scheduler_output = scheduler.schedule()
        output = create_model_runner_output([request])
        if request.num_computed_tokens < request.num_prompt_tokens:
            output.sampled_token_ids = [[]]
        scheduler.update_from_output(scheduler_output, output)

    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED
    delay_free, params = finished["result"]
    assert delay_free
    return scheduler, finished["block_ids"], params


def _schedule_decode(
    num_tokens: int, request_id: str, params: dict, sliding_window_only: bool = False
):
    """Admit the remote-prefilled request on a READ consumer for one step."""
    scheduler = _scheduler("kv_consumer", sliding_window_only)
    request = _make_request(num_tokens, request_id, do_remote_prefill=True)
    request.kv_transfer_params.update(params)
    scheduler.add_request(request)
    return scheduler, scheduler.schedule()


@pytest.mark.parametrize("num_tokens", PROMPT_LENGTHS)
def test_prefill_keeps_recomputed_token_window(num_tokens: int):
    """The block table the producer publishes still holds the block with the
    decoder's oldest in-window position, and the producer keeps it alive."""
    request_id = _new_request_id()
    scheduler, block_ids, _ = _run_prefill(num_tokens, request_id)
    sw_table = block_ids[_sliding_window_group(scheduler)]
    window_start = _window_start_block(num_tokens)
    block_id = sw_table[window_start]

    assert block_id != _null_block(scheduler), (
        f"N={num_tokens}: block {window_start} of {sw_table} was freed"
    )
    assert scheduler.kv_cache_manager.block_pool.blocks[block_id].ref_cnt > 0


@pytest.mark.parametrize("sliding_window_only", [False, True])
@pytest.mark.parametrize("num_tokens", PROMPT_LENGTHS)
def test_read_pairs_recomputed_token_window_with_prefill_kv(
    num_tokens: int, sliding_window_only: bool
):
    """Every decode block MoRIIO READs into is paired with the prefill block
    of the same logical position, and the recomputed token's oldest
    in-window block is read from a real prefill block."""
    request_id = _new_request_id()
    p_scheduler, p_block_ids, params = _run_prefill(
        num_tokens, request_id, sliding_window_only
    )
    d_scheduler, scheduler_output = _schedule_decode(
        num_tokens, request_id, params, sliding_window_only
    )
    metadata = scheduler_output.kv_connector_metadata
    assert isinstance(metadata, MoRIIOConnectorMetadata)
    read = metadata.reqs_to_recv[request_id]

    p_null, d_null = _null_block(p_scheduler), _null_block(d_scheduler)
    sw_group = _sliding_window_group(d_scheduler)
    for group_id, p_table in enumerate(p_block_ids):
        d_table = _block_table(d_scheduler, request_id, group_id)
        sources = {
            d_table.index(dst): src
            for dst, src in zip(
                read.local_block_ids[group_id],
                read.remote_block_ids[group_id],
                strict=True,
            )
            if dst != d_null
        }
        for position, src in sources.items():
            assert src != p_null and p_table.index(src) == position, (
                f"N={num_tokens} group {group_id}: decode block {position} "
                f"reads prefill block {src} (null block: {p_null}) of {p_table}"
            )
        if group_id == sw_group:
            assert _window_start_block(num_tokens) in sources, (
                f"N={num_tokens}: decode never reads block "
                f"{_window_start_block(num_tokens)}, reads {sorted(sources)}"
            )
