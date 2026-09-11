# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner


@contextmanager
def preserve_serving_state(
    runner: "GPUModelRunner", *, full_pool: bool = False
) -> Iterator[None]:
    """Isolate an MRV2 warmup from live request state and cached KV."""
    req_states = runner.req_states
    block_tables = getattr(runner, "block_tables", None)

    saved_free = list(req_states.free_indices)
    saved_id_to_index = dict(req_states.req_id_to_index)
    saved_index_to_id = dict(req_states.index_to_req_id)
    saved_eplb_suppressed = runner.eep_eplb_suppressed

    if full_pool:
        assert not saved_id_to_index, (
            f"full_pool warmup wanted an empty request pool, found "
            f"{len(saved_id_to_index)} live requests"
        )
    available_indices = (
        range(req_states.max_num_reqs) if full_pool else req_states.reserved_indices
    )
    req_states.free_indices[:] = available_indices

    if block_tables is not None:
        block_tables.redirect_writes_to_null_block = True
    runner.eep_eplb_suppressed = True

    try:
        yield
    finally:
        for req_id in list(req_states.req_id_to_index):
            if req_id not in saved_id_to_index:
                runner._remove_request(req_id)

        req_states.free_indices[:] = saved_free
        req_states.req_id_to_index.clear()
        req_states.req_id_to_index.update(saved_id_to_index)
        req_states.index_to_req_id.clear()
        req_states.index_to_req_id.update(saved_index_to_id)

        runner.eep_eplb_suppressed = saved_eplb_suppressed
        if block_tables is not None:
            block_tables.redirect_writes_to_null_block = False
        if runner.kv_block_zeroer is not None:
            runner.kv_block_zeroer.zero_block_ids([0])
