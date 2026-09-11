# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner


@contextmanager
def preserve_serving_state(runner: "GPUModelRunner") -> Iterator[None]:
    """Prepare MRV2 request state for warmup after requests have drained."""
    req_states = runner.req_states
    block_tables = getattr(runner, "block_tables", None)

    saved_free = list(req_states.free_indices)
    saved_id_to_index = dict(req_states.req_id_to_index)
    saved_index_to_id = dict(req_states.index_to_req_id)
    saved_eplb_suppressed = runner.eep_eplb_suppressed
    # Preserve the previous redirect state for nested contexts.
    saved_redirect = (
        block_tables.redirect_writes_to_null_block
        if block_tables is not None
        else False
    )

    assert (
        not saved_id_to_index
        and not saved_index_to_id
        and len(saved_free) == req_states.max_num_reqs
    ), (
        "MRV2 warmup requires an empty request pool, "
        f"found {len(saved_id_to_index)} request ids, "
        f"{len(saved_index_to_id)} request indices, and "
        f"{len(saved_free)}/{req_states.max_num_reqs} free slots"
    )
    req_states.free_indices[:] = range(req_states.max_num_reqs)

    if block_tables is not None:
        block_tables.redirect_writes_to_null_block = True
    runner.eep_eplb_suppressed = True

    try:
        yield
    finally:
        for req_id in list(req_states.req_id_to_index):
            runner._remove_request(req_id)

        req_states.free_indices[:] = saved_free
        req_states.req_id_to_index.clear()
        req_states.req_id_to_index.update(saved_id_to_index)
        req_states.index_to_req_id.clear()
        req_states.index_to_req_id.update(saved_index_to_id)

        runner.eep_eplb_suppressed = saved_eplb_suppressed
        if block_tables is not None:
            block_tables.redirect_writes_to_null_block = saved_redirect
        if runner.kv_block_zeroer is not None:
            runner.kv_block_zeroer.zero_block_ids([0])
