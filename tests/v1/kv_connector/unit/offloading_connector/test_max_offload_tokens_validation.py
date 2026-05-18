# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test validation of max_offload_tokens parameter."""

import pytest

from tests.v1.kv_connector.unit.offloading_connector.utils import (
    generate_store_output,
)
from tests.v1.kv_connector.unit.utils import EOS_TOKEN_ID

# Common runner configuration used across all tests.
_GPU_BLOCK_SIZE = 4
_BLOCK_SIZE_FACTOR = 3
_OFFLOADED_BLOCK_SIZE = _GPU_BLOCK_SIZE * _BLOCK_SIZE_FACTOR  # 12
_NUM_GPU_BLOCKS = 100


def _setup_request(runner, max_offload_tokens):
    runner.new_request(token_ids=[0] * _OFFLOADED_BLOCK_SIZE * 3)
    req = runner.scheduler.requests[str(runner.req_id)]
    req.kv_transfer_params = {"max_offload_tokens": max_offload_tokens}
    runner.manager.prepare_store.side_effect = (
        lambda keys, req_context: generate_store_output(keys)
    )


@pytest.mark.parametrize("async_scheduling", [True, False])
def test_max_offload_tokens_none_treated_as_unset(
    request_runner, async_scheduling: bool
):
    """None means no cap: it must not raise and all 3 offloaded blocks are stored.

    Verifies both that None is accepted as a valid value (not rejected by type
    validation) and that it imposes no limit — all 3 offloaded blocks
    (9 GPU block offsets) are offloaded.
    """
    r = request_runner(
        block_size=_GPU_BLOCK_SIZE,
        num_gpu_blocks=_NUM_GPU_BLOCKS,
        async_scheduling=async_scheduling,
        block_size_factor=_BLOCK_SIZE_FACTOR,
    )
    _setup_request(r, None)
    r.run(decoded_tokens=[EOS_TOKEN_ID], expected_stored=(0, 1, 2, 3, 4, 5, 6, 7, 8))


@pytest.mark.parametrize("async_scheduling", [True, False])
def test_max_offload_tokens_string_ignored(request_runner, async_scheduling: bool):
    """String value should warn and fall back to None (no cap)."""
    r = request_runner(
        block_size=_GPU_BLOCK_SIZE,
        num_gpu_blocks=_NUM_GPU_BLOCKS,
        async_scheduling=async_scheduling,
        block_size_factor=_BLOCK_SIZE_FACTOR,
    )
    _setup_request(r, "24")
    r.run(decoded_tokens=[EOS_TOKEN_ID], expected_stored=(0, 1, 2, 3, 4, 5, 6, 7, 8))


@pytest.mark.parametrize("async_scheduling", [True, False])
def test_max_offload_tokens_float_ignored(request_runner, async_scheduling: bool):
    """Float value should warn and fall back to None (no cap)."""
    r = request_runner(
        block_size=_GPU_BLOCK_SIZE,
        num_gpu_blocks=_NUM_GPU_BLOCKS,
        async_scheduling=async_scheduling,
        block_size_factor=_BLOCK_SIZE_FACTOR,
    )
    _setup_request(r, 24.5)
    r.run(decoded_tokens=[EOS_TOKEN_ID], expected_stored=(0, 1, 2, 3, 4, 5, 6, 7, 8))


@pytest.mark.parametrize("async_scheduling", [True, False])
def test_max_offload_tokens_bool_accepted_as_int(
    request_runner, async_scheduling: bool
):
    """bool is a subclass of int in Python so True/False pass type validation."""
    r = request_runner(
        block_size=_GPU_BLOCK_SIZE,
        num_gpu_blocks=_NUM_GPU_BLOCKS,
        async_scheduling=async_scheduling,
        block_size_factor=_BLOCK_SIZE_FACTOR,
    )
    _setup_request(r, True)  # True == 1; treated as max_offload_tokens=1
    r.run(decoded_tokens=[EOS_TOKEN_ID])


@pytest.mark.parametrize("async_scheduling", [True, False])
def test_max_offload_tokens_negative_ignored(request_runner, async_scheduling: bool):
    """Negative value should warn and fall back to None (no cap)."""
    r = request_runner(
        block_size=_GPU_BLOCK_SIZE,
        num_gpu_blocks=_NUM_GPU_BLOCKS,
        async_scheduling=async_scheduling,
        block_size_factor=_BLOCK_SIZE_FACTOR,
    )
    _setup_request(r, -1)
    r.run(decoded_tokens=[EOS_TOKEN_ID], expected_stored=(0, 1, 2, 3, 4, 5, 6, 7, 8))


@pytest.mark.parametrize("async_scheduling", [True, False])
def test_max_offload_tokens_zero_stores_nothing(request_runner, async_scheduling: bool):
    """Zero should be valid and result in no blocks being offloaded."""
    r = request_runner(
        block_size=_GPU_BLOCK_SIZE,
        num_gpu_blocks=_NUM_GPU_BLOCKS,
        async_scheduling=async_scheduling,
        block_size_factor=_BLOCK_SIZE_FACTOR,
    )
    _setup_request(r, 0)
    r.run(decoded_tokens=[EOS_TOKEN_ID], expected_stored=())


@pytest.mark.parametrize("async_scheduling", [True, False])
def test_max_offload_tokens_limits_stored_tokens(
    request_runner, async_scheduling: bool
):
    """A valid int cap is accepted and limits how many blocks are offloaded.

    With 3 offloaded blocks available (36 tokens total), capping at 24 tokens
    (= 2 offloaded blocks × 3 GPU blocks each = GPU offsets 0–5) leaves the
    third offloaded block (GPU offsets 6–8) un-offloaded.
    """
    r = request_runner(
        block_size=_GPU_BLOCK_SIZE,
        num_gpu_blocks=_NUM_GPU_BLOCKS,
        async_scheduling=async_scheduling,
        block_size_factor=_BLOCK_SIZE_FACTOR,
    )
    _setup_request(r, 24)
    # 2 offloaded blocks × 3 GPU blocks each = offsets 0–5 stored;
    # offloaded block 2 (offsets 6–8) is withheld because it exceeds the 24-token cap.
    r.run(decoded_tokens=[EOS_TOKEN_ID], expected_stored=(0, 1, 2, 3, 4, 5))
