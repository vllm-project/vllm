# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""pytest configuration for OBJ tier performance tests."""

import pytest

from .utils import S3_ENV_VARS, make_tier_with_buffer, s3_config_available, unique_key


def pytest_addoption(parser):
    """Register CLI options for OBJ tier performance tests."""
    parser.addoption(
        "--num-blocks",
        type=int,
        default=32,
        help="Number of KV cache blocks in the primary buffer (default: 32).",
    )
    parser.addoption(
        "--elements-per-block",
        type=int,
        default=4096,
        help=(
            "float32 elements per block, i.e. block size in bytes = "
            "elements × 4. Default: 4096 (16 KB per block)."
        ),
    )
    parser.addoption(
        "--num-passes",
        type=int,
        default=5,
        help="Number of load passes after the initial cold store (throughput test).",
    )
    parser.addoption(
        "--batch-size",
        type=int,
        default=8,
        help="Concurrent jobs per iteration (stress test).",
    )
    parser.addoption(
        "--num-iterations",
        type=int,
        default=3,
        help="Iterations per hot_ratio (stress test).",
    )
    parser.addoption(
        "--num-repeats",
        type=int,
        default=1,
        help=(
            "Number of full hot_ratio sweeps (stress test). "
            "Use >=2 to detect drift across repeated passes."
        ),
    )


@pytest.fixture(autouse=True)
def _skip_without_s3():
    """Skip all tests in this package if S3 env vars are not set."""
    if not s3_config_available():
        missing = [v for v in S3_ENV_VARS if not __import__("os").environ.get(v)]
        pytest.skip(f"S3 env vars not set: {missing}")


@pytest.fixture
def num_blocks(request) -> int:
    return request.config.getoption("--num-blocks")


@pytest.fixture
def elements_per_block(request) -> int:
    return request.config.getoption("--elements-per-block")


@pytest.fixture
def num_passes(request) -> int:
    return request.config.getoption("--num-passes")


@pytest.fixture
def batch_size(request) -> int:
    return request.config.getoption("--batch-size")


@pytest.fixture
def num_iterations(request) -> int:
    return request.config.getoption("--num-iterations")


@pytest.fixture
def num_repeats(request) -> int:
    return request.config.getoption("--num-repeats")


@pytest.fixture
def perf_tier(num_blocks, elements_per_block):
    """
    ObjSecondaryTier with a registered primary buffer.
    Yields (tier, tensor, pre_stored_keys) where pre_stored_keys is a list
    of `num_blocks` keys already written to S3 (for load-side tests).
    """
    import uuid
    import time
    from .utils import make_obj_tier, make_job, drain

    prefix = f"perf/{uuid.uuid4().hex[:8]}"
    tier, tensor = make_tier_with_buffer(
        num_blocks=num_blocks,
        elements_per_block=elements_per_block,
        key_prefix=prefix,
    )

    # Pre-populate S3 so load tests have data to read.
    import torch
    for bid in range(num_blocks):
        tensor[bid] = torch.rand((elements_per_block,), dtype=torch.float32)

    keys = [unique_key(i) for i in range(num_blocks)]
    tier.submit_store(make_job(0, keys, list(range(num_blocks))))
    drain(tier)

    yield tier, tensor, keys
    tier.shutdown()
