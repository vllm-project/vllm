# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in disable of EAGLE last-block drop on prefix-cache hits for always-K=0 DSD."""

import pytest

from tests.v1.core.test_prefix_caching import (
    make_kv_cache_config,
    make_kv_cache_config_hybrid_model,
    make_kv_cache_manager,
    make_request,
)
from tests.v1.core.utils import create_scheduler
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler, compute_drop_eagle_on_cache_hit
from vllm.v1.structured_output import StructuredOutputManager

pytestmark = pytest.mark.cpu_test

ALWAYS_K0 = [(1, 16, 0)]
MIXED_K = [(1, 2, 2), (3, 16, 0)]


@pytest.fixture(autouse=True)
def _auto_init_hash_fn():
    init_none_hash(sha256)


def _k0_lookup() -> list[int]:
    # Index 0 is unused; remaining entries are K for batch sizes 1..N.
    return [0, 0, 0, 0]


def _mixed_lookup() -> list[int]:
    return [0, 2, 2, 0]


@pytest.mark.parametrize(
    (
        "disable_flag",
        "enable_prefix_caching",
        "has_kv_connector",
        "use_eagle",
        "dynamic_sd_lookup",
        "expected",
    ),
    [
        pytest.param(False, True, False, True, _k0_lookup(), True, id="default-off"),
        pytest.param(
            True, True, False, True, _k0_lookup(), False, id="opt-in-always-k0"
        ),
        pytest.param(
            True, True, False, True, _mixed_lookup(), True, id="mixed-schedule"
        ),
        pytest.param(True, True, False, True, None, True, id="no-lookup"),
        pytest.param(True, True, False, True, [0], True, id="lookup-index0-only"),
        pytest.param(True, True, False, False, _k0_lookup(), True, id="not-eagle"),
        pytest.param(
            True, False, False, True, _k0_lookup(), True, id="no-prefix-cache"
        ),
        pytest.param(True, True, True, True, _k0_lookup(), True, id="kv-connector"),
    ],
)
def test_compute_drop_eagle_on_cache_hit(
    disable_flag: bool,
    enable_prefix_caching: bool,
    has_kv_connector: bool,
    use_eagle: bool,
    dynamic_sd_lookup: list[int] | None,
    expected: bool,
):
    assert (
        compute_drop_eagle_on_cache_hit(
            disable_eagle_cache_drop_for_k0=disable_flag,
            enable_prefix_caching=enable_prefix_caching,
            has_kv_connector=has_kv_connector,
            use_eagle=use_eagle,
            dynamic_sd_lookup=dynamic_sd_lookup,
        )
        is expected
    )


def _make_eagle_dsd_scheduler(
    schedule: list[tuple[int, int, int]] | None,
    *,
    disable_eagle_cache_drop_for_k0: bool = False,
    enable_prefix_caching: bool = True,
    use_kv_connector: bool = False,
    num_speculative_tokens: int | None = 3,
) -> Scheduler:
    """Build a scheduler with EAGLE/MTP `use_eagle()` and an optional DSD table.

    `create_scheduler` defaults to ngram, which is not EAGLE. Reconstruct after
    flipping `method` so the gate sees `use_eagle=True` without loading a draft
    model.
    """
    kwargs: dict = dict(
        enable_prefix_caching=enable_prefix_caching,
        use_kv_connector=use_kv_connector,
        disable_eagle_cache_drop_for_k0=disable_eagle_cache_drop_for_k0,
    )
    if num_speculative_tokens is not None:
        kwargs["num_speculative_tokens"] = num_speculative_tokens
        if schedule is not None:
            kwargs["num_speculative_tokens_per_batch_size"] = schedule
    base = create_scheduler(**kwargs)
    spec = base.vllm_config.speculative_config
    if spec is not None:
        spec.method = "mtp"
    return Scheduler(
        vllm_config=base.vllm_config,
        kv_cache_config=base.kv_cache_config,
        block_size=base.block_size,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(base.vllm_config),
    )


def test_scheduler_default_keeps_eagle_cache_drop():
    scheduler = _make_eagle_dsd_scheduler(ALWAYS_K0)
    assert scheduler.use_eagle is True
    assert scheduler.drop_eagle_on_cache_hit is True
    assert scheduler.kv_cache_manager.drop_eagle_on_cache_hit is True
    assert scheduler.kv_cache_manager.coordinator.drop_eagle_on_cache_hit is True


def test_scheduler_opt_in_always_k0_disables_eagle_cache_drop():
    scheduler = _make_eagle_dsd_scheduler(
        ALWAYS_K0, disable_eagle_cache_drop_for_k0=True
    )
    assert scheduler.use_eagle is True
    assert scheduler.dynamic_sd_lookup is not None
    assert max(scheduler.dynamic_sd_lookup[1:]) == 0
    assert scheduler.drop_eagle_on_cache_hit is False
    assert scheduler.kv_cache_manager.drop_eagle_on_cache_hit is False
    assert scheduler.kv_cache_manager.coordinator.drop_eagle_on_cache_hit is False


def test_scheduler_mixed_dsd_keeps_eagle_cache_drop():
    scheduler = _make_eagle_dsd_scheduler(MIXED_K, disable_eagle_cache_drop_for_k0=True)
    assert scheduler.use_eagle is True
    assert scheduler.drop_eagle_on_cache_hit is True


def test_scheduler_nospec_keeps_eagle_cache_drop():
    scheduler = create_scheduler(enable_prefix_caching=True)
    assert scheduler.use_eagle is False
    assert scheduler.drop_eagle_on_cache_hit is True


def test_scheduler_ngram_keeps_eagle_cache_drop():
    scheduler = create_scheduler(
        enable_prefix_caching=True,
        num_speculative_tokens=3,
        num_speculative_tokens_per_batch_size=ALWAYS_K0,
        disable_eagle_cache_drop_for_k0=True,
    )
    assert scheduler.use_eagle is False
    assert scheduler.drop_eagle_on_cache_hit is True


def test_scheduler_kv_connector_keeps_eagle_cache_drop():
    scheduler = _make_eagle_dsd_scheduler(
        ALWAYS_K0,
        disable_eagle_cache_drop_for_k0=True,
        use_kv_connector=True,
    )
    assert scheduler.connector is not None
    assert scheduler.use_eagle is True
    assert scheduler.drop_eagle_on_cache_hit is True


def _prime_and_lookup(
    manager, token_ids: list[int], block_size: int
) -> tuple[int, int]:
    req = make_request("prime", token_ids, block_size, sha256)
    computed_blocks, _, _ = manager.get_computed_blocks(req)
    manager.allocate_slots(
        req,
        len(token_ids),
        len(computed_blocks.blocks[0]) * block_size,
        computed_blocks,
    )
    manager.free(req)

    hit_req = make_request("hit", token_ids, block_size, sha256)
    computed_blocks, num_tokens, _ = manager.get_computed_blocks(hit_req)
    return len(computed_blocks.blocks[0]), num_tokens


def test_unitary_cache_hit_keeps_last_block_when_drop_disabled():
    block_size = 16
    token_ids = [0] * (3 * block_size)
    dropped = make_kv_cache_manager(
        make_kv_cache_config(block_size, num_blocks=10),
        max_model_len=8192,
        enable_caching=True,
        use_eagle=True,
        hash_block_size=block_size,
    )
    kept = make_kv_cache_manager(
        make_kv_cache_config(block_size, num_blocks=10),
        max_model_len=8192,
        enable_caching=True,
        use_eagle=True,
        drop_eagle_on_cache_hit=False,
        hash_block_size=block_size,
    )

    n_blocks_drop, n_tokens_drop = _prime_and_lookup(dropped, token_ids, block_size)
    n_blocks_keep, n_tokens_keep = _prime_and_lookup(kept, token_ids, block_size)

    assert n_blocks_drop == 1
    assert n_tokens_drop == 1 * block_size
    assert n_blocks_keep == 2
    assert n_tokens_keep == 2 * block_size


def test_hybrid_cache_hit_keeps_last_block_when_drop_disabled():
    block_size = 16
    kv_cache_config = make_kv_cache_config_hybrid_model(block_size, 31, 3)
    manager = make_kv_cache_manager(
        kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=block_size,
        use_eagle=True,
        drop_eagle_on_cache_hit=False,
    )

    num_full_blocks = 6
    common_token_ids = [i for i in range(num_full_blocks) for _ in range(block_size)]
    req0 = make_request("0", common_token_ids + [6] * 7, block_size, sha256)
    computed_blocks, num_computed_tokens, _ = manager.get_computed_blocks(req0)
    manager.allocate_slots(
        req0, len(req0.all_token_ids), num_computed_tokens, computed_blocks
    )

    req1 = make_request("1", common_token_ids + [6] * 5, block_size, sha256)
    computed_blocks, num_computed_tokens, _ = manager.get_computed_blocks(req1)
    assert num_computed_tokens == num_full_blocks * block_size
    assert len(computed_blocks.blocks[0]) == num_full_blocks
