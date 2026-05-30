# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for NixlConnectorScheduler with HMA and Mamba N-1 prefill."""

import gc
from unittest.mock import patch

import pytest
import torch

from tests.v1.attention.utils import MockMambaBuilder
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    SlidingWindowManager,
)

from .utils import (
    create_request,
    create_vllm_config,
    make_kv_cache_config,
    make_nixl_scheduler,
)


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "swa_enabled,expected_sw_sizes",
    [
        # SWA enabled: FullAttentionSpec (0) + SlidingWindowSpec (2048/16=128)
        (True, [0, 128 + 1]),
        # SWA disabled: only FullAttentionSpec (0)
        (False, [0]),
    ],
)
@patch("vllm.distributed.kv_transfer.kv_connector.v1.nixl.scheduler.current_platform")
def test_sw_sizes(mock_platform, swa_enabled, expected_sw_sizes):
    """Test sw_sizes is correctly computed based on SWA enabled/disabled."""
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.scheduler import (
        NixlConnectorScheduler,
    )

    mock_platform.device_type = "cpu"

    block_size = 16
    vllm_config = create_vllm_config(block_size=block_size)
    # SW 2048 tokens=>128 blocks
    kv_cache_config = make_kv_cache_config(
        block_size=block_size, swa_enabled=swa_enabled, sw_size=2048
    )

    scheduler = NixlConnectorScheduler(
        vllm_config=vllm_config,
        engine_id="test-engine",
        kv_cache_config=kv_cache_config,
    )
    # in number of blocks
    assert scheduler.blocks_per_sw == expected_sw_sizes, (
        f"Expected sw_sizes={expected_sw_sizes}, got {scheduler.blocks_per_sw}"
    )


@pytest.mark.cpu_test
def test_logical_to_kernel_block_ids_with_hma():
    """Test _logical_to_kernel_block_ids expands blocks when HMA is enabled.

    When HMA is enabled, the logical block size may differ from the kernel
    block size. Each logical block maps to multiple kernel blocks.
    """
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker import (
        NixlConnectorWorker,
    )

    # Create a mock worker with just the required attributes
    # (use __new__ to skip __init__)
    worker = object.__new__(NixlConnectorWorker)

    # Simulate HMA scenario: logical block size = 32, kernel block size = 16
    # So each logical block maps to 2 kernel blocks eg [0]->[0,1]
    worker._physical_blocks_per_logical_kv_block = 2
    # FA + SW groups (neither is MambaSpec, so both get expanded)
    worker.kv_cache_config = make_kv_cache_config(block_size=16, swa_enabled=True)

    # Test conversion: FA + SW group
    logical_block_ids = [[0, 1, 2], [3, 4]]
    kernel_block_ids = worker._logical_to_kernel_block_ids(logical_block_ids)

    expected_kernel_block_ids = [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9]]
    assert kernel_block_ids == expected_kernel_block_ids, (
        f"Expected {expected_kernel_block_ids}, got {kernel_block_ids}"
    )


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "group_spec_types,remote_physical_per_logical,"
    "local_physical_per_logical,tp_ratio,remote_block_ids,"
    "expected_remote_block_ids",
    [
        pytest.param(
            ("FullAttentionSpec", "SlidingWindowSpec"),
            2,
            2,
            1,
            ([0, 1, 2], [3, 4]),
            [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9]],
            id="dense_fa_swa",
        ),
        # Nemotron-3-Nano-30B-A3B 4p1d (P_TP=4, D_TP=1):
        # remote_physical_per_logical=34, local_physical_per_logical=66.
        # FA logical block 5 → kernel [170..203], block 6 → [204..237].
        # Mamba block unchanged.
        pytest.param(
            ("FullAttentionSpec", "MambaSpec"),
            34,
            66,
            -4,
            ([5, 6], [2]),
            [list(range(170, 238)), [2]],
            id="mamba_fa_ssm",
        ),
    ],
)
def test_read_blocks_for_req_expands_remote_ids(
    group_spec_types,
    remote_physical_per_logical,
    local_physical_per_logical,
    tp_ratio,
    remote_block_ids,
    expected_remote_block_ids,
):
    """_read_blocks_for_req must expand remote logical block IDs to kernel
    block IDs when kernel block size != logical block size.

    The hot path always calls _logical_to_remote_kernel_block_ids with
    remote_info.remote_physical_blocks_per_logical (model-agnostic).
    """
    from unittest.mock import MagicMock

    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
        NixlConnectorMetadata,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import (
        TPMapping,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker import (
        NixlConnectorWorker,
    )
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        MambaSpec,
        SlidingWindowSpec,
    )

    spec_name_to_type = {
        "FullAttentionSpec": FullAttentionSpec,
        "SlidingWindowSpec": SlidingWindowSpec,
        "MambaSpec": MambaSpec,
    }
    resolved_types = tuple(spec_name_to_type[n] for n in group_spec_types)

    worker = object.__new__(NixlConnectorWorker)
    worker._physical_blocks_per_logical_kv_block = local_physical_per_logical

    has_mamba = any(t is MambaSpec for t in resolved_types)
    has_swa = any(t is SlidingWindowSpec for t in resolved_types)
    worker.kv_cache_config = make_kv_cache_config(
        block_size=16, swa_enabled=has_swa, mamba_enabled=has_mamba
    )

    remote_engine_id = "remote-engine"

    worker.transfer_topo = MagicMock()
    # tp_ratio not exercised (all_source_ranks is empty so no reads run),
    # but set for realism.
    worker.transfer_topo.tp_ratio.return_value = tp_ratio
    remote_info = MagicMock()
    remote_info.remote_physical_blocks_per_logical = remote_physical_per_logical
    worker.transfer_topo.get_engine_info.return_value = remote_info
    worker.use_mla = False

    mock_plan = MagicMock(spec=TPMapping)
    mock_plan.all_source_ranks = ()
    mock_plan.source_ranks_per_group = ()
    worker.tp_mappings = {remote_engine_id: mock_plan}

    metadata = NixlConnectorMetadata()
    metadata.add_new_req_to_recv(
        request_id="test-req",
        local_block_ids=([0, 1], [2, 3]),
        kv_transfer_params={
            "remote_block_ids": remote_block_ids,
            "remote_engine_id": remote_engine_id,
            "remote_request_id": "prefill-test-req",
            "remote_host": "localhost",
            "remote_port": 1234,
            "tp_size": 1,
        },
    )

    meta = metadata.reqs_to_recv["test-req"]
    worker._read_blocks_for_req("test-req", meta)

    assert meta.remote.block_ids == expected_remote_block_ids, (
        f"Expected {expected_remote_block_ids}, got {meta.remote.block_ids}"
    )


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "local_physical_per_logical,remote_physical_per_logical,"
    "local_block_ids,remote_block_ids,"
    "expected_local,expected_remote",
    [
        # 10 kernel blocks of data, local has more logical blocks.
        # remote physical_per_logical=10 → 1 logical → 10 kernel blocks
        # local  physical_per_logical=6  → 2 logical → 12 kernel blocks
        # Trim local from 12 to 10.
        pytest.param(
            6,
            10,
            [list(range(12)), [42]],
            [list(range(10)), [42]],
            [list(range(10)), [42]],
            [list(range(10)), [42]],
            id="align_local6_remote10",
        ),
        # 10 kernel blocks of data, remote has more logical blocks.
        # remote physical_per_logical=6  → 2 logical → 12 kernel blocks
        # local  physical_per_logical=10 → 1 logical → 10 kernel blocks
        # Trim remote from 12 to 10.
        pytest.param(
            10,
            6,
            [list(range(10)), [42]],
            [list(range(12)), [42]],
            [list(range(10)), [42]],
            [list(range(10)), [42]],
            id="align_local10_remote6",
        ),
    ],
)
def test_apply_prefix_caching_mamba_hybrid(
    local_physical_per_logical,
    remote_physical_per_logical,
    local_block_ids,
    remote_block_ids,
    expected_local,
    expected_remote,
):
    """_apply_prefix_caching front-trims FA groups to
    min(local, remote) for Mamba hybrid models with heterogeneous TP.
    """
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker import (
        NixlConnectorWorker,
    )
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

    worker = object.__new__(NixlConnectorWorker)
    worker._has_mamba = True
    worker._physical_blocks_per_logical_kv_block = local_physical_per_logical
    worker._group_spec_types = (FullAttentionSpec, MambaSpec)
    worker.kv_cache_config = make_kv_cache_config(block_size=16, mamba_enabled=True)

    aligned_local, aligned_remote = worker._apply_prefix_caching(
        local_block_ids, remote_block_ids, remote_physical_per_logical
    )

    assert aligned_local == expected_local, (
        f"Expected local {expected_local}, got {aligned_local}"
    )
    assert aligned_remote == expected_remote, (
        f"Expected remote {expected_remote}, got {aligned_remote}"
    )


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "local_physical_per_logical,remote_physical_per_logical,"
    "remote_fa_blocks,local_fa_blocks,ssm_blocks,"
    "correct_remote_fa,correct_local_fa",
    [
        # 10 kernel blocks of data (640 tokens).
        # remote physical_per_logical=10 → 1 logical → 10 kernel [0..9]
        # local  physical_per_logical=6  → 2 logical → 12 kernel [0..11]
        # 1st local logical block cached → suffix [6..11]
        # Correct: transfer only uncached suffix tokens (384-639)
        #   = remote [6,7,8,9] → local [6,7,8,9].
        # Actual (front-trim): remote[:6]=[0..5] → local [6..11]. Wrong.
        pytest.param(
            6,
            10,
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [6, 7, 8, 9, 10, 11],
            [42],
            [6, 7, 8, 9],
            [6, 7, 8, 9],
            id="local6_remote10_fail",
        ),
        # 15 kernel blocks of data (960 tokens).
        # remote physical_per_logical=6  → 3 logical → 18 kernel [0..17]
        # local  physical_per_logical=10 → 2 logical → 20 kernel [0..19]
        # 1st local logical block cached → suffix [10..19]
        # Correct: transfer only uncached suffix tokens (640-959)
        #   = remote [10,11,12,13,14] → local [10,11,12,13,14].
        # Actual (front-trim): remote[:10]=[0..9] → local [10..19]. Wrong.
        pytest.param(
            10,
            6,
            list(range(18)),
            list(range(10, 20)),
            [42],
            [10, 11, 12, 13, 14],
            [10, 11, 12, 13, 14],
            id="local10_remote6_fail",
        ),
    ],
)
def test_mismatched_physical_per_logical_fails_with_prefix_caching(
    local_physical_per_logical,
    remote_physical_per_logical,
    remote_fa_blocks,
    local_fa_blocks,
    ssm_blocks,
    correct_remote_fa,
    correct_local_fa,
):
    """Demonstrate that _apply_prefix_caching front-trims ([:N])
    in the Mamba hybrid path, which fails when prefix caching produces
    suffix-only local blocks.

    Prefix caching operates at logical block granularity. When a logical
    block is cached locally, the decode side only allocates kernel blocks
    for the uncached suffix. The front-trim pairs remote prefix blocks
    with local suffix slots — a silent data corruption.
    """
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker import (
        NixlConnectorWorker,
    )

    worker = object.__new__(NixlConnectorWorker)
    worker._physical_blocks_per_logical_kv_block = local_physical_per_logical
    worker.kv_cache_config = make_kv_cache_config(
        block_size=16,
        mamba_enabled=True,
    )
    worker._has_mamba = True
    worker._group_spec_types = tuple(
        type(g.kv_cache_spec) for g in worker.kv_cache_config.kv_cache_groups
    )

    local_block_ids = (local_fa_blocks, ssm_blocks)
    remote_block_ids = (remote_fa_blocks, ssm_blocks)

    aligned_local, aligned_remote = worker._apply_prefix_caching(
        local_block_ids,
        remote_block_ids,
        remote_physical_per_logical,
    )

    assert (
        aligned_remote[0] != correct_remote_fa or aligned_local[0] != correct_local_fa
    ), (
        f"Prefix caching with mismatched physical_per_logical should not "
        f"produce correct transfer ids: "
        f"remote={aligned_remote[0]}, local={aligned_local[0]}, "
        f"correct_remote={correct_remote_fa}, correct_local={correct_local_fa}"
    )


@pytest.mark.parametrize("model_name, sw_size", [("google/gemma-3-1b-it", 512)])
def test_fewer_blocks_with_hma(monkeypatch, model_name, sw_size):
    """Test that a prefill instance returns fewer "remote blocks" for the SWA groups
    when sequence exceeds the sliding window.
    """
    kv_transfer_config = KVTransferConfig(
        kv_connector="NixlConnector",
        kv_role="kv_both",
    )
    block_size = 16
    llm_kwargs = {
        "model": model_name,
        "enforce_eager": True,
        "gpu_memory_utilization": 0.3,
        "kv_transfer_config": kv_transfer_config,
        "max_model_len": 2048,
        "max_num_seqs": 1,
        # NOTE: Make sure HMA is enabled
        "disable_hybrid_kv_cache_manager": False,
        "max_num_batched_tokens": 2048,
        "enable_prefix_caching": False,
        "block_size": block_size,
    }

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    def run_hma_test(llm: LLM):
        remote_prefill_opts = {
            "do_remote_decode": True,
            "do_remote_prefill": False,
            "remote_engine_id": None,
            "remote_block_ids": None,
            "remote_host": None,
            "remote_port": None,
        }
        # Simulate sidecar request
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            extra_args={"kv_transfer_params": remote_prefill_opts},
        )
        scheduler = llm.llm_engine.engine_core.engine_core.scheduler
        kv_managers = scheduler.kv_cache_manager.coordinator.single_type_managers
        # HMA enabled with FA + SWA groups
        assert len(kv_managers) > 2
        for kv_manager in kv_managers:
            assert isinstance(kv_manager, (SlidingWindowManager, FullAttentionManager))
        req_to_blocks = kv_managers[0].req_to_blocks
        assert len(req_to_blocks) == 0

        # Process some request with length exceeding the sliding window
        outputs = llm.generate(["hi" * 1401], sampling_params)
        kv_params = outputs[0].kv_transfer_params

        # +1 to account for overlapping window across blocks.
        expected_num_remote_blocks = sw_size // block_size + 1
        remote_block_ids = kv_params["remote_block_ids"]
        assert (
            len(remote_block_ids[0])
            == expected_num_remote_blocks
            < len(remote_block_ids[-1])
        )
        for group_block_ids in remote_block_ids[:-1]:
            assert len(group_block_ids) == expected_num_remote_blocks

    def run_test_and_cleanup():
        gc.collect()
        torch.accelerator.empty_cache()
        llm = LLM(**llm_kwargs)
        try:
            run_hma_test(llm)
        finally:
            llm.llm_engine.engine_core.shutdown()

    run_test_and_cleanup()


@pytest.mark.cpu_test
def test_nixl_metadata_hma_block_ids_structure():
    """
    Test that NixlConnectorMetadata correctly stores block IDs for multiple
    KV cache groups when HMA is enabled.
    """
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
        NixlConnectorMetadata,
    )

    metadata = NixlConnectorMetadata()

    # Add request with block IDs for 2 groups (FA + SW)
    fa_blocks = [0, 1, 2, 3, 4, 5, 6, 7]  # 8 blocks for FA
    sw_blocks = [8, 9, 10, 11]  # 4 blocks for SW (clipped)

    metadata.add_new_req_to_recv(
        request_id="test-req-hma",
        local_block_ids=(fa_blocks, sw_blocks),
        kv_transfer_params={
            "remote_block_ids": ([10, 11, 12, 13, 14, 15, 16, 17], [18, 19, 20, 21]),
            "remote_engine_id": "remote-engine",
            "remote_request_id": "prefill-test-req-hma",
            "remote_host": "localhost",
            "remote_port": 1234,
            "tp_size": 1,
        },
    )

    assert "test-req-hma" in metadata.reqs_to_recv
    req_meta = metadata.reqs_to_recv["test-req-hma"]

    # Verify local block IDs structure
    assert len(req_meta.local_block_ids) == 2
    assert list(req_meta.local_block_ids[0]) == fa_blocks
    assert list(req_meta.local_block_ids[1]) == sw_blocks

    # Verify remote block IDs structure
    assert req_meta.remote is not None
    assert len(req_meta.remote.block_ids) == 2
    assert list(req_meta.remote.block_ids[0]) == [10, 11, 12, 13, 14, 15, 16, 17]
    assert list(req_meta.remote.block_ids[1]) == [18, 19, 20, 21]


def _make_mock_worker_for_desc_ids(
    num_regions: int,
    has_mamba: bool,
    group_spec_types: tuple,
    block_len_per_layer: list[int] | None = None,
    ssm_regions_per_layer: int = 0,
):
    """Build a mock NixlConnectorWorker with attrs needed by _compute_desc_ids."""
    from unittest.mock import MagicMock

    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker import (
        NixlConnectorWorker,
    )

    worker = MagicMock(spec=NixlConnectorWorker)
    worker.num_regions = num_regions
    worker._has_mamba = has_mamba
    worker._group_spec_types = group_spec_types
    worker.block_len_per_layer = block_len_per_layer or [100]
    worker._ssm_regions_per_layer = ssm_regions_per_layer
    worker._compute_desc_ids = NixlConnectorWorker._compute_desc_ids.__get__(
        worker, NixlConnectorWorker
    )
    return worker


@pytest.mark.cpu_test
def test_get_block_descs_ids_hybrid_ssm():
    """Test _compute_desc_ids uses per-group strides for hybrid
    FA+SSM when ratio=1 (no kernel block size mismatch)."""
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

    worker = _make_mock_worker_for_desc_ids(
        num_regions=2,
        has_mamba=True,
        group_spec_types=(FullAttentionSpec, MambaSpec),
        block_len_per_layer=[100],
        ssm_regions_per_layer=4,
    )

    fa_blocks = [3, 5]
    ssm_blocks = [1, 2]
    result = worker._compute_desc_ids(
        block_ids=(fa_blocks, ssm_blocks),
        dst_num_blocks=100,
        block_size_ratio=None,
        physical_blocks_per_logical=1,
    )

    expected = [3, 5, 103, 105, 201, 202, 301, 302, 401, 402, 501, 502]
    assert list(result) == expected, f"Expected {expected}, got {list(result)}"


@pytest.mark.cpu_test
def test_get_block_descs_ids_kernel_block_mismatch():
    """Test _compute_desc_ids uses different strides for FA
    (kernel blocks) vs SSM (logical blocks) when ratio > 1."""
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

    ratio = 4
    logical_blocks = 100
    num_blocks = logical_blocks * ratio  # 400 kernel blocks

    worker = _make_mock_worker_for_desc_ids(
        num_regions=2,
        has_mamba=True,
        group_spec_types=(FullAttentionSpec, MambaSpec),
        block_len_per_layer=[100],
        ssm_regions_per_layer=4,
    )

    fa_blocks = [3, 7]
    ssm_blocks = [1, 2]
    result = worker._compute_desc_ids(
        block_ids=(fa_blocks, ssm_blocks),
        dst_num_blocks=num_blocks,
        block_size_ratio=None,
        physical_blocks_per_logical=ratio,
    )

    expected = [3, 7, 403, 407, 801, 802, 901, 902, 1001, 1002, 1101, 1102]
    assert list(result) == expected, f"Expected {expected}, got {list(result)}"


@pytest.mark.cpu_test
def test_nixl_metadata_hybrid_ssm_block_ids():
    """Test NixlConnectorMetadata correctly stores block IDs for FA + SSM
    groups with different block counts (kernel mismatch active)."""
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
        NixlConnectorMetadata,
    )

    metadata = NixlConnectorMetadata()

    # FA: 8 kernel blocks (2 logical * ratio=4), SSM: 2 logical blocks
    fa_blocks = [0, 1, 2, 3, 4, 5, 6, 7]
    ssm_blocks = [0, 1]

    metadata.add_new_req_to_recv(
        request_id="test-req-hybrid",
        local_block_ids=(fa_blocks, ssm_blocks),
        kv_transfer_params={
            "remote_block_ids": ([10, 11, 12, 13, 14, 15, 16, 17], [20, 21]),
            "remote_engine_id": "remote-engine",
            "remote_request_id": "prefill-test-req-hybrid",
            "remote_host": "localhost",
            "remote_port": 1234,
            "tp_size": 1,
        },
    )

    assert "test-req-hybrid" in metadata.reqs_to_recv
    req_meta = metadata.reqs_to_recv["test-req-hybrid"]

    # Verify local block IDs: different lengths per group
    assert len(req_meta.local_block_ids) == 2
    assert list(req_meta.local_block_ids[0]) == fa_blocks
    assert list(req_meta.local_block_ids[1]) == ssm_blocks
    assert len(req_meta.local_block_ids[0]) != len(req_meta.local_block_ids[1])

    # Verify remote block IDs: same asymmetry preserved
    assert req_meta.remote is not None
    assert len(req_meta.remote.block_ids) == 2
    assert list(req_meta.remote.block_ids[0]) == [10, 11, 12, 13, 14, 15, 16, 17]
    assert list(req_meta.remote.block_ids[1]) == [20, 21]
    assert len(req_meta.remote.block_ids[0]) != len(req_meta.remote.block_ids[1])


# ── Mamba N-1 prefill tests ──────────────────────────────────────────────


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "has_mamba,is_hma_required,expected_count",
    [
        (True, True, 9),
        (False, False, 10),
        (False, True, 10),
    ],
    ids=["mamba", "fa_only", "swa_only"],
)
def test_mamba_n1_d_side(has_mamba, is_hma_required, expected_count):
    """D-side: Mamba gets N-1 matched tokens, non-Mamba gets N."""
    sched = make_nixl_scheduler(has_mamba=has_mamba, is_hma_required=is_hma_required)
    req = create_request(num_tokens=10, do_remote_prefill=True)

    count, is_async = sched.get_num_new_matched_tokens(req, num_computed_tokens=0)
    assert count == expected_count
    assert is_async is True


@pytest.mark.cpu_test
def test_mamba_n1_d_side_builds_decode_metadata():
    req = create_request(num_tokens=10, do_remote_prefill=True)
    sched = make_nixl_scheduler(has_mamba=True, is_hma_required=True)

    num_computed_tokens, is_async = sched.get_num_new_matched_tokens(
        req, num_computed_tokens=0
    )

    assert num_computed_tokens == req.num_prompt_tokens - 1
    assert is_async is True

    vllm_config = create_vllm_config()
    metadata = MockMambaBuilder.build_mamba_metadata(
        vllm_config,
        seq_lens=[req.num_prompt_tokens],
        query_lens=[1],
        is_prefilling=[True],
    )

    assert metadata.num_decodes == 1
    assert metadata.num_prefills == 0


@pytest.mark.cpu_test
def test_mamba_n1_p_side_truncation():
    """P-side: Mamba truncates prompt to N-1, sets max_tokens=1.

    Also verifies idempotency (calling again is a no-op) which is
    needed for preemption safety via the _p_side_truncated guard,
    and that non-Mamba models skip truncation entirely.
    """
    sched = make_nixl_scheduler(has_mamba=True, is_hma_required=True)
    req = create_request(num_tokens=10, do_remote_decode=True)
    req.max_tokens = 128
    original_len = len(req.prompt_token_ids)

    count, is_async = sched.get_num_new_matched_tokens(req, num_computed_tokens=0)

    assert count == 0
    assert is_async is False
    assert len(req.prompt_token_ids) == original_len - 1
    assert req.num_prompt_tokens == original_len - 1
    assert req.max_tokens == 1
    assert req.kv_transfer_params["_p_side_truncated"] is True

    # Idempotency: second call must not truncate further
    sched.get_num_new_matched_tokens(req, num_computed_tokens=0)
    assert len(req.prompt_token_ids) == original_len - 1

    # Non-Mamba: truncation is skipped
    fa_sched = make_nixl_scheduler(has_mamba=False, is_hma_required=False)
    fa_req = create_request(num_tokens=10, do_remote_decode=True)
    fa_original = len(fa_req.prompt_token_ids)

    fa_sched.get_num_new_matched_tokens(fa_req, num_computed_tokens=0)
    assert len(fa_req.prompt_token_ids) == fa_original


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "swa_enabled,mamba_enabled,expected_has_mamba,expected_is_hma",
    [
        (True, True, True, True),
        (True, False, False, True),
        (False, False, False, False),
    ],
    ids=["fa_swa_mamba", "fa_swa_only", "fa_only"],
)
@patch("vllm.distributed.kv_transfer.kv_connector.v1.nixl.scheduler.current_platform")
def test_has_mamba_init(
    mock_platform,
    swa_enabled,
    mamba_enabled,
    expected_has_mamba,
    expected_is_hma,
):
    """Test _has_mamba / _is_hma_required derived from kv_cache_groups."""
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.scheduler import (
        NixlConnectorScheduler,
    )

    mock_platform.device_type = "cpu"

    block_size = 16
    vllm_config = create_vllm_config(block_size=block_size)
    # Explicitly enable HMA so we can test the scheduler's own derivation.
    vllm_config.scheduler_config.disable_hybrid_kv_cache_manager = False
    kv_cache_config = make_kv_cache_config(
        block_size=block_size,
        swa_enabled=swa_enabled,
        mamba_enabled=mamba_enabled,
    )

    scheduler = NixlConnectorScheduler(
        vllm_config=vllm_config,
        engine_id="test-engine",
        kv_cache_config=kv_cache_config,
    )
    assert scheduler._has_mamba is expected_has_mamba
    assert scheduler._is_hma_required is expected_is_hma


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "ssm_sizes,block_len,expected_ratio",
    [
        # Nemotron 30B TP=1: ceil((36864 + 2097152) / 8192) = 261
        ((36864, 2097152), 8192, 261),
        # Nemotron 30B TP=2: ceil((18432 + 1048576) / 4096) = 261
        ((18432, 1048576), 4096, 261),
        # Nemotron 30B TP=4: ceil((9216 + 524288) / 4096) = 131
        ((9216, 524288), 4096, 131),
    ],
)
def test_compute_physical_blocks_per_logical(ssm_sizes, block_len, expected_ratio):
    """Verify that compute_physical_blocks_per_logical is TP-dependent.

    With dimension-sharded Mamba state, the ratio differs across TP sizes
    (e.g. TP=1 → 261, TP=4 → 131 for Nemotron 30B). This is why
    _physical_blocks_per_logical must be stored per-engine.
    """
    from vllm.distributed.kv_transfer.kv_connector.v1.ssm_conv_transfer_utils import (
        compute_physical_blocks_per_logical,
    )

    assert compute_physical_blocks_per_logical(ssm_sizes, block_len) == expected_ratio


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "mamba_type,local_tp,conv_dim_local,conv_rows,temporal_shape,expected_proj_dims",
    [
        # nvidia/Nemotron-H-8B-Base-8K (Mamba2)
        # mamba_num_heads=128, head_dim=64, n_groups=8, ssm_state_size=128
        pytest.param(
            "mamba2",
            1,
            10240,
            3,
            (128, 64, 128),
            (8192, 1024, 1024),
            id="nemotron_h_8b_tp1",
        ),
        pytest.param(
            "mamba2",
            4,
            2560,
            3,
            (32, 64, 128),
            (2048, 256, 256),
            id="nemotron_h_8b_tp4",
        ),
        # nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B (Mamba2)
        # mamba_num_heads=64, head_dim=64, n_groups=8, ssm_state_size=128
        pytest.param(
            "mamba2",
            1,
            6144,
            3,
            (64, 64, 128),
            (4096, 1024, 1024),
            id="nemotron_nano_30b_tp1",
        ),
        # Qwen/Qwen3.5-0.8B (GDN, symmetric: num_v=num_k=16)
        # key_dim=2048, value_dim=2048, conv_dim=6144
        pytest.param(
            "gdn_attention",
            1,
            6144,
            3,
            (16, 128, 128),
            (2048, 2048, 2048),
            id="qwen35_08b_tp1",
        ),
        pytest.param(
            "gdn_attention",
            4,
            1536,
            3,
            (4, 128, 128),
            (512, 512, 512),
            id="qwen35_08b_tp4",
        ),
        # Qwen/Qwen3.5-4B (GDN, asymmetric: num_v=32, num_k=16, K:V=1:2)
        # key_dim=2048, value_dim=4096, conv_dim=8192
        pytest.param(
            "gdn_attention",
            1,
            8192,
            3,
            (32, 128, 128),
            (2048, 2048, 4096),
            id="qwen35_4b_tp1",
        ),
        # Qwen/Qwen3.5-27B (GDN, asymmetric: num_v=48, num_k=16, K:V=1:3)
        # key_dim=2048, value_dim=6144, conv_dim=10240
        pytest.param(
            "gdn_attention",
            1,
            10240,
            3,
            (48, 128, 128),
            (2048, 2048, 6144),
            id="qwen35_27b_tp1",
        ),
        pytest.param(
            "gdn_attention",
            8,
            1280,
            3,
            (6, 128, 128),
            (256, 256, 768),
            id="qwen35_27b_tp8",
        ),
    ],
)
def test_derive_mamba_conv_split(
    monkeypatch,
    mamba_type,
    local_tp,
    conv_dim_local,
    conv_rows,
    temporal_shape,
    expected_proj_dims,
):
    """Parametrized test for derive_mamba_conv_split with real model configs.

    Values generated by verify_conv_split.py which loads HuggingFace configs
    and calls vLLM's derive_mamba_conv_split directly.
    """
    from vllm.distributed.kv_transfer.kv_connector.v1.ssm_conv_transfer_utils import (
        derive_mamba_conv_split,
    )
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
    from vllm.v1.kv_cache_interface import MambaSpec

    _TYPE_MAP = {
        "mamba2": MambaAttentionBackendEnum.MAMBA2,
        "gdn_attention": MambaAttentionBackendEnum.GDN_ATTN,
    }
    mamba_type_enum = _TYPE_MAP[mamba_type]

    monkeypatch.setenv("VLLM_SSM_CONV_STATE_LAYOUT", "DS")
    spec = MambaSpec(
        block_size=64,
        shapes=((conv_dim_local, conv_rows), temporal_shape),
        dtypes=(torch.bfloat16, torch.bfloat16),
        mamba_type=mamba_type_enum,
    )
    out = derive_mamba_conv_split(spec, local_tp=local_tp)
    assert out.local_proj_dims == expected_proj_dims
    assert out.conv_rows == conv_rows


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "mamba_enabled,swa_enabled,"
    "local_physical_per_logical,remote_physical_per_logical,"
    "logical_block_ids,expected_kernel_block_ids",
    [
        # Qwen3.5-0.8B 4P2D (kernel_block_size=64):
        #   prefill TP=4: logical_block_size=384 → physical_per_logical=6
        #   decode  TP=2: logical_block_size=640 → physical_per_logical=10
        # FA logical [0] → remote kernel [0..9] (1 * 10)
        # SSM logical [10] → unchanged [10]
        pytest.param(
            True,
            False,
            6,
            10,
            ([0], [10]),
            [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10]],
            id="qwen35_4p2d",
        ),
        # Qwen3.5-0.8B 2P4D (kernel_block_size=64):
        #   prefill TP=2: logical_block_size=640 → physical_per_logical=10
        #   decode  TP=4: logical_block_size=384 → physical_per_logical=6
        # FA logical [0, 1] → remote kernel [0..5, 6..11] (2 * 6)
        # SSM logical [10] → unchanged [10]
        pytest.param(
            True,
            False,
            10,
            6,
            ([0, 1], [10]),
            [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [10]],
            id="qwen35_2p4d",
        ),
        # Homogeneous TP (kernel_block_size=64):
        #   both sides: logical_block_size=640 → physical_per_logical=10
        # FA logical [0] → kernel [0..9], SSM unchanged
        pytest.param(
            True,
            False,
            10,
            10,
            ([0], [10]),
            [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10]],
            id="homo_tp",
        ),
        # remote physical_per_logical=1: early return, no expansion
        pytest.param(
            True,
            False,
            10,
            1,
            ([0, 1, 2], [5]),
            [[0, 1, 2], [5]],
            id="mamba_remote_physical_per_logical_1",
        ),
        # Pure FA (no mamba): single group expanded with remote stride
        pytest.param(
            False,
            False,
            2,
            4,
            ([0, 1],),
            [[0, 1, 2, 3, 4, 5, 6, 7]],
            id="pure_fa",
        ),
        # FA + SWA (no mamba): both groups expanded
        pytest.param(
            False,
            True,
            2,
            3,
            ([0, 1], [2, 3]),
            [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]],
            id="fa_swa",
        ),
    ],
)
def test_logical_to_remote_kernel_block_ids(
    mamba_enabled,
    swa_enabled,
    local_physical_per_logical,
    remote_physical_per_logical,
    logical_block_ids,
    expected_kernel_block_ids,
):
    """Verify _logical_to_remote_kernel_block_ids uses the remote
    physical_per_logical for FA expansion, not the local one.

    This was the root cause of silent accuracy corruption in Qwen3.5
    heterogeneous TP (e.g. 4P2D): the old code used local physical_per_logical
    for the expansion arange, producing wrong kernel block indices.

    Qwen3.5-0.8B values verified by verify_conv_split.py (issue #13).
    """
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker import (
        NixlConnectorWorker,
    )

    worker = object.__new__(NixlConnectorWorker)
    worker._physical_blocks_per_logical_kv_block = local_physical_per_logical
    worker.kv_cache_config = make_kv_cache_config(
        block_size=16,
        mamba_enabled=mamba_enabled,
        swa_enabled=swa_enabled,
    )

    result = worker._logical_to_remote_kernel_block_ids(
        logical_block_ids,
        remote_physical_per_logical,
    )
    assert list(result) == expected_kernel_block_ids, (
        f"Expected {expected_kernel_block_ids}, got {result}"
    )


# ---- KDA (Kimi Delta Attention) state offset tests ----


@pytest.mark.cpu_test
def test_kda_state_split_info_local_offsets():
    """KDAStateSplitInfo.local_conv_offsets returns cumulative offsets for
    4 contiguous states within a page."""
    from vllm.distributed.kv_transfer.kv_connector.v1.ssm_conv_transfer_utils import (
        KDAStateSplitInfo,
    )

    info = KDAStateSplitInfo(state_sizes=(100, 200, 150, 512))
    offsets = info.local_conv_offsets
    assert offsets == [
        (0, 100),
        (100, 200),
        (300, 150),
        (450, 512),
    ]


@pytest.mark.cpu_test
def test_kda_state_split_info_remote_offsets_tp_positive():
    """tp_ratio >= 1 (D_TP >= P_TP): D rank reads its slice from a larger
    P page."""
    from vllm.distributed.kv_transfer.kv_connector.v1.ssm_conv_transfer_utils import (
        KDAStateSplitInfo,
    )

    # 4 states with known sizes. tp_ratio=2 means P has 2x the heads per rank.
    info = KDAStateSplitInfo(state_sizes=(100, 200, 150, 512))
    # D rank 0 reads its slice
    offsets_r0 = info.remote_state_offsets(local_rank_offset=0, tp_ratio=2)
    # Each state in P page is 2x local size. D rank 0 reads from offset 0.
    assert offsets_r0 == [
        (0, 100),
        (200, 200),
        (600, 150),
        (900, 512),
    ]

    # D rank 1 reads its slice (offset by 1 local-size within each region)
    offsets_r1 = info.remote_state_offsets(local_rank_offset=1, tp_ratio=2)
    assert offsets_r1 == [
        (100, 100),
        (400, 200),
        (750, 150),
        (1412, 512),
    ]


@pytest.mark.cpu_test
def test_kda_state_split_info_remote_offsets_tp_negative():
    """tp_ratio < 0 (P_TP > D_TP): P pages are smaller, D reads entire
    P page with scaled-down sizes."""
    from vllm.distributed.kv_transfer.kv_connector.v1.ssm_conv_transfer_utils import (
        KDAStateSplitInfo,
    )

    # D has 4x the state per rank. P pages are 4x smaller.
    info = KDAStateSplitInfo(state_sizes=(400, 800, 600, 2048))
    offsets = info.remote_state_offsets(local_rank_offset=0, tp_ratio=-4)
    # Each state size is divided by |tp_ratio|=4
    assert offsets == [
        (0, 100),
        (100, 200),
        (300, 150),
        (450, 512),
    ]


# ---- MambaConvSplitInfo additional tests ----


@pytest.mark.cpu_test
def test_mamba_conv_split_info_local_conv_offsets():
    """MambaConvSplitInfo.local_conv_offsets returns cumulative offsets for
    3 conv sub-projections within a page."""
    from vllm.distributed.kv_transfer.kv_connector.v1.ssm_conv_transfer_utils import (
        MambaConvSplitInfo,
    )

    info = MambaConvSplitInfo(
        conv_rows=3,
        local_proj_dims=(10, 5, 5),
        conv_dtype_size=2,
        ssm_sizes=(120, 200),
    )
    # proj_bytes: (10*3*2=60, 5*3*2=30, 5*3*2=30)
    offsets = info.local_conv_offsets
    assert offsets == [(0, 60), (60, 30), (90, 30)]


@pytest.mark.cpu_test
def test_mamba_conv_split_info_state_sizes_alias():
    """state_sizes is an alias for ssm_sizes."""
    from vllm.distributed.kv_transfer.kv_connector.v1.ssm_conv_transfer_utils import (
        MambaConvSplitInfo,
    )

    info = MambaConvSplitInfo(
        conv_rows=3,
        local_proj_dims=(10, 5, 5),
        conv_dtype_size=2,
        ssm_sizes=(120, 200),
    )
    assert info.state_sizes == (120, 200)
    assert info.state_sizes == info.ssm_sizes


@pytest.mark.cpu_test
def test_mamba_conv_split_info_remote_conv_offsets_tp_negative():
    """tp_ratio < 0 (P_TP > D_TP): P pages are smaller, D reads entire
    P page with scaled-down sub-projection sizes."""
    from vllm.distributed.kv_transfer.kv_connector.v1.ssm_conv_transfer_utils import (
        MambaConvSplitInfo,
    )

    # D has 2x the state per rank. local_proj_dims=(20, 10, 10), dtype_size=2.
    # proj_bytes: (20*3*2=120, 10*3*2=60, 10*3*2=60)
    info = MambaConvSplitInfo(
        conv_rows=3,
        local_proj_dims=(20, 10, 10),
        conv_dtype_size=2,
        ssm_sizes=(240, 400),
    )
    # |tp_ratio|=2: each sub-proj scaled down by 2 → (60, 30, 30)
    offsets = info.remote_conv_offsets(local_rank_offset=0, tp_ratio=-2)
    assert offsets == [(0, 60), (60, 30), (90, 30)]


# ---- KDA dispatch and 4-tuple tests ----


@pytest.mark.cpu_test
def test_derive_mamba_conv_split_kda_dispatch(monkeypatch):
    """derive_mamba_conv_split returns KDAStateSplitInfo for 4 shapes."""
    from vllm.distributed.kv_transfer.kv_connector.v1.ssm_conv_transfer_utils import (
        KDAStateSplitInfo,
        derive_mamba_conv_split,
    )
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
    from vllm.v1.kv_cache_interface import MambaSpec

    monkeypatch.setenv("VLLM_SSM_CONV_STATE_LAYOUT", "DS")

    spec = MambaSpec(
        block_size=64,
        shapes=((100, 3), (100, 3), (100, 3), (256,)),
        dtypes=(torch.bfloat16, torch.bfloat16, torch.bfloat16, torch.float32),
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
    )
    out = derive_mamba_conv_split(spec, local_tp=1)
    assert isinstance(out, KDAStateSplitInfo)
    # conv states: 100*3*2=600 each (bfloat16=2 bytes)
    # recurrent: 256*4=1024 (float32=4 bytes)
    assert out.state_sizes == (600, 600, 600, 1024)


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "ssm_sizes,block_len,expected_ratio",
    [
        pytest.param(
            (600, 600, 600, 1024),
            2824,
            1,
            id="kda_sum_equals_block",
        ),
        pytest.param(
            (600, 600, 600, 1024),
            1024,
            3,
            id="kda_smaller_block",
        ),
    ],
)
def test_compute_physical_blocks_per_logical_kda(ssm_sizes, block_len, expected_ratio):
    """compute_physical_blocks_per_logical works with 4-tuple (KDA) sizes."""
    from vllm.distributed.kv_transfer.kv_connector.v1.ssm_conv_transfer_utils import (
        compute_physical_blocks_per_logical,
    )

    assert compute_physical_blocks_per_logical(ssm_sizes, block_len) == expected_ratio


# ---- _ssm_regions_per_layer logic tests ----


@pytest.mark.cpu_test
def test_ssm_regions_per_layer_both_types():
    """Both KDA and Mamba2/GDN should have 4 SSM regions per layer."""
    from vllm.distributed.kv_transfer.kv_connector.v1.ssm_conv_transfer_utils import (
        KDAStateSplitInfo,
        MambaConvSplitInfo,
    )

    # KDA: 4 states → len(local_conv_offsets) = 4
    kda = KDAStateSplitInfo(state_sizes=(100, 200, 150, 512))
    assert len(kda.local_conv_offsets) == 4

    # Mamba2/GDN: 3 conv sub-projections + 1 SSM temporal = 4
    mamba = MambaConvSplitInfo(
        conv_rows=3,
        local_proj_dims=(10, 5, 5),
        conv_dtype_size=2,
        ssm_sizes=(120, 200),
    )
    assert len(mamba.local_conv_offsets) + 1 == 4


# ---- Worker descriptor building tests ----


def _make_mock_worker_for_desc_build(
    conv_decomp,
    mamba_ssm_size: tuple[int, ...],
    logical_num_blocks: int = 2,
    physical_per_logical: int = 1,
    block_len_per_layer: list[int] | None = None,
    device_id: int = 0,
    tp_rank: int = 0,
):
    """Build a mock NixlConnectorWorker with attrs needed by _build_mamba_*."""

    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker import (
        NixlConnectorWorker,
    )

    worker = object.__new__(NixlConnectorWorker)
    worker._logical_num_blocks = logical_num_blocks
    worker._physical_blocks_per_logical_kv_block = physical_per_logical
    worker.block_len_per_layer = block_len_per_layer or [sum(mamba_ssm_size)]
    worker._conv_decomp = conv_decomp
    worker._mamba_ssm_size = mamba_ssm_size
    worker.device_id = device_id
    worker.tp_rank = tp_rank
    return worker


@pytest.mark.cpu_test
def test_build_mamba_local_mamba2_includes_ssm_temporal():
    """Regression test: _build_mamba_local for Mamba2/GDN must produce 4
    descriptors per block per layer (3 conv sub-projections + 1 SSM temporal).

    This was the GDN regression bug: the SSM temporal state descriptor was
    dropped during the KDA generalization.
    """
    from vllm.distributed.kv_transfer.kv_connector.v1.ssm_conv_transfer_utils import (
        MambaConvSplitInfo,
    )

    # proj_bytes: (10*3*2=60, 5*3*2=30, 5*3*2=30)
    # ssm_sizes: (conv_state=120, ssm_temporal=200)
    decomp = MambaConvSplitInfo(
        conv_rows=3,
        local_proj_dims=(10, 5, 5),
        conv_dtype_size=2,
        ssm_sizes=(120, 200),
    )
    worker = _make_mock_worker_for_desc_build(
        conv_decomp=decomp,
        mamba_ssm_size=(120, 200),
        block_len_per_layer=[320],
    )

    result = worker._build_mamba_local(base_addresses=[1000], block_size_ratio=1)

    # 2 blocks * (3 conv + 1 SSM temporal) = 8 descriptors
    assert len(result) == 8, f"Expected 8 descriptors, got {len(result)}"

    # Block 0: conv sub-projections
    assert result[0] == (1000, 60, 0)
    assert result[1] == (1060, 30, 0)
    assert result[2] == (1090, 30, 0)
    # Block 0: SSM temporal (at conv_size=120 offset)
    assert result[3] == (1120, 200, 0)

    # Block 1: page_stride = 320
    assert result[4] == (1320, 60, 0)
    assert result[5] == (1380, 30, 0)
    assert result[6] == (1410, 30, 0)
    assert result[7] == (1440, 200, 0)


@pytest.mark.cpu_test
def test_build_mamba_local_kda():
    """_build_mamba_local for KDA produces 4 descriptors per block per layer,
    one for each state tensor (conv_q, conv_k, conv_v, recurrent)."""
    from vllm.distributed.kv_transfer.kv_connector.v1.ssm_conv_transfer_utils import (
        KDAStateSplitInfo,
    )

    decomp = KDAStateSplitInfo(state_sizes=(100, 200, 150, 512))
    worker = _make_mock_worker_for_desc_build(
        conv_decomp=decomp,
        mamba_ssm_size=(100, 200, 150, 512),
        block_len_per_layer=[962],
    )

    result = worker._build_mamba_local(base_addresses=[1000], block_size_ratio=1)

    # 2 blocks * 4 states = 8 descriptors
    assert len(result) == 8, f"Expected 8 descriptors, got {len(result)}"

    # Block 0
    assert result[0] == (1000, 100, 0)  # conv_q
    assert result[1] == (1100, 200, 0)  # conv_k
    assert result[2] == (1300, 150, 0)  # conv_v
    assert result[3] == (1450, 512, 0)  # recurrent

    # Block 1: page_stride = 962
    assert result[4] == (1962, 100, 0)
    assert result[5] == (2062, 200, 0)
    assert result[6] == (2262, 150, 0)
    assert result[7] == (2412, 512, 0)


@pytest.mark.cpu_test
def test_build_kda_remote_tp_positive():
    """_build_kda_remote with tp_ratio >= 1: D rank reads its slice from
    a larger P page."""
    from unittest.mock import MagicMock

    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
        NixlAgentMetadata,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.ssm_conv_transfer_utils import (
        KDAStateSplitInfo,
    )

    decomp = KDAStateSplitInfo(state_sizes=(100, 200, 150, 512))
    worker = _make_mock_worker_for_desc_build(
        conv_decomp=decomp,
        mamba_ssm_size=(100, 200, 150, 512),
        tp_rank=1,
    )

    # P has tp_ratio=2 (P page each state is 2x larger)
    meta = MagicMock(spec=NixlAgentMetadata)
    meta.kv_caches_base_addr = [5000]
    meta.num_blocks = 4  # 2 logical blocks (4 // 2)
    meta.block_lens = [1924]  # 200+400+300+1024
    meta.device_id = 7

    transfer_info = MagicMock()
    transfer_info.remote_physical_blocks_per_logical = 2

    result = worker._build_kda_remote(meta, tp_ratio=2, transfer_info=transfer_info)

    # 2 logical blocks * 4 states = 8 descriptors
    assert len(result) == 8

    # tp_ratio=2, effective_ratio=2, local_offset=1%2=1
    # remote_state_offsets(1, 2):
    #   conv_q:  (0+1*100=100, 100)
    #   conv_k:  (200+1*200=400, 200)
    #   conv_v:  (600+1*150=750, 150)
    #   rec:     (900+1*512=1412, 512)
    # page_stride = 1924 * 2 = 3848
    assert result[0] == (5100, 100, 7)
    assert result[1] == (5400, 200, 7)
    assert result[2] == (5750, 150, 7)
    assert result[3] == (6412, 512, 7)

    # Block 1
    assert result[4] == (5000 + 3848 + 100, 100, 7)
    assert result[5] == (5000 + 3848 + 400, 200, 7)
    assert result[6] == (5000 + 3848 + 750, 150, 7)
    assert result[7] == (5000 + 3848 + 1412, 512, 7)


@pytest.mark.cpu_test
def test_build_kda_remote_tp_negative():
    """_build_kda_remote with tp_ratio < 0: P pages are smaller, D reads
    entire P page with scaled-down sizes."""
    from unittest.mock import MagicMock

    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
        NixlAgentMetadata,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.ssm_conv_transfer_utils import (
        KDAStateSplitInfo,
    )

    # D has 4x state per rank
    decomp = KDAStateSplitInfo(state_sizes=(400, 800, 600, 2048))
    worker = _make_mock_worker_for_desc_build(
        conv_decomp=decomp,
        mamba_ssm_size=(400, 800, 600, 2048),
        tp_rank=0,
    )

    meta = MagicMock(spec=NixlAgentMetadata)
    meta.kv_caches_base_addr = [5000]
    meta.num_blocks = 2
    meta.block_lens = [962]  # 100+200+150+512 (P-sized, 1/4 of D)
    meta.device_id = 7

    transfer_info = MagicMock()
    transfer_info.remote_physical_blocks_per_logical = 1

    result = worker._build_kda_remote(meta, tp_ratio=-4, transfer_info=transfer_info)

    # 2 blocks * 4 states = 8 descriptors
    assert len(result) == 8

    # tp_ratio=-4: remote_state_offsets(0, -4) scales down by 4
    # → (100, 200, 150, 512) with cumulative offsets
    assert result[0] == (5000, 100, 7)
    assert result[1] == (5100, 200, 7)
    assert result[2] == (5300, 150, 7)
    assert result[3] == (5450, 512, 7)

    # Block 1: page_stride = 962 * 1 = 962
    assert result[4] == (5962, 100, 7)
    assert result[5] == (6062, 200, 7)
    assert result[6] == (6262, 150, 7)
    assert result[7] == (6412, 512, 7)


@pytest.mark.cpu_test
def test_build_mamba_remote_mamba2_includes_ssm_temporal():
    """Regression test: _build_mamba_remote for Mamba2/GDN must include
    SSM temporal state descriptors after conv sub-projections."""
    from unittest.mock import MagicMock

    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
        NixlAgentMetadata,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.ssm_conv_transfer_utils import (
        MambaConvSplitInfo,
    )

    decomp = MambaConvSplitInfo(
        conv_rows=3,
        local_proj_dims=(10, 5, 5),
        conv_dtype_size=2,
        ssm_sizes=(120, 200),
    )
    worker = _make_mock_worker_for_desc_build(
        conv_decomp=decomp,
        mamba_ssm_size=(120, 200),
        block_len_per_layer=[320],
        tp_rank=0,
    )

    meta = MagicMock(spec=NixlAgentMetadata)
    meta.ssm_sizes = (120, 200)
    meta.kv_caches_base_addr = [1000]
    meta.num_blocks = 2
    meta.block_lens = [320]
    meta.device_id = 0

    transfer_info = MagicMock()
    transfer_info.remote_physical_blocks_per_logical = 1

    result = worker._build_mamba_remote(meta, tp_ratio=1, transfer_info=transfer_info)

    # 2 blocks * (3 conv + 1 SSM temporal) = 8 descriptors
    assert len(result) == 8, f"Expected 8 descriptors, got {len(result)}"

    # Block 0: conv sub-projections
    assert result[0] == (1000, 60, 0)
    assert result[1] == (1060, 30, 0)
    assert result[2] == (1090, 30, 0)
    # Block 0: SSM temporal
    assert result[3] == (1120, 200, 0)

    # Block 1: page_stride = 320
    assert result[4] == (1320, 60, 0)
    assert result[5] == (1380, 30, 0)
    assert result[6] == (1410, 30, 0)
    assert result[7] == (1440, 200, 0)


@pytest.mark.cpu_test
def test_build_mamba_remote_mamba2_tp_negative():
    """_build_mamba_remote for Mamba2/GDN with tp_ratio < 0 uses remote
    SSM sizes and reads scaled-down conv sub-projections."""
    from unittest.mock import MagicMock

    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
        NixlAgentMetadata,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.ssm_conv_transfer_utils import (
        MambaConvSplitInfo,
    )

    # D has 2x state per rank
    decomp = MambaConvSplitInfo(
        conv_rows=3,
        local_proj_dims=(20, 10, 10),
        conv_dtype_size=2,
        ssm_sizes=(240, 400),
    )
    worker = _make_mock_worker_for_desc_build(
        conv_decomp=decomp,
        mamba_ssm_size=(240, 400),
        block_len_per_layer=[640],
        tp_rank=0,
    )

    meta = MagicMock(spec=NixlAgentMetadata)
    meta.ssm_sizes = (120, 200)  # P-sized (half of D)
    meta.kv_caches_base_addr = [1000]
    meta.num_blocks = 2
    meta.block_lens = [320]  # P page size
    meta.device_id = 0

    transfer_info = MagicMock()
    transfer_info.remote_physical_blocks_per_logical = 1

    result = worker._build_mamba_remote(meta, tp_ratio=-2, transfer_info=transfer_info)

    # 2 blocks * (3 conv + 1 SSM temporal) = 8 descriptors
    assert len(result) == 8

    # tp_ratio=-2: conv offsets scaled down by 2 → (60, 30, 30)
    # SSM: uses nixl_agent_meta.ssm_sizes[1]=200 (P-sized)
    # page_stride = 320 * 1 = 320
    assert result[0] == (1000, 60, 0)
    assert result[1] == (1060, 30, 0)
    assert result[2] == (1090, 30, 0)
    # SSM at conv_size_remote=120 offset
    assert result[3] == (1120, 200, 0)

    # Block 1
    assert result[4] == (1320, 60, 0)
    assert result[5] == (1380, 30, 0)
    assert result[6] == (1410, 30, 0)
    assert result[7] == (1440, 200, 0)
