# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.config import KVTransferConfig, ParallelConfig, SpeculativeConfig
from vllm.config.kv_transfer import KVRole
from vllm.config.speculative import SpeculativeMethod


def _spec_config(
    *, pp: int, role: KVRole, method: SpeculativeMethod = "dspark"
) -> SpeculativeConfig:
    config = object.__new__(SpeculativeConfig)
    config.method = method
    config.target_parallel_config = ParallelConfig(pipeline_parallel_size=pp)
    config.target_kv_transfer_config = KVTransferConfig(
        kv_connector="NixlConnector",
        kv_role=role,
    )
    return config


@pytest.mark.parametrize(
    ("pp", "role", "method", "expected"),
    [
        (2, "kv_producer", "dspark", True),
        (4, "kv_producer", "dspark", True),
        (1, "kv_producer", "dspark", False),
        (2, "kv_consumer", "dspark", False),
        (2, "kv_both", "dspark", False),
        (2, "kv_producer", "dflash", False),
    ],
)
def test_dspark_prefill_only_role_detection(pp, role, method, expected):
    assert _spec_config(pp=pp, role=role, method=method).is_dspark_prefill_only() is (
        expected
    )


def test_dspark_prefill_materializer_uses_pp1_draft_config():
    target = ParallelConfig(pipeline_parallel_size=4, tensor_parallel_size=1)

    draft = SpeculativeConfig.create_draft_parallel_config(
        target,
        speculative_draft_tensor_parallel_size=1,
        pipeline_parallel_size=1,
    )

    assert draft.pipeline_parallel_size == 1
    assert draft.tensor_parallel_size == 1


def _spec_config_no_kv(
    *, pp: int, method: SpeculativeMethod = "dspark"
) -> SpeculativeConfig:
    config = object.__new__(SpeculativeConfig)
    config.method = method
    config.target_parallel_config = ParallelConfig(pipeline_parallel_size=pp)
    config.target_kv_transfer_config = None  # type: ignore[assignment]
    return config


@pytest.mark.parametrize(
    ("pp", "method", "expected"),
    [
        (2, "dspark", True),
        (4, "dspark", True),
        (1, "dspark", False),
        (2, "dflash", False),
        (2, "mtp", False),
    ],
)
def test_dspark_last_stage_drafter_aggregated(pp, method, expected):
    # Aggregated (IFB) serving: no KV transfer config at all.
    assert _spec_config_no_kv(pp=pp, method=method).use_dspark_last_stage_drafter() is (
        expected
    )


def test_dspark_last_stage_drafter_covers_pd_roles():
    # kv_producer (prefill-only) and kv_consumer/aggregated all take the
    # last-stage drafter path once the target is pipeline-parallel.
    assert (
        _spec_config(pp=2, role="kv_producer").use_dspark_last_stage_drafter() is True
    )
    assert (
        _spec_config(pp=2, role="kv_consumer").use_dspark_last_stage_drafter() is True
    )
