# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.distributed.kv_transfer.kv_connector.utils import (
    EngineTransferInfo,
    TransferTopology,
)

pytestmark = pytest.mark.cpu_test


class _FakeAttentionBackend:
    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[int, int, int, int]:
        return (num_blocks, num_kv_heads, block_size, 2 * head_size)


def _make_topology(
    *,
    tp_rank: int = 1,
    tp_size: int = 4,
    total_num_kv_heads: int = 8,
) -> TransferTopology:
    return TransferTopology(
        tp_rank=tp_rank,
        tp_size=tp_size,
        block_size=16,
        engine_id="local-engine",
        is_mla=False,
        is_mamba=False,
        total_num_kv_heads=total_num_kv_heads,
        attn_backends=[_FakeAttentionBackend],
    )


def test_legacy_register_remote_engine_uses_pp_rank_zero() -> None:
    topology = _make_topology()
    info = EngineTransferInfo(
        remote_tp_size=2,
        remote_block_len=1024,
        remote_block_size=16,
        remote_physical_blocks_per_logical=1,
    )

    registered = topology.register_remote_engine("remote-engine", info)

    assert registered == info
    assert registered.remote_pp_rank == 0
    assert topology.get_engine_info("remote-engine") == info
    assert topology._engines[("remote-engine", 0)] == info
    assert topology.target_remote_ranks("remote-engine") == [0]


def test_register_remote_engine_stores_pp_ranks_separately() -> None:
    topology = _make_topology(tp_rank=0, tp_size=2)

    info_0 = EngineTransferInfo(
        remote_tp_size=2,
        remote_block_len=1024,
        remote_block_size=16,
        remote_physical_blocks_per_logical=1,
        remote_pp_rank=0,
        start_layer=0,
        end_layer=16,
    )
    info_1 = EngineTransferInfo(
        remote_tp_size=1,
        remote_block_len=512,
        remote_block_size=8,
        remote_physical_blocks_per_logical=2,
        remote_pp_rank=1,
        start_layer=16,
        end_layer=32,
    )

    registered_0 = topology.register_remote_engine("remote-engine", info_0)
    registered_1 = topology.register_remote_engine("remote-engine", info_1)

    assert registered_0 == info_0
    assert registered_1 == info_1
    assert topology.get_engine_info("remote-engine") == info_0
    assert topology.get_engine_info("remote-engine", 0) == info_0
    assert topology.get_engine_info("remote-engine", 1) == info_1
    assert set(topology._engines) == {
        ("remote-engine", 0),
        ("remote-engine", 1),
    }


def test_helpers_use_requested_pp_rank() -> None:
    topology = _make_topology(tp_rank=1, tp_size=2, total_num_kv_heads=2)
    topology.register_remote_engine(
        "remote-engine",
        EngineTransferInfo(
            remote_tp_size=1,
            remote_block_len=1024,
            remote_block_size=16,
            remote_physical_blocks_per_logical=1,
            remote_pp_rank=0,
            start_layer=0,
            end_layer=8,
        ),
    )
    topology.register_remote_engine(
        "remote-engine",
        EngineTransferInfo(
            remote_tp_size=4,
            remote_block_len=1024,
            remote_block_size=16,
            remote_physical_blocks_per_logical=1,
            remote_pp_rank=1,
            start_layer=8,
            end_layer=16,
        ),
    )

    assert not topology.is_kv_replicated("remote-engine", 0)
    assert topology.is_kv_replicated("remote-engine", 1)
    assert topology.replicates_kv_cache("remote-engine", 1)
    assert topology.target_remote_ranks("remote-engine", 0) == [0]
    assert topology.target_remote_ranks("remote-engine", 1) == [2, 3]
    assert "remote_pp=1" in topology.describe("remote-engine", 1)


def test_engine_info_fields_have_backward_compatible_defaults() -> None:
    topology = _make_topology()
    info = EngineTransferInfo(
        remote_tp_size=2,
        remote_block_len=1024,
        remote_block_size=16,
        remote_physical_blocks_per_logical=1,
    )

    registered = topology.register_remote_engine("remote-engine", info)

    assert topology.get_engine_info("remote-engine") == registered
    assert registered.remote_pp_rank == 0
    assert registered.start_layer == 0
    assert registered.end_layer == 0
