# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    init_distributed_environment,
)


def test_file_store_preserves_raw_path(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_DISTRIBUTED_USE_SPLIT_GROUP", "0")
    store_path = tmp_path / "store#?"

    try:
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            backend="gloo",
            distributed_init_method=f"file://{store_path}",
        )
        assert store_path.exists()
    finally:
        destroy_distributed_environment()
