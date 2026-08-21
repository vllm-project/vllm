# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vllm.v1.worker.gpu.model_runner as model_runner_module
from vllm.v1.core.sched.output import CachedRequestData
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.model_runner import GPUModelRunner


def test_update_requests_applies_only_explicit_device_rewind():
    class StagedOffsets:
        def __init__(self):
            self.gpu = np.asarray([19, 18, 7], dtype=np.int32)
            self.pending: list[tuple[int, int]] = []
            self.staged: list[tuple[int, int]] = []
            self.apply_count = 0

        def stage_write_elem(self, index: int, value: int) -> None:
            self.pending.append((index, value))
            self.staged.append((index, value))

        def apply_write(self) -> None:
            self.apply_count += 1
            for index, value in self.pending:
                self.gpu[index] = value
            self.pending.clear()

    offsets = StagedOffsets()
    runner = SimpleNamespace(
        req_states=SimpleNamespace(
            req_id_to_index={"rewound": 0, "spec_rejected": 1, "forward": 2},
            num_computed_tokens_np=np.asarray([19, 19, 3], dtype=np.int32),
            num_computed_tokens=offsets,
            prefill_len=SimpleNamespace(np=np.asarray([96, 96, 96], dtype=np.int32)),
            num_computed_prefill_tokens=np.zeros(3, dtype=np.int32),
        ),
        block_tables=SimpleNamespace(
            append_block_ids=lambda *_args, **_kwargs: pytest.fail(
                "no block append expected"
            )
        ),
    )
    cached = CachedRequestData(
        req_ids=["rewound", "spec_rejected", "forward"],
        resumed_req_ids=set(),
        new_token_ids=[],
        all_token_ids={},
        new_block_ids=[None, None, None],
        num_computed_tokens=[0, 18, 7],
        num_output_tokens=[0, 0, 0],
        rewound_req_ids={"rewound"},
    )
    scheduler_output = SimpleNamespace(
        scheduled_cached_reqs=cached,
        new_block_ids_to_zero=None,
        kv_cache_block_copies=None,
    )

    GPUModelRunner.update_requests(runner, scheduler_output)

    assert runner.req_states.num_computed_tokens_np.tolist() == [0, 18, 7]
    assert offsets.gpu.tolist() == [0, 18, 7]
    assert offsets.staged == [(0, 0)]
    assert offsets.apply_count == 1
    assert runner.req_states.num_computed_prefill_tokens.tolist() == [0, 18, 7]


@pytest.mark.parametrize(
    ("mamba_cache_mode", "num_speculative_blocks", "expected"),
    [
        pytest.param("align", 0, 65_536, id="align-prefix-cache"),
        pytest.param("none", 7, 8, id="no-prefix-cache-with-speculation"),
    ],
)
def test_initialize_kv_cache_does_not_dcp_shard_mamba_block_table(
    monkeypatch,
    mamba_cache_mode: str,
    num_speculative_blocks: int,
    expected: int,
):
    """Mamba/GDN block-table rows index global positions, unlike DCP KV."""

    max_model_len = 1_048_576
    attention_block_size = 1_536
    mamba_block_size = 16
    dcp_size = 8
    full_attention_spec = FullAttentionSpec(
        block_size=attention_block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.bfloat16,
    )
    mamba_spec = MambaSpec(
        shapes=((1,),),
        dtypes=(torch.bfloat16,),
        block_size=mamba_block_size,
        mamba_cache_mode=mamba_cache_mode,
        num_speculative_blocks=num_speculative_blocks,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["attention"], full_attention_spec),
            KVCacheGroupSpec(["kda"], mamba_spec),
        ],
    )
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=dcp_size),
        cache_config=SimpleNamespace(mamba_cache_mode=mamba_cache_mode),
    )
    runner = SimpleNamespace(
        max_model_len=max_model_len,
        is_encoder_decoder=False,
        vllm_config=vllm_config,
    )

    class _CapturedWidths(Exception):
        pass

    captured: list[int] = []

    def capture_width(max_num_blocks: int, *_args, **_kwargs) -> int:
        captured.append(max_num_blocks)
        if len(captured) == 2:
            raise _CapturedWidths
        return max_num_blocks

    monkeypatch.setattr(model_runner_module, "get_block_table_width", capture_width)

    with pytest.raises(_CapturedWidths):
        GPUModelRunner.initialize_kv_cache(runner, kv_cache_config)

    # Attention KV is local to one of eight DCP ranks; KDA state is replicated
    # and therefore needs one table entry for every global 16-token page.
    assert captured == [86, expected]


def test_append_block_ids_rejects_write_past_row_capacity():
    """Reject an oversized staged write before it can corrupt the next row."""

    class _BlockTable:
        gpu = torch.empty((2, 4), dtype=torch.int32)

        def stage_write(self, *_args):
            pytest.fail("an oversized write must not be staged")

    block_tables = BlockTables.__new__(BlockTables)
    block_tables.num_kv_cache_groups = 1
    block_tables.blocks_per_kv_block = [1]
    block_tables.block_tables = [_BlockTable()]
    block_tables.num_blocks = SimpleNamespace(
        np=torch.tensor([[0, 3]], dtype=torch.int32)
    )

    with pytest.raises(
        RuntimeError,
        match=r"request 1, group 0 exceeds row capacity \(5 > 4\)",
    ):
        block_tables.append_block_ids(
            req_index=1,
            new_block_ids=([4, 5],),
            overwrite=False,
        )

    assert block_tables.num_blocks.np[0, 1] == 3
