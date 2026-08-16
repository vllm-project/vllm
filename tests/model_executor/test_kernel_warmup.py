# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from vllm.model_executor.layers.mamba.ops import ssu_dispatch
from vllm.model_executor.warmup import kernel_warmup as warmup


@pytest.mark.parametrize(
    "backend_name, expected_calls", [("flashinfer", 1), ("triton", 0)]
)
def test_replayssm_autotune_uses_uniform_decode(
    monkeypatch, backend_name, expected_calls
):
    block_ids = np.zeros((32, 1), dtype=np.int32)
    block_table = SimpleNamespace(block_table=SimpleNamespace(np=block_ids))
    multi_group_block_table = SimpleNamespace(
        block_tables=[block_table], commit_block_table=Mock()
    )

    def dummy_run(**kwargs):
        assert block_ids[:16, 0].tolist() == list(range(1, 17))

    dummy_run = Mock(side_effect=dummy_run)
    runner = SimpleNamespace(
        uniform_decode_query_len=6,
        max_num_tokens=100,
        scheduler_config=SimpleNamespace(max_num_seqs=32),
        input_batch=SimpleNamespace(block_table=multi_group_block_table),
        get_model=lambda: SimpleNamespace(modules=lambda: ()),
        _dummy_run=dummy_run,
    )
    monkeypatch.setattr(
        ssu_dispatch,
        "get_replayssm_backend",
        lambda: SimpleNamespace(name=backend_name),
    )

    warmup._flashinfer_replayssm_autotune_dummy_run(runner)

    assert dummy_run.call_count == expected_calls
    if expected_calls:
        dummy_run.assert_called_once_with(
            num_tokens=96,
            uniform_decode=True,
            allow_microbatching=False,
            skip_eplb=True,
            is_profile=True,
            randomize_inputs=True,
            force_attention=True,
            profile_seq_lens=7,
        )
        assert not block_ids.any()
        multi_group_block_table.commit_block_table.assert_called_once_with(16)


def test_replayssm_autotune_skips_uninitialized_backend(monkeypatch):
    runner = SimpleNamespace(_dummy_run=Mock())

    def get_backend():
        raise RuntimeError

    monkeypatch.setattr(ssu_dispatch, "get_replayssm_backend", get_backend)

    warmup._flashinfer_replayssm_autotune_dummy_run(runner)

    runner._dummy_run.assert_not_called()
