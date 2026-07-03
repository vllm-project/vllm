# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.config.parallel import ParallelConfig
from vllm.platforms import current_platform
from vllm.v1.worker.cpu_worker import _get_cpushm_dist_ident
from vllm.v1.worker.dp_utils import _synchronize_dp_ranks

if not current_platform.is_cpu():
    pytest.skip("CPU-only test", allow_module_level=True)


def _make_parallel_config(dp_port: int) -> ParallelConfig:
    return ParallelConfig(
        tensor_parallel_size=1,
        data_parallel_size=2,
        data_parallel_master_ip="127.0.0.1",
        _data_parallel_master_port_list=[dp_port],
    )


def test_get_cpushm_dist_ident_uses_dp_rendezvous_for_single_node_dp():
    ident_rank0 = _get_cpushm_dist_ident(
        _make_parallel_config(26001),
        "tcp://127.0.0.1:11001",
    )
    ident_rank1 = _get_cpushm_dist_ident(
        _make_parallel_config(26001),
        "tcp://127.0.0.1:11002",
    )

    assert ident_rank0 == ident_rank1 == "127.0.0.1:26001"


def test_get_cpushm_dist_ident_differs_for_different_dp_rendezvous():
    ident_a = _get_cpushm_dist_ident(
        _make_parallel_config(26001),
        "tcp://127.0.0.1:11001",
    )
    ident_b = _get_cpushm_dist_ident(
        _make_parallel_config(26002),
        "tcp://127.0.0.1:11002",
    )

    assert ident_a != ident_b


def test_synchronize_dp_ranks_cpu_ep_moe_does_not_force_padding(monkeypatch):
    parallel_config = ParallelConfig(
        tensor_parallel_size=1,
        data_parallel_size=2,
        data_parallel_rank=0,
        enable_expert_parallel=True,
        is_moe_model=True,
    )

    def fake_run_ar(*args, **kwargs):
        return torch.tensor(
            [
                [3, 5],
                [3, 5],
                [0, 0],
                [0, 0],
            ],
            dtype=torch.int32,
        )

    monkeypatch.setattr("vllm.v1.worker.dp_utils._run_ar", fake_run_ar)

    should_ubatch, num_tokens_after_padding, synced_cudagraph_mode = (
        _synchronize_dp_ranks(
            num_tokens_unpadded=3,
            num_tokens_padded=3,
            should_attempt_ubatching=False,
            cudagraph_mode=0,
            parallel_config=parallel_config,
        )
    )

    assert not should_ubatch
    assert synced_cudagraph_mode == 0
    assert num_tokens_after_padding is not None
    assert num_tokens_after_padding.device.type == "cpu"
    torch.testing.assert_close(
        num_tokens_after_padding,
        torch.tensor([3, 5], dtype=torch.int32),
    )


def test_synchronize_dp_ranks_cpu_ep_moe_pads_small_uniform_decode_batch(
    monkeypatch,
):
    parallel_config = ParallelConfig(
        tensor_parallel_size=1,
        data_parallel_size=2,
        data_parallel_rank=0,
        enable_expert_parallel=True,
        is_moe_model=True,
    )

    def fake_run_ar(*args, **kwargs):
        return torch.tensor(
            [
                [1, 2],
                [1, 2],
                [0, 0],
                [0, 0],
            ],
            dtype=torch.int32,
        )

    monkeypatch.setattr("vllm.v1.worker.dp_utils._run_ar", fake_run_ar)

    should_ubatch, num_tokens_after_padding, synced_cudagraph_mode = (
        _synchronize_dp_ranks(
            num_tokens_unpadded=1,
            num_tokens_padded=1,
            should_attempt_ubatching=False,
            cudagraph_mode=0,
            parallel_config=parallel_config,
        )
    )

    assert not should_ubatch
    assert synced_cudagraph_mode == 0
    assert num_tokens_after_padding is not None
    assert num_tokens_after_padding.device.type == "cpu"
    torch.testing.assert_close(
        num_tokens_after_padding,
        torch.tensor([2, 2], dtype=torch.int32),
    )
