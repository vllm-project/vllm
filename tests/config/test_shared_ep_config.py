# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import cast

import pytest
import torch

from vllm.config import ParallelConfig, SchedulerConfig, VllmConfig
from vllm.model_executor.layers.fused_moe.shared_ep import SharedEPMemory
from vllm.platforms import current_platform


def _shared_ep_parallel_config(**overrides) -> ParallelConfig:
    values = {
        "all2all_backend": "shared_ep",
        "enable_expert_parallel": True,
        "data_parallel_size": 4,
        "data_parallel_size_local": 4,
    }
    values.update(overrides)
    return ParallelConfig(**values)


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"enable_expert_parallel": False}, "enable_expert_parallel"),
        ({"data_parallel_size": 1, "data_parallel_size_local": 1}, "data_parallel"),
        ({"data_parallel_size_local": 2}, "every DP/EP rank to be local"),
        ({"tensor_parallel_size": 2}, "tensor_parallel"),
        ({"prefill_context_parallel_size": 2}, "prefill_context_parallel"),
        ({"nnodes": 2}, "single node"),
        ({"enable_dbo": True}, "ubatching"),
        ({"ubatch_size": 2}, "ubatching"),
        ({"enable_eplb": True}, "EPLB"),
        ({"enable_elastic_ep": True}, "Elastic EP"),
        ({"enable_fault_tolerance": True}, "fault tolerance"),
    ],
)
def test_shared_ep_parallel_admission(overrides, error):
    with pytest.raises(ValueError, match=error):
        _shared_ep_parallel_config(**overrides)


def test_shared_ep_scheduler_admission():
    parallel_config = _shared_ep_parallel_config()
    scheduler_config = SchedulerConfig(
        max_model_len=8192,
        is_encoder_decoder=False,
        max_num_batched_tokens=32,
        max_num_seqs=32,
        enable_chunked_prefill=True,
    )
    config = VllmConfig(
        parallel_config=parallel_config,
        scheduler_config=scheduler_config,
    )
    assert config.scheduler_config.max_num_batched_tokens == 32

    with pytest.raises(ValueError, match="at most 32"):
        VllmConfig(
            parallel_config=parallel_config,
            scheduler_config=SchedulerConfig(
                max_model_len=32,
                is_encoder_decoder=False,
                max_num_batched_tokens=33,
                max_num_seqs=32,
            ),
        )

    with pytest.raises(ValueError, match="chunked prefill"):
        VllmConfig(
            parallel_config=parallel_config,
            scheduler_config=SchedulerConfig(
                max_model_len=32,
                is_encoder_decoder=False,
                max_num_batched_tokens=32,
                max_num_seqs=32,
                enable_chunked_prefill=False,
            ),
        )


def test_shared_ep_allows_offline_same_host_dp(monkeypatch):
    monkeypatch.setenv("VLLM_DP_SIZE", "4")
    monkeypatch.setenv("VLLM_DP_MASTER_IP", "127.0.0.1")
    config = _shared_ep_parallel_config(data_parallel_size_local=1)
    assert config.data_parallel_size == 4


def test_shared_ep_rejects_non_sm100(monkeypatch):
    monkeypatch.setattr(
        current_platform,
        "is_device_capability_family",
        lambda capability: capability != 100,
    )
    with pytest.raises(NotImplementedError, match="SM100"):
        SharedEPMemory.create(
            max_tokens=32,
            hidden_size=4096,
            top_k=8,
            quant_dtype="nvfp4",
            group=cast(torch.distributed.ProcessGroup, None),
            device=torch.device("cuda"),
        )


def test_shared_ep_rejects_non_native_activation(monkeypatch):
    monkeypatch.setattr(
        current_platform,
        "is_device_capability_family",
        lambda capability: capability == 100,
    )
    with pytest.raises(ValueError, match="native NVFP4 or native MXFP8"):
        SharedEPMemory.create(
            max_tokens=32,
            hidden_size=4096,
            top_k=8,
            quant_dtype="fp8",
            group=cast(torch.distributed.ProcessGroup, None),
            device=torch.device("cuda"),
        )
