# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from types import SimpleNamespace
from vllm.config.mamba import MambaBackendEnum
from vllm.model_executor.layers.mamba.ops.mamba_ssm import (
    _try_get_optimal_ssm_config_cached,
    get_ssm_configs,
)
from vllm.model_executor.warmup import mamba_ssu_autotune
from vllm.model_executor.warmup.triton_autotune import run_config_tuning
SHAPE = mamba_ssu_autotune.SSUShape(
    headdim=64, dstate=16, nheads=8, ngroups=1,
    dtype=torch.bfloat16, state_dtype=torch.float32,
)
from vllm.model_executor.layers.mamba.ops.ssu_tuning import (
    NUM_WARPS_CHOICES,
    SSUTuningCase,
    block_size_m_choices,
    tune_ssu_case,
    valid_request_batches,
    validate_ssu_config,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="GPU")

CASE = SSUTuningCase(
    batch=4, nheads=8, headdim=64, dstate=16, ngroups=1,
    dtype=torch.bfloat16, state_dtype=torch.float32,
    device=torch.device(current_platform.device_type),
)


def test_request_batches_are_clipped_and_include_the_max():
    assert valid_request_batches(20) == [1, 8, 16, 20]


def test_tuned_config_is_a_candidate_and_matches_the_default_output():
    config = tune_ssu_case(CASE, num_iters=3, num_warmup=1)
    assert config is not None
    assert config["BLOCK_SIZE_M"] in block_size_m_choices(CASE.headdim)
    assert config["num_warps"] in NUM_WARPS_CHOICES
    assert validate_ssu_config(CASE, config["BLOCK_SIZE_M"], config["num_warps"])

def test_table_fills_missing_configs_then_has_nothing_left(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_TRITON_AUTOTUNE_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(mamba_ssu_autotune, "discover_ssu_shapes", lambda w: {SHAPE})
    get_ssm_configs.cache_clear()
    _try_get_optimal_ssm_config_cached.cache_clear()
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(
            mamba_config=SimpleNamespace(backend=MambaBackendEnum.TRITON)
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=8),
    )
    table = mamba_ssu_autotune.MambaSSUConfigTable()
    world = SimpleNamespace(rank_in_group=0, world_size=1, cpu_group=None)
    results = run_config_tuning([table], worker, world)
    assert len(results) == 2  # request batches 1 and 8
    assert set(get_ssm_configs(64, 16, "float32")) == {8, 64}  # batch * nheads
    assert table.pending_items(worker) == []