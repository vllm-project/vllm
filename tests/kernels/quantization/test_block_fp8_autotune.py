# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.kernels.linear.scaled_mm.triton import (
    TritonFp8BlockScaledMMKernel,
)
from vllm.model_executor.layers.quantization.utils.fp8_block_tuning import (
    BlockFp8TuningCase,
    search_space,
    tune_block_fp8_case,
    valid_m_buckets,
    validate_block_fp8_config,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    get_w8a8_block_fp8_config_file_name,
    get_w8a8_block_fp8_configs,
    save_block_fp8_configs,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.model_executor.warmup import block_fp8_autotune
from vllm.model_executor.warmup.triton_autotune import run_config_tuning
from vllm.platforms import current_platform

requires_gpu = pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="GPU")

# No bundled config exists for this shape, so lookups only see test files.
_N, _K = 256, 512
_SHAPE = block_fp8_autotune.BlockFp8Shape(_N, _K, 128, 128, torch.bfloat16)
_CONFIG = {
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 1,
    "num_warps": 4,
    "num_stages": 2,
}


def _case(M: int) -> BlockFp8TuningCase:
    return BlockFp8TuningCase(
        M=M,
        N=_N,
        K=_K,
        block_n=128,
        block_k=128,
        out_dtype=torch.bfloat16,
        device=torch.device(current_platform.device_type),
    )


@pytest.fixture(autouse=True)
def isolated_config_dirs(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_TRITON_AUTOTUNE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("VLLM_TUNED_CONFIG_FOLDER", raising=False)
    monkeypatch.delenv("VLLM_TRITON_AUTOTUNE_FORCE", raising=False)
    get_w8a8_block_fp8_configs.cache_clear()
    yield
    get_w8a8_block_fp8_configs.cache_clear()


def test_m_buckets_are_clipped_to_the_token_limit():
    assert valid_m_buckets(100) == [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 100]
    assert valid_m_buckets(8192)[-1] == 4096


def test_search_space_prunes_only_configs_that_cannot_differ():
    small = search_space(_case(1))
    assert len(small) == 64
    assert {c["BLOCK_SIZE_M"] for c in small} == {16}
    assert {c["GROUP_SIZE_M"] for c in small} == {1}
    assert len(search_space(_case(4096))) == 1280


def test_save_then_load_round_trip():
    save_block_fp8_configs(_N, _K, 128, 128, {8: _CONFIG})
    assert get_w8a8_block_fp8_configs(_N, _K, 128, 128) == {8: _CONFIG}


def test_user_folder_beats_autotune_cache(monkeypatch, tmp_path):
    save_block_fp8_configs(_N, _K, 128, 128, {8: _CONFIG})
    user_config = {**_CONFIG, "num_warps": 8}
    file_name = get_w8a8_block_fp8_config_file_name(_N, _K, 128, 128)
    (tmp_path / file_name).write_text(json.dumps({"8": user_config}))
    monkeypatch.setenv("VLLM_TUNED_CONFIG_FOLDER", str(tmp_path))
    get_w8a8_block_fp8_configs.cache_clear()
    assert get_w8a8_block_fp8_configs(_N, _K, 128, 128) == {8: user_config}


def test_no_config_returns_none():
    assert get_w8a8_block_fp8_configs(_N, _K, 128, 128) is None


def test_discovery_finds_only_triton_block_fp8_layers():
    kernel = object.__new__(TritonFp8BlockScaledMMKernel)
    kernel.weight_group_shape = GroupShape(128, 128)
    kernel.config = SimpleNamespace(out_dtype=torch.bfloat16)

    def layer(owner_name: str, fp8_linear: object, n: int, k: int) -> torch.nn.Module:
        module = torch.nn.Module()
        module.weight = torch.nn.Parameter(torch.empty(n, k), requires_grad=False)
        setattr(module, owner_name, SimpleNamespace(fp8_linear=fp8_linear))
        return module

    model = torch.nn.Sequential(
        layer("quant_method", kernel, 256, 512),
        layer("scheme", kernel, 1024, 512),  # compressed-tensors layout
        layer("quant_method", SimpleNamespace(), 2048, 512),  # another kernel
    )
    worker = SimpleNamespace(get_model=lambda: model)
    assert block_fp8_autotune.discover_block_fp8_shapes(worker) == {
        block_fp8_autotune.BlockFp8Shape(256, 512, 128, 128, torch.bfloat16),
        block_fp8_autotune.BlockFp8Shape(1024, 512, 128, 128, torch.bfloat16),
    }


@requires_gpu
def test_tuned_config_is_a_candidate_and_matches_the_default_output():
    case = _case(64)
    candidates = search_space(case)[:8]
    config = tune_block_fp8_case(case, candidates=candidates, num_iters=3)
    assert config in candidates
    assert validate_block_fp8_config(case, config)


@requires_gpu
def test_validation_rejects_a_config_that_misreads_the_scales():
    # BLOCK_SIZE_K larger than the quantization block applies one scale to two
    # blocks, so the output is wrong.
    case = _case(64)
    broken = {**_CONFIG, "BLOCK_SIZE_K": 256}
    assert not validate_block_fp8_config(case, broken)
    assert tune_block_fp8_case(case, candidates=[broken], num_iters=3) is None


@requires_gpu
def test_table_fills_missing_configs_then_has_nothing_left(monkeypatch):
    monkeypatch.setattr(
        block_fp8_autotune, "discover_block_fp8_shapes", lambda worker: {_SHAPE}
    )
    monkeypatch.setattr(
        block_fp8_autotune,
        "tune_block_fp8_case",
        lambda case: tune_block_fp8_case(
            case, candidates=search_space(case)[:4], num_iters=3
        ),
    )
    worker = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16)
    )
    table = block_fp8_autotune.BlockFp8ConfigTable()
    world = SimpleNamespace(rank_in_group=0, world_size=1, cpu_group=None)
    results = run_config_tuning([table], worker, world)
    assert len(results) == 5  # M buckets 1, 2, 4, 8, 16
    assert set(get_w8a8_block_fp8_configs(_N, _K, 128, 128)) == {1, 2, 4, 8, 16}
    assert table.pending_items(worker) == []
