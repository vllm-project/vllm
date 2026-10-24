# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import math
import random
from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.quantization.utils.fp8_utils as fp8_utils
from vllm.model_executor.kernels.linear.scaled_mm.triton import (
    TritonFp8BlockScaledMMKernel,
)
from vllm.model_executor.layers.quantization.utils import fp8_block_tuning
from vllm.model_executor.layers.quantization.utils.fp8_block_tuning import (
    BlockFp8TuningCase,
    TritonBackend,
    default_config,
    normalize_config,
    search_axes,
    starting_configs,
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
from vllm.model_executor.warmup.triton_autotune import TuningItem, run_config_tuning
from vllm.platforms import current_platform

requires_gpu = pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="GPU")

# Backends with and without optional launch parameters.
_WITH_OPTIONS = TritonBackend(
    max_shared_mem=64 * 1024,
    options={
        "waves_per_eu": (0, 1, 2, 4),
        "matrix_instr_nonkdim": (0, 16, 32),
        "kpack": (1, 2),
    },
    option_defaults={"waves_per_eu": 0, "matrix_instr_nonkdim": 0, "kpack": 1},
)
_WITHOUT_OPTIONS = TritonBackend(
    max_shared_mem=227 * 1024,
    options={},
    option_defaults={},
)

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


def _case(M: int, N: int = _N) -> BlockFp8TuningCase:
    return BlockFp8TuningCase(
        M=M,
        N=N,
        K=_K,
        block_n=128,
        block_k=128,
        out_dtype=torch.bfloat16,
        device=torch.device(current_platform.device_type),
    )


def _key(config: dict) -> tuple:
    return tuple(sorted(config.items()))


@pytest.fixture
def bundled_dir(tmp_path):
    path = tmp_path / "bundled"
    path.mkdir()
    return path


@pytest.fixture(autouse=True)
def isolated_config_dirs(monkeypatch, tmp_path, bundled_dir):
    monkeypatch.setattr(fp8_utils, "_BLOCK_FP8_CONFIGS_DIR", str(bundled_dir))
    monkeypatch.setattr(
        fp8_utils, "get_device_name_as_file_name", lambda: "Test_Device"
    )
    monkeypatch.setenv("VLLM_TRITON_AUTOTUNE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("VLLM_TUNED_CONFIG_FOLDER", raising=False)
    monkeypatch.delenv("VLLM_TRITON_AUTOTUNE_FORCE", raising=False)
    get_w8a8_block_fp8_configs.cache_clear()
    yield
    get_w8a8_block_fp8_configs.cache_clear()


def _table_path(folder):
    file_name = get_w8a8_block_fp8_config_file_name(_N, _K, 128, 128)
    return folder / file_name


def _write_table(folder, table: dict[int, dict]) -> None:
    _table_path(folder).write_text(
        json.dumps({str(m): config for m, config in table.items()})
    )


def _use_backend(monkeypatch, backend: TritonBackend):
    monkeypatch.setattr(fp8_block_tuning, "triton_backend", lambda: backend)


def _stub_kernels(monkeypatch, fake_benchmark):
    monkeypatch.setattr(fp8_block_tuning, "make_block_fp8_inputs", lambda case: None)
    monkeypatch.setattr(fp8_block_tuning, "_run", lambda *args: None)
    monkeypatch.setattr(
        fp8_block_tuning, "validate_block_fp8_config", lambda *a, **k: True
    )
    monkeypatch.setattr(fp8_block_tuning, "benchmark_block_fp8_config", fake_benchmark)


def test_m_buckets_are_clipped_to_the_token_limit():
    assert valid_m_buckets(100) == [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 100]
    assert valid_m_buckets(8192)[-1] == 4096
    assert valid_m_buckets(0) == []


_BACKENDS = pytest.mark.parametrize(
    "backend", [_WITH_OPTIONS, _WITHOUT_OPTIONS], ids=["options", "no_options"]
)


@_BACKENDS
def test_search_axes_come_from_the_triton_backend(monkeypatch, backend):
    _use_backend(monkeypatch, backend)
    axes = search_axes(_case(4096))
    option_names = fp8_block_tuning._BACKEND_OPTION_VALUES.keys()
    assert option_names & axes.keys() == backend.options.keys()
    assert all(axes[name] == values for name, values in backend.options.items())


def test_m_tiles_are_limited_by_the_token_count(monkeypatch):
    _use_backend(monkeypatch, _WITH_OPTIONS)
    assert search_axes(_case(1))["BLOCK_SIZE_M"] == (16,)
    assert search_axes(_case(48))["BLOCK_SIZE_M"] == (16, 32, 64)
    assert search_axes(_case(4096))["BLOCK_SIZE_M"] == (16, 32, 64, 128, 256)


def test_only_shared_memory_is_pre_filtered(monkeypatch):
    _use_backend(monkeypatch, _WITH_OPTIONS)
    assert fp8_block_tuning._fits_shared_memory(_CONFIG)
    assert fp8_block_tuning._fits_shared_memory(
        {**_CONFIG, "BLOCK_SIZE_N": 16, "num_warps": 8}
    )
    assert fp8_block_tuning._fits_shared_memory({**_CONFIG, "matrix_instr_nonkdim": 32})
    assert not fp8_block_tuning._fits_shared_memory(
        {**_CONFIG, "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "num_stages": 3}
    )


def test_normalize_rejects_k_tiles_across_quant_blocks_and_merges_group_sizes():
    assert normalize_config({**_CONFIG, "BLOCK_SIZE_K": 256}, _case(64)) is None
    grouped = {**_CONFIG, "GROUP_SIZE_M": 8}
    # 16 tokens fit one 16-row tile, 64 tokens need four.
    assert normalize_config(grouped, _case(16)) == _CONFIG
    assert normalize_config(grouped, _case(64)) == grouped


@_BACKENDS
def test_searched_configs_respect_the_quant_block(monkeypatch, backend):
    _use_backend(monkeypatch, backend)
    case = _case(256)
    axes = search_axes(case)
    configs = fp8_block_tuning._random_configs(axes, case, 20, random.Random(0))
    configs += [n for c in configs for n in fp8_block_tuning._neighbors(c, axes, case)]
    assert len(configs) > 20
    for config in configs:
        assert 128 % config["BLOCK_SIZE_K"] == 0
        assert fp8_block_tuning._fits_shared_memory(config)


def test_starting_configs_are_the_default_then_the_seeds(bundled_dir):
    case = _case(64)
    default = default_config(case)
    seed = {**_CONFIG, "GROUP_SIZE_M": 32, "num_warps": 1}
    # Config files, even a table for this exact shape, are never starting points.
    _write_table(bundled_dir, {64: {**_CONFIG, "num_warps": 2}})
    wide_k = {**seed, "BLOCK_SIZE_K": 256}
    assert starting_configs(case, [seed, default, wide_k]) == [default, seed]


def test_search_beats_every_starting_point(monkeypatch):
    _use_backend(monkeypatch, _WITH_OPTIONS)
    case = _case(256)
    axes = search_axes(case)
    target = {name: values[len(values) // 2] for name, values in axes.items()}

    def fake_time(config):
        values = {**_WITH_OPTIONS.option_defaults, **config}
        return 1.0 + sum(
            abs(math.log2(max(values[name], 1)) - math.log2(max(target[name], 1)))
            for name in axes
        )

    searched = []

    def fake_benchmark(case, config, cutoff_us=None, **kwargs):
        if cutoff_us is not None:
            searched.append(config)
        return fake_time(config)

    _stub_kernels(monkeypatch, fake_benchmark)
    tuned = tune_block_fp8_case(case)
    assert len({_key(c) for c in searched}) == fp8_block_tuning.MAX_CONFIGS_PER_BUCKET
    best_start = min(
        fake_time(c) for c in searched[: fp8_block_tuning._RANDOM_STARTS + 1]
    )
    assert fake_time(fp8_block_tuning._complete(tuned, axes, case)) < best_start


@pytest.mark.parametrize(
    "side_by_side_us,keeps_winner",
    [(1.0, True), (1.95, True), (1.97, False), (3.0, False)],
)
def test_winner_must_clearly_beat_the_default_side_by_side(
    monkeypatch, side_by_side_us, keeps_winner
):
    _use_backend(monkeypatch, _WITHOUT_OPTIONS)
    monkeypatch.setattr(fp8_block_tuning, "_RANDOM_STARTS", 0)
    case = _case(64)
    default = default_config(case)
    other = {**default, "num_warps": 8}
    calls = {_key(default): 0, _key(other): 0}

    def fake_benchmark(case, config, **kwargs):
        calls[_key(config)] += 1
        if config == default:
            return 2.0
        return 1.0 if calls[_key(other)] == 1 else side_by_side_us

    _stub_kernels(monkeypatch, fake_benchmark)
    tuned = tune_block_fp8_case(case, seeds=[other], max_configs=2)
    assert tuned == (other if keeps_winner else default)
    assert calls[_key(other)] == 1 + fp8_block_tuning._HEAD_TO_HEAD_ROUNDS


def test_save_then_load_round_trip():
    path = save_block_fp8_configs(_N, _K, 128, 128, {8: _CONFIG, 1: _CONFIG})
    with open(path) as f:
        assert list(json.load(f)) == ["triton_version", "1", "8"]
    assert get_w8a8_block_fp8_configs(_N, _K, 128, 128) == {1: _CONFIG, 8: _CONFIG}


def test_user_folder_beats_autotune_cache(monkeypatch, tmp_path):
    save_block_fp8_configs(_N, _K, 128, 128, {8: _CONFIG})
    user_config = {**_CONFIG, "num_warps": 8}
    _write_table(tmp_path, {8: user_config})
    monkeypatch.setenv("VLLM_TUNED_CONFIG_FOLDER", str(tmp_path))
    get_w8a8_block_fp8_configs.cache_clear()
    assert get_w8a8_block_fp8_configs(_N, _K, 128, 128) == {8: user_config}


def test_autotune_cache_beats_bundled_config(bundled_dir):
    bundled_config = {**_CONFIG, "num_warps": 8}
    _write_table(bundled_dir, {8: bundled_config})
    assert get_w8a8_block_fp8_configs(_N, _K, 128, 128) == {8: bundled_config}
    save_block_fp8_configs(_N, _K, 128, 128, {8: _CONFIG})
    assert get_w8a8_block_fp8_configs(_N, _K, 128, 128) == {8: _CONFIG}


def test_no_config_returns_none():
    assert get_w8a8_block_fp8_configs(_N, _K, 128, 128) is None


def test_non_table_config_is_ignored(bundled_dir):
    _table_path(bundled_dir).write_text("[]")
    assert get_w8a8_block_fp8_configs(_N, _K, 128, 128) is None
    assert not fp8_utils.has_fixed_w8a8_block_fp8_config(_N, _K, 128, 128)


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


def _pending(table, max_tokens: int = 16) -> list[int]:
    worker = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=max_tokens)
    )
    return sorted(item.bucket for item in table.pending_items(worker))


def test_pending_items_fill_gaps_skip_fixed_files_and_honor_force(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        block_fp8_autotune, "discover_block_fp8_shapes", lambda worker: {_SHAPE}
    )
    table = block_fp8_autotune.BlockFp8ConfigTable()
    assert _pending(table) == [1, 2, 4, 8, 16]

    save_block_fp8_configs(_N, _K, 128, 128, {1: _CONFIG, 2: _CONFIG})
    assert _pending(table) == [4, 8, 16]

    monkeypatch.setenv("VLLM_TRITON_AUTOTUNE_FORCE", "1")
    assert _pending(table) == [1, 2, 4, 8, 16]
    monkeypatch.delenv("VLLM_TRITON_AUTOTUNE_FORCE")

    _write_table(tmp_path, {})
    monkeypatch.setenv("VLLM_TUNED_CONFIG_FOLDER", str(tmp_path))
    assert _pending(table) == []


def test_pending_items_skip_shapes_with_a_bundled_table(monkeypatch, bundled_dir):
    _write_table(bundled_dir, {1: _CONFIG})
    monkeypatch.setattr(
        block_fp8_autotune, "discover_block_fp8_shapes", lambda worker: {_SHAPE}
    )
    assert _pending(block_fp8_autotune.BlockFp8ConfigTable()) == []


def test_each_bucket_starts_from_the_nearest_buckets_tuned_before(monkeypatch):
    seeds_by_item = {}

    def fake_tune(case, seeds):
        seeds_by_item[case.N, case.M] = seeds
        return {"N": case.N, "M": case.M}

    monkeypatch.setattr(block_fp8_autotune, "tune_block_fp8_case", fake_tune)
    monkeypatch.setattr(block_fp8_autotune, "block_fp8_scratch_bytes", lambda case: 0)
    monkeypatch.setattr(torch.accelerator, "get_memory_info", lambda: (1, 1))
    table = block_fp8_autotune.BlockFp8ConfigTable()
    other_shape = block_fp8_autotune.BlockFp8Shape(1024, _K, 128, 128, torch.bfloat16)
    for shape, m in [(_SHAPE, 1), (_SHAPE, 2), (_SHAPE, 4), (_SHAPE, 16)]:
        table.tune(TuningItem(table.name, shape, m))
    for m in (4, 2):
        table.tune(TuningItem(table.name, other_shape, m))
    assert seeds_by_item[_N, 1] == []
    assert seeds_by_item[_N, 16] == [{"N": _N, "M": 4}, {"N": _N, "M": 2}]
    # Another shape's winner at the same M comes first; on equal distance the
    # shape's own winner beats the other shape's.
    assert seeds_by_item[1024, 4] == [{"N": _N, "M": 4}, {"N": _N, "M": 2}]
    assert seeds_by_item[1024, 2] == [{"N": _N, "M": 2}, {"N": 1024, "M": 4}]


@requires_gpu
def test_tuned_config_matches_the_default_output():
    case = _case(64)
    config = tune_block_fp8_case(case, max_configs=6, num_iters=3)
    assert config is not None
    assert validate_block_fp8_config(case, config)


@requires_gpu
def test_validation_rejects_a_config_that_misreads_the_scales():
    # BLOCK_SIZE_K larger than the quantization block applies one scale to two
    # blocks, so the output is wrong. The search never offers such a config.
    case = _case(64)
    broken = {**_CONFIG, "BLOCK_SIZE_K": 256}
    assert not validate_block_fp8_config(case, broken)
    assert normalize_config(broken, case) is None


@requires_gpu
def test_a_candidate_with_wrong_output_is_never_chosen(monkeypatch):
    monkeypatch.setattr(fp8_block_tuning, "_RANDOM_STARTS", 0)
    case = _case(64)
    default = default_config(case)
    broken = {**default, "num_warps": 8}
    run = fp8_block_tuning._run

    def corrupt_broken(case, inputs, config):
        out = run(case, inputs, config)
        return torch.full_like(out, float("nan")) if config["num_warps"] == 8 else out

    monkeypatch.setattr(fp8_block_tuning, "_run", corrupt_broken)
    assert not validate_block_fp8_config(case, broken)
    tuned = tune_block_fp8_case(case, seeds=[broken], max_configs=2, num_iters=3)
    assert tuned == default


@requires_gpu
def test_table_fills_missing_configs_then_has_nothing_left(monkeypatch):
    monkeypatch.setattr(
        block_fp8_autotune, "discover_block_fp8_shapes", lambda worker: {_SHAPE}
    )
    monkeypatch.setattr(
        block_fp8_autotune,
        "tune_block_fp8_case",
        lambda case, seeds: tune_block_fp8_case(
            case, seeds=seeds, max_configs=3, num_iters=3
        ),
    )
    worker = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16)
    )
    table = block_fp8_autotune.BlockFp8ConfigTable()
    world = SimpleNamespace(rank_in_group=0, world_size=1, cpu_group=None)
    results = run_config_tuning([table], worker, world)
    assert sorted(item.bucket for item in results) == [1, 2, 4, 8, 16]
    assert set(get_w8a8_block_fp8_configs(_N, _K, 128, 128)) == {1, 2, 4, 8, 16}
    tuned_m8 = next(cfg for item, cfg in results.items() if item.bucket == 8)
    assert get_w8a8_block_fp8_configs(_N, _K, 128, 128)[8] == tuned_m8
    assert table.pending_items(worker) == []
