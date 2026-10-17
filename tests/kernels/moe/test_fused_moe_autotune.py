# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import dataclasses
import json
import math
import os
import random
from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.fused_moe.fused_moe as fused_moe_module
from vllm.model_executor.layers.fused_moe import get_config, moe_tuning
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG,
    fp8_w8a8_moe_quant_config,
    int8_w8a8_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.experts.triton_moe import (
    TritonExperts,
    TritonWNA16Experts,
)
from vllm.model_executor.layers.fused_moe.fused_moe import (
    get_config_file_name,
    get_moe_autotune_config_file_path,
    get_moe_configs,
    save_moe_configs,
    try_get_optimal_moe_config,
)
from vllm.model_executor.layers.fused_moe.moe_tuning import (
    FusedMoETuningCase,
    TritonBackend,
    benchmark_moe_config,
    default_config,
    fits_hardware,
    make_moe_inputs,
    normalize_config,
    search_axes,
    starting_configs,
    tune_fused_moe_case,
    valid_m_buckets,
    validate_moe_config,
)
from vllm.model_executor.warmup import fused_moe_autotune
from vllm.model_executor.warmup.triton_autotune import TuningItem, run_config_tuning
from vllm.platforms import current_platform

requires_gpu = pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="GPU")

# Fake Triton backends, so search space tests do not depend on the GPU running
# them: one that accepts extra launch options and one that accepts none.
_WITH_OPTIONS = TritonBackend(
    warp_size=64,
    max_shared_mem=64 * 1024,
    options={
        "waves_per_eu": (0, 1, 2, 4),
        "matrix_instr_nonkdim": (0, 16, 32),
        "kpack": (1, 2),
    },
    option_defaults={"waves_per_eu": 0, "matrix_instr_nonkdim": 0, "kpack": 1},
)
_WITHOUT_OPTIONS = TritonBackend(
    warp_size=32,
    max_shared_mem=227 * 1024,
    options={},
    option_defaults={},
)

# E=8, N=256, K=512, top-2. The fixture below renames the device so lookups
# only see the test's own files.
_E, _N, _K, _TOPK = 8, 256, 512, 2
_W1, _W2 = (_E, 2 * _N, _K), (_E, _K, _N)
_CONFIG = {
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 1,
    "num_warps": 4,
    "num_stages": 2,
}


def _contiguous_stride(shape: tuple[int, int, int]) -> tuple[int, int, int]:
    return (shape[1] * shape[2], shape[2], 1)


def _shape(
    use_fp8_w8a8: bool = False,
    per_act_token_quant: bool = False,
    block_shape: tuple[int, int] | None = None,
    w1_shape: tuple[int, int, int] = _W1,
    w2_shape: tuple[int, int, int] = _W2,
) -> fused_moe_autotune.FusedMoEShape:
    return fused_moe_autotune.FusedMoEShape(
        w1_shape=w1_shape,
        w2_shape=w2_shape,
        w1_stride=_contiguous_stride(w1_shape),
        w2_stride=_contiguous_stride(w2_shape),
        top_k=_TOPK,
        dtype=torch.bfloat16,
        use_fp8_w8a8=use_fp8_w8a8,
        per_act_token_quant=per_act_token_quant,
        block_shape=block_shape,
        activation=MoEActivation.SILU,
    )


def _case(M: int, shape: fused_moe_autotune.FusedMoEShape | None = None):
    shape = shape or _shape()
    return FusedMoETuningCase(
        M=M,
        w1_shape=shape.w1_shape,
        w2_shape=shape.w2_shape,
        w1_stride=shape.w1_stride,
        w2_stride=shape.w2_stride,
        top_k=shape.top_k,
        dtype=shape.dtype,
        use_fp8_w8a8=shape.use_fp8_w8a8,
        per_act_token_quant=shape.per_act_token_quant,
        block_shape=shape.block_shape,
        activation=shape.activation,
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
    # Tests only see the bundled tables they write, under a fake device name.
    for module in (fused_moe_module, fused_moe_autotune):
        monkeypatch.setattr(module, "MOE_CONFIGS_DIR", str(bundled_dir))
    monkeypatch.setenv("VLLM_TRITON_AUTOTUNE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("VLLM_TUNED_CONFIG_FOLDER", raising=False)
    monkeypatch.delenv("VLLM_TRITON_AUTOTUNE_FORCE", raising=False)
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    monkeypatch.setattr(
        fused_moe_module, "get_device_name_as_file_name", lambda: "Test_Device"
    )
    get_moe_configs.cache_clear()
    yield
    get_moe_configs.cache_clear()


def _write_table(folder, file_name: str, table: dict[int, dict]) -> None:
    (folder / file_name).write_text(json.dumps({str(m): c for m, c in table.items()}))


def _use_backend(monkeypatch, backend: TritonBackend):
    monkeypatch.setattr(moe_tuning, "triton_backend", lambda: backend)


def _stub_kernels(monkeypatch, fake_benchmark):
    monkeypatch.setattr(moe_tuning, "make_moe_inputs", lambda case: None)
    monkeypatch.setattr(moe_tuning, "run_moe", lambda *args: None)
    monkeypatch.setattr(moe_tuning, "validate_moe_config", lambda *a, **k: True)
    monkeypatch.setattr(moe_tuning, "benchmark_moe_config", fake_benchmark)


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
    option_names = moe_tuning._BACKEND_OPTION_VALUES.keys()
    assert option_names & axes.keys() == backend.options.keys()
    assert all(axes[name] == values for name, values in backend.options.items())


def test_m_tiles_are_limited_by_the_rows_per_expert(monkeypatch):
    _use_backend(monkeypatch, _WITH_OPTIONS)
    # With 8 experts and top-2, M tokens give each expert M / 4 rows on average.
    assert search_axes(_case(1))["BLOCK_SIZE_M"] == (16,)
    assert search_axes(_case(64))["BLOCK_SIZE_M"] == (16, 32)
    assert search_axes(_case(4096))["BLOCK_SIZE_M"] == (16, 32, 64, 128, 256)


def test_configs_beyond_registers_or_shared_memory_are_rejected(monkeypatch):
    _use_backend(monkeypatch, _WITH_OPTIONS)
    case = _case(64)
    assert fits_hardware(_CONFIG, case)
    # 16x256 by 256x256 bf16 tiles need 136 KiB of shared memory.
    assert not fits_hardware(
        {**_CONFIG, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 256}, case
    )
    # A 16x16 tile has fewer accumulators than 8 warps have threads.
    assert not fits_hardware({**_CONFIG, "BLOCK_SIZE_N": 16, "num_warps": 8}, case)
    # A 256x256 tile on one warp needs 1024 accumulators per thread.
    huge_tile = {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32}
    assert not fits_hardware({**_CONFIG, **huge_tile, "num_warps": 1}, case)
    assert not fits_hardware({**_CONFIG, "matrix_instr_nonkdim": 32}, case)


@_BACKENDS
def test_searched_configs_respect_the_quant_block(monkeypatch, backend):
    _use_backend(monkeypatch, backend)
    case = _case(256, _shape(use_fp8_w8a8=True, block_shape=(128, 128)))
    axes = search_axes(case)
    configs = moe_tuning._random_configs(axes, case, 20, random.Random(0))
    configs += [n for c in configs for n in moe_tuning._neighbors(c, axes, case)]
    assert len(configs) > 20
    for config in configs:
        assert config["BLOCK_SIZE_K"] <= 128
        assert 128 % config["BLOCK_SIZE_K"] == 0
        n = config["BLOCK_SIZE_N"]
        assert n % 128 == 0 or 128 % n == 0
        assert fits_hardware(config, case)


def test_normalize_clamps_k_and_rejects_misaligned_tiles():
    config = {**_CONFIG, "BLOCK_SIZE_K": 256, "SPLIT_K": 1}
    assert normalize_config(config, (128, 128)) == {**_CONFIG, "BLOCK_SIZE_K": 128}
    assert normalize_config({**_CONFIG, "BLOCK_SIZE_N": 96}, (128, 128)) is None


def test_starting_configs_are_the_default_then_the_seeds(bundled_dir):
    case = _case(64)
    default = default_config(case)
    seed = {**_CONFIG, "GROUP_SIZE_M": 32, "num_warps": 1}
    # Config files, even a table for this exact shape, are never starting points.
    _write_table(
        bundled_dir,
        get_config_file_name(_E, _N, None),
        {64: {**_CONFIG, "num_warps": 2}},
    )
    assert starting_configs(case, [seed, default]) == [default, seed]

    block_case = _case(64, _shape(use_fp8_w8a8=True, block_shape=(128, 128)))
    misaligned = {**_CONFIG, "BLOCK_SIZE_N": 96}
    wide_k = {**seed, "BLOCK_SIZE_K": 256}
    assert starting_configs(block_case, [misaligned, wide_k]) == [
        default_config(block_case),
        {**seed, "BLOCK_SIZE_K": 128},
    ]


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
    tuned = tune_fused_moe_case(case)
    assert len({_key(c) for c in searched}) == moe_tuning.MAX_CONFIGS_PER_BUCKET
    best_start = min(fake_time(c) for c in searched[: moe_tuning._RANDOM_STARTS + 1])
    assert fake_time(moe_tuning._complete(tuned, axes, case)) < best_start


def test_autotune_file_names_follow_the_bundled_naming():
    path = get_moe_autotune_config_file_path(256, 256, "fp8_w8a8", [128, 128])
    assert os.path.basename(path) == (
        "E=256,N=256,device_name=Test_Device,dtype=fp8_w8a8,block_shape=[128,128].json"
    )


def test_save_then_load_round_trip():
    path = save_moe_configs(_E, _N, None, None, {8: _CONFIG, 1: _CONFIG})
    with open(path) as f:
        assert list(json.load(f)) == ["triton_version", "1", "8"]
    assert get_moe_configs(_E, _N, None) == {1: _CONFIG, 8: _CONFIG}


def test_user_folder_beats_autotune_cache(monkeypatch, tmp_path):
    save_moe_configs(_E, _N, None, None, {8: _CONFIG})
    user_config = {**_CONFIG, "num_warps": 8}
    _write_table(tmp_path, get_config_file_name(_E, _N, None), {8: user_config})
    monkeypatch.setenv("VLLM_TUNED_CONFIG_FOLDER", str(tmp_path))
    get_moe_configs.cache_clear()
    assert get_moe_configs(_E, _N, None) == {8: user_config}


def test_autotune_cache_beats_bundled_config(bundled_dir):
    bundled_config = {**_CONFIG, "num_warps": 8}
    _write_table(bundled_dir, get_config_file_name(_E, _N, None), {8: bundled_config})
    assert get_moe_configs(_E, _N, None) == {8: bundled_config}
    save_moe_configs(_E, _N, None, None, {8: _CONFIG})
    assert get_moe_configs(_E, _N, None) == {8: _CONFIG}


def test_no_config_returns_none():
    assert get_moe_configs(_E, _N, None) is None


def _experts(cls: type, quant_config) -> TritonExperts:
    experts = object.__new__(cls)
    experts.quant_config = quant_config
    return experts


def _moe_layer(experts, w1: torch.Tensor, w2: torch.Tensor) -> torch.nn.Module:
    module = torch.nn.Module()
    module.w13_weight = torch.nn.Parameter(w1, requires_grad=False)
    module.w2_weight = torch.nn.Parameter(w2, requires_grad=False)
    module.top_k = _TOPK
    module.activation = MoEActivation.SILU
    module.quant_method = SimpleNamespace(
        moe_kernel=SimpleNamespace(fused_experts=experts)
    )
    return module


def test_discovery_keeps_only_plain_triton_bf16_and_fp8_layers():
    ones = torch.ones(1)
    fp8_block = fp8_w8a8_moe_quant_config(
        w1_scale=ones, w2_scale=ones, block_shape=[128, 128]
    )
    int8 = int8_w8a8_moe_quant_config(ones, ones, None, None)
    # Padded MoE weights are a strided view of a wider buffer.
    padded_w1 = torch.empty(_E, 2 * _N, _K + 128)[..., :_K]
    fp8_w1, fp8_w2 = torch.empty(16, 2 * _N, _K), torch.empty(16, _K, _N)

    def plain():
        return _experts(TritonExperts, FUSED_MOE_UNQUANTIZED_CONFIG)

    model = torch.nn.Sequential(
        _moe_layer(plain(), padded_w1, torch.empty(_W2)),
        _moe_layer(plain(), padded_w1, torch.empty(_W2)),  # same file key
        _moe_layer(_experts(TritonExperts, fp8_block), fp8_w1, fp8_w2),
        _moe_layer(
            _experts(TritonWNA16Experts, FUSED_MOE_UNQUANTIZED_CONFIG),
            torch.empty(32, 2 * _N, _K),
            torch.empty(32, _K, _N),
        ),
        _moe_layer(
            _experts(TritonExperts, int8),
            torch.empty(64, 2 * _N, _K),
            torch.empty(64, _K, _N),
        ),
        _moe_layer(
            SimpleNamespace(), torch.empty(128, 2 * _N, _K), torch.empty(128, _K, _N)
        ),
    )
    worker = SimpleNamespace(
        get_model=lambda: model, model_config=SimpleNamespace(dtype=torch.bfloat16)
    )
    shapes = fused_moe_autotune.discover_fused_moe_shapes(worker)
    assert [s.file_key for s in shapes] == [
        (_E, _N, None, None),
        (16, _N, "fp8_w8a8", (128, 128)),
    ]
    assert shapes[0].w1_stride == padded_w1.stride()
    assert shapes[1].use_fp8_w8a8 and not shapes[1].per_act_token_quant


def _pending(table, max_tokens: int = 16) -> list[int]:
    worker = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=max_tokens)
    )
    return sorted(item.bucket for item in table.pending_items(worker))


def test_pending_items_fill_gaps_skip_fixed_files_and_honor_force(
    monkeypatch, tmp_path
):
    shape = _shape()
    monkeypatch.setattr(
        fused_moe_autotune, "discover_fused_moe_shapes", lambda worker: [shape]
    )
    table = fused_moe_autotune.FusedMoEConfigTable()
    assert _pending(table) == [1, 2, 4, 8, 16]

    save_moe_configs(_E, _N, None, None, {1: _CONFIG, 2: _CONFIG})
    assert _pending(table) == [4, 8, 16]

    monkeypatch.setenv("VLLM_TRITON_AUTOTUNE_FORCE", "1")
    assert _pending(table) == [1, 2, 4, 8, 16]
    monkeypatch.delenv("VLLM_TRITON_AUTOTUNE_FORCE")

    (tmp_path / get_config_file_name(_E, _N, None)).write_text("{}")
    monkeypatch.setenv("VLLM_TUNED_CONFIG_FOLDER", str(tmp_path))
    assert _pending(table) == []
    monkeypatch.delenv("VLLM_TUNED_CONFIG_FOLDER")

    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    assert _pending(table) == []


def test_pending_items_skip_shapes_with_a_bundled_table(monkeypatch, bundled_dir):
    _write_table(bundled_dir, get_config_file_name(_E, _N, None), {1: _CONFIG})
    monkeypatch.setattr(
        fused_moe_autotune, "discover_fused_moe_shapes", lambda worker: [_shape()]
    )
    assert _pending(fused_moe_autotune.FusedMoEConfigTable()) == []


def test_each_bucket_starts_from_the_nearest_buckets_tuned_before(monkeypatch):
    seeds_by_m = {}

    def fake_tune(case, seeds):
        seeds_by_m[case.M] = seeds
        return {**_CONFIG, "GROUP_SIZE_M": case.M}

    monkeypatch.setattr(fused_moe_autotune, "tune_fused_moe_case", fake_tune)
    monkeypatch.setattr(fused_moe_autotune, "moe_scratch_bytes", lambda case: 0)
    monkeypatch.setattr(torch.accelerator, "get_memory_info", lambda: (1, 1))
    table = fused_moe_autotune.FusedMoEConfigTable()
    for m in (1, 2, 4, 16):
        table.tune(TuningItem(table.name, _shape(), m))
    assert seeds_by_m[1] == []
    assert [seed["GROUP_SIZE_M"] for seed in seeds_by_m[16]] == [4, 2]


@pytest.mark.parametrize(
    "side_by_side_us,keeps_winner", [(1.0, True), (1.96, False), (3.0, False)]
)
def test_winner_must_clearly_beat_the_default_side_by_side(
    monkeypatch, side_by_side_us, keeps_winner
):
    _use_backend(monkeypatch, _WITHOUT_OPTIONS)
    monkeypatch.setattr(moe_tuning, "_RANDOM_STARTS", 0)
    case = _case(64)
    default = default_config(case)
    other = {**default, "num_warps": 8 if default["num_warps"] != 8 else 4}
    calls = {_key(default): 0, _key(other): 0}

    def fake_benchmark(case, config, **kwargs):
        calls[_key(config)] += 1
        if config == default:
            return 2.0
        return 1.0 if calls[_key(other)] == 1 else side_by_side_us

    _stub_kernels(monkeypatch, fake_benchmark)
    tuned = tune_fused_moe_case(case, seeds=[other], max_configs=2)
    assert tuned == (other if keeps_winner else default)
    assert calls[_key(other)] == 1 + moe_tuning._HEAD_TO_HEAD_ROUNDS


def test_a_failing_candidate_does_not_leave_its_config_active(monkeypatch):
    def out_of_resources(*args, **kwargs):
        assert get_config() == _CONFIG
        raise RuntimeError("OutOfResources")

    monkeypatch.setattr(moe_tuning, "fused_experts", out_of_resources)
    inputs = SimpleNamespace(
        hidden_states=None, w1=None, w2=None, routings=[(None, None)], quant_config=None
    )
    with pytest.raises(RuntimeError, match="OutOfResources"):
        moe_tuning.run_moe(_case(16), inputs, _CONFIG)
    assert get_config() is None


_QUANT_SHAPES = {
    "bf16": _shape(),
    "fp8_tensor": _shape(use_fp8_w8a8=True),
    "fp8_channel": _shape(use_fp8_w8a8=True, per_act_token_quant=True),
    "fp8_block": _shape(use_fp8_w8a8=True, block_shape=(128, 128)),
}


@requires_gpu
@pytest.mark.parametrize("quant", list(_QUANT_SHAPES))
def test_tuned_config_matches_the_default_output(quant, default_vllm_config):
    case = _case(64, _QUANT_SHAPES[quant])
    config = tune_fused_moe_case(case, max_configs=6, num_iters=3)
    assert config is not None
    assert validate_moe_config(case, config)


@requires_gpu
def test_scratch_weights_mirror_the_live_strides(default_vllm_config):
    stride = (2 * _N * (_K + 128), _K + 128, 1)
    case = dataclasses.replace(_case(16), w1_stride=stride)
    inputs = make_moe_inputs(case)
    assert inputs.w1.stride() == stride
    assert validate_moe_config(case, _CONFIG, inputs=inputs)


@requires_gpu
def test_timing_cycles_through_different_routings(monkeypatch, default_vllm_config):
    case = _case(16)
    inputs = make_moe_inputs(case)
    first_ids = inputs.routings[0][1]
    assert any(not torch.equal(ids, first_ids) for _, ids in inputs.routings[1:])
    used = []
    run_moe = moe_tuning.run_moe

    def recording(case, inputs, config, routing=0):
        used.append(routing)
        return run_moe(case, inputs, config, routing)

    monkeypatch.setattr(moe_tuning, "run_moe", recording)
    assert benchmark_moe_config(case, _CONFIG, inputs=inputs, num_iters=1)
    assert set(used) == set(range(len(inputs.routings)))


@requires_gpu
def test_a_candidate_with_wrong_output_is_never_chosen(
    monkeypatch, default_vllm_config
):
    monkeypatch.setattr(moe_tuning, "_RANDOM_STARTS", 0)
    case = _case(64)
    default = default_config(case)
    broken_warps = 8 if default["num_warps"] != 8 else 4
    broken = {**default, "num_warps": broken_warps}
    run_moe = moe_tuning.run_moe

    def corrupt_broken(case, inputs, config, routing=0):
        out = run_moe(case, inputs, config, routing)
        return out + 1 if config["num_warps"] == broken_warps else out

    monkeypatch.setattr(moe_tuning, "run_moe", corrupt_broken)
    assert not validate_moe_config(case, broken)
    tuned = tune_fused_moe_case(case, seeds=[broken], max_configs=2, num_iters=3)
    assert tuned == default


@requires_gpu
def test_an_out_of_resources_candidate_is_skipped(monkeypatch, default_vllm_config):
    monkeypatch.setattr(moe_tuning, "_RANDOM_STARTS", 0)
    case = _case(64)
    huge = {
        "BLOCK_SIZE_M": 256,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 256,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 4,
    }
    assert moe_tuning.benchmark_moe_config(case, huge, num_iters=1) is None
    assert get_config() is None
    tuned = tune_fused_moe_case(case, seeds=[huge], max_configs=2, num_iters=3)
    assert tuned == default_config(case)


@requires_gpu
def test_table_fills_missing_configs_then_has_nothing_left(
    monkeypatch, default_vllm_config
):
    shape = _shape()
    monkeypatch.setattr(
        fused_moe_autotune, "discover_fused_moe_shapes", lambda worker: [shape]
    )
    monkeypatch.setattr(
        fused_moe_autotune,
        "tune_fused_moe_case",
        lambda case, seeds: tune_fused_moe_case(
            case, seeds=seeds, max_configs=3, num_iters=3
        ),
    )
    worker = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16)
    )
    table = fused_moe_autotune.FusedMoEConfigTable()
    world = SimpleNamespace(rank_in_group=0, world_size=1, cpu_group=None)
    results = run_config_tuning([table], worker, world)
    assert sorted(item.bucket for item in results) == [1, 2, 4, 8, 16]
    assert set(get_moe_configs(_E, _N, None)) == {1, 2, 4, 8, 16}
    tuned_m8 = next(cfg for item, cfg in results.items() if item.bucket == 8)
    assert try_get_optimal_moe_config(_W1, _W2, _TOPK, None, 8) == tuned_m8
    assert table.pending_items(worker) == []
