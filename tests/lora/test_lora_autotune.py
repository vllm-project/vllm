# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import os
from types import SimpleNamespace

import pytest
import torch

from vllm.lora.layers import (
    LogitsProcessorWithLoRA,
    MergedQKVParallelLinearWithLoRA,
    ReplicatedLinearWithLoRA,
    RowParallelLinearWithLoRA,
)
from vllm.lora.ops.triton_ops.lora_tuning import (
    LoRATuningCase,
    default_config,
    lora_m_buckets,
    make_lora_inputs,
    search_space,
    tune_lora_case,
    validate_lora_config,
)
from vllm.lora.ops.triton_ops.utils import (
    _LORA_A_PTR_DICT,
    _LORA_B_PTR_DICT,
    _default_lora_op_config,
    _get_lora_op_configs_cached,
    get_lora_autotune_config_file_path,
    get_lora_op_config_file_name,
    get_lora_op_configs,
    load_lora_op_config,
    override_lora_op_config,
    save_lora_configs,
)
from vllm.model_executor.warmup import lora_autotune
from vllm.model_executor.warmup.lora_autotune import LoRAShape
from vllm.model_executor.warmup.triton_autotune import run_config_tuning
from vllm.platforms import current_platform

pytestmark = pytest.mark.skip_global_cleanup

requires_gpu = pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="GPU")

_SHRINK_CONFIG = {
    "block_m": 16,
    "block_n": 16,
    "block_k": 64,
    "split_k": 4,
    "num_warps": 4,
    "num_ctas": 1,
    "group_size_m": 8,
    "num_stages": 2,
}
# config_data[max_loras][num_slices][m][k][n] for max_loras=1 (key 2), one
# slice, M=16, hidden size 4096 and rank 16.
_SHRINK_TABLE = {"2": {"1": {"16": {"4096": {"16": _SHRINK_CONFIG}}}}}

_SHRINK_SHAPE = LoRAShape("shrink", 1, 16, (512, 512), None, torch.bfloat16)
_EXPAND_SHAPE = LoRAShape("expand", 1, 16, (512, 256), True, torch.bfloat16)


def _clear_caches() -> None:
    load_lora_op_config.cache_clear()
    _get_lora_op_configs_cached.cache_clear()


def _shrink_lookup(num_slices: int = 1, batch: int = 16, hidden_size: int = 4096):
    return get_lora_op_configs(
        "shrink",
        max_loras=2,
        batch=batch,
        hidden_size=hidden_size,
        rank=16,
        num_slices=num_slices,
    )


def _case(shape: LoRAShape, M: int) -> LoRATuningCase:
    return LoRATuningCase(
        op_type=shape.op_type,
        M=M,
        max_loras=shape.max_loras,
        rank=shape.rank,
        slice_sizes=shape.slice_sizes,
        add_inputs=shape.add_inputs,
        dtype=shape.dtype,
        device=torch.device(current_platform.device_type),
    )


@pytest.fixture(autouse=True)
def isolated_config_dirs(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_TRITON_AUTOTUNE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("VLLM_TUNED_CONFIG_FOLDER", raising=False)
    monkeypatch.delenv("VLLM_TRITON_AUTOTUNE_FORCE", raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: "Test GPU-1")
    _clear_caches()
    yield
    _clear_caches()


def test_file_names_match_the_tuning_readme():
    assert get_lora_op_config_file_name("shrink", None) == "Test_GPU_1_SHRINK.json"
    assert get_lora_op_config_file_name("expand", True) == "Test_GPU_1_EXPAND_TRUE.json"
    assert (
        get_lora_op_config_file_name("expand", False) == "Test_GPU_1_EXPAND_FALSE.json"
    )


def test_autotune_cache_file_uses_the_platform_device_name(monkeypatch):
    # torch.cuda.get_device_name() returns "" on some ROCm builds.
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: "")
    monkeypatch.setattr(
        "vllm.lora.ops.triton_ops.utils.get_device_name_as_file_name",
        lambda: "AMD_Instinct_MI300X",
    )
    path = get_lora_autotune_config_file_path("expand", False)
    assert os.path.basename(path) == "AMD_Instinct_MI300X_EXPAND_FALSE.json"


def test_m_buckets_stop_at_the_first_entry_covering_the_limit():
    assert lora_m_buckets(256) == [1, 16, 32, 64, 128, 256]
    assert lora_m_buckets(300) == [1, 16, 32, 64, 128, 256, 512]
    assert lora_m_buckets(8192)[-1] == 4096
    assert lora_m_buckets(0) == []


@requires_gpu
def test_save_then_lookup_round_trip():
    save_lora_configs("shrink", None, _SHRINK_TABLE)
    with open(get_lora_autotune_config_file_path("shrink", None)) as f:
        assert "triton_version" in json.load(f)
    assert _shrink_lookup() == _SHRINK_CONFIG
    # M and K resolve to the nearest tuned entry.
    assert _shrink_lookup(batch=20, hidden_size=4000) == _SHRINK_CONFIG


@requires_gpu
def test_user_folder_beats_autotune_cache(monkeypatch, tmp_path):
    save_lora_configs("shrink", None, _SHRINK_TABLE)
    user_config = {**_SHRINK_CONFIG, "num_warps": 8}
    user_table = {"2": {"1": {"16": {"4096": {"16": user_config}}}}}
    (tmp_path / "Test_GPU_1_SHRINK.json").write_text(json.dumps(user_table))
    monkeypatch.setenv("VLLM_TUNED_CONFIG_FOLDER", str(tmp_path))
    _clear_caches()
    assert _shrink_lookup() == user_config


def test_missing_num_slices_falls_back_to_the_default(monkeypatch, tmp_path):
    (tmp_path / "Test_GPU_1_SHRINK.json").write_text(json.dumps(_SHRINK_TABLE))
    monkeypatch.setenv("VLLM_TUNED_CONFIG_FOLDER", str(tmp_path))
    _clear_caches()
    assert _shrink_lookup(num_slices=1) == _SHRINK_CONFIG
    assert _shrink_lookup(num_slices=3) == _default_lora_op_config("shrink", 16, 3, 16)


def test_no_config_returns_the_default():
    assert _shrink_lookup() == _default_lora_op_config("shrink", 16, 1, 16)


def test_override_applies_only_to_its_op():
    expand_default = _default_lora_op_config("expand", 16, 1, 16)
    expand_kwargs = dict(
        max_loras=2, batch=16, hidden_size=4096, rank=16, num_slices=1, add_inputs=True
    )
    with override_lora_op_config("shrink", _SHRINK_CONFIG):
        assert _shrink_lookup() == _SHRINK_CONFIG
        assert get_lora_op_configs("expand", **expand_kwargs) == expand_default
    assert _shrink_lookup() == _default_lora_op_config("shrink", 16, 1, 16)


def test_search_space_starts_with_the_default_and_prunes_block_m():
    small = _case(_SHRINK_SHAPE, 1)
    candidates = search_space(small)
    assert candidates[0] == default_config(small)
    assert {c["block_m"] for c in candidates[1:]} == {16}
    assert all(c["block_k"] * c["split_k"] <= 512 for c in candidates[1:])
    assert len(candidates) == 6
    assert len(search_space(_case(_SHRINK_SHAPE, 4096))) == 21
    # Shrink also tries BLOCK_N equal to a rank above 16.
    rank_64 = LoRAShape("shrink", 1, 64, (512,), None, torch.bfloat16)
    assert len(search_space(_case(rank_64, 4096))) == 41

    expand = search_space(_case(_EXPAND_SHAPE, 4096))
    assert expand[0] == default_config(_case(_EXPAND_SHAPE, 4096))
    assert len(expand) == len({json.dumps(c, sort_keys=True) for c in expand})
    assert len(expand) == 37
    assert len(search_space(_case(_EXPAND_SHAPE, 1))) == 13


def _fake_layer(cls, lora_a, lora_b, dual_stream: bool = False):
    layer = object.__new__(cls)
    layer.lora_a_stacked = lora_a
    layer.lora_b_stacked = lora_b
    layer._enable_aux_cuda_stream = dual_stream
    return layer


def _weights(*shape: int) -> torch.Tensor:
    return torch.empty(*shape, dtype=torch.bfloat16, device="meta")


def test_discovery_reads_shapes_from_the_lora_weights():
    qkv = _fake_layer(
        MergedQKVParallelLinearWithLoRA,
        tuple(_weights(2, 1, 16, 4096) for _ in range(3)),
        (_weights(2, 1, 4096, 16), _weights(2, 1, 1024, 16), _weights(2, 1, 1024, 16)),
    )
    # Dual stream writes LoRA-only output for layers on the shared apply() path.
    o_proj = _fake_layer(
        RowParallelLinearWithLoRA,
        (_weights(2, 1, 16, 4096),),
        (_weights(2, 1, 4096, 16),),
        dual_stream=True,
    )
    gate = _fake_layer(
        ReplicatedLinearWithLoRA,
        (_weights(2, 1, 16, 4096),),
        (_weights(2, 1, 64, 16),),
        dual_stream=True,
    )
    logits = object.__new__(LogitsProcessorWithLoRA)
    logits.lora_a_stacked = _weights(2, 1, 16, 4096)
    logits.lora_b_stacked = _weights(2, 1, 32000, 16)
    model = SimpleNamespace(modules=lambda: [qkv, o_proj, gate, logits])
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(
            lora_config=SimpleNamespace(max_loras=2),
            speculative_config=SimpleNamespace(num_speculative_tokens=1),
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=2048, max_num_seqs=64),
        get_model=lambda: model,
    )

    def shape(op_type, slice_sizes, add_inputs):
        return LoRAShape(op_type, 2, 16, slice_sizes, add_inputs, torch.bfloat16)

    assert lora_autotune.discover_lora_shapes(worker) == {
        shape("shrink", (4096, 4096, 4096), None): 2048,
        shape("expand", (4096, 1024, 1024), True): 2048,
        # o_proj, gate and lm_head share one shrink shape.
        shape("shrink", (4096,), None): 2048,
        shape("expand", (4096,), False): 2048,
        shape("expand", (64,), True): 2048,
        # lm_head runs on one row per sampled token.
        shape("expand", (32000,), True): 128,
    }


def _table_worker() -> SimpleNamespace:
    return SimpleNamespace(
        vllm_config=SimpleNamespace(lora_config=SimpleNamespace(max_loras=1))
    )


@requires_gpu
def test_pending_items_are_per_shape_and_bucket(monkeypatch, tmp_path):
    monkeypatch.setattr(
        lora_autotune, "discover_lora_shapes", lambda worker: {_SHRINK_SHAPE: 16}
    )
    table = lora_autotune.LoRAConfigTable()
    worker = _table_worker()
    assert [item.bucket for item in table.pending_items(worker)] == [1, 16]

    entry = {"2": {"2": {"1": {"512": {"16": _SHRINK_CONFIG}}}}}
    save_lora_configs("shrink", None, entry)
    assert [item.bucket for item in table.pending_items(worker)] == [16]

    monkeypatch.setenv("VLLM_TRITON_AUTOTUNE_FORCE", "1")
    assert len(table.pending_items(worker)) == 2
    monkeypatch.delenv("VLLM_TRITON_AUTOTUNE_FORCE")

    # A user-supplied file replaces the autotune cache, so nothing is pending.
    (tmp_path / "Test_GPU_1_SHRINK.json").write_text(json.dumps(entry))
    monkeypatch.setenv("VLLM_TUNED_CONFIG_FOLDER", str(tmp_path))
    assert table.pending_items(worker) == []


@requires_gpu
@pytest.mark.parametrize("shape", [_SHRINK_SHAPE, _EXPAND_SHAPE])
def test_tuned_config_is_a_candidate_and_matches_the_default_output(shape):
    a_keys, b_keys = set(_LORA_A_PTR_DICT), set(_LORA_B_PTR_DICT)
    case = _case(shape, 64)
    candidates = search_space(case)[:4]
    config = tune_lora_case(case, candidates=candidates, num_iters=3)
    assert config in candidates
    assert validate_lora_config(case, config)
    # Scratch weights must not stay in the data_ptr-keyed pointer caches.
    assert set(_LORA_A_PTR_DICT) == a_keys
    assert set(_LORA_B_PTR_DICT) == b_keys


@requires_gpu
def test_validation_rejects_an_output_that_differs_from_the_reference():
    case = _case(_EXPAND_SHAPE, 16)
    inputs = make_lora_inputs(case)
    reference = inputs.out_init + 1.0
    assert not validate_lora_config(
        case, default_config(case), inputs=inputs, reference=reference
    )


@requires_gpu
def test_table_fills_missing_configs_then_has_nothing_left(monkeypatch):
    shapes = {_SHRINK_SHAPE: 16, _EXPAND_SHAPE: 16}
    monkeypatch.setattr(lora_autotune, "discover_lora_shapes", lambda worker: shapes)
    monkeypatch.setattr(
        lora_autotune,
        "tune_lora_case",
        lambda case: tune_lora_case(
            case, candidates=search_space(case)[:3], num_iters=3
        ),
    )
    table = lora_autotune.LoRAConfigTable()
    worker = _table_worker()
    world = SimpleNamespace(rank_in_group=0, world_size=1, cpu_group=None)
    results = run_config_tuning([table], worker, world)
    assert len(results) == 4  # two shapes at M = 1 and 16

    for item, config in results.items():
        shape = item.shape
        lookup = get_lora_op_configs(
            shape.op_type,
            max_loras=shape.max_loras + 1,
            batch=item.bucket,
            hidden_size=max(shape.slice_sizes),
            rank=shape.rank,
            num_slices=len(shape.slice_sizes),
            add_inputs=shape.add_inputs,
        )
        assert lookup == config
    assert table.pending_items(worker) == []
