# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.device import DeviceConfig
from vllm.config.lora import LoRAConfig
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.lora.layers import (
    MergedColumnParallelLinearWithLoRA,
    MergedQKVParallelLinearWithLoRA,
)
from vllm.lora.layers.column_parallel_linear import (
    MergedColumnParallelLinearVariableSliceWithLoRA,
)
from vllm.lora.lora_model import LoRAModel
from vllm.lora.lora_weights import LoRALayerWeights
from vllm.lora.model_manager import LoRAModelManager, LRUCacheLoRAModelManager
from vllm.lora.peft_helper import PEFTHelper
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
)
from vllm.model_executor.models.deepseek_v2 import DeepSeekV2FusedQkvAProjLinear
from vllm.model_executor.models.interfaces import SupportsLoRA

pytestmark = pytest.mark.skip_global_cleanup


class LocalAdapterTestModel(torch.nn.Module, SupportsLoRA):
    packed_modules_mapping = {
        "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "three_proj": ["first_proj", "second_proj", "third_proj"],
    }

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(architectures=["GlmMoeDsaForCausalLM"])
        self.o_proj = ReplicatedLinear(16, 24, bias=False, disable_tp=True)
        self.fused_qkv_a_proj = DeepSeekV2FusedQkvAProjLinear(16, [24, 16])
        self.qkv_proj = QKVParallelLinear(
            16, 4, 4, total_num_kv_heads=2, bias=False, disable_tp=True
        )
        self.three_proj = MergedColumnParallelLinear(
            16, [8, 16, 24], bias=False, disable_tp=True
        )


@pytest.fixture(scope="module")
def cpu_parallel(tmp_path_factory):
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    rendezvous = tmp_path_factory.mktemp("local-manager-gloo") / "init"
    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=rendezvous.as_uri(),
            backend="gloo",
        )
        initialize_model_parallel(1, 1, backend="gloo")
        yield
        destroy_model_parallel()
        destroy_distributed_environment()


@pytest.fixture(params=[LoRAModelManager, LRUCacheLoRAModelManager])
def manager(request, monkeypatch, cpu_parallel):
    def init_cpu_punica(self, *_):
        self.supports_mm = False
        self.punica_wrapper_mapping = {"language_model": object()}

    monkeypatch.setattr(LoRAModelManager, "_init_punica_wrapper", init_cpu_punica)
    monkeypatch.setattr("vllm.lora.model_manager.PIN_MEMORY", False)
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    with set_current_vllm_config(config):
        yield request.param(
            LocalAdapterTestModel(),
            1,
            8,
            16,
            LoRAConfig(
                max_lora_rank=8, max_cpu_loras=3, max_loras=2, lora_dtype=torch.bfloat16
            ),
            torch.device("cpu"),
            config,
        )


def make_payload(manager, targets=None):
    helper = PEFTHelper(
        r=4,
        lora_alpha=12,
        target_modules=targets or ["o_proj", "q_a_proj", "kv_a_proj_with_mqa"],
    )
    plan = manager.get_local_adapter_plan(helper)
    factors = {}
    for module in plan.modules:
        a, b = [], []
        for index, (a_shape, b_shape) in enumerate(module.factor_shapes):
            a.append(torch.full(a_shape, index + 1.25, dtype=torch.bfloat16))
            b.append(torch.full(b_shape, index + 2.5, dtype=torch.bfloat16))
        factors[module.module_name] = (a, b)
    return plan, factors


def test_native_mla_and_variable_slice_plan_uses_actual_wrapper_metadata(manager):
    plan, _ = make_payload(
        manager,
        [
            "q_a_proj",
            "kv_a_proj_with_mqa",
            "first_proj",
            "second_proj",
            "third_proj",
        ],
    )
    mla, variable = plan.modules
    assert (
        type(manager.modules["fused_qkv_a_proj"]) is MergedColumnParallelLinearWithLoRA
    )
    assert (
        type(manager.modules["three_proj"])
        is MergedColumnParallelLinearVariableSliceWithLoRA
    )
    assert mla.tp_size == variable.tp_size == 1
    assert mla.global_output_sizes == (24, 16)
    assert mla.source_names == (("q_a_proj",), ("kv_a_proj_with_mqa",))
    assert variable.global_output_sizes == (8, 16, 24)
    assert variable.output_shard_ids == (0, 0, 0)
    assert variable.factor_shapes[-1] == ((4, 16), (24, 4))


def test_native_merged_qkv_plan_uses_distinct_qkv_slices(manager):
    plan, _ = make_payload(manager, ["q_proj", "k_proj", "v_proj"])

    assert len(plan.modules) == 1
    qkv = plan.modules[0]
    assert type(manager.modules["qkv_proj"]) is MergedQKVParallelLinearWithLoRA
    assert qkv.source_layout == "merged"
    assert qkv.source_names == (("q_proj",), ("k_proj",), ("v_proj",))
    assert qkv.global_output_sizes == (16, 8, 8)
    assert qkv.output_shard_ids == (0, 0, 0)
    assert qkv.factor_shapes == (
        ((4, 16), (16, 4)),
        ((4, 16), (8, 4)),
        ((4, 16), (8, 4)),
    )


def test_local_registration_owns_values_scales_once_and_preserves_slot_lifecycle(
    manager,
):
    plan, factors = make_payload(manager)
    expected = {
        name: ([a.clone() for a in pair[0]], [b.clone() * 3 for b in pair[1]])
        for name, pair in factors.items()
    }
    assert manager.add_local_adapter(1, plan, factors)
    for a, b in factors.values():
        for tensor in a + b:
            tensor.zero_()
    registered = manager.get_adapter(1)
    assert registered.tensor_extent == "local"
    assert registered.clone(2).local_plan == plan
    for _ in range(2):
        assert manager.activate_adapter(1)
        for name, (a, b) in expected.items():
            buffers = manager.modules[name]._get_lora_shard_buffers(0)
            for (actual_a, actual_b), expected_a, expected_b in zip(buffers, a, b):
                torch.testing.assert_close(actual_a[:4], expected_a, rtol=0, atol=0)
                torch.testing.assert_close(
                    actual_b[..., :4], expected_b, rtol=0, atol=0
                )
        assert manager.lora_index_to_id == [1, None]
        assert not manager.activate_adapter(1)
        assert manager.deactivate_adapter(1)
    assert manager.remove_adapter(1)
    assert manager.lora_index_to_id == [None, None]
    assert manager.list_adapters() == {}


def test_local_registration_does_not_clone_factors(manager, monkeypatch):
    plan, factors = make_payload(manager)

    def fail_clone(*_args, **_kwargs):
        raise AssertionError("local registration must not clone factor tensors")

    monkeypatch.setattr(torch.Tensor, "clone", fail_clone)
    assert manager.add_local_adapter(1, plan, factors)
    assert manager.lora_index_to_id == [None, None]
    assert manager._staged_local_adapters == {1}
    assert manager.activate_adapter(1)
    assert manager.lora_index_to_id == [1, None]


@pytest.mark.parametrize(
    "failure", ["layout", "tp_rank", "missing", "extra", "shape", "dtype", "rank"]
)
def test_bad_local_payload_cannot_mutate_cache_or_active_slot(manager, failure):
    plan, factors = make_payload(manager)
    manager.add_local_adapter(1, plan, factors)
    manager.activate_adapter(1)
    before = [
        pair[0].clone()
        for module in manager.modules.values()
        for pair in module._get_lora_shard_buffers(0)
    ]
    if failure == "layout":
        plan = replace(plan, modules=plan.modules[::-1])
    elif failure == "tp_rank":
        plan = replace(
            plan, modules=(replace(plan.modules[0], tp_rank=7), *plan.modules[1:])
        )
    elif failure == "missing":
        factors.pop("o_proj")
    elif failure == "extra":
        factors["unknown"] = factors["o_proj"]
    elif failure == "shape":
        factors["o_proj"][1][0] = torch.ones(1, 4, dtype=torch.bfloat16)
    elif failure == "dtype":
        factors["o_proj"][0][0] = factors["o_proj"][0][0].float()
    else:
        plan = replace(plan, rank=2)
    with pytest.raises(ValueError):
        manager.add_local_adapter(2, plan, factors)
    assert set(manager.list_adapters()) == {1}
    assert manager.lora_index_to_id == [1, None]
    after = [
        pair[0]
        for module in manager.modules.values()
        for pair in module._get_lora_shard_buffers(0)
    ]
    assert all(torch.equal(a, b) for a, b in zip(before, after))


def test_local_cache_and_gpu_capacity_never_evict_active_generation(manager):
    plan, factors = make_payload(manager)
    for adapter_id in (1, 2):
        manager.add_local_adapter(adapter_id, plan, factors)
        manager.activate_adapter(adapter_id)
    with pytest.raises(RuntimeError, match="GPU slots"):
        manager.add_local_adapter(3, plan, factors)
    assert manager.lora_index_to_id == [1, 2]
    assert set(manager.list_adapters()) == {1, 2}


def test_lru_eviction_releases_local_slot_and_receiver_buffers(manager):
    if not isinstance(manager, LRUCacheLoRAModelManager):
        pytest.skip("LRU eviction is specific to LRUCacheLoRAModelManager")
    plan, factors = make_payload(manager)

    def make_global_adapter(adapter_id: int) -> LoRAModel:
        a = torch.ones(4, 16, dtype=torch.bfloat16)
        b = torch.ones(24, 4, dtype=torch.bfloat16)
        return LoRAModel(
            adapter_id,
            4,
            {"o_proj": LoRALayerWeights("o_proj", 4, 12, a, b)},
        )

    manager.add_local_adapter(1, plan, factors)
    manager.activate_adapter(1)
    manager.deactivate_adapter(1)

    for adapter_id in (2, 3):
        manager.add_adapter(make_global_adapter(adapter_id))
    manager.activate_adapter(1)
    manager.deactivate_adapter(1)
    manager.add_adapter(make_global_adapter(4))
    assert set(manager.list_adapters()) == {1, 3, 4}

    for adapter_id in (5, 6):
        manager.add_adapter(make_global_adapter(adapter_id))

    assert 1 not in manager.list_adapters()
    assert 1 not in manager._local_adapter_slots
    assert 1 not in manager._staged_local_adapters
    assert manager.lora_index_to_id == [None, None]
    assert all(
        torch.count_nonzero(a) == torch.count_nonzero(b) == 0
        for module in manager.modules.values()
        for a, b in module._get_lora_shard_buffers(0)
    )

    manager.remove_adapter(4)
    manager.add_local_adapter(7, plan, factors)
    assert manager._local_adapter_slots[7] == 0


def test_global_adapter_keeps_existing_scaling_and_loading_path(manager):
    a, b = (
        torch.ones(4, 16, dtype=torch.bfloat16),
        torch.ones(24, 4, dtype=torch.bfloat16),
    )
    global_adapter = LoRAModel(
        1, 4, {"o_proj": LoRALayerWeights("o_proj", 4, 12, a, b)}
    )
    assert global_adapter.tensor_extent == "global"
    manager.add_adapter(global_adapter)
    manager.activate_adapter(1)
    actual_a, actual_b = manager.modules["o_proj"]._get_lora_shard_buffers(0)[0]
    torch.testing.assert_close(actual_a[:4], torch.ones_like(a))
    torch.testing.assert_close(actual_b[..., :4], torch.full_like(b, 3))


def test_partial_packed_target_fails_before_registration(manager):
    with pytest.raises(NotImplementedError, match="every packed target"):
        make_payload(manager, ["q_a_proj"])
    assert manager.list_adapters() == {}


def test_failed_local_copy_keeps_previous_adapter_and_slot_map(manager, monkeypatch):
    plan, factors = make_payload(manager)
    manager.add_local_adapter(1, plan, factors)
    manager.activate_adapter(1)
    before = [
        (a.clone(), b.clone())
        for module in manager.modules.values()
        for a, b in module._get_lora_shard_buffers(0)
    ]

    def fail_copy(*_):
        raise RuntimeError("injected copy failure")

    with monkeypatch.context() as stage:
        stage.setattr(manager.modules["fused_qkv_a_proj"], "set_lora_shard", fail_copy)
        with pytest.raises(RuntimeError, match="injected"):
            manager.add_local_adapter(2, plan, factors)
    assert manager.lora_index_to_id == [1, None]
    after = [
        (a, b)
        for module in manager.modules.values()
        for a, b in module._get_lora_shard_buffers(0)
    ]
    assert all(
        torch.equal(old_a, new_a) and torch.equal(old_b, new_b)
        for (old_a, old_b), (new_a, new_b) in zip(before, after)
    )
    assert all(
        torch.count_nonzero(a) == torch.count_nonzero(b) == 0
        for module in manager.modules.values()
        for a, b in module._get_lora_shard_buffers(1)
    )
    manager.add_local_adapter(2, plan, factors)
    assert manager.activate_adapter(2)
    assert manager.lora_index_to_id == [1, 2]


@pytest.mark.parametrize("fused_3d", [False, True])
def test_local_moe_plan_binds_expert_order_and_tp_partition(manager, fused_3d):
    from tests.lora import test_local_shard_loading as layer_tests
    from vllm.lora.layers import FusedMoE3DWithLoRA, FusedMoEWithLoRA

    layer_type = FusedMoE3DWithLoRA if fused_3d else FusedMoEWithLoRA
    layer = layer_tests.local_shard_layer.__wrapped__(SimpleNamespace(param=layer_type))
    layer.enable_moe_shared_loras = False
    layer.moe_config = SimpleNamespace(
        hidden_dim=16,
        num_experts=3,
        num_local_experts=3,
        intermediate_size_per_partition=2,
        moe_parallel_config=SimpleNamespace(use_ep=False, ep_rank=0),
    )
    manager.register_module("experts", layer)
    if not fused_3d:
        manager.packed_modules["experts"] = [
            f"experts.{expert}.{projection}"
            for expert in range(3)
            for projection in ("gate_proj", "down_proj", "up_proj")
        ]
    plan, factors = make_payload(manager, ["experts"])
    module_plan = plan.modules[0]
    assert module_plan.expert_ids == (0, 1, 2)
    assert module_plan.tp_rank == 3 and module_plan.tp_size == 8
    if fused_3d:
        assert module_plan.source_names == (("experts.base_layer",), ("experts",))
        assert module_plan.global_output_sizes == (32, 16)
    else:
        assert module_plan.source_names[0] == tuple(
            f"experts.{expert}.gate_proj" for expert in range(3)
        )
        assert module_plan.global_output_sizes == (16, 16, 16)
    manager.add_local_adapter(1, plan, factors)
    manager.activate_adapter(1)
    for (a_buffer, b_buffer), a, b in zip(
        layer._get_lora_shard_buffers(0), *factors["experts"]
    ):
        torch.testing.assert_close(a_buffer[..., :4, :], a, rtol=0, atol=0)
        torch.testing.assert_close(b_buffer[..., :4], b * 3, rtol=0, atol=0)
    if fused_3d:
        layer._base_model = "GptOssForCausalLM"
        with pytest.raises(NotImplementedError, match="concatenated gate/up"):
            manager.get_local_adapter_plan(
                PEFTHelper(r=4, lora_alpha=12, target_modules=["experts"])
            )
        assert manager.lora_index_to_id == [1, None]
        layer._base_model = "GlmMoeDsaForCausalLM"
    layer.moe_config.moe_parallel_config.use_ep = True
    with pytest.raises(NotImplementedError, match="EP size 1"):
        manager.get_local_adapter_plan(
            PEFTHelper(r=4, lora_alpha=12, target_modules=["experts"])
        )


def test_local_replacement_updates_existing_mapping_invalidation(manager, monkeypatch):
    plan, factors = make_payload(manager)
    manager.add_local_adapter(1, plan, factors)
    manager.add_local_adapter(2, plan, factors)
    manager.activate_adapter(1)
    observed = []
    monkeypatch.setattr(
        manager,
        "_set_adapter_mapping",
        lambda _: observed.append(tuple(manager.lora_index_to_id)),
    )
    mapping = object()
    manager.set_adapter_mapping(mapping)
    manager.deactivate_adapter(1)
    manager.activate_adapter(2)
    manager.set_adapter_mapping(mapping)
    assert observed == [(1, None), (None, 2)]


@pytest.mark.parametrize(
    "unsupported", ["rslora", "custom_scaling", "fully_sharded", "regex"]
)
def test_unsupported_local_scaling_or_target_modes_fail_closed(manager, unsupported):
    helper = PEFTHelper(r=4, lora_alpha=12, target_modules=["o_proj"])
    if unsupported == "rslora":
        helper = PEFTHelper(
            r=4, lora_alpha=12, target_modules=["o_proj"], use_rslora=True
        )
    elif unsupported == "custom_scaling":
        helper.vllm_lora_scaling_factor = 7.0
    elif unsupported == "fully_sharded":
        manager.lora_config.fully_sharded_loras = True
    else:
        helper.target_modules = ".*proj"
    with pytest.raises((NotImplementedError, ValueError)):
        manager.get_local_adapter_plan(helper)
    assert manager.list_adapters() == {}
