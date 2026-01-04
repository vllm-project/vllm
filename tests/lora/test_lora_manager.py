# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import pytest
import torch
from safetensors.torch import load_file
from torch import nn

from vllm.config import ModelConfig, VllmConfig
from vllm.config.lora import LoRAConfig
from vllm.lora.layers import (
    ColumnParallelLinearWithLoRA,
    MergedColumnParallelLinearWithLoRA,
    RowParallelLinearWithLoRA,
)
from vllm.lora.lora_model import LoRAModel
from vllm.lora.lora_weights import LoRALayerWeights, PackedLoRALayerWeights
from vllm.lora.model_manager import (
    DEFAULT_LANGUAGE_WRAPPER_KEY,
    LoRAMapping,
    LoRAModelManager,
    LRUCacheLoRAModelManager,
)
from vllm.lora.peft_helper import PEFTHelper
from vllm.lora.request import LoRARequest
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager, WorkerLoRAManager
from vllm.platforms import current_platform

from .utils import create_peft_lora

EMBEDDING_MODULES = {
    "embed_tokens": "input_embeddings",
    "lm_head": "output_embeddings",
}


DEVICES = (
    [f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)]
    if current_platform.is_cuda_alike()
    else ["cpu"]
)

DEFAULT_DTYPE = torch.get_default_dtype()


@pytest.mark.parametrize("device", DEVICES)
def test_from_lora_tensors(qwen3_lora_files, device):
    tensors = load_file(os.path.join(qwen3_lora_files, "adapter_model.safetensors"))

    peft_helper = PEFTHelper.from_local_dir(
        qwen3_lora_files, max_position_embeddings=4096
    )
    lora_model = LoRAModel.from_lora_tensors(
        1,
        tensors,
        peft_helper=peft_helper,
        device=device,
    )
    for module_name, lora in lora_model.loras.items():
        assert lora.module_name == module_name
        assert lora.rank == 8
        assert lora.lora_alpha == 32
        assert lora.lora_a is not None
        assert lora.lora_b is not None
        assert lora.lora_a.device == torch.device(device)
        assert lora.lora_b.device == torch.device(device)
        assert lora.lora_a.shape[0] == lora.lora_b.shape[1], (
            f"{lora.lora_a.shape=}, {lora.lora_b.shape=}"
        )
        assert lora.lora_a.shape[0] == 8


def create_lora(
    lora_id: int, model: nn.Module, sub_modules: list[str], device: torch.device
) -> LoRAModel:
    loras: dict[str, LoRALayerWeights] = {}
    for name in sub_modules:
        w = model.get_submodule(name).weight
        loras[name] = LoRALayerWeights(
            name,
            8,
            16,
            torch.rand([8, w.shape[1]], device=device),
            torch.rand([w.shape[0], 8], device=device),
        )
    return LoRAModel(lora_id, 8, loras)


def create_packed_lora(
    lora_id: int,
    model: nn.Module,
    module_name,
    replaced_module_names,
    device: torch.device,
    empty_replaced_module_name=None,
) -> LoRAModel:
    w = model.get_submodule(module_name).weight
    loras: dict[str, LoRALayerWeights] = {}
    for replaced_module_name in replaced_module_names:
        if replaced_module_name == empty_replaced_module_name:
            continue
        loras[replaced_module_name] = LoRALayerWeights(
            replaced_module_name,
            8,
            16,
            torch.rand([8, w.shape[1]], device=device),
            torch.rand([w.shape[0] // len(replaced_module_names), 8], device=device),
        )
    return LoRAModel(lora_id, 8, loras)


def test_replace_submodules(dist_init, dummy_model):
    model = dummy_model
    manager = LoRAModelManager(
        model,
        1,
        1,
        1,
        LoRAConfig(
            max_lora_rank=8, max_cpu_loras=8, max_loras=8, lora_dtype=DEFAULT_DTYPE
        ),
        torch.device(DEVICES[0]),
    )
    model = manager.model
    assert isinstance(model.get_submodule("dense1"), ColumnParallelLinearWithLoRA)
    assert isinstance(
        model.get_submodule("layer1.dense1"), ColumnParallelLinearWithLoRA
    )
    assert isinstance(model.get_submodule("dense2"), RowParallelLinearWithLoRA)
    assert isinstance(model.get_submodule("layer1.dense2"), RowParallelLinearWithLoRA)


@pytest.mark.parametrize(
    "target_modules,exclude_modules,expected_lora_modules",
    [
        (
            None,
            None,
            ["dense1", "layer1.dense1", "dense2", "layer1.dense2", "output"],
        ),  # All modules
        (
            "dense1",
            None,
            ["dense1", "layer1.dense1"],
        ),  # Suffix match
        (
            ["dense1", "dense2"],
            None,
            ["dense1", "layer1.dense1", "dense2", "layer1.dense2"],
        ),
        (
            [".*dense1"],
            None,
            ["dense1", "layer1.dense1"],
        ),  # Regex match in list
        (
            ["dense1", ".*dense2"],
            None,
            ["dense1", "layer1.dense1", "dense2", "layer1.dense2"],
        ),  # Mixed suffix and regex in list
        (
            None,
            ["dense1"],
            ["dense2", "layer1.dense2", "output"],
        ),  # Exclude suffix match
        (
            None,
            [".*dense1"],
            ["dense2", "layer1.dense2", "output"],
        ),  # Exclude regex match in list
        (
            ["dense1", "dense2"],
            ["layer1.dense1"],
            ["dense1", "dense2", "layer1.dense2"],
        ),  # Target with exclude
    ],
)
def test_replace_submodules_with_target_modules(
    dist_init,
    dummy_model,
    target_modules,
    exclude_modules,
    expected_lora_modules,
):
    model = dummy_model
    # The dummy model has these modules that support LoRA:
    # dense1, layer1.dense1, dense2, layer1.dense2, output, lm_head
    # Note: lm_head is excluded from expected_lora_modules check in this test
    # because it's handled differently in some contexts, but let's stick to
    # the linear layers for simplicity unless specified.

    manager = LoRAModelManager(
        model,
        1,
        1,
        1,
        LoRAConfig(
            max_lora_rank=8,
            max_cpu_loras=8,
            max_loras=8,
            lora_dtype=DEFAULT_DTYPE,
            lora_target_modules=target_modules,
            lora_exclude_modules=exclude_modules,
        ),
        torch.device(DEVICES[0]),
    )
    model = manager.model

    # Check that expected modules are replaced with LoRA wrappers
    for module_name in expected_lora_modules:
        module = model.get_submodule(module_name)
        assert isinstance(
            module,
            (
                ColumnParallelLinearWithLoRA,
                RowParallelLinearWithLoRA,
                MergedColumnParallelLinearWithLoRA,
            ),
        ), f"Module {module_name} should be a LoRA layer"

    # Check that other modules are NOT replaced
    all_lora_candidates = [
        "dense1",
        "layer1.dense1",
        "dense2",
        "layer1.dense2",
        "output",
    ]
    for module_name in all_lora_candidates:
        if module_name not in expected_lora_modules:
            module = model.get_submodule(module_name)
            assert not isinstance(
                module,
                (
                    ColumnParallelLinearWithLoRA,
                    RowParallelLinearWithLoRA,
                    MergedColumnParallelLinearWithLoRA,
                ),
            ), f"Module {module_name} should NOT be a LoRA layer"


@pytest.mark.parametrize("device", DEVICES)
def test_lora_model_manager(dist_init, dummy_model, device):
    model = dummy_model
    model_lora1 = create_lora(
        1, model, ["layer1.dense1", "dense2", "lm_head"], device=device
    )
    model_lora2 = create_lora(2, model, ["dense1", "dense2", "lm_head"], device=device)
    model_lora3 = create_lora(3, model, ["dense1", "dense2", "lm_head"], device=device)
    manager = LoRAModelManager(
        model,
        2,
        2,
        2,
        LoRAConfig(
            max_lora_rank=8, max_cpu_loras=3, max_loras=2, lora_dtype=DEFAULT_DTYPE
        ),
        device=device,
    )
    assert all(x is None for x in manager.lora_index_to_id)
    assert manager.add_adapter(model_lora1)
    assert manager.activate_adapter(1)
    assert manager.lora_index_to_id[0] == 1
    assert not manager.add_adapter(model_lora1)
    assert not manager.activate_adapter(1)
    assert manager.add_adapter(model_lora2)
    assert manager.activate_adapter(2)
    assert manager.lora_index_to_id[0] == 1
    assert manager.lora_index_to_id[1] == 2
    assert not manager.add_adapter(model_lora2)
    assert not manager.activate_adapter(2)
    assert manager.add_adapter(model_lora3)
    assert manager.lora_index_to_id[0] == 1
    assert manager.lora_index_to_id[1] == 2
    with pytest.raises(ValueError):
        assert manager.activate_adapter(3)
    assert manager.lora_index_to_id[0] == 1
    assert manager.lora_index_to_id[1] == 2
    assert manager.remove_adapter(model_lora2.id)
    assert manager.lora_index_to_id[1] is None
    assert not manager.remove_adapter(model_lora2.id)
    assert manager.remove_adapter(model_lora1.id)
    assert not manager.remove_adapter(model_lora1.id)
    assert manager.add_adapter(model_lora1)
    assert manager.lora_index_to_id[0] is None
    assert manager.lora_index_to_id[1] is None
    assert manager.add_adapter(model_lora2)
    assert manager.activate_adapter(3)
    assert manager.lora_index_to_id[0] == 3
    assert manager.lora_index_to_id[1] is None
    assert manager.activate_adapter(2)
    assert manager.lora_index_to_id[0] == 3
    assert manager.lora_index_to_id[1] == 2
    assert manager.device == device
    assert (
        manager.punica_wrapper_mapping.get(DEFAULT_LANGUAGE_WRAPPER_KEY).device
        == device
    )
    assert hasattr(manager, "supported_lora_modules")
    assert sorted(manager.supported_lora_modules) == [
        "dense1",
        "dense2",
        "lm_head",
        "output",
    ]


@pytest.mark.parametrize("device", DEVICES)
def test_lora_lru_cache_model_manager(dist_init, dummy_model, device):
    model = dummy_model
    model_lora1 = create_lora(
        1, model, ["layer1.dense1", "dense2", "lm_head"], device=device
    )
    model_lora2 = create_lora(2, model, ["dense1", "dense2", "lm_head"], device=device)
    model_lora3 = create_lora(3, model, ["dense1", "dense2", "lm_head"], device=device)
    manager = LRUCacheLoRAModelManager(
        model,
        2,
        2,
        2,
        LoRAConfig(
            max_lora_rank=8, max_cpu_loras=3, max_loras=2, lora_dtype=DEFAULT_DTYPE
        ),
        device=device,
    )
    assert all(x is None for x in manager.lora_index_to_id)
    assert manager.add_adapter(model_lora1)
    assert manager.activate_adapter(1)
    assert manager.lora_index_to_id[0] == 1
    assert not manager.add_adapter(model_lora1)
    assert not manager.activate_adapter(1)
    assert manager.add_adapter(model_lora2)
    assert manager.activate_adapter(2)
    assert manager.lora_index_to_id[0] == 1
    assert manager.lora_index_to_id[1] == 2
    assert not manager.add_adapter(model_lora2)
    assert not manager.activate_adapter(2)
    assert manager.add_adapter(model_lora3)
    assert manager.lora_index_to_id[0] == 1
    assert manager.lora_index_to_id[1] == 2
    assert manager.activate_adapter(3)
    assert manager.lora_index_to_id[0] == 3
    assert manager.lora_index_to_id[1] == 2
    assert manager.remove_adapter(model_lora2.id)
    assert manager.lora_index_to_id[1] is None
    assert not manager.remove_adapter(model_lora2.id)
    assert manager.remove_adapter(model_lora1.id)
    assert not manager.remove_adapter(model_lora1.id)
    assert manager.add_adapter(model_lora1)
    assert manager.activate_adapter(1)
    assert manager.lora_index_to_id[0] == 3
    assert manager.lora_index_to_id[1] == 1
    assert manager.add_adapter(model_lora2)
    assert manager.deactivate_adapter(3)
    assert manager.lora_index_to_id[0] is None
    assert manager.lora_index_to_id[1] == 1
    assert manager.activate_adapter(2)
    assert manager.lora_index_to_id[0] == 2
    assert manager.lora_index_to_id[1] == 1
    assert manager.activate_adapter(3)
    assert manager.lora_index_to_id[0] == 2
    assert manager.lora_index_to_id[1] == 3
    assert manager.pin_adapter(2)
    assert manager.lora_index_to_id[0] == 2
    assert manager.lora_index_to_id[1] == 3
    assert manager.activate_adapter(1)
    assert manager.lora_index_to_id[0] == 2
    assert manager.lora_index_to_id[1] == 1
    assert manager.deactivate_adapter(2)
    assert manager.lora_index_to_id[0] is None
    assert manager.lora_index_to_id[1] == 1
    assert manager.activate_adapter(3)
    assert manager.lora_index_to_id[0] == 3
    assert manager.lora_index_to_id[1] == 1
    assert manager.pin_adapter(3)
    assert manager.pin_adapter(1)
    with pytest.raises(RuntimeError):
        assert manager.pin_adapter(2)
    assert manager.lora_index_to_id[0] == 3
    assert manager.lora_index_to_id[1] == 1
    with pytest.raises(RuntimeError):
        assert manager.activate_adapter(2)

    assert manager.deactivate_adapter(3)
    assert manager.pin_adapter(2)
    assert manager.lora_index_to_id[0] == 2
    assert manager.lora_index_to_id[1] == 1
    assert manager.remove_adapter(3)
    with pytest.raises(ValueError):
        assert manager.pin_adapter(3)
    assert (
        manager.punica_wrapper_mapping.get(DEFAULT_LANGUAGE_WRAPPER_KEY).device
        == device
    )
    assert manager.device == device


@pytest.mark.parametrize("device", DEVICES)
def test_lru_lora_model_manager(dist_init, dummy_model, device):
    # This tests just the LRU cache functionality, everything else is
    # tested in test_lora_model_manager
    model = dummy_model
    model_lora1 = create_lora(
        1, model, ["layer1.dense1", "dense2", "lm_head"], device=device
    )
    model_lora2 = create_lora(2, model, ["dense1", "dense2", "lm_head"], device=device)
    model_lora3 = create_lora(3, model, ["dense1", "dense2", "lm_head"], device=device)
    model_lora4 = create_lora(4, model, ["dense1", "dense2", "lm_head"], device=device)
    manager = LRUCacheLoRAModelManager(
        model,
        2,
        2,
        2,
        LoRAConfig(
            max_lora_rank=8, max_cpu_loras=2, max_loras=2, lora_dtype=DEFAULT_DTYPE
        ),
        device=device,
    )
    assert all(x is None for x in manager.lora_index_to_id)

    # Add up to capacity
    assert manager.add_adapter(model_lora1)
    assert manager.add_adapter(model_lora2)
    assert manager.activate_adapter(1)
    assert manager.activate_adapter(2)

    assert set(manager.list_adapters()) == {1, 2}
    assert manager.lora_index_to_id[0] == 1
    assert manager.lora_index_to_id[1] == 2

    # Add over capacity
    assert manager.add_adapter(model_lora3)
    assert manager.add_adapter(model_lora4)
    assert manager.activate_adapter(3)
    assert manager.activate_adapter(4)

    assert set(manager.list_adapters()) == {3, 4}
    assert manager.lora_index_to_id[0] == 3
    assert manager.lora_index_to_id[1] == 4

    # Add 3 again to move it to the top and then add 2
    # should return false since it's in already
    assert not manager.add_adapter(model_lora3)
    assert not manager.activate_adapter(3)
    assert manager.add_adapter(model_lora2)
    assert manager.activate_adapter(2)

    assert set(manager.list_adapters()) == {3, 2}
    assert manager.lora_index_to_id[0] == 3
    assert manager.lora_index_to_id[1] == 2

    # Remove manually
    assert manager.remove_adapter(3)
    assert not manager.remove_adapter(3)

    assert set(manager.list_adapters()) == {2}
    assert manager.lora_index_to_id[0] is None
    assert manager.lora_index_to_id[1] == 2

    assert manager.add_adapter(model_lora3)
    assert manager.activate_adapter(3)
    assert manager.add_adapter(model_lora4)
    assert manager.activate_adapter(4)

    assert set(manager.list_adapters()) == {3, 4}
    assert manager.lora_index_to_id[0] == 3
    assert manager.lora_index_to_id[1] == 4

    assert manager.remove_oldest_adapter()
    assert set(manager.list_adapters()) == {4}
    assert manager.lora_index_to_id[0] is None
    assert manager.lora_index_to_id[1] == 4

    assert manager.remove_oldest_adapter()
    assert set(manager.list_adapters()) == set()
    assert all(x is None for x in manager.lora_index_to_id)

    assert not manager.remove_oldest_adapter()
    assert set(manager.list_adapters()) == set()
    assert all(x is None for x in manager.lora_index_to_id)

    # pinning
    assert manager.add_adapter(model_lora3)
    assert manager.activate_adapter(3)
    assert manager.add_adapter(model_lora4)
    assert manager.activate_adapter(4)
    assert set(manager.list_adapters()) == {3, 4}
    with pytest.raises(ValueError):
        assert manager.pin_adapter(1)
    assert manager.pin_adapter(3)
    # Remove manually
    assert manager.remove_adapter(3)
    assert not manager.remove_adapter(3)

    assert set(manager.list_adapters()) == {4}
    assert manager.lora_index_to_id[0] is None
    assert manager.lora_index_to_id[1] == 4

    assert manager.add_adapter(model_lora1)
    assert manager.pin_adapter(1)
    assert manager.add_adapter(model_lora2)
    assert manager.activate_adapter(2)

    assert set(manager.list_adapters()) == {1, 2}
    assert manager.lora_index_to_id[0] == 1
    assert manager.lora_index_to_id[1] == 2

    assert manager.remove_oldest_adapter()
    assert set(manager.list_adapters()) == {1}
    assert manager.lora_index_to_id[0] == 1
    assert manager.lora_index_to_id[1] is None

    with pytest.raises(RuntimeError):
        assert manager.remove_oldest_adapter()

    assert set(manager.list_adapters()) == {1}
    assert (
        manager.punica_wrapper_mapping.get(DEFAULT_LANGUAGE_WRAPPER_KEY).device
        == device
    )
    assert manager.device == device


@pytest.mark.parametrize("device", DEVICES)
def test_lru_cache_worker_adapter_manager(dist_init, dummy_model, device, tmp_path):
    lora_config = LoRAConfig(
        max_lora_rank=8, max_cpu_loras=4, max_loras=4, lora_dtype=DEFAULT_DTYPE
    )

    dummy_lora_files = f"{tmp_path}/lora_adapter"
    os.makedirs(dummy_lora_files, exist_ok=True)
    create_peft_lora(
        dummy_model,
        save_dir=dummy_lora_files,
        target_modules=["layer1.dense1", "dense2"],
        lora_dtype=DEFAULT_DTYPE,
    )

    model_config = ModelConfig(max_model_len=16)
    vllm_config = VllmConfig(model_config=model_config, lora_config=lora_config)

    vllm_config.scheduler_config.max_num_seqs = 4
    vllm_config.scheduler_config.max_num_batched_tokens = 2
    worker_adapter_manager = LRUCacheWorkerLoRAManager(
        vllm_config, device, EMBEDDING_MODULES
    )

    worker_adapter_manager.max_num_seqs = 4
    worker_adapter_manager.max_num_batched_tokens = 2

    worker_adapter_manager.create_lora_manager(dummy_model)

    mapping = LoRAMapping([], [])
    worker_adapter_manager.set_active_adapters(
        [LoRARequest("1", 1, dummy_lora_files), LoRARequest("2", 2, dummy_lora_files)],
        mapping,
    )
    assert worker_adapter_manager.list_adapters() == {1, 2}
    assert worker_adapter_manager._adapter_manager.lora_index_to_id[0] == 1
    assert worker_adapter_manager._adapter_manager.lora_index_to_id[1] == 2

    worker_adapter_manager.set_active_adapters(
        [
            LoRARequest("1", 1, dummy_lora_files),
            LoRARequest("3", 3, dummy_lora_files),
            LoRARequest("4", 4, dummy_lora_files),
        ],
        mapping,
    )
    assert worker_adapter_manager.list_adapters() == {1, 2, 3, 4}
    assert worker_adapter_manager._adapter_manager.lora_index_to_id[0] == 1
    assert worker_adapter_manager._adapter_manager.lora_index_to_id[1] == 2
    assert worker_adapter_manager._adapter_manager.lora_index_to_id[2] == 3
    assert worker_adapter_manager._adapter_manager.lora_index_to_id[3] == 4

    worker_adapter_manager.set_active_adapters(
        [
            LoRARequest("1", 1, dummy_lora_files),
            LoRARequest("2", 2, dummy_lora_files),
            LoRARequest("5", 5, dummy_lora_files),
        ],
        mapping,
    )
    assert worker_adapter_manager.list_adapters() == {1, 2, 4, 5}
    assert worker_adapter_manager._adapter_manager.lora_index_to_id[0] == 1
    assert worker_adapter_manager._adapter_manager.lora_index_to_id[1] == 2
    assert worker_adapter_manager._adapter_manager.lora_index_to_id[2] == 5
    assert worker_adapter_manager._adapter_manager.lora_index_to_id[3] == 4

    worker_adapter_manager.set_active_adapters(
        [
            LoRARequest("1", 1, dummy_lora_files),
            LoRARequest("1", 1, dummy_lora_files),
            LoRARequest("1", 1, dummy_lora_files),
        ],
        mapping,
    )
    assert worker_adapter_manager.list_adapters() == {1, 2, 4, 5}
    assert worker_adapter_manager._adapter_manager.lora_index_to_id[0] == 1
    assert worker_adapter_manager._adapter_manager.lora_index_to_id[1] == 2
    assert worker_adapter_manager._adapter_manager.lora_index_to_id[2] == 5
    assert worker_adapter_manager._adapter_manager.lora_index_to_id[3] == 4

    worker_adapter_manager.set_active_adapters(
        [
            LoRARequest("6", 6, dummy_lora_files),
            LoRARequest("7", 7, dummy_lora_files),
            LoRARequest("8", 8, dummy_lora_files),
        ],
        mapping,
    )
    assert worker_adapter_manager.list_adapters() == {1, 6, 7, 8}
    assert worker_adapter_manager._adapter_manager.lora_index_to_id[0] == 1
    assert worker_adapter_manager._adapter_manager.lora_index_to_id[1] == 7
    assert worker_adapter_manager._adapter_manager.lora_index_to_id[2] == 8
    assert worker_adapter_manager._adapter_manager.lora_index_to_id[3] == 6

    # Over capacity
    with pytest.raises(RuntimeError):
        worker_adapter_manager.set_active_adapters(
            [
                LoRARequest("10", 10, dummy_lora_files),
                LoRARequest("11", 11, dummy_lora_files),
                LoRARequest("12", 12, dummy_lora_files),
                LoRARequest("13", 13, dummy_lora_files),
                LoRARequest("14", 14, dummy_lora_files),
            ],
            mapping,
        )

    assert worker_adapter_manager.device == device
    punica_wrapper = worker_adapter_manager._adapter_manager.punica_wrapper_mapping.get(
        DEFAULT_LANGUAGE_WRAPPER_KEY
    )
    assert punica_wrapper.device == device


@pytest.mark.parametrize("device", DEVICES)
def test_worker_adapter_manager(dist_init, dummy_model_gate_up, device, tmp_path):
    # Should remove every LoRA not specified in the request.
    lora_config = LoRAConfig(
        max_lora_rank=8, max_cpu_loras=4, max_loras=4, lora_dtype=DEFAULT_DTYPE
    )

    model_config = ModelConfig(max_model_len=16)
    vllm_config = VllmConfig(model_config=model_config, lora_config=lora_config)

    vllm_config.scheduler_config.max_num_seqs = 4
    vllm_config.scheduler_config.max_num_batched_tokens = 2

    worker_adapter_manager = WorkerLoRAManager(vllm_config, device, EMBEDDING_MODULES)
    worker_adapter_manager.vocab_size = dummy_model_gate_up.unpadded_vocab_size
    worker_adapter_manager.create_lora_manager(dummy_model_gate_up)

    dummy_lora_files = f"{tmp_path}/lora_adapter"
    os.makedirs(dummy_lora_files, exist_ok=True)
    create_peft_lora(
        dummy_model_gate_up,
        save_dir=dummy_lora_files,
        target_modules=["layer1.dense1", "dense2"],
        lora_dtype=DEFAULT_DTYPE,
    )

    mapping = LoRAMapping([], [])
    worker_adapter_manager.set_active_adapters(
        [LoRARequest("1", 1, dummy_lora_files), LoRARequest("2", 2, dummy_lora_files)],
        mapping,
    )
    assert worker_adapter_manager.list_adapters() == {1, 2}
    assert worker_adapter_manager._adapter_manager.lora_index_to_id[0] == 1
    assert worker_adapter_manager._adapter_manager.lora_index_to_id[1] == 2

    worker_adapter_manager.set_active_adapters(
        [
            LoRARequest("1", 1, dummy_lora_files),
            LoRARequest("3", 3, dummy_lora_files),
            LoRARequest("4", 4, dummy_lora_files),
        ],
        mapping,
    )
    assert worker_adapter_manager.list_adapters() == {1, 3, 4}
    assert worker_adapter_manager._adapter_manager.lora_index_to_id[0] == 1
    assert worker_adapter_manager._adapter_manager.lora_index_to_id[1] == 3
    assert worker_adapter_manager._adapter_manager.lora_index_to_id[2] == 4

    worker_adapter_manager.set_active_adapters(
        [
            LoRARequest("1", 1, dummy_lora_files),
            LoRARequest("2", 2, dummy_lora_files),
            LoRARequest("5", 5, dummy_lora_files),
        ],
        mapping,
    )
    assert worker_adapter_manager.list_adapters() == {1, 2, 5}
    assert worker_adapter_manager._adapter_manager.lora_index_to_id[0] == 1
    assert worker_adapter_manager._adapter_manager.lora_index_to_id[1] == 2
    assert worker_adapter_manager._adapter_manager.lora_index_to_id[2] == 5

    worker_adapter_manager.set_active_adapters(
        [
            LoRARequest("1", 1, dummy_lora_files),
            LoRARequest("1", 1, dummy_lora_files),
            LoRARequest("1", 1, dummy_lora_files),
        ],
        mapping,
    )
    assert worker_adapter_manager.list_adapters() == {1}
    assert worker_adapter_manager._adapter_manager.lora_index_to_id[0] == 1
    assert worker_adapter_manager._adapter_manager.lora_index_to_id[1] is None
    assert worker_adapter_manager._adapter_manager.lora_index_to_id[2] is None

    worker_adapter_manager.set_active_adapters(
        [
            LoRARequest("6", 6, dummy_lora_files),
            LoRARequest("7", 7, dummy_lora_files),
            LoRARequest("8", 8, dummy_lora_files),
        ],
        mapping,
    )
    assert worker_adapter_manager.list_adapters() == {6, 7, 8}
    assert worker_adapter_manager._adapter_manager.lora_index_to_id[0] == 8
    assert worker_adapter_manager._adapter_manager.lora_index_to_id[1] == 6
    assert worker_adapter_manager._adapter_manager.lora_index_to_id[2] == 7

    # Over capacity
    with pytest.raises(RuntimeError):
        worker_adapter_manager.set_active_adapters(
            [
                LoRARequest("10", 10, dummy_lora_files),
                LoRARequest("11", 11, dummy_lora_files),
                LoRARequest("12", 12, dummy_lora_files),
                LoRARequest("13", 13, dummy_lora_files),
                LoRARequest("14", 14, dummy_lora_files),
            ],
            mapping,
        )

    assert worker_adapter_manager.device == device
    punica_wrapper = worker_adapter_manager._adapter_manager.punica_wrapper_mapping.get(
        DEFAULT_LANGUAGE_WRAPPER_KEY
    )
    assert punica_wrapper.device == device


@pytest.mark.parametrize("device", DEVICES)
def test_packed_loras(dist_init, dummy_model_gate_up, device):
    model = dummy_model_gate_up
    model_lora = create_packed_lora(
        1,
        model,
        module_name="gate_up_proj",
        replaced_module_names=["gate_proj", "up_proj"],
        device=device,
    )
    model_lora1 = create_packed_lora(
        2,
        model,
        module_name="gate_up_proj",
        replaced_module_names=["gate_proj", "up_proj"],
        device=device,
        empty_replaced_module_name="gate_proj",
    )

    manager = LoRAModelManager(
        model,
        2,
        2,
        2,
        LoRAConfig(
            max_lora_rank=8, max_cpu_loras=2, max_loras=2, lora_dtype=DEFAULT_DTYPE
        ),
        device=device,
    )
    model = manager.model

    assert isinstance(
        model.get_submodule("gate_up_proj"), MergedColumnParallelLinearWithLoRA
    )
    # Verify packed lora is correct
    model_lora_clone = model_lora.clone(1)
    model_lora_clone1 = model_lora1.clone(1)
    assert manager.add_adapter(model_lora)
    assert manager.add_adapter(model_lora1)

    assert model_lora.get_lora("gate_proj") is None
    assert model_lora.get_lora("up_proj") is None
    assert model_lora1.get_lora("up_proj") is None
    packed_lora = model_lora.get_lora("gate_up_proj")
    assert packed_lora and isinstance(packed_lora, PackedLoRALayerWeights)

    torch.testing.assert_close(
        packed_lora.lora_a[0], model_lora_clone.get_lora("gate_proj").lora_a
    )
    torch.testing.assert_close(
        packed_lora.lora_b[0], model_lora_clone.get_lora("gate_proj").lora_b
    )
    torch.testing.assert_close(
        packed_lora.lora_a[1], model_lora_clone.get_lora("up_proj").lora_a
    )
    torch.testing.assert_close(
        packed_lora.lora_b[1], model_lora_clone.get_lora("up_proj").lora_b
    )

    packed_lora1 = model_lora1.get_lora("gate_up_proj")
    assert packed_lora1 and isinstance(packed_lora1, PackedLoRALayerWeights)

    assert packed_lora1.lora_a[0] is None
    assert packed_lora1.lora_b[0] is None
    torch.testing.assert_close(
        packed_lora1.lora_a[1], model_lora_clone1.get_lora("up_proj").lora_a
    )
    torch.testing.assert_close(
        packed_lora1.lora_b[1], model_lora_clone1.get_lora("up_proj").lora_b
    )
