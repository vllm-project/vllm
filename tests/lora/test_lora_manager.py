# SPDX-License-Identifier: Apache-2.0

import os
from typing import Dict, List

import pytest
import torch
from safetensors.torch import load_file
from torch import nn

from vllm.config import LoRAConfig
from vllm.lora.layers import (ColumnParallelLinearWithLoRA,
                              MergedColumnParallelLinearWithLoRA,
                              RowParallelLinearWithLoRA)
from vllm.lora.lora import LoRALayerWeights, PackedLoRALayerWeights
from vllm.lora.models import (LoRAMapping, LoRAModel, LoRAModelManager,
                              LRUCacheLoRAModelManager)
from vllm.lora.peft_helper import PEFTHelper
from vllm.lora.request import LoRARequest
from vllm.lora.worker_manager import (LRUCacheWorkerLoRAManager,
                                      WorkerLoRAManager)
from vllm.platforms import current_platform

EMBEDDING_MODULES = {
    "embed_tokens": "input_embeddings",
    "lm_head": "output_embeddings",
}

EMBEDDING_PADDING_MODULES = ["lm_head"]

DEVICES = ([
    f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)
] if current_platform.is_cuda_alike() else ["cpu"])


@pytest.mark.parametrize("device", DEVICES)
def test_from_lora_tensors(sql_lora_files, device):
    tensors = load_file(
        os.path.join(sql_lora_files, "adapter_model.safetensors"))
    new_embeddings = load_file(
        os.path.join(sql_lora_files, "new_embeddings.safetensors"))

    peft_helper = PEFTHelper.from_local_dir(sql_lora_files,
                                            max_position_embeddings=4096)
    lora_model = LoRAModel.from_lora_tensors(
        1,
        tensors,
        peft_helper=peft_helper,
        device=device,
        embeddings=new_embeddings,
        embedding_modules=EMBEDDING_MODULES,
        embedding_padding_modules=EMBEDDING_PADDING_MODULES,
    )
    for module_name, lora in lora_model.loras.items():
        assert lora.module_name == module_name
        assert lora.rank == 8
        assert lora.lora_alpha == 16
        assert lora.lora_a is not None
        assert lora.lora_b is not None
        assert lora.lora_a.device == torch.device(device)
        assert lora.lora_b.device == torch.device(device)
        assert (lora.lora_a.shape[1] == lora.lora_b.shape[0]
                ), f"{lora.lora_a.shape=}, {lora.lora_b.shape=}"
        assert lora.lora_a.shape[1] == 8
        embeddings_module = next(
            (k for k in EMBEDDING_MODULES if k in module_name), None)
        if embeddings_module:
            assert torch.equal(
                lora.embeddings_tensor,
                new_embeddings[EMBEDDING_MODULES[embeddings_module]].to(
                    device=lora.embeddings_tensor.device),
            )
        else:
            assert lora.embeddings_tensor is None


def create_lora(
    lora_id: int,
    model: nn.Module,
    sub_modules: List[str],
    device: torch.device,
    use_dora: bool = False,
) -> LoRAModel:
    loras: Dict[str, LoRALayerWeights] = {}
    for name in sub_modules:
        w = model.get_submodule(name).weight

        # For DoRA, also create magnitude parameters
        magnitude_param = None
        if use_dora:
            magnitude_param = torch.abs(torch.rand([w.shape[0]],
                                                   device=device))

        loras[name] = LoRALayerWeights(
            name,
            8,
            16,
            torch.rand([w.shape[1], 8], device=device),
            torch.rand([8, w.shape[0]], device=device),
            magnitude_param=magnitude_param,
        )
    return LoRAModel(lora_id, 8, loras)


def create_packed_lora(
    lora_id: int,
    model: nn.Module,
    module_name,
    replaced_module_names,
    device: torch.device,
    empty_replaced_module_name=None,
    use_dora: bool = False,
) -> LoRAModel:
    w = model.get_submodule(module_name).weight
    loras: dict[str, LoRALayerWeights] = {}
    for replaced_module_name in replaced_module_names:
        if replaced_module_name == empty_replaced_module_name:
            continue

        # For DoRA, create magnitude parameters
        magnitude_param = None
        if use_dora:
            magnitude_param = torch.abs(
                torch.rand([w.shape[0] // len(replaced_module_names)],
                           device=device))

        loras[replaced_module_name] = LoRALayerWeights(
            replaced_module_name,
            8,
            16,
            torch.rand([w.shape[1], 8], device=device),
            torch.rand([8, w.shape[0] // len(replaced_module_names)],
                       device=device),
            magnitude_param=magnitude_param,
        )
    return LoRAModel(lora_id, 8, loras)


def test_replace_submodules(dist_init, dummy_model):
    model = dummy_model
    manager = LoRAModelManager(
        model,
        1,
        1,
        1,
        LoRAConfig(max_lora_rank=8, max_cpu_loras=8, max_loras=8),
        torch.device(DEVICES[0]),
    )
    model = manager.model
    assert isinstance(model.get_submodule("dense1"),
                      ColumnParallelLinearWithLoRA)
    assert isinstance(model.get_submodule("layer1.dense1"),
                      ColumnParallelLinearWithLoRA)
    assert isinstance(model.get_submodule("dense2"), RowParallelLinearWithLoRA)
    assert isinstance(model.get_submodule("layer1.dense2"),
                      RowParallelLinearWithLoRA)


@pytest.mark.parametrize("device", DEVICES)
def test_lora_model_manager(dist_init, dummy_model, device):
    model = dummy_model
    model_lora1 = create_lora(1,
                              model, ["layer1.dense1", "dense2", "lm_head"],
                              device=device)
    model_lora2 = create_lora(2,
                              model, ["dense1", "dense2", "lm_head"],
                              device=device)
    model_lora3 = create_lora(3,
                              model, ["dense1", "dense2", "lm_head"],
                              device=device)
    manager = LoRAModelManager(
        model,
        2,
        2,
        2,
        LoRAConfig(max_lora_rank=8, max_cpu_loras=3, max_loras=2),
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
    assert manager.punica_wrapper.device == device
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
    model_lora1 = create_lora(1,
                              model, ["layer1.dense1", "dense2", "lm_head"],
                              device=device)
    model_lora2 = create_lora(2,
                              model, ["dense1", "dense2", "lm_head"],
                              device=device)
    model_lora3 = create_lora(3,
                              model, ["dense1", "dense2", "lm_head"],
                              device=device)
    manager = LRUCacheLoRAModelManager(
        model,
        2,
        2,
        2,
        LoRAConfig(max_lora_rank=8, max_cpu_loras=3, max_loras=2),
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

    assert manager.punica_wrapper.device == device
    assert manager.device == device


@pytest.mark.parametrize("device", DEVICES)
def test_lru_lora_model_manager(dist_init, dummy_model, device):
    # This tests just the LRU cache functionality, everything else is
    # tested in test_lora_model_manager
    model = dummy_model
    model_lora1 = create_lora(1,
                              model, ["layer1.dense1", "dense2", "lm_head"],
                              device=device)
    model_lora2 = create_lora(2,
                              model, ["dense1", "dense2", "lm_head"],
                              device=device)
    model_lora3 = create_lora(3,
                              model, ["dense1", "dense2", "lm_head"],
                              device=device)
    model_lora4 = create_lora(4,
                              model, ["dense1", "dense2", "lm_head"],
                              device=device)
    manager = LRUCacheLoRAModelManager(
        model,
        2,
        2,
        2,
        LoRAConfig(max_lora_rank=8, max_cpu_loras=2, max_loras=2),
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
    assert manager.punica_wrapper.device == device
    assert manager.device == device


@pytest.mark.parametrize("device", DEVICES)
def test_lru_cache_worker_adapter_manager(llama_2_7b_model_extra_embeddings,
                                          sql_lora_files, device):
    lora_config = LoRAConfig(max_lora_rank=8, max_cpu_loras=4, max_loras=4)
    worker_adapter_manager = LRUCacheWorkerLoRAManager(
        4,
        2,
        llama_2_7b_model_extra_embeddings.unpadded_vocab_size -
        lora_config.lora_extra_vocab_size,
        lora_config,
        device,
        EMBEDDING_MODULES,
        EMBEDDING_PADDING_MODULES,
    )
    worker_adapter_manager.create_lora_manager(
        llama_2_7b_model_extra_embeddings)

    mapping = LoRAMapping([], [])
    worker_adapter_manager.set_active_adapters(
        [
            LoRARequest("1", 1, sql_lora_files),
            LoRARequest("2", 2, sql_lora_files)
        ],
        mapping,
    )
    assert worker_adapter_manager.list_adapters() == {1, 2}
    assert worker_adapter_manager._adapter_manager.lora_index_to_id[0] == 1
    assert worker_adapter_manager._adapter_manager.lora_index_to_id[1] == 2

    worker_adapter_manager.set_active_adapters(
        [
            LoRARequest("1", 1, sql_lora_files),
            LoRARequest("3", 3, sql_lora_files),
            LoRARequest("4", 4, sql_lora_files),
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
            LoRARequest("1", 1, sql_lora_files),
            LoRARequest("2", 2, sql_lora_files),
            LoRARequest("5", 5, sql_lora_files),
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
            LoRARequest("1", 1, sql_lora_files),
            LoRARequest("1", 1, sql_lora_files),
            LoRARequest("1", 1, sql_lora_files),
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
            LoRARequest("6", 6, sql_lora_files),
            LoRARequest("7", 7, sql_lora_files),
            LoRARequest("8", 8, sql_lora_files),
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
                LoRARequest("10", 10, sql_lora_files),
                LoRARequest("11", 11, sql_lora_files),
                LoRARequest("12", 12, sql_lora_files),
                LoRARequest("13", 13, sql_lora_files),
                LoRARequest("14", 14, sql_lora_files),
            ],
            mapping,
        )

    assert worker_adapter_manager.device == device
    assert worker_adapter_manager._adapter_manager.punica_wrapper.device == device


@pytest.mark.parametrize("device", DEVICES)
def test_worker_adapter_manager(llama_2_7b_model_extra_embeddings,
                                sql_lora_files, device):
    # Should remove every LoRA not specified in the request.
    lora_config = LoRAConfig(max_lora_rank=8, max_cpu_loras=4, max_loras=4)
    worker_adapter_manager = WorkerLoRAManager(
        4,
        2,
        llama_2_7b_model_extra_embeddings.unpadded_vocab_size -
        lora_config.lora_extra_vocab_size,
        lora_config,
        device,
        EMBEDDING_MODULES,
        EMBEDDING_PADDING_MODULES,
    )
    worker_adapter_manager.create_lora_manager(
        llama_2_7b_model_extra_embeddings)

    mapping = LoRAMapping([], [])
    worker_adapter_manager.set_active_adapters(
        [
            LoRARequest("1", 1, sql_lora_files),
            LoRARequest("2", 2, sql_lora_files)
        ],
        mapping,
    )
    assert worker_adapter_manager.list_adapters() == {1, 2}
    assert worker_adapter_manager._adapter_manager.lora_index_to_id[0] == 1
    assert worker_adapter_manager._adapter_manager.lora_index_to_id[1] == 2

    worker_adapter_manager.set_active_adapters(
        [
            LoRARequest("1", 1, sql_lora_files),
            LoRARequest("3", 3, sql_lora_files),
            LoRARequest("4", 4, sql_lora_files),
        ],
        mapping,
    )
    assert worker_adapter_manager.list_adapters() == {1, 3, 4}
    assert worker_adapter_manager._adapter_manager.lora_index_to_id[0] == 1
    assert worker_adapter_manager._adapter_manager.lora_index_to_id[1] == 3
    assert worker_adapter_manager._adapter_manager.lora_index_to_id[2] == 4

    worker_adapter_manager.set_active_adapters(
        [
            LoRARequest("1", 1, sql_lora_files),
            LoRARequest("2", 2, sql_lora_files),
            LoRARequest("5", 5, sql_lora_files),
        ],
        mapping,
    )
    assert worker_adapter_manager.list_adapters() == {1, 2, 5}
    assert worker_adapter_manager._adapter_manager.lora_index_to_id[0] == 1
    assert worker_adapter_manager._adapter_manager.lora_index_to_id[1] == 2
    assert worker_adapter_manager._adapter_manager.lora_index_to_id[2] == 5

    worker_adapter_manager.set_active_adapters(
        [
            LoRARequest("1", 1, sql_lora_files),
            LoRARequest("1", 1, sql_lora_files),
            LoRARequest("1", 1, sql_lora_files),
        ],
        mapping,
    )
    assert worker_adapter_manager.list_adapters() == {1}
    assert worker_adapter_manager._adapter_manager.lora_index_to_id[0] == 1
    assert worker_adapter_manager._adapter_manager.lora_index_to_id[1] is None
    assert worker_adapter_manager._adapter_manager.lora_index_to_id[2] is None

    worker_adapter_manager.set_active_adapters(
        [
            LoRARequest("6", 6, sql_lora_files),
            LoRARequest("7", 7, sql_lora_files),
            LoRARequest("8", 8, sql_lora_files),
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
                LoRARequest("10", 10, sql_lora_files),
                LoRARequest("11", 11, sql_lora_files),
                LoRARequest("12", 12, sql_lora_files),
                LoRARequest("13", 13, sql_lora_files),
                LoRARequest("14", 14, sql_lora_files),
            ],
            mapping,
        )

    assert worker_adapter_manager.device == device
    assert worker_adapter_manager._adapter_manager.punica_wrapper.device == device


@pytest.mark.parametrize("device", DEVICES)
def test_worker_adapter_manager_with_dora(llama_2_7b_model_extra_embeddings,
                                          dora_files, device):
    """Test worker adapter manager with a real DoRA adapter."""
    # Initialize worker manager
    lora_config = LoRAConfig(max_lora_rank=16, max_cpu_loras=4, max_loras=4)
    worker_adapter_manager = WorkerLoRAManager(
        4,
        2,
        llama_2_7b_model_extra_embeddings.unpadded_vocab_size -
        lora_config.lora_extra_vocab_size,
        lora_config,
        device,
        EMBEDDING_MODULES,
        EMBEDDING_PADDING_MODULES,
    )
    worker_adapter_manager.create_lora_manager(
        llama_2_7b_model_extra_embeddings)

    # Create mapping for requests
    mapping = LoRAMapping([], [])

    # Set a DoRA adapter as active
    worker_adapter_manager.set_active_adapters(
        [LoRARequest("dora_adapter", 1, dora_files)], mapping)

    # Verify the adapter was loaded
    assert worker_adapter_manager.list_adapters() == {1}
    assert worker_adapter_manager._adapter_manager.lora_index_to_id[0] == 1

    # Get the LoRA model to check properties
    lora_model = worker_adapter_manager._adapter_manager.get_adapter(1)
    assert lora_model is not None

    # Check that it has expected DoRA properties:
    # 1. Some modules should have magnitude parameters
    has_magnitudes = False
    for module_name, lora_weights in lora_model.loras.items():
        if (hasattr(lora_weights, "magnitude_param")
                and lora_weights.magnitude_param is not None):
            has_magnitudes = True

            # Different implementations handle magnitudes differently
            if isinstance(lora_weights.magnitude_param, list):
                # If it's a list, check the first non-None element
                for mag in lora_weights.magnitude_param:
                    if mag is not None:
                        # Just validate it's a tensor and has data
                        # We don't check device since it might be on CPU or GPU
                        # in different implementations
                        assert isinstance(mag, torch.Tensor)
                        assert mag.numel() > 0
                        break
            else:
                # If it's a tensor, direct comparison works
                # Check shapes are reasonable
                assert (lora_weights.magnitude_param.shape[0] ==
                        lora_weights.lora_b.shape[1])
                # We don't check device since it might be on CPU or GPU
                # in different implementations
                assert isinstance(lora_weights.magnitude_param, torch.Tensor)

    # Verify at least some modules have magnitude parameters
    assert has_magnitudes, "DoRA adapter should have magnitude parameters"


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
        LoRAConfig(max_lora_rank=8, max_cpu_loras=2, max_loras=2),
        device=device,
    )
    model = manager.model

    assert isinstance(model.get_submodule("gate_up_proj"),
                      MergedColumnParallelLinearWithLoRA)
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

    torch.testing.assert_close(packed_lora.lora_a[0],
                               model_lora_clone.get_lora("gate_proj").lora_a)
    torch.testing.assert_close(packed_lora.lora_b[0],
                               model_lora_clone.get_lora("gate_proj").lora_b)
    torch.testing.assert_close(packed_lora.lora_a[1],
                               model_lora_clone.get_lora("up_proj").lora_a)
    torch.testing.assert_close(packed_lora.lora_b[1],
                               model_lora_clone.get_lora("up_proj").lora_b)

    packed_lora1 = model_lora1.get_lora("gate_up_proj")
    assert packed_lora1 and isinstance(packed_lora1, PackedLoRALayerWeights)

    assert packed_lora1.lora_a[0] is None
    assert packed_lora1.lora_b[0] is None
    torch.testing.assert_close(packed_lora1.lora_a[1],
                               model_lora_clone1.get_lora("up_proj").lora_a)
    torch.testing.assert_close(packed_lora1.lora_b[1],
                               model_lora_clone1.get_lora("up_proj").lora_b)


@pytest.mark.parametrize("device", DEVICES)
def test_dora_model_manager(dist_init, dummy_model, device):
    """Test that the LoRA model manager handles DoRA adapters correctly."""
    model = dummy_model

    # Create LoRA adapters with DoRA enabled
    dora_lora1 = create_lora(1,
                             model, ["layer1.dense1", "dense2", "lm_head"],
                             device=device,
                             use_dora=True)
    dora_lora2 = create_lora(2,
                             model, ["dense1", "dense2", "lm_head"],
                             device=device,
                             use_dora=True)

    # Create a regular LoRA adapter for comparison
    std_lora3 = create_lora(3,
                            model, ["dense1", "dense2", "lm_head"],
                            device=device,
                            use_dora=False)

    # Initialize the manager with a model
    manager = LoRAModelManager(
        model,
        2,
        2,
        2,
        LoRAConfig(max_lora_rank=8, max_cpu_loras=3, max_loras=2),
        device=device,
    )

    # Check that the slots are empty initially
    assert all(x is None for x in manager.lora_index_to_id)

    # Add and activate the first DoRA adapter
    assert manager.add_adapter(dora_lora1)
    assert manager.activate_adapter(1)
    assert manager.lora_index_to_id[0] == 1

    # Verify that modules have magnitude parameters
    for module_name, lora_module in dora_lora1.loras.items():
        assert lora_module.magnitude_param is not None
        # Check that magnitude parameter has the right shape
        assert lora_module.magnitude_param.shape[
            0] == lora_module.lora_b.shape[1]
        # Verify that the magnitude parameter is on the right device
        assert lora_module.magnitude_param.device == torch.device(device)

    # Add and activate the second DoRA adapter
    assert manager.add_adapter(dora_lora2)
    assert manager.activate_adapter(2)
    assert manager.lora_index_to_id[0] == 1
    assert manager.lora_index_to_id[1] == 2

    # Add standard LoRA and check it can coexist with DoRA adapters
    assert manager.add_adapter(std_lora3)

    # Need to free a slot first
    assert manager.deactivate_adapter(1)
    assert manager.lora_index_to_id[0] is None
    assert manager.activate_adapter(3)
    assert manager.lora_index_to_id[0] == 3

    # Verify the standard LoRA doesn't have magnitude parameters
    for module_name, lora_module in std_lora3.loras.items():
        assert lora_module.magnitude_param is None

    # Clean up
    assert manager.remove_adapter(2)
    assert manager.remove_adapter(3)


@pytest.mark.parametrize("device", DEVICES)
def test_packed_dora_loras(dist_init, dummy_model_gate_up, device):
    """Test that packed DoRA LoRAs work correctly in the manager."""
    model = dummy_model_gate_up

    # Create a packed DoRA LoRA
    dora_model_lora = create_packed_lora(
        1,
        model,
        module_name="gate_up_proj",
        replaced_module_names=["gate_proj", "up_proj"],
        device=device,
        use_dora=True,
    )

    # Create a non-DoRA packed LoRA for comparison
    std_model_lora = create_packed_lora(
        2,
        model,
        module_name="gate_up_proj",
        replaced_module_names=["gate_proj", "up_proj"],
        device=device,
        use_dora=False,
    )

    # Set up the manager
    manager = LoRAModelManager(
        model,
        2,
        2,
        2,
        LoRAConfig(max_lora_rank=8, max_cpu_loras=2, max_loras=2),
        device=device,
    )
    model = manager.model

    # Verify manager setup
    assert isinstance(model.get_submodule("gate_up_proj"),
                      MergedColumnParallelLinearWithLoRA)

    # Create clones before adding to manager
    dora_model_lora_clone = dora_model_lora.clone(1)
    std_model_lora_clone = std_model_lora.clone(2)

    # Add both adapters to the manager
    assert manager.add_adapter(dora_model_lora)
    assert manager.add_adapter(std_model_lora)

    # Verify DoRA adapter has magnitude parameters
    packed_dora_lora = dora_model_lora.get_lora("gate_up_proj")
    assert packed_dora_lora and isinstance(packed_dora_lora,
                                           PackedLoRALayerWeights)

    # Check that magnitudes are properly packed
    for i, replaced_module in enumerate(["gate_proj", "up_proj"]):
        # Verify magnitude parameters exist for DoRA
        original_lora = dora_model_lora_clone.get_lora(replaced_module)
        assert original_lora.magnitude_param is not None

        # The implementation might pack magnitudes as a list or as tensors
        # Handle both cases
        if isinstance(packed_dora_lora.magnitude_param, list):
            assert packed_dora_lora.magnitude_param[i] is not None
            torch.testing.assert_close(packed_dora_lora.magnitude_param[i],
                                       original_lora.magnitude_param)
        else:
            # If it's a tensor, compare the relevant slice
            assert packed_dora_lora.magnitude_param is not None
            # Get the appropriate slice based on output dimensions
            slice_size = original_lora.magnitude_param.shape[0]
            start_idx = i * slice_size
            end_idx = (i + 1) * slice_size
            torch.testing.assert_close(
                packed_dora_lora.magnitude_param[start_idx:end_idx],
                original_lora.magnitude_param,
            )

    # Verify standard LoRA doesn't have magnitude parameters
    packed_std_lora = std_model_lora.get_lora("gate_up_proj")
    assert packed_std_lora and isinstance(packed_std_lora,
                                          PackedLoRALayerWeights)

    # For standard LoRA, magnitude parameters should be None
    assert packed_std_lora.magnitude_param is None or all(
        m is None for m in packed_std_lora.magnitude_param)
