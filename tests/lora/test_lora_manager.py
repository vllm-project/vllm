import os
from typing import List

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
from vllm.lora.request import LoRARequest
from vllm.lora.worker_manager import (LRUCacheWorkerLoRAManager,
                                      WorkerLoRAManager)
from vllm.model_executor.layers.linear import RowParallelLinear

EMBEDDING_MODULES = {
    "embed_tokens": "input_embeddings",
    "lm_head": "output_embeddings",
}

EMBEDDING_PADDING_MODULES = ["lm_head"]


def test_from_lora_tensors(sql_lora_files):
    tensors = load_file(
        os.path.join(sql_lora_files, "adapter_model.safetensors"))
    new_embeddings = load_file(
        os.path.join(sql_lora_files, "new_embeddings.safetensors"))
    lora_model = LoRAModel.from_lora_tensors(
        1,
        8,
        16,
        tensors,
        "cuda",
        embeddings=new_embeddings,
        embedding_modules=EMBEDDING_MODULES,
        embedding_padding_modules=EMBEDDING_PADDING_MODULES)
    for module_name, lora in lora_model.loras.items():
        assert lora.module_name == module_name
        assert lora.rank == 8
        assert lora.lora_alpha == 16
        assert lora.lora_a is not None
        assert lora.lora_b is not None
        assert (lora.lora_a.shape[1] == lora.lora_b.shape[0]
                ), f"{lora.lora_a.shape=}, {lora.lora_b.shape=}"
        assert lora.lora_a.shape[1] == 8
        embeddings_module = next(
            (k for k in EMBEDDING_MODULES if k in module_name), None)
        if embeddings_module:
            assert torch.equal(
                lora.embeddings_tensor,
                new_embeddings[EMBEDDING_MODULES[embeddings_module]].to(
                    device=lora.embeddings_tensor.device))
        else:
            assert lora.embeddings_tensor is None


def create_lora(lora_id: int, model: nn.Module,
                sub_modules: List[str]) -> LoRAModel:
    loras = {}
    for name in sub_modules:
        w = model.get_submodule(name).weight
        loras[name] = LoRALayerWeights(
            name,
            8,
            16,
            torch.rand([w.shape[1], 8], device="cuda"),
            torch.rand([8, w.shape[0]], device="cuda"),
        )
    return LoRAModel(lora_id, 8, loras)


def create_packed_lora(
    lora_id: int,
    model: nn.Module,
    module_name,
    replaced_module_names,
    empty_replaced_module_name=None,
) -> LoRAModel:
    w = model.get_submodule(module_name).weight
    loras = {}
    for replaced_module_name in replaced_module_names:
        if replaced_module_name == empty_replaced_module_name:
            continue
        loras[replaced_module_name] = LoRALayerWeights(
            replaced_module_name,
            8,
            16,
            torch.rand([w.shape[1], 8], device="cuda"),
            torch.rand([8, w.shape[0] // len(replaced_module_names)],
                       device="cuda"),
        )
    return LoRAModel(lora_id, 8, loras)


def test_replace_submodules(dist_init, dummy_model):
    model = dummy_model
    model.supported_lora_modules = ["dense1", "layer1.dense2"]
    model.packed_modules_mapping = {}
    manager = LoRAModelManager(
        model, 1, 1, 1,
        LoRAConfig(max_lora_rank=8, max_cpu_loras=8, max_loras=8))
    model = manager.model

    assert isinstance(model.get_submodule("dense1"),
                      ColumnParallelLinearWithLoRA)
    assert isinstance(model.get_submodule("layer1.dense1"),
                      ColumnParallelLinearWithLoRA)
    assert isinstance(model.get_submodule("dense2"), RowParallelLinear)
    assert isinstance(model.get_submodule("layer1.dense2"),
                      RowParallelLinearWithLoRA)


def test_lora_model_manager(dist_init, dummy_model):
    model = dummy_model
    model.supported_lora_modules = ["dense1", "dense2", "lm_head"]
    model.packed_modules_mapping = {}
    model_lora1 = create_lora(1, model, ["layer1.dense1", "dense2", "lm_head"])
    model_lora2 = create_lora(2, model, ["dense1", "dense2", "lm_head"])
    model_lora3 = create_lora(3, model, ["dense1", "dense2", "lm_head"])
    manager = LoRAModelManager(
        model, 2, 2, 2,
        LoRAConfig(max_lora_rank=8, max_cpu_loras=3, max_loras=2))
    assert all(x is None for x in manager.lora_index_to_id)
    assert manager.add_lora(model_lora1)
    assert manager.activate_lora(1)
    assert manager.lora_index_to_id[0] == 1
    assert not manager.add_lora(model_lora1)
    assert not manager.activate_lora(1)
    assert manager.add_lora(model_lora2)
    assert manager.activate_lora(2)
    assert manager.lora_index_to_id[0] == 1
    assert manager.lora_index_to_id[1] == 2
    assert not manager.add_lora(model_lora2)
    assert not manager.activate_lora(2)
    assert manager.add_lora(model_lora3)
    assert manager.lora_index_to_id[0] == 1
    assert manager.lora_index_to_id[1] == 2
    with pytest.raises(ValueError):
        assert manager.activate_lora(3)
    assert manager.lora_index_to_id[0] == 1
    assert manager.lora_index_to_id[1] == 2
    assert manager.remove_lora(model_lora2.id)
    assert manager.lora_index_to_id[1] is None
    assert not manager.remove_lora(model_lora2.id)
    assert manager.remove_lora(model_lora1.id)
    assert not manager.remove_lora(model_lora1.id)
    assert manager.add_lora(model_lora1)
    assert manager.lora_index_to_id[0] is None
    assert manager.lora_index_to_id[1] is None
    assert manager.add_lora(model_lora2)
    assert manager.activate_lora(3)
    assert manager.lora_index_to_id[0] == 3
    assert manager.lora_index_to_id[1] is None
    assert manager.activate_lora(2)
    assert manager.lora_index_to_id[0] == 3
    assert manager.lora_index_to_id[1] == 2


def test_lora_lru_cache_model_manager(dist_init, dummy_model):
    model = dummy_model
    model.supported_lora_modules = ["dense1", "dense2", "lm_head"]
    model.packed_modules_mapping = {}
    model_lora1 = create_lora(1, model, ["layer1.dense1", "dense2", "lm_head"])
    model_lora2 = create_lora(2, model, ["dense1", "dense2", "lm_head"])
    model_lora3 = create_lora(3, model, ["dense1", "dense2", "lm_head"])
    manager = LRUCacheLoRAModelManager(
        model, 2, 2, 2,
        LoRAConfig(max_lora_rank=8, max_cpu_loras=3, max_loras=2))
    assert all(x is None for x in manager.lora_index_to_id)
    assert manager.add_lora(model_lora1)
    assert manager.activate_lora(1)
    assert manager.lora_index_to_id[0] == 1
    assert not manager.add_lora(model_lora1)
    assert not manager.activate_lora(1)
    assert manager.add_lora(model_lora2)
    assert manager.activate_lora(2)
    assert manager.lora_index_to_id[0] == 1
    assert manager.lora_index_to_id[1] == 2
    assert not manager.add_lora(model_lora2)
    assert not manager.activate_lora(2)
    assert manager.add_lora(model_lora3)
    assert manager.lora_index_to_id[0] == 1
    assert manager.lora_index_to_id[1] == 2
    assert manager.activate_lora(3)
    assert manager.lora_index_to_id[0] == 3
    assert manager.lora_index_to_id[1] == 2
    assert manager.remove_lora(model_lora2.id)
    assert manager.lora_index_to_id[1] is None
    assert not manager.remove_lora(model_lora2.id)
    assert manager.remove_lora(model_lora1.id)
    assert not manager.remove_lora(model_lora1.id)
    assert manager.add_lora(model_lora1)
    assert manager.activate_lora(1)
    assert manager.lora_index_to_id[0] == 3
    assert manager.lora_index_to_id[1] == 1
    assert manager.add_lora(model_lora2)
    assert manager.deactivate_lora(3)
    assert manager.lora_index_to_id[0] is None
    assert manager.lora_index_to_id[1] == 1
    assert manager.activate_lora(2)
    assert manager.lora_index_to_id[0] == 2
    assert manager.lora_index_to_id[1] == 1
    assert manager.activate_lora(3)
    assert manager.lora_index_to_id[0] == 2
    assert manager.lora_index_to_id[1] == 3


def test_lru_lora_model_manager(dist_init, dummy_model):
    # This tests just the LRU cache functionality, everything else is
    # tested in test_lora_model_manager
    model = dummy_model
    model.supported_lora_modules = ["dense1", "dense2", "lm_head"]
    model.packed_modules_mapping = {}
    model_lora1 = create_lora(1, model, ["layer1.dense1", "dense2", "lm_head"])
    model_lora2 = create_lora(2, model, ["dense1", "dense2", "lm_head"])
    model_lora3 = create_lora(3, model, ["dense1", "dense2", "lm_head"])
    model_lora4 = create_lora(4, model, ["dense1", "dense2", "lm_head"])
    manager = LRUCacheLoRAModelManager(
        model, 2, 2, 2,
        LoRAConfig(max_lora_rank=8, max_cpu_loras=2, max_loras=2))

    assert all(x is None for x in manager.lora_index_to_id)

    # Add up to capacity
    assert manager.add_lora(model_lora1)
    assert manager.add_lora(model_lora2)
    assert manager.activate_lora(1)
    assert manager.activate_lora(2)

    assert set(manager.list_loras()) == {1, 2}
    assert manager.lora_index_to_id[0] == 1
    assert manager.lora_index_to_id[1] == 2

    # Add over capacity
    assert manager.add_lora(model_lora3)
    assert manager.add_lora(model_lora4)
    assert manager.activate_lora(3)
    assert manager.activate_lora(4)

    assert set(manager.list_loras()) == {3, 4}
    assert manager.lora_index_to_id[0] == 3
    assert manager.lora_index_to_id[1] == 4

    # Add 3 again to move it to the top and then add 2
    # should return false since it's in already
    assert not manager.add_lora(model_lora3)
    assert not manager.activate_lora(3)
    assert manager.add_lora(model_lora2)
    assert manager.activate_lora(2)

    assert set(manager.list_loras()) == {3, 2}
    assert manager.lora_index_to_id[0] == 3
    assert manager.lora_index_to_id[1] == 2

    # Remove manually
    assert manager.remove_lora(3)
    assert not manager.remove_lora(3)

    assert set(manager.list_loras()) == {2}
    assert manager.lora_index_to_id[0] is None
    assert manager.lora_index_to_id[1] == 2

    assert manager.add_lora(model_lora3)
    assert manager.activate_lora(3)
    assert manager.add_lora(model_lora4)
    assert manager.activate_lora(4)

    assert set(manager.list_loras()) == {3, 4}
    assert manager.lora_index_to_id[0] == 3
    assert manager.lora_index_to_id[1] == 4

    assert manager.remove_oldest_lora()
    assert set(manager.list_loras()) == {4}
    assert manager.lora_index_to_id[0] is None
    assert manager.lora_index_to_id[1] == 4

    assert manager.remove_oldest_lora()
    assert set(manager.list_loras()) == set()
    assert all(x is None for x in manager.lora_index_to_id)

    assert not manager.remove_oldest_lora()
    assert set(manager.list_loras()) == set()
    assert all(x is None for x in manager.lora_index_to_id)


def test_lru_cache_worker_lora_manager(llama_2_7b_model_extra_embeddings,
                                       sql_lora_files):
    lora_config = LoRAConfig(max_lora_rank=8, max_cpu_loras=4, max_loras=4)
    worker_lora_manager = LRUCacheWorkerLoRAManager(
        4, 2, llama_2_7b_model_extra_embeddings.unpadded_vocab_size -
        lora_config.lora_extra_vocab_size, lora_config, torch.device("cuda"),
        EMBEDDING_MODULES, EMBEDDING_PADDING_MODULES)
    worker_lora_manager.create_lora_manager(llama_2_7b_model_extra_embeddings)

    mapping = LoRAMapping([], [])
    worker_lora_manager.set_active_loras([
        LoRARequest("1", 1, sql_lora_files),
        LoRARequest("2", 2, sql_lora_files)
    ], mapping)
    assert worker_lora_manager.list_loras() == {1, 2}
    assert worker_lora_manager._lora_manager.lora_index_to_id[0] == 1
    assert worker_lora_manager._lora_manager.lora_index_to_id[1] == 2

    worker_lora_manager.set_active_loras([
        LoRARequest("1", 1, sql_lora_files),
        LoRARequest("3", 3, sql_lora_files),
        LoRARequest("4", 4, sql_lora_files)
    ], mapping)
    assert worker_lora_manager.list_loras() == {1, 2, 3, 4}
    assert worker_lora_manager._lora_manager.lora_index_to_id[0] == 1
    assert worker_lora_manager._lora_manager.lora_index_to_id[1] == 2
    assert worker_lora_manager._lora_manager.lora_index_to_id[2] == 3
    assert worker_lora_manager._lora_manager.lora_index_to_id[3] == 4

    worker_lora_manager.set_active_loras([
        LoRARequest("1", 1, sql_lora_files),
        LoRARequest("2", 2, sql_lora_files),
        LoRARequest("5", 5, sql_lora_files)
    ], mapping)
    assert worker_lora_manager.list_loras() == {1, 2, 4, 5}
    assert worker_lora_manager._lora_manager.lora_index_to_id[0] == 1
    assert worker_lora_manager._lora_manager.lora_index_to_id[1] == 2
    assert worker_lora_manager._lora_manager.lora_index_to_id[2] == 5
    assert worker_lora_manager._lora_manager.lora_index_to_id[3] == 4

    worker_lora_manager.set_active_loras([
        LoRARequest("1", 1, sql_lora_files),
        LoRARequest("1", 1, sql_lora_files),
        LoRARequest("1", 1, sql_lora_files)
    ], mapping)
    assert worker_lora_manager.list_loras() == {1, 2, 4, 5}
    assert worker_lora_manager._lora_manager.lora_index_to_id[0] == 1
    assert worker_lora_manager._lora_manager.lora_index_to_id[1] == 2
    assert worker_lora_manager._lora_manager.lora_index_to_id[2] == 5
    assert worker_lora_manager._lora_manager.lora_index_to_id[3] == 4

    worker_lora_manager.set_active_loras([
        LoRARequest("6", 6, sql_lora_files),
        LoRARequest("7", 7, sql_lora_files),
        LoRARequest("8", 8, sql_lora_files)
    ], mapping)
    assert worker_lora_manager.list_loras() == {1, 6, 7, 8}
    assert worker_lora_manager._lora_manager.lora_index_to_id[0] == 1
    assert worker_lora_manager._lora_manager.lora_index_to_id[1] == 7
    assert worker_lora_manager._lora_manager.lora_index_to_id[2] == 8
    assert worker_lora_manager._lora_manager.lora_index_to_id[3] == 6

    # Over capacity
    with pytest.raises(RuntimeError):
        worker_lora_manager.set_active_loras([
            LoRARequest("10", 10, sql_lora_files),
            LoRARequest("11", 11, sql_lora_files),
            LoRARequest("12", 12, sql_lora_files),
            LoRARequest("13", 13, sql_lora_files),
            LoRARequest("14", 14, sql_lora_files)
        ], mapping)


def test_worker_lora_manager(llama_2_7b_model_extra_embeddings,
                             sql_lora_files):
    # Should remove every LoRA not specified in the request.
    lora_config = LoRAConfig(max_lora_rank=8, max_cpu_loras=4, max_loras=4)
    worker_lora_manager = WorkerLoRAManager(
        4, 2, llama_2_7b_model_extra_embeddings.unpadded_vocab_size -
        lora_config.lora_extra_vocab_size, lora_config, torch.device("cuda"),
        EMBEDDING_MODULES, EMBEDDING_PADDING_MODULES)
    worker_lora_manager.create_lora_manager(llama_2_7b_model_extra_embeddings)

    mapping = LoRAMapping([], [])
    worker_lora_manager.set_active_loras([
        LoRARequest("1", 1, sql_lora_files),
        LoRARequest("2", 2, sql_lora_files)
    ], mapping)
    assert worker_lora_manager.list_loras() == {1, 2}
    assert worker_lora_manager._lora_manager.lora_index_to_id[0] == 1
    assert worker_lora_manager._lora_manager.lora_index_to_id[1] == 2

    worker_lora_manager.set_active_loras([
        LoRARequest("1", 1, sql_lora_files),
        LoRARequest("3", 3, sql_lora_files),
        LoRARequest("4", 4, sql_lora_files)
    ], mapping)
    assert worker_lora_manager.list_loras() == {1, 3, 4}
    assert worker_lora_manager._lora_manager.lora_index_to_id[0] == 1
    assert worker_lora_manager._lora_manager.lora_index_to_id[1] == 3
    assert worker_lora_manager._lora_manager.lora_index_to_id[2] == 4

    worker_lora_manager.set_active_loras([
        LoRARequest("1", 1, sql_lora_files),
        LoRARequest("2", 2, sql_lora_files),
        LoRARequest("5", 5, sql_lora_files)
    ], mapping)
    assert worker_lora_manager.list_loras() == {1, 2, 5}
    assert worker_lora_manager._lora_manager.lora_index_to_id[0] == 1
    assert worker_lora_manager._lora_manager.lora_index_to_id[1] == 2
    assert worker_lora_manager._lora_manager.lora_index_to_id[2] == 5

    worker_lora_manager.set_active_loras([
        LoRARequest("1", 1, sql_lora_files),
        LoRARequest("1", 1, sql_lora_files),
        LoRARequest("1", 1, sql_lora_files)
    ], mapping)
    assert worker_lora_manager.list_loras() == {1}
    assert worker_lora_manager._lora_manager.lora_index_to_id[0] == 1
    assert worker_lora_manager._lora_manager.lora_index_to_id[1] is None
    assert worker_lora_manager._lora_manager.lora_index_to_id[2] is None

    worker_lora_manager.set_active_loras([
        LoRARequest("6", 6, sql_lora_files),
        LoRARequest("7", 7, sql_lora_files),
        LoRARequest("8", 8, sql_lora_files)
    ], mapping)
    assert worker_lora_manager.list_loras() == {6, 7, 8}
    assert worker_lora_manager._lora_manager.lora_index_to_id[0] == 8
    assert worker_lora_manager._lora_manager.lora_index_to_id[1] == 6
    assert worker_lora_manager._lora_manager.lora_index_to_id[2] == 7

    # Over capacity
    with pytest.raises(RuntimeError):
        worker_lora_manager.set_active_loras([
            LoRARequest("10", 10, sql_lora_files),
            LoRARequest("11", 11, sql_lora_files),
            LoRARequest("12", 12, sql_lora_files),
            LoRARequest("13", 13, sql_lora_files),
            LoRARequest("14", 14, sql_lora_files)
        ], mapping)


def test_packed_loras(dist_init, dummy_model_gate_up):
    model = dummy_model_gate_up
    model.supported_lora_modules = ["gate_up_proj"]
    model.packed_modules_mapping = {
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }
    model_lora = create_packed_lora(
        1,
        model,
        module_name="gate_up_proj",
        replaced_module_names=["gate_proj", "up_proj"])
    model_lora1 = create_packed_lora(
        2,
        model,
        module_name="gate_up_proj",
        replaced_module_names=["gate_proj", "up_proj"],
        empty_replaced_module_name="gate_proj",
    )

    manager = LoRAModelManager(
        model, 2, 2, 2,
        LoRAConfig(max_lora_rank=8, max_cpu_loras=2, max_loras=2))
    model = manager.model

    assert isinstance(model.get_submodule("gate_up_proj"),
                      MergedColumnParallelLinearWithLoRA)
    assert manager.add_lora(model_lora)
    assert manager.add_lora(model_lora1)

    packed_lora = model_lora.get_lora("gate_up_proj")
    assert packed_lora and isinstance(packed_lora, PackedLoRALayerWeights)

    assert torch.allclose(packed_lora.lora_a[0],
                          model_lora.get_lora("gate_proj").lora_a)
    assert torch.allclose(packed_lora.lora_b[0],
                          model_lora.get_lora("gate_proj").lora_b)
    assert torch.allclose(packed_lora.lora_a[1],
                          model_lora.get_lora("up_proj").lora_a)
    assert torch.allclose(packed_lora.lora_b[1],
                          model_lora.get_lora("up_proj").lora_b)

    packed_lora1 = model_lora1.get_lora("gate_up_proj")
    assert packed_lora1 and isinstance(packed_lora1, PackedLoRALayerWeights)

    assert packed_lora1.lora_a[0] is None
    assert packed_lora1.lora_b[0] is None
    assert torch.allclose(packed_lora1.lora_a[1],
                          model_lora1.get_lora("up_proj").lora_a)
    assert torch.allclose(packed_lora1.lora_b[1],
                          model_lora1.get_lora("up_proj").lora_b)
