# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tempfile
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.lora import LoRAConfig
from vllm.distributed import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.lora.lora_model import LoRAModel
from vllm.lora.lora_weights import LoRALayerWeights
from vllm.lora.model_manager import LoRAModelManager
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.models.interfaces import SupportsLoRA
from vllm.platforms import current_platform

CANONICAL_PATH = "layers.0.proj"
ALIASED_PATH = "self_decoder.decoder_layers.0.proj"


class DecoderAliasWrapper(nn.Module):
    def __init__(self, decoder_layers: nn.ModuleList):
        super().__init__()
        self.decoder_layers = decoder_layers


class ToyGemma4DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = ColumnParallelLinear(16, 8)


class ToyGemma4AliasedLoRAModel(nn.Module, SupportsLoRA):
    """Minimal Gemma4-style aliasing setup.

    The same decoder child module is reachable through both:
    - layers.0.proj
    - self_decoder.decoder_layers.0.proj
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([ToyGemma4DecoderLayer()])
        self.self_decoder = DecoderAliasWrapper(self.layers[:1])
        self.config = MagicMock()
        self.embedding_modules = {}
        self.unpadded_vocab_size = 128


@pytest.fixture()
def should_do_global_cleanup_after_test() -> bool:
    return False


@pytest.fixture()
def dist_init_local():
    temp_file = tempfile.mkstemp()[1]

    backend = "nccl"
    if current_platform.is_cpu() or current_platform.is_tpu():
        backend = "gloo"

    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"file://{temp_file}",
            local_rank=0,
            backend=backend,
        )
        initialize_model_parallel(1, 1)
        yield

    cleanup_dist_env_and_memory(shutdown_ray=False)


def _make_lora_config() -> LoRAConfig:
    return LoRAConfig(
        max_lora_rank=8,
        max_cpu_loras=2,
        max_loras=1,
        lora_dtype=torch.float32,
    )


def _make_nonzero_lora(module_name: str, module: ColumnParallelLinear) -> LoRAModel:
    lora_a = torch.ones((8, module.base_layer.weight.shape[1]))
    lora_b = torch.ones((module.base_layer.weight.shape[0], 8))

    assert lora_a.abs().sum().item() > 0
    assert lora_b.abs().sum().item() > 0

    return LoRAModel(
        1,
        8,
        {module_name: LoRALayerWeights(module_name, 8, 16, lora_a, lora_b)},
    )


def _stacked_lora_sums(module: ColumnParallelLinear) -> tuple[float, float]:
    return (
        module.lora_a_stacked[0].abs().sum().item(),
        module.lora_b_stacked[0].abs().sum().item(),
    )


def _make_manager() -> LoRAModelManager:
    return LoRAModelManager(
        ToyGemma4AliasedLoRAModel(),
        max_num_seqs=1,
        max_num_batched_tokens=8,
        vocab_size=128,
        lora_config=_make_lora_config(),
        device="cpu",
    )


def test_toy_model_exposes_aliased_linear_paths(dist_init_local):
    model = ToyGemma4AliasedLoRAModel()

    aliased_paths = {
        name
        for name, _ in model.named_modules(remove_duplicate=False)
        if name in {CANONICAL_PATH, ALIASED_PATH}
    }

    assert aliased_paths == {CANONICAL_PATH, ALIASED_PATH}
    assert model.layers[0] is model.self_decoder.decoder_layers[0]
    assert model.layers[0].proj is model.self_decoder.decoder_layers[0].proj


def test_lora_manager_keeps_aliased_modules(dist_init_local):
    manager = _make_manager()

    assert set(manager.modules) == {CANONICAL_PATH, ALIASED_PATH}
    assert manager.model.layers[0].proj is manager.modules[CANONICAL_PATH]
    assert (
        manager.model.self_decoder.decoder_layers[0].proj
        is manager.modules[ALIASED_PATH]
    )
    assert manager.modules[CANONICAL_PATH] is manager.modules[ALIASED_PATH]


def test_activation_prefers_weighted_duplicate_lora_wrapper(dist_init_local):
    manager = _make_manager()

    assert set(manager.modules) == {CANONICAL_PATH, ALIASED_PATH}
    assert manager.modules[CANONICAL_PATH] is manager.modules[ALIASED_PATH]

    shared_module = manager.modules[CANONICAL_PATH]
    manager.add_adapter(_make_nonzero_lora(CANONICAL_PATH, shared_module))
    manager.modules = {
        ALIASED_PATH: manager.modules[ALIASED_PATH],
        CANONICAL_PATH: manager.modules[CANONICAL_PATH],
    }

    events: list[str] = []
    set_lora = shared_module.set_lora
    reset_lora = shared_module.reset_lora

    def wrapped_set_lora(*args, **kwargs):
        events.append("set")
        return set_lora(*args, **kwargs)

    def wrapped_reset_lora(*args, **kwargs):
        events.append("reset")
        return reset_lora(*args, **kwargs)

    shared_module.set_lora = MagicMock(side_effect=wrapped_set_lora)
    shared_module.reset_lora = MagicMock(side_effect=wrapped_reset_lora)

    manager.activate_adapter(1)

    assert shared_module.set_lora.call_count == 1
    assert shared_module.reset_lora.call_count == 1
    assert events == ["set", "reset"]
    assert _stacked_lora_sums(shared_module) != (0.0, 0.0)
