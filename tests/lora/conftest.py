# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tempfile
from collections import OrderedDict
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download

from vllm.distributed import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.interfaces import SupportsLoRA
from vllm.platforms import current_platform


@pytest.fixture()
def should_do_global_cleanup_after_test(request) -> bool:
    """Allow subdirectories to skip global cleanup by overriding this fixture.
    This can provide a ~10x speedup for non-GPU unit tests since they don't need
    to initialize torch.
    """

    return not request.node.get_closest_marker("skip_global_cleanup")


@pytest.fixture(autouse=True)
def cleanup_fixture(should_do_global_cleanup_after_test: bool):
    yield
    if should_do_global_cleanup_after_test:
        cleanup_dist_env_and_memory(shutdown_ray=True)


@pytest.fixture
def dist_init():
    temp_file = tempfile.mkstemp()[1]

    backend = "nccl"
    if current_platform.is_cpu() or current_platform.is_tpu():
        backend = "gloo"

    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method=f"file://{temp_file}",
        local_rank=0,
        backend=backend,
    )
    initialize_model_parallel(1, 1)
    yield
    cleanup_dist_env_and_memory(shutdown_ray=True)


@pytest.fixture
def dist_init_torch_only():
    if torch.distributed.is_initialized():
        return
    backend = "nccl"
    if current_platform.is_cpu():
        backend = "gloo"

    temp_file = tempfile.mkstemp()[1]
    torch.distributed.init_process_group(
        world_size=1, rank=0, init_method=f"file://{temp_file}", backend=backend
    )


class DummyLoRAModel(nn.Sequential, SupportsLoRA):
    pass


@pytest.fixture
def dummy_model() -> nn.Module:
    model = DummyLoRAModel(
        OrderedDict(
            [
                ("dense1", ColumnParallelLinear(764, 100)),
                ("dense2", RowParallelLinear(100, 50)),
                (
                    "layer1",
                    nn.Sequential(
                        OrderedDict(
                            [
                                ("dense1", ColumnParallelLinear(100, 10)),
                                ("dense2", RowParallelLinear(10, 50)),
                            ]
                        )
                    ),
                ),
                ("act2", nn.ReLU()),
                ("output", ColumnParallelLinear(50, 10)),
                ("outact", nn.Sigmoid()),
                # Special handling for lm_head & sampler
                ("lm_head", ParallelLMHead(512, 10)),
                ("logits_processor", LogitsProcessor(512)),
            ]
        )
    )
    model.config = MagicMock()
    model.embedding_modules = {"lm_head": "lm_head"}
    model.unpadded_vocab_size = 32000
    return model


@pytest.fixture
def dummy_model_gate_up() -> nn.Module:
    model = DummyLoRAModel(
        OrderedDict(
            [
                ("dense1", ColumnParallelLinear(764, 100)),
                ("dense2", RowParallelLinear(100, 50)),
                (
                    "layer1",
                    nn.Sequential(
                        OrderedDict(
                            [
                                ("dense1", ColumnParallelLinear(100, 10)),
                                ("dense2", RowParallelLinear(10, 50)),
                            ]
                        )
                    ),
                ),
                ("act2", nn.ReLU()),
                ("gate_up_proj", MergedColumnParallelLinear(50, [5, 5])),
                ("outact", nn.Sigmoid()),
                # Special handling for lm_head & sampler
                ("lm_head", ParallelLMHead(512, 10)),
                ("logits_processor", LogitsProcessor(512)),
            ]
        )
    )
    model.config = MagicMock()
    model.packed_modules_mapping = {
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }
    model.embedding_modules = {"lm_head": "lm_head"}
    model.unpadded_vocab_size = 32000

    return model


@pytest.fixture(scope="session")
def llama_2_7b_base_huggingface_id():
    # used as a base model for testing with sql lora adapter
    return "meta-llama/Llama-2-7b-hf"


@pytest.fixture(scope="session")
def sql_lora_huggingface_id():
    # huggingface repo id is used to test lora runtime downloading.
    return "yard1/llama-2-7b-sql-lora-test"


@pytest.fixture(scope="session")
def sql_lora_files(sql_lora_huggingface_id):
    return snapshot_download(repo_id=sql_lora_huggingface_id)


@pytest.fixture(scope="session")
def mixtral_lora_files():
    # Note: this module has incorrect adapter_config.json to test
    # https://github.com/vllm-project/vllm/pull/5909/files.
    return snapshot_download(repo_id="SangBinCho/mixtral-lora")


@pytest.fixture(scope="session")
def chatglm3_lora_files():
    return snapshot_download(repo_id="jeeejeee/chatglm3-text2sql-spider")


@pytest.fixture(scope="session")
def baichuan_lora_files():
    return snapshot_download(repo_id="jeeejeee/baichuan7b-text2sql-spider")


@pytest.fixture(scope="session")
def baichuan_zero_lora_files():
    # all the lora_B weights are initialized to zero.
    return snapshot_download(repo_id="jeeejeee/baichuan7b-zero-init")


@pytest.fixture(scope="session")
def baichuan_regex_lora_files():
    return snapshot_download(repo_id="jeeejeee/baichuan-7b-lora-zero-regex")


@pytest.fixture(scope="session")
def ilama_lora_files():
    return snapshot_download(repo_id="jeeejeee/ilama-text2sql-spider")


@pytest.fixture(scope="session")
def minicpmv_lora_files():
    return snapshot_download(repo_id="jeeejeee/minicpmv25-lora-pokemon")


@pytest.fixture(scope="session")
def qwen2vl_lora_files():
    return snapshot_download(repo_id="jeeejeee/qwen2-vl-lora-pokemon")


@pytest.fixture(scope="session")
def qwen25vl_base_huggingface_id():
    # used as a base model for testing with qwen25vl lora adapter
    return "Qwen/Qwen2.5-VL-3B-Instruct"


@pytest.fixture(scope="session")
def qwen25vl_lora_files():
    return snapshot_download(repo_id="jeeejeee/qwen25-vl-lora-pokemon")


@pytest.fixture(scope="session")
def tinyllama_lora_files():
    return snapshot_download(repo_id="jashing/tinyllama-colorist-lora")


@pytest.fixture(scope="session")
def deepseekv2_lora_files():
    return snapshot_download(repo_id="wuchen01/DeepSeek-V2-Lite-Chat-All-LoRA")


@pytest.fixture(scope="session")
def gptoss20b_lora_files():
    return snapshot_download(repo_id="jeeejeee/gpt-oss-20b-lora-adapter-text2sql")


@pytest.fixture(scope="session")
def qwen3moe_lora_files():
    return snapshot_download(repo_id="jeeejeee/qwen3-moe-text2sql-spider")


@pytest.fixture(scope="session")
def olmoe_lora_files():
    return snapshot_download(repo_id="jeeejeee/olmoe-instruct-text2sql-spider")


@pytest.fixture(scope="session")
def qwen3_lora_files():
    return snapshot_download(repo_id="charent/self_cognition_Alice")


@pytest.fixture(scope="session")
def llama32_lora_files():
    return snapshot_download(repo_id="jeeejeee/llama32-3b-text2sql-spider")


@pytest.fixture
def reset_default_device():
    """
    Some tests, such as `test_punica_ops.py`, explicitly set the
    default device, which can affect subsequent tests. Adding this fixture
    helps avoid this problem.
    """
    original_device = torch.get_default_device()
    yield
    torch.set_default_device(original_device)
