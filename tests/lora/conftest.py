import tempfile
from collections import OrderedDict
from typing import Dict, List, TypedDict
from unittest.mock import MagicMock, patch

import pytest
import safetensors
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download

import vllm
from vllm.config import LoRAConfig
from vllm.distributed import (cleanup_dist_env_and_memory,
                              init_distributed_environment,
                              initialize_model_parallel)
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader import get_model
from vllm.platforms import current_platform


class ContextIDInfo(TypedDict):
    lora_id: int
    context_length: str


class ContextInfo(TypedDict):
    lora: str
    context_length: str


LONG_LORA_INFOS: List[ContextIDInfo] = [{
    "lora_id": 1,
    "context_length": "16k",
}, {
    "lora_id": 2,
    "context_length": "16k",
}, {
    "lora_id": 3,
    "context_length": "32k",
}]


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
    if current_platform.is_cpu():
        backend = "gloo"

    init_distributed_environment(world_size=1,
                                 rank=0,
                                 distributed_init_method=f"file://{temp_file}",
                                 local_rank=0,
                                 backend=backend)
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
    torch.distributed.init_process_group(world_size=1,
                                         rank=0,
                                         init_method=f"file://{temp_file}",
                                         backend=backend)


@pytest.fixture
def dummy_model() -> nn.Module:
    model = nn.Sequential(
        OrderedDict([
            ("dense1", ColumnParallelLinear(764, 100)),
            ("dense2", RowParallelLinear(100, 50)),
            (
                "layer1",
                nn.Sequential(
                    OrderedDict([
                        ("dense1", ColumnParallelLinear(100, 10)),
                        ("dense2", RowParallelLinear(10, 50)),
                    ])),
            ),
            ("act2", nn.ReLU()),
            ("output", ColumnParallelLinear(50, 10)),
            ("outact", nn.Sigmoid()),
            # Special handling for lm_head & sampler
            ("lm_head", ParallelLMHead(512, 10)),
            ("logits_processor", LogitsProcessor(512)),
            ("sampler", Sampler())
        ]))
    model.config = MagicMock()
    return model


@pytest.fixture
def dummy_model_gate_up() -> nn.Module:
    model = nn.Sequential(
        OrderedDict([
            ("dense1", ColumnParallelLinear(764, 100)),
            ("dense2", RowParallelLinear(100, 50)),
            (
                "layer1",
                nn.Sequential(
                    OrderedDict([
                        ("dense1", ColumnParallelLinear(100, 10)),
                        ("dense2", RowParallelLinear(10, 50)),
                    ])),
            ),
            ("act2", nn.ReLU()),
            ("gate_up_proj", MergedColumnParallelLinear(50, [5, 5])),
            ("outact", nn.Sigmoid()),
            # Special handling for lm_head & sampler
            ("lm_head", ParallelLMHead(512, 10)),
            ("logits_processor", LogitsProcessor(512)),
            ("sampler", Sampler())
        ]))
    model.config = MagicMock()
    return model


@pytest.fixture(scope="session")
def sql_lora_huggingface_id():
    # huggingface repo id is used to test lora runtime downloading.
    return "yard1/llama-2-7b-sql-lora-test"


@pytest.fixture(scope="session")
def sql_lora_files(sql_lora_huggingface_id):
    return snapshot_download(repo_id=sql_lora_huggingface_id)


@pytest.fixture(scope="session")
def lora_bias_files():
    return snapshot_download(repo_id="followumesh/granite-3b-lora8-bias")


@pytest.fixture(scope="session")
def mixtral_lora_files():
    # Note: this module has incorrect adapter_config.json to test
    # https://github.com/vllm-project/vllm/pull/5909/files.
    return snapshot_download(repo_id="SangBinCho/mixtral-lora")


@pytest.fixture(scope="session")
def mixtral_lora_files_all_target_modules():
    return snapshot_download(repo_id="dyang415/mixtral-lora-v0")


@pytest.fixture(scope="session")
def jamba_lora_files():
    #   some of the adapters have unnecessary weights for serving,
    #   hence we remove them
    def remove_unnecessary_weights(path):
        lora_path = f"{adapter_path}/adapter_model.safetensors"
        tensors = safetensors.torch.load_file(lora_path)
        nonlora_keys = []
        for k in list(tensors.keys()):
            if "lora" not in k:
                nonlora_keys.append(k)
        for k in nonlora_keys:
            del tensors[k]
        safetensors.torch.save_file(tensors, lora_path)

    adapter_path = snapshot_download(
        repo_id=
        "hf-100/Jamba-1.5-mini-Spellbound-StoryWriter-0.1-6583896-ckpt53-lora")

    remove_unnecessary_weights(adapter_path)
    return adapter_path


@pytest.fixture(scope="session")
def gemma_lora_files():
    return snapshot_download(repo_id="wskwon/gemma-7b-test-lora")


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
def minicpmv_lora_files():
    return snapshot_download(repo_id="jeeejeee/minicpmv25-lora-pokemon")


@pytest.fixture(scope="session")
def qwen2vl_lora_files():
    return snapshot_download(repo_id="jeeejeee/qwen2-vl-lora-pokemon")


@pytest.fixture(scope="session")
def tinyllama_lora_files():
    return snapshot_download(repo_id="jashing/tinyllama-colorist-lora")


@pytest.fixture(scope="session")
def phi2_lora_files():
    return snapshot_download(repo_id="isotr0py/phi-2-test-sql-lora")


@pytest.fixture(scope="session")
def long_context_lora_files_16k_1():
    return snapshot_download(repo_id="SangBinCho/long_context_16k_testing_1")


@pytest.fixture(scope="session")
def long_context_lora_files_16k_2():
    return snapshot_download(repo_id="SangBinCho/long_context_16k_testing_2")


@pytest.fixture(scope="session")
def long_context_lora_files_32k():
    return snapshot_download(repo_id="SangBinCho/long_context_32k_testing")


@pytest.fixture(scope="session")
def long_context_infos(long_context_lora_files_16k_1,
                       long_context_lora_files_16k_2,
                       long_context_lora_files_32k):
    cleanup_dist_env_and_memory(shutdown_ray=True)
    infos: Dict[int, ContextInfo] = {}
    for lora_checkpoint_info in LONG_LORA_INFOS:
        lora_id = lora_checkpoint_info["lora_id"]
        if lora_id == 1:
            lora = long_context_lora_files_16k_1
        elif lora_id == 2:
            lora = long_context_lora_files_16k_2
        elif lora_id == 3:
            lora = long_context_lora_files_32k
        else:
            raise AssertionError("Unknown lora id")
        infos[lora_id] = {
            "context_length": lora_checkpoint_info["context_length"],
            "lora": lora,
        }
    return infos


@pytest.fixture
def llama_2_7b_engine_extra_embeddings():
    cleanup_dist_env_and_memory(shutdown_ray=True)
    get_model_old = get_model

    def get_model_patched(**kwargs):
        kwargs["vllm_config"].lora_config = LoRAConfig(max_loras=4,
                                                       max_lora_rank=8)
        return get_model_old(**kwargs)

    with patch("vllm.worker.model_runner.get_model", get_model_patched):
        engine = vllm.LLM("meta-llama/Llama-2-7b-hf", enable_lora=False)
    yield engine.llm_engine
    del engine
    cleanup_dist_env_and_memory(shutdown_ray=True)


@pytest.fixture
def llama_2_7b_model_extra_embeddings(llama_2_7b_engine_extra_embeddings):
    yield (llama_2_7b_engine_extra_embeddings.model_executor.driver_worker.
           model_runner.model)
