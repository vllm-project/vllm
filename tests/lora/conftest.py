import contextlib
import gc
import tempfile
from collections import OrderedDict
from unittest.mock import MagicMock, patch

import pytest
import ray
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download

import vllm
from vllm.config import LoRAConfig
from vllm.distributed import destroy_model_parallel, initialize_model_parallel
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader import get_model

LONG_LORA_INFOS = [{
    "lora_id": 1,
    "context_length": "16k",
}, {
    "lora_id": 2,
    "context_length": "16k",
}, {
    "lora_id": 3,
    "context_length": "32k",
}]


def cleanup():
    destroy_model_parallel()
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    gc.collect()
    torch.cuda.empty_cache()
    ray.shutdown()


@pytest.fixture()
def should_do_global_cleanup_after_test(request) -> bool:
    """Allow subdirectories to skip global cleanup by overriding this fixture.
    This can provide a ~10x speedup for non-GPU unit tests since they don't need
    to initialize torch.
    """

    if request.node.get_closest_marker("skip_global_cleanup"):
        return False

    return True


@pytest.fixture(autouse=True)
def cleanup_fixture(should_do_global_cleanup_after_test: bool):
    yield
    if should_do_global_cleanup_after_test:
        cleanup()


@pytest.fixture
def dist_init():
    if not torch.distributed.is_initialized():
        temp_file = tempfile.mkstemp()[1]
        torch.distributed.init_process_group(
            backend="nccl",
            world_size=1,
            rank=0,
            init_method=f"file://{temp_file}",
        )
        torch.distributed.all_reduce(torch.zeros(1).cuda())
    initialize_model_parallel(1, 1)
    yield
    cleanup()


@pytest.fixture
def dist_init_torch_only():
    if torch.distributed.is_initialized():
        return
    temp_file = tempfile.mkstemp()[1]
    torch.distributed.init_process_group(
        backend="nccl",
        world_size=1,
        rank=0,
        init_method=f"file://{temp_file}",
    )


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
def sql_lora_files():
    return snapshot_download(repo_id="yard1/llama-2-7b-sql-lora-test")


@pytest.fixture(scope="session")
def mixtral_lora_files():
    return snapshot_download(repo_id="terrysun/mixtral-lora-adapter")


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
    cleanup()
    infos = {}
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
def llama_2_7b_engine_extra_embeddings() -> nn.Module:
    cleanup()
    get_model_old = get_model

    def get_model_patched(*, model_config, device_config, **kwargs):
        kwargs["lora_config"] = LoRAConfig(max_loras=4, max_lora_rank=8)
        return get_model_old(model_config=model_config,
                             device_config=device_config,
                             **kwargs)

    with patch("vllm.worker.model_runner.get_model", get_model_patched):
        engine = vllm.LLM("meta-llama/Llama-2-7b-hf", enable_lora=False)
    yield engine.llm_engine
    del engine
    cleanup()


@pytest.fixture
def llama_2_7b_model_extra_embeddings(
        llama_2_7b_engine_extra_embeddings) -> nn.Module:
    yield (llama_2_7b_engine_extra_embeddings.model_executor.driver_worker.
           model_runner.model)
