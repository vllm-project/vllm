import contextlib
import gc
import tempfile
from collections import OrderedDict
from unittest.mock import patch, MagicMock

import pytest
import ray
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download

import vllm
from vllm.config import LoRAConfig
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.parallel_utils.parallel_state import (
    destroy_model_parallel, initialize_model_parallel)


def cleanup():
    destroy_model_parallel()
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    gc.collect()
    torch.cuda.empty_cache()
    ray.shutdown()


@pytest.fixture(autouse=True)
def cleanup_fixture():
    yield
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
            ("sampler", Sampler(512))
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
            ("sampler", Sampler(512))
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


@pytest.fixture
def llama_2_7b_engine_extra_embeddings() -> nn.Module:
    cleanup()
    get_model_old = get_model

    def get_model_patched(model_config, device_config, **kwargs):
        return get_model_old(model_config,
                             device_config,
                             lora_config=LoRAConfig(max_loras=4,
                                                    max_lora_rank=8))

    with patch("vllm.worker.model_runner.get_model", get_model_patched):
        engine = vllm.LLM("meta-llama/Llama-2-7b-hf", enable_lora=False)
    yield engine.llm_engine
    del engine
    cleanup()


@pytest.fixture
def llama_2_7b_model_extra_embeddings(
        llama_2_7b_engine_extra_embeddings) -> nn.Module:
    yield llama_2_7b_engine_extra_embeddings.driver_worker.model_runner.model
