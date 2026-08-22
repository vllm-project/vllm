# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import fnmatch
import multiprocessing as mp
import os
import shutil
import sys
import types
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm import LLM, SamplingParams
from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader import (
    FastSafetensorsShardedStateLoader,
    ShardedStateLoader,
    get_model_loader,
)
from vllm.platforms import current_platform
from vllm.transformers_utils.repo_utils import hf_api

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# Create a sampling params object.
sampling_params = SamplingParams(
    temperature=0,
    max_tokens=256,
    ignore_eos=True,
)


def test_filter_subtensors():
    state_dict = {
        "a": torch.empty(2),
        "b": torch.empty((2, 4)),
        "c": torch.empty((2, 4, 8)),
    }
    state_dict.update(
        {
            "x": state_dict["b"],
            "y": state_dict["c"][1, 2, :],
            "z": state_dict["c"][1, :, 4],
        }
    )
    filtered_state_dict = ShardedStateLoader._filter_subtensors(state_dict)
    assert tuple(filtered_state_dict.keys()) == ("a", "b", "c")
    for key, tensor in filtered_state_dict.items():
        # NOTE: don't use `equal` here, as the tensor might contain NaNs
        assert tensor is state_dict[key]


def _fastsafetensors_module(loader_factory):
    module: Any = types.ModuleType("fastsafetensors")
    module.SafeTensorsFileLoader = loader_factory
    module.SingleGroup = MagicMock(return_value=object())
    return module


def test_fastsafetensors_sharded_loader_routes_and_validates_config():
    loader = get_model_loader(
        LoadConfig(
            load_format="fastsafetensors_sharded",
            model_loader_extra_config={
                "pattern": "rank-{rank}-{part}.safetensors",
                "nogds": True,
                "bbuf_size_kb": 4096,
                "max_threads": 4,
                "max_copy_block_size": 1024,
                "debug_log": True,
            },
        )
    )
    assert isinstance(loader, FastSafetensorsShardedStateLoader)
    assert loader.pattern == "rank-{rank}-{part}.safetensors"
    assert loader.nogds is True
    assert loader.bbuf_size_kb == 4096
    assert loader.max_threads == 4
    assert loader.max_copy_block_size == 1024
    assert loader.debug_log is True

    with pytest.raises(ValueError, match="nogds must be a boolean or null"):
        FastSafetensorsShardedStateLoader(
            LoadConfig(
                load_format="fastsafetensors_sharded",
                model_loader_extra_config={"nogds": "true"},
            )
        )


def test_fastsafetensors_sharded_loads_all_rank_files_together():
    loader = FastSafetensorsShardedStateLoader(
        LoadConfig(
            load_format="fastsafetensors_sharded",
            model_loader_extra_config={"nogds": True},
        )
    )
    file_buffer = MagicMock()
    file_buffer.key_to_rank_lidx = {"weight": None}
    file_buffer.get_tensor.return_value = torch.ones(1)
    fast_loader = MagicMock()
    fast_loader.copy_files_to_device.return_value = file_buffer
    module = _fastsafetensors_module(MagicMock(return_value=fast_loader))
    stream = MagicMock()

    with (
        patch.dict(sys.modules, {"fastsafetensors": module}),
        patch.object(current_platform, "is_cuda_alike", return_value=True),
        patch.object(current_platform, "current_device", return_value=0),
        patch("torch.accelerator.synchronize") as synchronize,
        patch("torch.accelerator.current_stream", return_value=stream),
    ):
        tensors = list(loader.iterate_over_files(["part-1", "part-0"]))

    assert tensors[0][0] == "weight"
    fast_loader.add_filenames.assert_called_once_with({0: ["part-0", "part-1"]})
    fast_loader.copy_files_to_device.assert_called_once_with(
        max_copy_block_size=loader.max_copy_block_size
    )
    synchronize.assert_called_once_with(torch.device("cuda:0"))
    stream.synchronize.assert_called_once()
    file_buffer.close.assert_called_once()
    fast_loader.close.assert_called_once()


def test_fastsafetensors_sharded_retries_gds_failure_before_copy():
    loader = FastSafetensorsShardedStateLoader(
        LoadConfig(load_format="fastsafetensors_sharded")
    )
    file_buffer = MagicMock()
    file_buffer.key_to_rank_lidx = {"weight": None}
    file_buffer.get_tensor.return_value = torch.ones(1)
    loaders = [MagicMock(), MagicMock()]
    loaders[0].copy_files_to_device.side_effect = RuntimeError(
        "cuFile GDS initialization failed"
    )
    loaders[1].copy_files_to_device.return_value = file_buffer
    loader_factory = MagicMock(side_effect=loaders)
    module = _fastsafetensors_module(loader_factory)

    with (
        patch.dict(sys.modules, {"fastsafetensors": module}),
        patch.object(current_platform, "is_cuda_alike", return_value=True),
        patch.object(current_platform, "current_device", return_value=0),
        patch("torch.accelerator.synchronize"),
        patch("torch.accelerator.current_stream", return_value=MagicMock()),
    ):
        tensors = list(loader.iterate_over_files(["part-0"]))

    assert tensors[0][0] == "weight"
    assert loader.nogds is True
    assert [call.kwargs["nogds"] for call in loader_factory.call_args_list] == [
        False,
        True,
    ]
    for fast_loader in loaders:
        fast_loader.close.assert_called_once()


@pytest.fixture(scope="module")
def llama_3p2_1b_files():
    input_dir = hf_api().snapshot_download(
        "meta-llama/Llama-3.2-1B-Instruct", ignore_patterns=["*.bin*", "original/*"]
    )

    yield input_dir


def _run_writer(input_dir, output_dir, weights_patterns, **kwargs):
    llm_sharded_writer = LLM(model=input_dir, **kwargs)

    # Dump worker states to output directory
    llm_sharded_writer.llm_engine.engine_core.save_sharded_state(path=output_dir)

    # Copy metadata files to output directory
    for file in os.listdir(input_dir):
        if os.path.isdir(os.path.join(input_dir, file)):
            shutil.copytree(
                os.path.join(input_dir, file), os.path.join(output_dir, file)
            )
        elif not any(fnmatch.fnmatch(file, ext) for ext in weights_patterns):
            shutil.copy(os.path.join(input_dir, file), output_dir)


def _run_generate(input_dir, queue: mp.Queue, **kwargs):
    llm = LLM(model=input_dir, **kwargs)
    gen = llm.generate(prompts, sampling_params)
    queue.put([g.outputs[0].__dict__ for g in gen])
    queue.close()
    queue.join_thread()


@pytest.mark.parametrize("enable_lora", [False, True])
@pytest.mark.parametrize("tp_size", [1, 2])
def test_sharded_state_loader(
    enable_lora, tp_size, num_gpus_available, llama_3p2_1b_files
):
    if num_gpus_available < tp_size:
        pytest.skip(f"Not enough GPUs for tensor parallelism {tp_size}")

    weights_patterns = ("*.safetensors",)
    gpu_memory_utilization = 0.8
    input_dir = llama_3p2_1b_files
    ctx = mp.get_context("spawn")

    # Keep batching deterministic: this test compares exact greedy outputs
    # across separate engine processes, whose schedulers may otherwise form
    # different batches depending on prompt-processing timing.
    platform_args = {"max_num_seqs": 1}

    # Run in separate processes for memory & CUDA isolation
    with TemporaryDirectory() as output_dir:
        p = ctx.Process(
            target=_run_writer,
            args=(input_dir, output_dir, weights_patterns),
            kwargs=dict(
                tensor_parallel_size=tp_size,
                gpu_memory_utilization=gpu_memory_utilization,
                enforce_eager=True,
                **platform_args,
            ),
        )
        p.start()
        p.join()

        queue = ctx.Queue()

        p = ctx.Process(
            target=_run_generate,
            args=(input_dir, queue),
            kwargs=dict(
                enable_lora=enable_lora,
                gpu_memory_utilization=gpu_memory_utilization,
                tensor_parallel_size=tp_size,
                **platform_args,
            ),
        )
        p.start()
        # Call queue.get() before p.join() to prevent deadlock:
        # If p.join() is called before queue.get() and the queue is full,
        # the child process may block while writing to the queue and never
        # terminate, causing the parent to wait indefinitely on p.join().
        # See: https://github.com/vllm-project/vllm/pull/22371#discussion_r2257773814
        out_before = queue.get()
        p.join()
        queue.close()
        queue.join_thread()

        queue = ctx.Queue()

        p = ctx.Process(
            target=_run_generate,
            args=(output_dir, queue),
            kwargs=dict(
                enable_lora=enable_lora,
                gpu_memory_utilization=gpu_memory_utilization,
                tensor_parallel_size=tp_size,
                load_format="sharded_state",
                **platform_args,
            ),
        )
        p.start()
        # Call queue.get() before p.join() to prevent deadlock:
        # If p.join() is called before queue.get() and the queue is full,
        # the child process may block while writing to the queue and never
        # terminate, causing the parent to wait indefinitely on p.join().
        # See: https://github.com/vllm-project/vllm/pull/22371#discussion_r2257773814
        out_after = queue.get()
        p.join()
        queue.close()
        queue.join_thread()

        assert out_before == out_after
