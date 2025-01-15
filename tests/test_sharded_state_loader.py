import multiprocessing as mp
import os
import shutil
from tempfile import TemporaryDirectory

import pytest
import torch
from huggingface_hub import snapshot_download

from vllm import LLM, SamplingParams
from vllm.model_executor.model_loader.loader import ShardedStateLoader

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
    state_dict.update({
        "x": state_dict["b"],
        "y": state_dict["c"][1, 2, :],
        "z": state_dict["c"][1, :, 4],
    })
    filtered_state_dict = ShardedStateLoader._filter_subtensors(state_dict)
    assert tuple(filtered_state_dict.keys()) == ("a", "b", "c")
    for key, tensor in filtered_state_dict.items():
        # NOTE: don't use `equal` here, as the tensor might contain NaNs
        assert tensor is state_dict[key]


@pytest.fixture(scope="module")
def llama_2_7b_files():
    with TemporaryDirectory() as cache_dir:
        input_dir = snapshot_download("meta-llama/Llama-3.2-1B",
                                      cache_dir=cache_dir,
                                      ignore_patterns=["*.bin*", "original/*"])

        yield input_dir


def _run_writer(input_dir, output_dir, weights_patterns, **kwargs):
    llm_sharded_writer = LLM(model=input_dir, **kwargs)

    # Dump worker states to output directory
    llm_sharded_writer.llm_engine.model_executor.save_sharded_state(
        path=output_dir)

    # Copy metadata files to output directory
    for file in os.listdir(input_dir):
        if not any(
                file.endswith(ext) and not os.path.isdir(file)
                for ext in weights_patterns):
            shutil.copy(f"{input_dir}/{file}", output_dir)


def _run_generate(input_dir, queue: mp.Queue, **kwargs):
    llm = LLM(model=input_dir, **kwargs)
    gen = llm.generate(prompts, sampling_params)
    queue.put([g.outputs[0].__dict__ for g in gen])
    queue.close()
    queue.join_thread()


@pytest.mark.parametrize("enable_lora", [False, True])
@pytest.mark.parametrize("tp_size", [1, 2])
def test_sharded_state_loader(enable_lora, tp_size, num_gpus_available,
                              llama_2_7b_files):
    if num_gpus_available < tp_size:
        pytest.skip(f"Not enough GPUs for tensor parallelism {tp_size}")

    weights_patterns = ("*.safetensors", )
    gpu_memory_utilization = 0.8
    input_dir = llama_2_7b_files
    ctx = mp.get_context("spawn")

    # Run in separate processes for memory & CUDA isolation
    with TemporaryDirectory() as output_dir:
        p = ctx.Process(target=_run_writer,
                        args=(input_dir, output_dir, weights_patterns),
                        kwargs=dict(
                            tensor_parallel_size=tp_size,
                            distributed_executor_backend="mp",
                            gpu_memory_utilization=gpu_memory_utilization,
                            enforce_eager=True,
                        ))
        p.start()
        p.join()

        queue = ctx.Queue()

        p = ctx.Process(target=_run_generate,
                        args=(input_dir, queue),
                        kwargs=dict(
                            distributed_executor_backend="mp",
                            enable_lora=enable_lora,
                            gpu_memory_utilization=gpu_memory_utilization,
                            tensor_parallel_size=tp_size,
                        ))
        p.start()
        p.join()
        out_before = queue.get()

        p = ctx.Process(target=_run_generate,
                        args=(output_dir, queue),
                        kwargs=dict(
                            distributed_executor_backend="mp",
                            enable_lora=enable_lora,
                            gpu_memory_utilization=gpu_memory_utilization,
                            tensor_parallel_size=tp_size,
                            load_format="sharded_state",
                        ))
        p.start()
        p.join()
        out_after = queue.get()

        assert out_before == out_after
