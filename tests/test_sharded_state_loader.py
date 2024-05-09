import os
import shutil
from tempfile import TemporaryDirectory

from huggingface_hub import snapshot_download

from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, seed=0)


def test_sharded_state_loader():
    weights_patterns = ("*.bin", "*.pt", "*.safetensors")

    cache_dir = TemporaryDirectory()
    input_dir = snapshot_download("facebook/opt-125m",
                                  cache_dir=cache_dir.name)

    llm_before = LLM(
        model=input_dir,
        worker_use_ray=True,
        gpu_memory_utilization=0.1,
    )
    gen_before = llm_before.generate(prompts, sampling_params)
    out_before = [gen.outputs[0].__dict__ for gen in gen_before]

    # Dump worker states to output directory
    model_executor = llm_before.llm_engine.model_executor
    output_dir = TemporaryDirectory()
    model_executor.save_sharded_state(path=output_dir.name)
    # Copy metadata files to output directory
    for file in os.listdir(input_dir):
        if not any(file.endswith(ext) for ext in weights_patterns):
            shutil.copy(f"{input_dir}/{file}", output_dir.name)

    cache_dir.cleanup()

    llm_after = LLM(
        model=output_dir.name,
        worker_use_ray=True,
        gpu_memory_utilization=0.1,
        load_format="sharded_state",
    )
    gen_after = llm_after.generate(prompts, sampling_params)
    out_after = [gen.outputs[0].__dict__ for gen in gen_after]

    assert out_before == out_after
