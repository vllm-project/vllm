# SPDX-License-Identifier: Apache-2.0
# usage: torchrun --nproc-per-node=2 examples/offline_inference/data_parallel.py
# we need to have a launcher like torchrun to create multiple data parallel
# ranks. And each rank will create a vLLM instance to process its own prompts.

import os

from vllm import LLM, SamplingParams
from vllm.utils import cancel_torchrun_envs

# convert torchrun envs to vllm envs, and then delete torchrun envs

del os.environ["LOCAL_RANK"]  # not used in DP

os.environ["VLLM_DP_RANK"] = os.environ["RANK"]
del os.environ["RANK"]
os.environ["VLLM_DP_SIZE"] = os.environ["WORLD_SIZE"]
del os.environ["WORLD_SIZE"]
os.environ["VLLM_DP_MASTER_IP"] = os.environ["MASTER_ADDR"]
del os.environ["MASTER_ADDR"]
os.environ["VLLM_DP_MASTER_PORT"] = os.environ["MASTER_PORT"]
del os.environ["MASTER_PORT"]

dp_rank = int(os.environ["VLLM_DP_RANK"])
dp_size = int(os.environ["VLLM_DP_SIZE"])
GPUs_per_dp_rank = 2
# set devices for each dp_rank
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
    str(i) for i in range(dp_rank * GPUs_per_dp_rank, (dp_rank + 1) *
                          GPUs_per_dp_rank))

cancel_torchrun_envs()

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# with DP, each rank should process different prompts.
# usually all the DP ranks process a full dataset,
# and each rank processes a different part of the dataset.
promts_per_rank = len(prompts) // dp_size
start = dp_rank * promts_per_rank
end = start + promts_per_rank
prompts = prompts[start:end]

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.
llm = LLM(model="facebook/opt-125m", tensor_parallel_size=2)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
