# SPDX-License-Identifier: Apache-2.0

from vllm import LLM, SamplingParams
import pandas as pd
import numpy as np

# read HarmBench/data/behavior_datasets/harmbench_behaviors_text_test.csv, select Behavior column as prompts
prompts = pd.read_csv('HarmBench/data/behavior_datasets/harmbench_behaviors_text_test.csv')['Behavior'].tolist()
with open('stats.log', 'w') as f:
    f.write('opt_facotr,blk_in_count,blk_out_count,swap_in_count,swap_out_count,in_spd,out_spd\n')

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95,  max_tokens=100)

# Create an LLM.
# llm = LLM(model="facebook/opt-125m", gpu_memory_utilization=0.011, max_model_len=384, preemption_mode="swap", scheduling_policy='priority', enable_prefix_caching=True)
llm = LLM(model="facebook/opt-125m", gpu_memory_utilization=0.15, max_model_len=384, preemption_mode="swap", scheduling_policy='priority')

for factor in np.arange(0, 4, 0.1):
    with open('config.log', 'w') as f:
        f.write(f'{factor}')
    
    for _ in range(5):
        # Generate texts from the prompts. The output is a list of RequestOutput objects
        # that contain the prompt, generated text, and other information.
        outputs = llm.generate(prompts, sampling_params)

# # Write the outputs to a file.
# with open('results.txt', 'w') as f:
#     for output in outputs:
#         prompt = output.prompt
#         generated_text = output.outputs[0].text
#         f.write(f"Prompt: {prompt!r}, Generated text: {generated_text!r}\n")
