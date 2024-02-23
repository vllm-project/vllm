import time
import os, sys
import json
import matplotlib.pyplot as plt

start_test_id = 0
end_test_id = 10
max_tokens = 2048
model_path = "/data/zyh/llm-finetune/llama2-hf/7B"
directory_path = "/data/leili/datasets/longbench/"
files = os.listdir(directory_path)
save_file_path = "longbench_llama2_7B.txt"
save_png_path = "longbench_llama2_7B.png"
prompts = []
output_length = []

for file_name in files:
    json_file_path = directory_path + file_name
    with open(json_file_path) as json_file:
        for line in json_file:
            data = json.loads(line)
            if len(data['input']) >= 2046 or len(data['input']) == 0:
                continue
        
            trunc = "\n\n" + data['input']
            context_len = 2047 - len(trunc)
            trunc = data['context'][:context_len] + trunc
            prompts.append(trunc)

print(f"number of requests produced by human: {len(prompts)}\n\
max prompts: {max(len(prompt) for prompt in prompts)}\n\
min prompts: {min(len(prompt) for prompt in prompts)}\n\
avg prompts: {sum(len(prompt) for prompt in prompts)/len(prompts)}")

from vllm import LLM, SamplingParams
# get the output of the prompt using given model.
sampling_params = SamplingParams(temperature=0.8, \
                top_p=0.95, max_tokens=max_tokens)
llm = LLM(model=model_path)
start_time = time.time()
outputs = llm.generate(prompts[start_test_id:end_test_id], \
                       sampling_params)
end_time = time.time()
print("execution time: ", end_time - start_time)
for output in outputs:
    prompt = output.prompt
    generated_token = output.outputs[0].token_ids
    output_length.append(len(generated_token))
    # print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    # print(len(generated_text))
print(f"num of output: {len(output_length)}, max_len: {max(output_length)}\n\
min_len: {min(output_length)}, avg_len: {sum(output_length)/len(output_length)}")
# write the output list into a file in case of wrong graphing.

with open(save_file_path, 'w') as file:
    for idx in range(len(output_length)):
        number = output_length[idx]
        file.write(f"{len(prompts[idx])} {number}\n")


# plot the graph of the output length.
plt.hist(output_length, bins=max(output_length)-min(output_length)+1,\
            align='left', edgecolor='orange')

plt.title('Output Length Distribution')
plt.xlabel('Prompt')
plt.ylabel('Frequency')

plt.savefig(save_png_path)
