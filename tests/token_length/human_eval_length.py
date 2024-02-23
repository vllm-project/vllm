import time
import sys
import json
import matplotlib.pyplot as plt

start_test_id = 0
end_test_id = 10
max_tokens = 2048
file_path = '/data/leili/datasets/human_eval.json'
# 'human_eval.jsonl'
model_path = "/data/zyh/llm-finetune/llama2-hf/7B"
save_file_path = "human_eval_llama2_7B.txt"
save_png_path = "human_eval_llama2_7B.png"
prompts = []
token_length = []

# 打开文件并逐行读取
with open(file_path, 'r') as file:
    for line in file:
        json_data = json.loads(line)
        task_id = json_data['task_id']
        request = json_data['completion']
        prompts.append(request)


from vllm import LLM, SamplingParams
sampling_params = SamplingParams(temperature=0.8, 
                    top_p=0.95, max_tokens=max_tokens)
llm = LLM(model=model_path)
start_time = time.time()
outputs = llm.generate(prompts[start_test_id:end_test_id],\
                    sampling_params)
end_time = time.time()
print("execution time: ", end_time - start_time)
for output in outputs:
    prompt = output.prompt
    generated_token = output.outputs[0].token_ids
    token_length.append(len(generated_token))
print(f"num of output: {len(token_length)}, max_len: {max(token_length)}\n\
min_len: {min(token_length)}, avg_len: {sum(token_length)/len(token_length)}")

with open(save_file_path, 'w') as file:
    for idx in range(len(token_length)):
        number = token_length[idx]
        file.write(f"{len(prompts[idx])} {number}\n")

# plot the graph of the output length.
plt.hist(token_length, bins=max(token_length)-min(token_length)+1,\
        align='left', edgecolor='orange')

plt.title('Output Length Distribution')
plt.xlabel('Prompt')
plt.ylabel('Frequency')

plt.savefig(save_png_path)
 