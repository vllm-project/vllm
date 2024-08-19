import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' 

# torch-2.1.0+cuda11.8-cp310-cp310-linux_aarch64.whl

import torch
from PIL import Image
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

max_model_len, tp_size = 8192, 1
model_name = "THUDM/glm-4v-9b"

llm = LLM(
    model=model_name,
    tensor_parallel_size=tp_size,
    max_model_len=max_model_len,
    trust_remote_code=True,
    dtype=torch.bfloat16,
    enforce_eager=True,
    load_format='bitsandbytes',
    quantization='bitsandbytes'
)
stop_token_ids = [151329, 151336, 151338]
sampling_params = SamplingParams(temperature=0, max_tokens=1024, stop_token_ids=stop_token_ids)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

query = 'Describe this picture.'

image = Image.open(os.path.join(os.path.dirname(__file__), "../../docs/source/assets/logos/vllm-logo-text-light.png")).convert('RGB')
inputs = tokenizer.apply_chat_template(
    [{"role": "user", "image": image, "content": query}],
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt",
    return_dict=True
)

image_tensor = inputs['images']

input_ids = inputs['input_ids'][0].tolist()

outputs = llm.generate(
    TokensPrompt(**{
        "prompt_token_ids": input_ids,
        "multi_modal_data":  {"image": image_tensor},
    }),
    sampling_params=sampling_params
)

print(outputs[0].outputs[0].text)


# from transformers import AutoTokenizer
# from vllm import LLM, SamplingParams

# # GLM-4-9B-Chat-1M
# # max_model_len, tp_size = 1048576, 4
# # 如果遇见 OOM 现象，建议减少max_model_len，或者增加tp_size
# max_model_len, tp_size = 60000, 1
# model_name = "THUDM/glm-4-9b-chat"
# prompt = [{"role": "user", "content": "你好"}]

# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# llm = LLM(
#     model=model_name,
#     tensor_parallel_size=tp_size,
#     max_model_len=max_model_len,
#     trust_remote_code=True,
#     enforce_eager=True,
#     load_format='bitsandbytes',
#     quantization='bitsandbytes'
#     # GLM-4-9B-Chat-1M 如果遇见 OOM 现象，建议开启下述参数
#     # enable_chunked_prefill=True,
#     # max_num_batched_tokens=8192
# )
# stop_token_ids = [151329, 151336, 151338]
# sampling_params = SamplingParams(temperature=0.95, max_tokens=1024, stop_token_ids=stop_token_ids)

# inputs = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
# outputs = llm.generate(prompts=inputs, sampling_params=sampling_params)

# print(outputs[0].outputs[0].text)

# from vllm import LLM, SamplingParams


# prompts = [
#     "Hello, China is a"
# ]
# sampling_params = SamplingParams(temperature=0.8, top_p=0.95)


# llm = LLM(
#     model="huggyllama/llama-7b",
#     trust_remote_code=True,
#     enforce_eager=True,
#     load_format='bitsandbytes',
#     quantization='bitsandbytes'
# )

# outputs = llm.generate(prompts, sampling_params)

# # Print the outputs.
# for output in outputs:
#     prompt = output.prompt
#     generated_text = output.outputs[0].text
#     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")