import vllm
from vllm import LLM, SamplingParams
import time
from datasets import load_dataset
llm = LLM(
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    tensor_parallel_size=1,
    enable_prefix_caching=True,
    enable_chunked_prefill = True,
    trust_remote_code=True,  # Required for some HuggingFace models like Qwen
)
# Load a few GSM8K examples
ds = load_dataset("openai/gsm8k", "main", split="test")
ds = ds.select(range(2))  # Select a few examples for testing

# Sampling parameters
sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=2048,
    skip_special_tokens=True,
)

responses = []

# Run inference
for i, q in enumerate(ds['question']):
    prompt = f"""You are a helpful and accurate math tutor. Provide step-by-step reasoning and the final answer.

Question: {q}
Answer:"""
    
    start_time = time.time()
    outputs = llm.generate([prompt], sampling_params=sampling_params, use_tqdm=False)
    duration = time.time() - start_time
    
    output_text = outputs[0].outputs[0].text.strip()
    
    responses.append({
        "tid": i,
        "query": q,
        "duration": duration,
        "num_input_tokens": len(outputs[0].prompt_token_ids),
        "num_output_tokens": len(outputs[0].outputs[0].token_ids),
        "response": output_text,
    })
    
    print(f"Response {i}:\n{output_text}\n{'-'*60}")
# import pandas as pd
# llm = LLM(model="facebook/opt-125m", distributed_executor_backend="uni")
# sampling_params = SamplingParams(
#     stop=["Question:", "Explanation:", "Note:"],
#     skip_special_tokens=False,
#     temperature=0.0,
#     max_tokens=1024,
#     seed=42,
# )

# tokenizer = llm.get_tokenizer()

# from datasets import load_dataset

# ds = load_dataset("openai/gsm8k", "main", split="test")
# ds = ds.select(range(2))
import time
# questions = ['What is 2^3?', 'What is 2^10?']
# responses = []
# for i, d in enumerate(questions):
#     prompt = f"You are a helpful assistant.\n\n{d}"
#     start_time = time.time()
#     outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
#     duration = time.time() - start_time
#     responses.append({
#         "tid": i,
#         "query": d,
#         "duration": duration,
#         "num_input_tokens": len(outputs[0].prompt_token_ids),
#         "num_output_tokens": len(outputs[0].outputs[0].token_ids),
#         "response": outputs[0].outputs[0].text,
#     })
#     print(f"Response {i}: {outputs[0].outputs[0].text}")
# df=pd.DataFrame(responses)
# df.to_csv("gsm8k.csv", index=False)
