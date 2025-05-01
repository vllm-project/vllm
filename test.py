import vllm
from vllm import LLM, SamplingParams
import pandas as pd
vllm.__version__
llm = LLM(model="facebook/opt-125m", distributed_executor_backend="uni")


sampling_params = SamplingParams(
    stop=["Question:", "Explanation:", "Note:"],
    skip_special_tokens=False,
    temperature=0.0,
    max_tokens=100,
    seed=42,
)

tokenizer = llm.get_tokenizer()

from datasets import load_dataset

ds = load_dataset("openai/gsm8k", "main", split="test")
ds = ds.select(range(2))
import time
questions = ['What is 2^3?', 'What is 2^10?']
responses = []
for i, d in enumerate(questions):
    prompt = f"You are a helpful assistant.\n\n{d}"
    start_time = time.time()
    outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
    duration = time.time() - start_time
    responses.append({
        "tid": i,
        "query": d,
        "duration": duration,
        "num_input_tokens": len(outputs[0].prompt_token_ids),
        "num_output_tokens": len(outputs[0].outputs[0].token_ids),
        "response": outputs[0].outputs[0].text,
    })
    # print(f"Response {i}: {outputs[0].outputs[0].text}")
# df=pd.DataFrame(responses)
# df.to_csv("gsm8k.csv", index=False)
