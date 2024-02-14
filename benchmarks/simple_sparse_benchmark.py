from vllm import LLM, SamplingParams
import copy
import time
#os.environ["CUDA_VISIBLE_DEVICES"] = "0" # Use only cuda:0

model = LLM(
    "nm-testing/zephyr-50sparse-24",
    #sparsity="sparse_w16a16", # If left off, model will be loaded as dense
    enforce_eager=True,  # Does not work with cudagraphs yet
    dtype="float16",  # bfloat16
    tensor_parallel_size=2,
    max_model_len=1024)

num_prompts = 64
input_len = 3072 * 2
prompts = [
    copy.deepcopy(prompt) for prompt in (["Hi im a prompt"] * num_prompts)
]

sampling_params = SamplingParams(max_tokens=100, temperature=0)

start_time = time.time()

outputs = model.generate(prompts, sampling_params=sampling_params)

end_time = time.time()  # Capture end time
duration = end_time - start_time  # Calculate duration

print(f"Elapsed time: {duration} seconds")

#print(outputs[0])
#print(outputs)
#print(outputs[0].outputs[0].text)
