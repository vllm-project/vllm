from vllm import LLM, SamplingParams

model = LLM("nm-testing/zephyr-50sparse-24",
            sparsity="semi_structured_sparse_w16a16",
            enforce_eager=True,
            dtype="float16",
            tensor_parallel_size=1,
            max_model_len=1024)

sampling_params = SamplingParams(max_tokens=100, temperature=0)
outputs = model.generate("Hello my name is", sampling_params=sampling_params)
print(outputs[0].outputs[0].text)
