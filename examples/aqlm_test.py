from vllm import LLM, SamplingParams

#model = LLM("nm-testing/llama2.c-stories110M-pruned2.4")
model = LLM("BlackSamorez/Llama-2-7b-AQLM-2Bit-1x16-hf")

sampling_params = SamplingParams(max_tokens=100, temperature=0)
outputs = model.generate("Hello my name is", sampling_params=sampling_params)
print(outputs[0].outputs[0].text)
