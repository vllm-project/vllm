from vllm import LLM, SamplingParams

# 1x16
model = LLM("BlackSamorez/Llama-2-7b-AQLM-2Bit-1x16-hf", enforce_eager=True)

# 2 x 8
#model = LLM("BlackSamorez/Llama-2-7b-AQLM-2Bit-2x8-hf", enforce_eager=True)

sampling_params = SamplingParams(max_tokens=100, temperature=0)
outputs = model.generate("Hello my name is", sampling_params=sampling_params)
print(outputs[0].outputs[0].text)
