from vllm import LLM, SamplingParams

#model = LLM("BlackSamorez/Llama-2-7b-AQLM-2Bit-1x16-hf", enforce_eager=True)

#model = LLM("BlackSamorez/Llama-2-7b-AQLM-2Bit-2x8-hf", enforce_eager=True)

model = LLM("BlackSamorez/TinyLlama-1_1B-Chat-v1_0-AQLM-2Bit-1x16-hf", enforce_eager=True, tensor_parallel_size=2)

# These have custom code and the old format, and puzzling and conflicting stats, which probably I shouldn't even try to support.
#model = LLM("BlackSamorez/Llama-2-7b-AQLM-2Bit-8x8-hf", enforce_eager=True)
#model = LLM("BlackSamorez/Llama-2-13b-AQLM-2Bit-1x16-hf", enforce_eager=True, trust_remote_code=True)

sampling_params = SamplingParams(max_tokens=100, temperature=0)
outputs = model.generate("Hello my name is", sampling_params=sampling_params)
print(outputs[0].outputs[0].text)
