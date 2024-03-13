.. _fp8_e4m3_kvcache:

FP8 E4M3 KV Cache
==================

The int8/int4 quantization scheme requires additional scale GPU memory storage, which reduces the expected GPU memory benefits.
The FP8 data format retains 3 mantissa bits and can convert float/fp16/bflaot16 and fp8 to each other.

Here is an example of how to enable this feature:

.. code-block:: python


        from vllm import LLM, SamplingParams
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
        llm = LLM(model="/data/models/llama-2-70b-chat-hf", kv_cache_dtype="fp8", scales_path="./tests/fp8_kv/llama2-70b-fp8-kv/kv_cache_scales.json")
        prompt = "London is the capital of"
        out = llm.generate(prompt, sampling_params)[0].outputs[0].text
        print(out)
