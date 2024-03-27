.. _fp8_e4m3_kvcache:

FP8 E4M3 KV Cache
==================

The FP8 quantized KV cache reduces memory footprint and bandwidth consumption, leading to better LLM serve performance overall.
The FP8 (OCP E4M3) data format retains 4 exponent bits and 3 mantissa bits, it can be converted to or from float/fp16/bfloat16,
and the conversions are accelerated on recent silicons - AMD MI300, nVIDIA Hopper or later.

Here is an example of how to enable this feature:

.. code-block:: python


        from vllm import LLM, SamplingParams
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
        llm = LLM(model="/data/models/llama-2-70b-chat-hf", kv_cache_dtype="fp8", scales_path="./tests/fp8_kv/llama2-7b-fp8-kv/kv_cache_scales.json")
        prompt = "London is the capital of"
        out = llm.generate(prompt, sampling_params)[0].outputs[0].text
        print(out)
