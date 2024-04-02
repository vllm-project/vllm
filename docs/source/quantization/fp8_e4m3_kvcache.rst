.. _fp8_e4m3_kvcache:

FP8 E4M3 KV Cache
==================

Quantizing the KV cache to FP8 reduces its memory footprint. This increases the number of tokens that can be stored in the cache, 
improving throughput. OCP (Open Compute Project www.opencompute.org) specifies two common 8-bit floating point data formats: E5M2 
(5 exponent bits and 2 mantissa bits) and E4M3FN (4 exponent bits and 3 mantissa bits), often shortened as E4M3. One benefit of 
the E4M3 format over E5M2 is that floating point numbers are represented in higher precision. However, the small dynamic range of 
FP8 E4M3 (±240.0 can be represented) typically necessitates the use of a higher-precision (typically FP32) scaling factor alongside 
each quantized tensor. For now, only per-tensor (scalar) scaling factors are supported. Development is ongoing to support scaling 
factors of a finer granularity (e.g. per-channel).

These scaling factors can be specified by passing an optional quantization param JSON to the LLM engine at load time. If 
this JSON is not specified, scaling factors default to 1.0. These scaling factors are typically obtained when running an 
unquantized model through a quantizer tool (e.g. AMD quantizer or NVIDIA AMMO). 

To install AMMO (AlgorithMic Model Optimization):

.. code-block:: console

        $ pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com nvidia-ammo

Studies have shown that FP8 E4M3 quantization typically only minimally degrades inference accuracy. The most recent silicon 
offerings e.g. AMD MI300, NVIDIA Hopper or later support native hardware conversion to and from fp32, fp16, bf16, etc. 
Thus, LLM inference is greatly accelerated with minimal accuracy loss.


Here is an example of how to enable this feature:

.. code-block:: python

        # two float8_e4m3fn kv cache scaling factor files are provided under tests/fp8_kv, please refer to 
        # https://github.com/vllm-project/vllm/blob/main/examples/fp8/README.md to generate kv_cache_scales.json of your own.

        from vllm import LLM, SamplingParams
        sampling_params = SamplingParams(temperature=1.3, top_p=0.8)
        llm = LLM(model="meta-llama/Llama-2-7b-chat-hf",
                  kv_cache_dtype="fp8",
                  quantization_param_path="./tests/fp8_kv/llama2-7b-fp8-kv/kv_cache_scales.json")
        prompt = "London is the capital of"
        out = llm.generate(prompt, sampling_params)[0].outputs[0].text
        print(out)

        # output w/ scaling factors:  England, the United Kingdom, and one of the world's leading financial,
        # output w/o scaling factors:  England, located in the southeastern part of the country. It is known 

Note, current prefix caching doesn't work with FP8 KV cache enabled, forward_prefix kernel should handle different KV and cache type.

