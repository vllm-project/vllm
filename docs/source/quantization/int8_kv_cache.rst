.. _int8_kv_cache:

INT8 KV Cache
==================

The kv cache is quantized to INT8 dtype from float/fp16/bflaot16 to save GPU memory.
To use it, you first need to export scales and zero points with a calibration dataset like pileval and save these quantization parameters at a certain path.
Then you can enable the int8 kv cache in the vllm settings.
Note that INT8 KV Cache only supports Llama model for now.


Here is an example of how to export quantization scales and zero points:

First, you should capture kv cache states for subsequent calculation of scales and zero points.

.. code-block:: console

    $ python3 vllm/kv_quant/calibrate.py --model facebook/llama-13b --calib_dataset pileval 
    --calib_samples 128 --calib_seqlen 2048 --work_dir kv_cache_states/llama-13b

Second, export quantization scales and zero points with the captured kv cache states.

.. code-block:: console

    $ python3 vllm/kv_quant/export_kv_params.py --work_dir kv_cache_states/llama-13b 
    --kv_params_dir quant_params/llama-13b


Here is an example of how to enable int8 kv cache:

.. code-block:: python

    from vllm import LLM, SamplingParams
    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    # Create an LLM.
    llm = LLM(model="facebook/llama-13b", kv_cache_dtype="int8", kv_quant_params_path="quant_params/llama-13b")
    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

