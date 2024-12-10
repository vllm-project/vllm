.. _INC:

FP8 INC
==================

vLLM supports FP8 (8-bit floating point) weight and activation quantization using INC (Intel Neural Compressor) on hardware acceleration of Intel Gaudi (HPU).
Currently, only Llama models quntization are supported.

Please visit the Intel Gaudi documentation of `Run Inference Using FP8  <https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Using_FP8.html>`_.

In order to run Inference it is required to have Measurements/Scales files:

Retrieve Measurements
---------------------

To obtain measurement files:
* Use the "inc" quantization method (as parameter to the LLM object).
* Call shutdown_inc and shutdown methods of the model_executor in the end of the run.

.. code-block:: python

    from vllm import LLM
    llm = LLM("llama3.1/Meta-Llama-3.1-8B-Instruct", quantization="inc")
    ...
    # Call llm.generate on the required prompts and sampling params.
    ...
    llm.llm_engine.model_executor.shutdown_inc()
    llm.llm_engine.model_executor.shutdown()

.. note::

   Make sure to supply the "QUANT_CONFIG" enviornment variable which points to the `Json config file <https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Using_FP8.html#supported-json-config-file-options>`_ with MEASURE mode.

Run Inference Using FP8
--------------------------------------------

Inte Gaudi supports quantization of Linear Layers, KV-Cache and functions like Matmul and Softamx as shown in:
`Supported Modules <https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Using_FP8.html#supported-modules>`_.
`Supported Functions <https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Using_FP8.html#supported-functions>`_.

In order to run Inference it requies to have Scales which located in scale files according to the `Json config file <https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Using_FP8.html#supported-json-config-file-options>`_ dump_stats_path.
If none exist they can be generated during inference run using the measurement files (should be loacted in the same folder).

To run inference (and obtain scale files):
* Use the "inc" quantization method (as parameter to the LLM object).
* Use the "fp8_inc" kv cache dtype (as parameter to the LLM object).
* Call shutdown method of the model_executor in the end of the run.

.. code-block:: python

    from vllm import LLM
    llm = LLM("llama3.1/Meta-Llama-3.1-8B-Instruct", quantization="inc", kv_cache_dtype="fp8_inc")
    ...
    # Call llm.generate on the required prompts and sampling params.
    ...
    llm.llm_engine.model_executor.shutdown()

.. note::

    Make sure to supply the "QUANT_CONFIG" enviornment variable which points to the `Json config file <https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Using_FP8.html#supported-json-config-file-options>`_ with QUANTIZE mode.

Specifying Device for the Model's Weights Uploading
---------------------------------------------------

It is possible to upload the (unquantized) weights on a different device before qunantizing them 
and moving to the device on which the model will run.
Use the weights_load_device parameter for the LLM object to specify this device.

.. code-block:: python

    from vllm import LLM
    llm = LLM("llama3.1/Meta-Llama-3.1-8B-Instruct", quantization="inc", kv_cache_dtype="fp8_inc", weights_load_device="cpu")

