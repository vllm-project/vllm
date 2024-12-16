.. _INC:

FP8 INC
=======

vLLM supports FP8 (8-bit floating point) weight and activation quantization using INC (Intel Neural Compressor) on hardware acceleration of Intel Gaudi (HPU).
Currently, quantization is supported only for Llama models.

Please visit the Intel Gaudi documentation of `Run Inference Using FP8  <https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Using_FP8.html>`_.

In order to run inference it is required to have measurements/scales files:

Obtain Measurements
-------------------

To obtain measurement files:
* Set the "QUANT_CONFIG" environment variable which points to the `JSON config file <https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Using_FP8.html#supported-json-config-file-options>`_ with MEASURE mode.
* Pass ``quantization=inc`` as parameter to the ``LLM`` object.
* Call ``shutdown_inc`` and ``shutdown`` methods of the ``model_executor`` at the end of the run.

.. code-block:: python

    from vllm import LLM
    llm = LLM("llama3.1/Meta-Llama-3.1-8B-Instruct", quantization="inc")
    ...
    # Call llm.generate on the required prompts and sampling params.
    ...
    llm.llm_engine.model_executor.shutdown_inc()
    llm.llm_engine.model_executor.shutdown()

Run Inference Using FP8
-----------------------

Intel Gaudi supports quantization of various modules and functions, including, but not limited to ``Linear``, ``KVCache``, ``Matmul`` and ``Softmax``. For more information, please refer to:
`Supported Modules <https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Using_FP8.html#supported-modules>`_.
`Supported Functions <https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Using_FP8.html#supported-functions>`_.

In order to run inference it requires to have Scales which located in scale files according to the `JSON config file <https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Using_FP8.html#supported-json-config-file-options>`_ ``dump_stats_path``.
If none exist, they can be generated during inference run using the measurement files (should be located in the same folder).

To run inference (and obtain scale files):
* Set the "QUANT_CONFIG" environment variable which points to the `JSON config file <https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Using_FP8.html#supported-json-config-file-options>`_ with QUANTIZE mode.
* Pass ``quantization=inc`` as parameter to the ``LLM`` object.
* Pass ``fp8_inc`` as KV cache data type:
   * Offline inference: pass ``kv_cache_dtype=fp8_inc`` as parameter to the ``LLM`` object. 
   * Online inference: pass ``--kv-cache-dtype=fp8_inc`` as command line parameter.
* Call shutdown method of the model_executor at the end of the run.

.. code-block:: python

    from vllm import LLM
    llm = LLM("llama3.1/Meta-Llama-3.1-8B-Instruct", quantization="inc", kv_cache_dtype="fp8_inc")
    ...
    # Call llm.generate on the required prompts and sampling params.
    ...
    llm.llm_engine.model_executor.shutdown()

Specifying Device for the Model's Weights Uploading
---------------------------------------------------

It is possible to load the unquantized weights on a different device before quantizing them,
and moving to the device on which the model will run. This reduces the device memory footprint of model weights, as only quantized weights are stored in device memory.
To set the load device, use the ``weights_load_device`` parameter for the ``LLM`` object, or ``--weights-load-device`` command line parameter in online mode.

.. code-block:: python

    from vllm import LLM
    llm = LLM("llama3.1/Meta-Llama-3.1-8B-Instruct", quantization="inc", kv_cache_dtype="fp8_inc", weights_load_device="cpu")
