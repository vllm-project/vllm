Profiling vLLM 
=================================

We support tracing vLLM workers using the ``torch.profiler`` module. You can enable tracing by setting the ``VLLM_TORCH_PROFILER_DIR`` environment variable to the directory where you want to save the traces: ``VLLM_TORCH_PROFILER_DIR=/mnt/traces/``

The OpenAI server also needs to be started with the ``VLLM_TORCH_PROFILER_DIR`` environment variable set.

When using ``benchmarks/benchmark_serving.py``, you can enable profiling by passing the ``--profile`` flag.

.. warning::

   Only enable profiling in a development environment. 


Traces can be visualized using https://ui.perfetto.dev/.

.. tip::

   Only send a few requests through vLLM when profiling, as the traces can get quite large. Also, no need to untar the traces, they can be viewed directly.
   
Example commands:

OpenAI Server:

.. code-block:: bash

    VLLM_TORCH_PROFILER_DIR=/mnt/traces/ python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-70B 

benchmark_serving.py:

.. code-block:: bash

    python benchmarks/benchmark_serving.py --backend vllm --model meta-llama/Meta-Llama-3-70B --dataset-name sharegpt --dataset-path sharegpt.json --profile --num-prompts 2 