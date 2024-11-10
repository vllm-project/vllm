.. _tensorizer:

Loading Models with Run:ai Model Streamer
=========================================
Run:ai Model Streamer is a library to read tensors in concurrency, while streaming it to GPU memory.
Further reading can be found in `Run:ai Model Streamer Documentation <https://github.com/run-ai/runai-model-streamer/blob/master/docs/README.md>`_.

vLLM supports loading weights in Safetensors format using the Run:ai Model Streamer.

To run it as an OpenAI-compatible server, add the `--load-format runai_streamer` flag:

.. code-block:: console

    $ vllm serve /home/meta-llama/Llama-3.2-3B-Instruct --load-format runai_streamer

Tunable parameters
------------------

You can control the level of concurrency by using the `concurrency` parameter in `--model-loader-extra-config`:

 .. code-block:: console

    $ vllm serve /home/meta-llama/Llama-3.2-3B-Instruct --load-format runai_streamer --model-loader-extra-config '{"concurrency":16}'

You can control the amount of CPU memory used to stream tensors by using the `memory_limit` parameter in `--model-loader-extra-config`:

 .. code-block:: console

    $ vllm serve /home/meta-llama/Llama-3.2-3B-Instruct --load-format runai_streamer --model-loader-extra-config '{"memory_limit":5368709120}'

.. note::
  For further instructions about tunable parameters and additional parameters configurable through environment variables, read the `Environment Variables Documentation <https://github.com/run-ai/runai-model-streamer/blob/master/docs/src/env-vars.md>`_.
