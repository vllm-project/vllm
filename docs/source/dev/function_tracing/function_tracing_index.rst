Function Tracing
=================================

You can enable tracing vLLM python function calls for debugging and performance tuning by setting the environment variable ``VLLM_TRACE_FUNCTION=1``.

After vLLM program exits, you will find the logs in /tmp directory with the name pattern like: ``VLLM_TRACE_FUNCTION_for_process_254159_thread_137882107995648_at_2024-10-19_16_55_44.973810.log``

Reading the traces is inefficient, so you can use the script ``tools/perfetto_trace_gen.py`` to combine and convert the traces to perfetto events, and
they can be visualized using https://ui.perfetto.dev/.

.. warning::

   Only enable tracing in a development environment.

Example commands and usage:
===========================

Convert traces:
------------------

.. code-block:: bash

    python tools/perfetto_trace_gen.py VLLM_TRACE_FUNCTION_for_process_254159_thread_137882107995648_at_2024-10-19_16_55_44.973810.logvllm_example_trace VLLM_TRACE_FUNCTION_for_process_254239_thread_137882108997432_at_2024-10-19_16_55_44.973823.log