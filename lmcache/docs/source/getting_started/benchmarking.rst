Benchmarking
============

This is a simple tutorial on how to deploy and benchmark LMCache using the
``lmcache bench engine`` CLI.

The ``lmcache bench engine`` command is a flexible traffic simulator that
sends configurable workloads to your inference engine and reports TTFT,
decoding speed, and throughput metrics. This tutorial walks through a
long-document Q&A benchmark that exercises LMCache's CPU offloading path.

For the full CLI reference -- including every flag, every workload type, and
config-file usage -- see :doc:`/cli/bench`.

Long Doc QA workload
--------------------

The ``long-doc-qa`` workload simulates repeated Q&A over long synthetic
documents: a warmup round primes the KV cache with each document, then a
benchmark round dispatches the questions. The number of documents is
derived from ``--kv-cache-volume`` and the model's tokens-per-GB rather
than set directly. See :doc:`/cli/bench` for the full flag list.

Example
-------

To measure the benefit of LMCache, run the **same benchmark against two
setups** and compare the results:

- **Setup A (baseline)** -- vLLM alone.
- **Setup B (with LMCache)** -- vLLM plus a standalone LMCache server.

The steps below reproduce both runs on ``Qwen/Qwen3-8B``. Adjust the sizes
to match your hardware.

Setup A: vLLM alone (baseline)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    vllm serve Qwen/Qwen3-8B

Setup B: vLLM with LMCache
~~~~~~~~~~~~~~~~~~~~~~~~~~

Run LMCache as a standalone service and point vLLM at it with the
``LMCacheMPConnector``. See :doc:`/getting_started/quickstart` for the full MP-mode
walkthrough.

**Start the LMCache server:**

.. code-block:: bash

    lmcache server \
        --l1-size-gb 66 --eviction-policy LRU

The ZMQ port defaults to **5555** (used by vLLM) and the HTTP frontend
defaults to **8080** (used by ``lmcache bench engine --lmcache-url``).

**Start vLLM with the MP connector in a separate terminal:**

.. code-block:: bash

    vllm serve Qwen/Qwen3-8B \
        --kv-transfer-config \
        '{"kv_connector": "LMCacheMPConnector", "kv_role": "kv_both"}'

Run the benchmark
~~~~~~~~~~~~~~~~~

To make the comparison fair, capture the benchmark settings **once** in a
config file, then replay the same config against both setups.

**Step 1 -- export a shared config.** With the LMCache server from Setup B
still running, launch ``lmcache bench engine`` in interactive mode:

.. code-block:: bash

    lmcache bench engine --lmcache-url http://localhost:8080

Interactive mode triggers because ``--engine-url`` and ``--workload`` are
missing, and ``--lmcache-url`` auto-detects ``tokens-per-gb-kvcache`` from
the server. Walk through the prompts and pick:

- Engine URL: ``http://localhost:8000``
- Workload: ``long-doc-qa``
- Model: auto-detected from the engine (or type ``Qwen/Qwen3-8B``)
- KV cache volume (GB): ``10``
- ``ldqa-query-per-document``: ``1``
- ``ldqa-shuffle-policy``: ``tile``
- ``ldqa-num-inflight-requests``: ``4``
- Leave the rest at their defaults.

At the **summary** step, choose **"Export configuration for later use and
exit"** and save to ``bench_config.json``. The file looks like:

.. code-block:: json

    {
      "model": "Qwen/Qwen3-8B",
      "workload": "long-doc-qa",
      "kv_cache_volume": 10.0,
      "tokens_per_gb_kvcache": 46020,
      "ldqa_document_length": 10000,
      "ldqa_query_per_document": 1,
      "ldqa_shuffle_policy": "tile",
      "ldqa_num_inflight_requests": 4
    }

Note that the exported config stores ``tokens_per_gb_kvcache`` (resolved
from ``--lmcache-url``) but **not** the engine URL or the LMCache URL, so
the same file is portable across environments.

**Step 2 -- replay against each setup.** Point ``--engine-url`` at whichever
vLLM you want to benchmark and pass the shared config:

.. code-block:: bash

    lmcache bench engine \
        --engine-url http://localhost:8000 \
        --config bench_config.json

Run this once against Setup A's vLLM and once against Setup B's
vLLM-plus-LMCache, recording the metrics from each run.

Results
~~~~~~~

Pull the headline numbers out of each run's
``Engine Benchmark Result (long-doc-qa)`` summary:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Metric
     - Setup A (vLLM)
     - Setup B (+ LMCache)
   * - Successful requests
     - 46
     - 46
   * - Benchmark duration (s)
     - 23.47
     - 13.79
   * - Mean TTFT (ms)
     - 757.00
     - 185.00

That's a **75%** reduction in Mean TTFT (757 ms → 185 ms) and a **41%**
reduction in benchmark duration (23.47 s → 13.79 s) from LMCache offloading.

.. note::
   Without LMCache, once the benchmark's working set overflows the GPU KV
   cache the second round has to recompute every prefix, so TTFT and
   throughput don't improve even when content repeats. LMCache keeps the
   evicted blocks on CPU RAM and restores them on demand -- that's where
   the speedup comes from.
