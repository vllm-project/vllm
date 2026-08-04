lmcache query
=============

The ``lmcache query`` command sends a single OpenAI-compatible inference
request and reports token and latency metrics. It has two targets:

.. code-block:: bash

   lmcache query {engine,kvcache} [options]

* ``engine`` — send one request to a serving engine's HTTP API.
* ``kvcache`` — query KV-cache endpoints (not implemented yet).


query engine
------------

The ``query engine`` subcommand sends one request to the engine API and
reports metrics. ``--prompt`` supports placeholders: ``{lmcache}`` loads
``lmcache/cli/documents/lmcache.txt``, and custom documents can be passed with
``--documents NAME=PATH``. The prompt token count is taken directly from the
usage data reported by the engine (``stream_options: {include_usage: true}``).

.. code-block:: bash

   lmcache query engine --url http://localhost:8000/v1 \
     --prompt "{lmcache} Summarize LMCache usage." \
     --format terminal \
     --max-tokens 128

.. code-block:: text

   ================= Query Engine =================
   Model:                         facebook/opt-125m
   Input tokens:                                618
   --------------- Latency Metrics ----------------
   Output tokens:                                 9
   TTFT (ms):                                 26.88
   TPOT (ms/token):                            0.91
   Total latency (ms):                        35.05
   Throughput (tokens/s):                   1100.64
   ================================================

Options
~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Flag
     - Required
     - Description
   * - ``--url URL``
     - Yes
     - Serving engine base URL (e.g. ``http://localhost:8000/v1``).
   * - ``--prompt TEXT``
     - Yes
     - Prompt text with optional ``{name}`` placeholders. ``{lmcache}``
       expands to the bundled sample document.
   * - ``--model ID``
     - No
     - Model ID for the serving engine. Auto-detected from the engine's
       reported usage if omitted.
   * - ``--max-tokens N``
     - No
     - Maximum completion tokens (default: 128).
   * - ``--timeout SECS``
     - No
     - HTTP timeout in seconds (default: 30).
   * - ``--documents NAME=PATH``
     - No
     - Load file text for ``{NAME}`` in ``--prompt``. Accepts one or more
       ``NAME=PATH`` values.
   * - ``--completions``
     - No
     - Use ``POST /v1/completions`` only.
   * - ``--chat-first``
     - No
     - Try ``/v1/chat/completions`` first, then fall back to
       ``/v1/completions``.
   * - ``--format``
     - No
     - Output format: ``terminal`` (default) or ``json``.
   * - ``--output PATH``
     - No
     - Save metrics to a file (format follows ``--format``).
   * - ``-q`` / ``--quiet``
     - No
     - Suppress stdout output. Exit code only.
