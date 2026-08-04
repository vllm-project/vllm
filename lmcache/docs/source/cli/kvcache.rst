lmcache kvcache
===============

The ``lmcache kvcache`` command manages KV cache state on a running LMCache
server.

.. code-block:: bash

   lmcache kvcache <sub-command> [options]

.. code-block:: text

   $ lmcache kvcache -h
   usage: lmcache kvcache [-h] [--format FORMAT] [--output PATH] [-q] {clear} ...

   Manage KV cache state.

   subcommands:
     clear          Clear all cached KV data in L1 (CPU)

   options:
     -h, --help       show this help message and exit
     --format FORMAT  Stdout output format (default: terminal). Available: terminal, json.
     --output PATH    Save metrics to a file at PATH (format chosen by --format).
     -q, --quiet      Suppress stdout output. Exit code only.

clear
-----

Clear all cached KV data in **L1 (CPU memory)** on the target LMCache server.

.. code-block:: bash

   lmcache kvcache clear --url <MP_HTTP_URL>

**Example:**

.. code-block:: bash

   $ lmcache kvcache clear --url http://localhost:8000

   ================ KV Cache Clear ================
   Status:                                       OK
   ================================================

**JSON output** (for scripting with ``jq``):

.. code-block:: bash

   $ lmcache kvcache clear --url http://localhost:8000 --format json
   {
     "title": "KV Cache Clear",
     "metrics": {
       "status": "OK"
     }
   }

**Quiet mode** (exit code only, no output):

.. code-block:: bash

   $ lmcache kvcache clear -q --url http://localhost:8000
   $ echo $?
   0

Options
~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 10 65

   * - Flag
     - Required
     - Description
   * - ``--url``
     - Yes
     - URL of the LMCache MP HTTP server (e.g. ``http://localhost:8000``).
   * - ``--format``
     - No
     - Output format: ``terminal`` (default) or ``json``.
   * - ``--output``
     - No
     - Save output to a file (uses the format chosen by ``--format``).
   * - ``-q`` / ``--quiet``
     - No
     - Suppress stdout. Useful in scripts where you only need the exit code.

Exit Codes
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Code
     - Meaning
   * - ``0``
     - Success.
   * - ``1``
     - Error (connection failure, server error, bad arguments).

Common Patterns
---------------

**Handle temporary server unavailability:**

If the server is temporarily unreachable (e.g. due to network issue), the command
fails with exit code 1. For persistent connectivity issues, use ``lmcache ping``
to diagnose.

.. code-block:: bash

   if lmcache kvcache clear -q --url http://localhost:8000; then
       echo "Cache cleared"
   else
       echo "Clear failed — server temporarily unreachable, retrying later"
   fi

**Clear cache and capture JSON result:**

.. code-block:: bash

   RESULT=$(lmcache kvcache clear --url http://localhost:8000 --format json)
   STATUS=$(echo "$RESULT" | jq -r '.metrics.status')
   echo "Clear status: $STATUS"
