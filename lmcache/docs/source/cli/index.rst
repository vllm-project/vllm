CLI Reference
=============

The ``lmcache`` command-line interface provides tools for launching,
managing, inspecting, and benchmarking LMCache servers and the inference
engines in front of them.

.. code-block:: bash

   lmcache <command> [options]

After installing LMCache, the ``lmcache`` command is available globally.
Run ``lmcache -h`` to see all commands, or ``lmcache <command> -h`` for a
specific command.

Installation
------------

The ``lmcache`` CLI ships in two packages:

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Package
     - Install
     - When to use
   * - ``lmcache``
     - ``pip install lmcache``
     - Full install: server, CLI, and CUDA extensions. Required for
       ``server``, ``bench server``, ``bench l2``, and ``trace``.
       Linux + GPU.
   * - ``lmcache-cli``
     - ``pip install lmcache-cli``
     - CLI only: ``ping``, ``query``, ``describe``, ``kvcache``,
       ``quota``, ``bench engine``. No GPU required, any OS.

.. note::

   Do not install both packages in the same environment — they both provide
   the ``lmcache`` entry point.

Available Commands
------------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Command
     - Description
   * - :doc:`server`
     - Launch the LMCache MP server (ZMQ + HTTP). Requires the full install.
   * - :doc:`coordinator`
     - Launch the LMCache MP coordinator (HTTP instance registry).
   * - :doc:`describe`
     - Show detailed status of a running LMCache service.
   * - :doc:`ping`
     - Liveness check for LMCache or vLLM servers.
   * - :doc:`query`
     - Single-shot query interface for the serving engine.
   * - :doc:`bench`
     - Run sustained benchmarks against an inference engine
       (``engine``), an LMCache MP server (``server``), or an L2 cache
       adapter (``l2``).
   * - :doc:`kvcache`
     - Manage KV cache state (e.g. clear L1 cache) on a running server.
   * - :doc:`quota`
     - Manage per-salt cache quotas (set, get, list, delete).
   * - :doc:`trace`
     - Inspect and replay storage-level trace files.
   * - :doc:`tool`
     - Run offline analysis tools (e.g. the cache simulator).

Output Formats
--------------

Commands that produce metrics share three common flags:

* ``--format {terminal,json}`` — stdout format (default: ``terminal``).
* ``--output PATH`` — also write metrics to a file (uses ``--format``).
* ``-q`` / ``--quiet`` — suppress stdout; rely on the exit code.

The terminal output uses human-readable labels (e.g. ``"Round trip time
(ms)"``), while JSON uses machine-readable keys (e.g.
``"round_trip_time_ms"``).

Adding New Commands
-------------------

New CLI subcommands are added by creating a ``BaseCommand`` subclass under
``lmcache/cli/commands/``; they are discovered and registered automatically.
See :doc:`/developer_guide/cli` for details.

.. toctree::
   :maxdepth: 1

   server
   coordinator
   describe
   ping
   query
   bench
   kvcache
   quota
   trace
   tool

