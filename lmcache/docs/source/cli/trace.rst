lmcache trace
=============

The ``lmcache trace`` command inspects and replays LMCache storage-level
trace files (``.lct``). It has two sub-commands:

.. code-block:: bash

   lmcache trace {info,replay} FILE [options]

.. note::

   ``lmcache trace`` needs the full ``lmcache`` package (StorageManager,
   trace codecs, ``TraceReader``). It is not available in the lightweight
   ``lmcache-cli`` install and exits with status ``2`` if those modules are
   missing.

Trace *capture* is not a ``trace`` sub-command — recording is bound to a live
server via ``lmcache server --trace-level storage [--trace-output ...]`` (see
:doc:`server`).


info
----

Print a one-screen summary of a trace file: header metadata plus per-qualname
record counts.

.. code-block:: bash

   lmcache trace info path/to/trace.lct

.. code-block:: text

   Trace file: path/to/trace.lct
     level:                storage
     format_version:       1
     trace_schema_version: 1
     duration:             12.345s
     sm_config_digest:     a1b2c3d4
     total_records:        2048
     ops:
       StorageManager.store: 1024
       StorageManager.retrieve: 1024

The only argument is the positional ``FILE`` (path to a ``.lct`` trace file).


replay
------

Reissue every recorded call against a fresh ``StorageManager``, honoring the
recorded inter-call timings.

.. code-block:: bash

   lmcache trace replay path/to/trace.lct \
       --l1-size-gb 10 --eviction-policy LRU

``replay`` accepts the standard storage-manager configuration flags
(``--l1-size-gb``, ``--eviction-policy``, ``--l2-...``); see
``lmcache server --help`` for the full list. The replay-side config may
differ from the config recorded in the trace, which can legitimately cause
retrieve misses.

.. warning::

   A replay environment mismatch may cause retrieve misses. Replay uses the
   replay-side StorageManager config (which may differ from the recorded
   config), runs on a host whose performance may differ from the recording
   host, and StorageManager reads/writes are async. Treat retrieve-miss
   counts as a signal about the replay environment, not a defect in the trace.

Options
~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Flag
     - Description
   * - ``FILE``
     - Path to a ``.lct`` trace file (positional, required).
   * - ``--verbose``
     - Print one line per replayed record.
   * - ``--jsonl-out PATH``
     - Write one JSON object per replayed record to ``PATH`` (qualname,
       latency_ms, failed).
   * - ``--output-dir DIR``
     - Directory for aggregated CSV / JSON summary output (default: current
       directory).
   * - ``--no-csv``
     - Skip the aggregated CSV summary export.
   * - ``--json``
     - Also export an aggregated JSON summary.
   * - ``-q`` / ``--quiet``
     - Suppress the terminal metrics table (files are still written).

The terminal summary reports overall replay stats (records replayed /
skipped / failed, duration, config-digest match) and per-op latency
percentiles. ``replay`` exits with status ``1`` if any record failed.
