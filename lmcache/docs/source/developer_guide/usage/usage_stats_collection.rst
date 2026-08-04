.. _usage-stats-collection:

Usage Stats Collection
======================

LMCache collects anonymous usage data by default to help the engineering team understand real-world workloads, prioritize optimizations, and improve reliability. All collected data is aggregated and contains no sensitive user information.

A sanitized subset of the aggregated data may be publicly released for the community’s benefit (for example, see a daily usage report `here <https://github.com/Hanchenli/OSS_Growth_Toolkit/tree/main/usage_tracker/report>`_).

What data is collected?
-----------------------

Usage stats are implemented in the ``lmcache/usage_telemetry/`` package. The
one-shot messages sent at startup depend on how LMCache runs.

When LMCache runs **inside the serving engine** (the single-process
integrations for vLLM, SGLang, and TensorRT-LLM), engine startup sends:

- **EnvMessage**
  Captures environment details such as cloud provider, CPU info, total memory, architecture, GPU count/type, and execution source.

- **EngineMessage**
  Records engine configuration and metadata, including cache settings (chunk size, local device, cache limits), remote backend parameters, blending settings, model name, world size, and KV-cache dtype/shape.

- **MetadataMessage**
  Reports execution metadata: the timestamp when the run started and total duration in seconds.

When LMCache runs as a **standalone multiprocess (MP) cache server**
(``lmcache server``), server startup sends:

- **EnvMessage**
  Same environment snapshot as above, taken on the cache-server host.

- **MPServerMessage**
  Records the server configuration: LMCache version, chunk size, hash
  algorithm, engine type, transfer mode, worker pool sizes, whether P2P is
  enabled, L1 size and medium (DRAM / GDS / DRAM+DevDAX), eviction policy,
  the configured L2 adapter and serde types, and the L2 store/prefetch
  policies.

  The MP server's ``--instance-id`` is **never** sent: it can be
  operator-chosen and therefore identifying. Model names are not known at
  server startup (vLLM instances register later) and are not part of this
  message.

In addition to the one-shot messages above, both deployment modes
periodically send interval counters (**ContinuousContextMessage**: tokens
retrieved, tokens stored, stored KV bytes, process uptime). Idle
intervals are still sent and double as a liveness heartbeat.
Single-process mode additionally sends a cache-lifespan histogram
(**CacheLifespanMessage**). The flush interval is
``LMCACHE_USAGE_TRACK_INTERVAL`` seconds (default 600).

Every payload carries four common fields:

- ``session_id`` -- a random UUID minted once per process, joining the
  one-shot context with the continuous messages of the same run.
- ``machine_id`` -- a random UUID persisted at
  ``~/.config/lmcache/machine_id``, grouping sessions from the same machine.
  It is never derived from hardware identifiers (MAC address, hostname).
- ``schema_version`` -- the version of the message schema.
- ``deployment_mode`` -- ``single_process`` (LMCache inside the serving
  engine) or ``mp_server`` (standalone MP cache server).

These messages are serialized to JSON and POSTed to the LMCache usage server.

Example JSON payload
~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

   {
     "message_type": "EnvMessage",
     "provider": "GCP",
     "num_cpu": 24,
     "cpu_type": "Intel(R) Xeon(R) CPU @ 2.20GHz",
     "cpu_family_model_stepping": "6,85,7",
     "total_memory": 101261135872,
     "architecture": ["64bit", "ELF"],
     "platforms": "Linux-5.10.0-28-cloud-amd64-x86_64-with-glibc2.31",
     "gpu_count": 2,
     "gpu_type": "NVIDIA L4",
     "gpu_memory_per_device": 23580639232,
     "source": "DOCKER"
   }

Previewing collected data
-------------------------

If you enable **local logging**, usage messages are appended to your specified log file. To inspect the most recent entries:

.. code-block:: bash

   tail ~/.config/lmcache/usage.log

Configuration & Opt-out
-----------------------

By default, usage tracking is **enabled**. Any one of the following opt-outs
disables all usage stats collection:

.. code-block:: bash

   # LMCache-specific opt-out
   export LMCACHE_TRACK_USAGE=false

   # The cross-tool "do not track" convention (1/true/yes)
   export DO_NOT_TRACK=1

When tracking is disabled, ``InitializeUsageContext`` will return ``None`` and
no data will be sent or logged, and no state files (such as ``machine_id``)
will be created.

Reference
~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 34 14 52

   * - Environment variable / file
     - Default
     - Effect
   * - ``LMCACHE_TRACK_USAGE``
     - unset
     - Set to ``false`` to disable all usage stats collection.
   * - ``DO_NOT_TRACK``
     - unset
     - Set to ``1``/``true``/``yes`` to disable collection (cross-tool
       convention).
   * - ``LMCACHE_USAGE_TRACK_URL``
     - ``http://stats.lmcache.ai:8080``
     - Override the stats server endpoint (e.g. for a private sink).
   * - ``LMCACHE_USAGE_TRACK_INTERVAL``
     - ``600``
     - Seconds between continuous-message flushes.
   * - ``~/.config/lmcache/machine_id``
     - created on first send
     - Holds the random anonymous machine UUID. Delete it to rotate the
       identifier.

Local logging
~~~~~~~~~~~~~

If you would like to log to a file in addition to (or instead of) sending data to the server, pass a local-log path when initializing:

.. code-block:: python

   from lmcache.usage_telemetry import InitializeUsageContext

   usage_ctx = InitializeUsageContext(
       config=engine_config,
       metadata=engine_metadata,
       local_log="~/.config/lmcache/usage.log"
   )

Omitting the ``local_log`` argument (or passing ``None``) disables local file logging.
