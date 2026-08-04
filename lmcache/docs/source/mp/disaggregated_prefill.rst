.. _mp_disaggregated_prefill:

Disaggregated Prefill
=====================

Overview
--------

Disaggregated prefill (P/D) runs prefill and decode on separate vLLM instances:
a **prefill** instance computes the prompt's KV, hands it to a **decode**
instance over NIXL, and the decoder generates tokens without recomputing the
prompt. Prefill- and decode-heavy work then scale independently.

**LMCache MP adds cross-request KV reuse on top.** Each vLLM instance also
offloads its KV to a co-located LMCache server, so a recurring prefix — on a
later request, or after eviction from GPU HBM — is loaded from LMCache instead
of recomputed. KV-source order on the prefill path: local GPU cache → LMCache →
recompute.

Both are composed with vLLM's ``MultiConnector``, which runs two connectors per
instance:

* ``NixlConnector`` — the prefill→decode KV handoff for the current
  request, over NIXL (UCX / RDMA).
* ``LMCacheMPConnector`` — offload/load to the instance's LMCache
  server for cross-request reuse.

A vLLM **router** in P/D mode sends each request to a prefill then a decode
instance and threads the NIXL handshake between them.

How it works
------------

A minimal deployment is five processes:

* **Prefill vLLM (producer)** — ``MultiConnector[NixlConnector (kv_producer),
  LMCacheMPConnector]``; computes prompt KV, stores it to its LMCache server,
  and exposes it for the decoder to pull over NIXL.
* **Decode vLLM (consumer)** — ``MultiConnector[NixlConnector (kv_consumer),
  LMCacheMPConnector]``; pulls prompt KV from the producer and generates output.
* **Two LMCache servers** — one ``lmcache server`` per vLLM instance (they must
  not share one).
* **Router** — ``vllm-router --vllm-pd-disaggregation`` routes prefill then
  decode.

To share reuse across the prefill and decode pools too, layer
:doc:`P2P KV cache sharing <p2p>` on top (give both servers a coordinator and
``--p2p-advertise-url``).

Requirements
------------

.. important::

   `vllm-project/vllm#46865
   <https://github.com/vllm-project/vllm/pull/46865>`_ (the MultiConnector
   real-blocks fix) unblocks LMCache offload under ``MultiConnector``. Without
   it, offload silently never triggers.

* **NIXL** available in the vLLM environment (``lmcache[nixl]`` extra, which
  pulls ``nixl>=1.3.0``).

Configuration
-------------

Each vLLM instance takes a ``--kv-transfer-config`` selecting ``MultiConnector``
with two sub-connectors:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Key
     - Description
   * - ``kv_connector`` = ``MultiConnector``
     - Runs the ``kv_connector_extra_config.connectors`` side by side.
   * - ``kv_role``
     - ``kv_producer`` (prefill) or ``kv_consumer`` (decode), on both the
       top-level ``MultiConnector`` and the nested ``NixlConnector``.
   * - ``NixlConnector``
     - The prefill→decode handoff. Set ``kv_load_failure_policy`` = ``fail`` so
       a failed transfer surfaces instead of silently recomputing.
   * - ``LMCacheMPConnector``
     - Offload/load to the local LMCache server; use ``kv_role`` = ``kv_both``.
   * - ``lmcache.mp.host`` / ``lmcache.mp.port``
     - Transport+host and ZMQ ``--port`` of the instance's LMCache server
       (distinct per instance).

The NIXL transfer is tuned via environment variables on the vLLM instances:
``VLLM_NIXL_SIDE_CHANNEL_HOST`` (host advertised for the handshake, reachable by
the peer), ``VLLM_NIXL_SIDE_CHANNEL_PORT`` (**must differ between instances on
the same host**; default ``5600``), plus ``UCX_NET_DEVICES=all`` and
``NCCL_CUMEM_ENABLE=1``. See :doc:`/cli/server` for the full ``lmcache server``
flag list.

Running a deployment
--------------------

The example below places the prefill pool, the decode pool, and the router on
separate hosts. Replace ``<PREFILL_IP>`` / ``<DECODE_IP>`` with routable
addresses and ``<model>`` with the model path (identical on both instances).

**Step 1 — prefill LMCache server** (on the prefill host):

.. code-block:: bash

   lmcache server \
       --port 6555 --http-port 8090 \
       --l1-size-gb 100 --eviction-policy LRU --chunk-size 256 \
       --instance-id prefiller

**Step 2 — prefill vLLM (producer)** (on the prefill host):

.. code-block:: bash

   VLLM_NIXL_SIDE_CHANNEL_HOST=<PREFILL_IP> VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
   UCX_NET_DEVICES=all NCCL_CUMEM_ENABLE=1 \
   vllm serve <model> \
       --port 8001 --tensor-parallel-size 1 \
       --kv-transfer-config '{"kv_connector":"MultiConnector","kv_role":"kv_producer","kv_connector_extra_config":{"connectors":[{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"},{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.host":"tcp://localhost","lmcache.mp.port":6555}}]}}'

The producer advertises its NIXL side channel on ``<PREFILL_IP>:5600`` and
offloads to the LMCache server on port ``6555``.

**Step 3 — decode LMCache server** (on the decode host):

.. code-block:: bash

   lmcache server \
       --port 6556 --http-port 8091 \
       --l1-size-gb 100 --eviction-policy LRU --chunk-size 256 \
       --instance-id decoder

**Step 4 — decode vLLM (consumer)** (on the decode host):

.. code-block:: bash

   VLLM_NIXL_SIDE_CHANNEL_HOST=<DECODE_IP> VLLM_NIXL_SIDE_CHANNEL_PORT=5558 \
   UCX_NET_DEVICES=all NCCL_CUMEM_ENABLE=1 \
   vllm serve <model> \
       --port 8002 --tensor-parallel-size 1 \
       --kv-transfer-config '{"kv_connector":"MultiConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"connectors":[{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"},{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.host":"tcp://localhost","lmcache.mp.port":6556}}]}}'

The consumer offloads to the LMCache server on port ``6556``; its side-channel
port (``5558``) differs from the producer's so the two can share a host.

**Step 5 — router** (on any host that can reach both vLLM instances):

.. code-block:: bash

   vllm-router \
       --policy round_robin \
       --vllm-pd-disaggregation \
       --prefill http://<PREFILL_IP>:8001 \
       --decode http://<DECODE_IP>:8002 \
       --host 0.0.0.0 --port 30000

Send requests to the router (``http://<ROUTER_IP>:30000/v1/...``); it handles
the prefill→decode split transparently.

Running on a single node (testing & debugging)
----------------------------------------------

Same five processes over ``localhost``, prefill on GPU 6 and decode on GPU 7.
The instances must differ in **every** shared-host port: the LMCache ``--port``
and ``--http-port``, the vLLM ``--port``, and ``VLLM_NIXL_SIDE_CHANNEL_PORT``.

.. code-block:: bash

   lmcache server --port 6555 --http-port 8090 \
       --l1-size-gb 100 --eviction-policy LRU --chunk-size 256 --instance-id prefiller

   CUDA_VISIBLE_DEVICES=6 VLLM_NIXL_SIDE_CHANNEL_HOST=127.0.0.1 VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
   UCX_NET_DEVICES=all NCCL_CUMEM_ENABLE=1 \
   vllm serve <model> --port 8001 --enforce-eager --max-model-len 16384 --gpu-memory-utilization 0.4 \
       --kv-transfer-config '{"kv_connector":"MultiConnector","kv_role":"kv_producer","kv_connector_extra_config":{"connectors":[{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"},{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.host":"tcp://localhost","lmcache.mp.port":6555}}]}}'

   lmcache server --port 6556 --http-port 8091 \
       --l1-size-gb 100 --eviction-policy LRU --chunk-size 256 --instance-id decoder

   CUDA_VISIBLE_DEVICES=7 VLLM_NIXL_SIDE_CHANNEL_HOST=127.0.0.1 VLLM_NIXL_SIDE_CHANNEL_PORT=5558 \
   UCX_NET_DEVICES=all NCCL_CUMEM_ENABLE=1 \
   vllm serve <model> --port 8002 --enforce-eager --max-model-len 16384 --gpu-memory-utilization 0.4 \
       --kv-transfer-config '{"kv_connector":"MultiConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"connectors":[{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"},{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.host":"tcp://localhost","lmcache.mp.port":6556}}]}}'

   vllm-router --policy round_robin --vllm-pd-disaggregation \
       --prefill http://localhost:8001 --decode http://localhost:8002 --host 0.0.0.0 --port 30000

Add ``--enable-tracing --otlp-endpoint http://localhost:4317`` to each
``lmcache server`` to export traces/metrics to a local OpenTelemetry collector.

.. note::

   On a single host, NIXL over ``localhost`` uses loopback/TCP, not RDMA, so
   latencies are not representative — single-node mode is for **functional**
   testing only.

Verifying it is working
-----------------------

.. code-block:: bash

   curl -s http://localhost:30000/v1/completions -H "Content-Type: application/json" \
       -d '{"model":"<model>","prompt":"The capital of France is","max_tokens":16}'

* **P/D routing** — the response ``id`` is stamped with the workers, e.g.
  ``cmpl-___prefill_addr_localhost:8001___decode_addr_localhost:8002_...``.
* **NIXL transfer** — the decode vLLM logs ``KV Transfer metrics:
  NixlConnector={'Num successful transfers': N, ...}``.
* **LMCache offload** — ``storage_manager.l1_manager.total_object_count`` in
  ``curl -s http://localhost:8090/status`` grows (prompts must be at least
  ``--chunk-size`` tokens to store anything).
* **LMCache hit rate** — read vLLM's **"External prefix cache hit rate"** log
  line (the fraction of prefill tokens served from LMCache), *not* the LMCache
  server ``/status``. It stays ``0`` while the working set fits in GPU HBM and
  rises once reuse spills out of the GPU cache.

Limitations and tuning
----------------------

* **Keep cold KV transfers bounded under high concurrency.** vLLM's native NIXL
  P/D path is sensitive to many large *cold* transfers (full-prompt, nothing
  cached) in flight at once. Keep the decode instance's GPU KV cache large
  enough to hold the working set so transfers stay small and cache-served, or
  reduce concurrency when per-request transfers are large.
