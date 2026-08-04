.. _mp_p2p:

P2P KV Cache Sharing
====================

Overview
--------

In a multi-node deployment, every node runs its own LMCache server process,
and each one caches the KV of the requests it has served in its local memory.
Without sharing, a prefix that was computed on one node has to be recomputed
from scratch when the same prefix arrives on a different node.

**Peer-to-peer (P2P) KV cache sharing turns the per-node caches into one
logical cache.** When a node looks up a prefix that is not in its own memory,
it can read that prefix's KV directly from the memory of the peer node that
holds it, over the datacenter network using RDMA. The result is a much higher
effective cache hit rate across the fleet, without any central storage tier in
the hot path.

The transfer is a one-sided RDMA read from the requesting node into its own L1
buffer; the node that owns the data is not interrupted to serve it. On an
RDMA-capable network (InfiniBand / RoCE) this is dramatically faster than
recomputing the prefix or round-tripping through a shared object store.

How it works
------------

P2P involves three pieces:

* **Coordinator** — a small HTTP service (one per deployment) that tracks which
  LMCache servers are alive. Each server registers with it and heartbeats; the
  coordinator answers "who are my live peers?" queries. It only manages
  membership — it never sees KV data or participates in lookups. See
  :doc:`coordinator`.
* **LMCache server** — each server runs a P2P controller that periodically asks
  the coordinator for the current peer list and, for every live peer, opens a
  connection used to look up and read that peer's KV.
* **Transfer channel** — the RDMA layer that performs the actual remote memory
  reads. Each server registers its L1 buffer once at startup so peers can read
  from it.

On a cache miss, a node asks the peer that owns the prefix to *lock and locate*
it, receives the remote addresses, RDMA-reads the KV into its own L1, and serves
the request from there. Peers are discovered and connected automatically as
they join, and disconnected automatically as they leave — no static peer lists.

Requirements
------------

* **A coordinator.** P2P needs the coordinator for peer discovery. A server
  started with ``--p2p-advertise-url`` but no ``--coordinator-url`` refuses to
  start.
* **An RDMA-capable network** (InfiniBand / RoCE) is strongly recommended for
  production performance. By default P2P uses the ``nixl`` transfer engine,
  which ships as the optional ``lmcache[nixl]`` extra — install it with
  ``uv pip install lmcache[nixl]`` (or ``pip install lmcache[nixl]``) before
  enabling P2P.
* **A single, contiguous L1 region.** The transfer channel registers the whole
  L1 buffer for RDMA, so P2P is incompatible with the GDS L1 tier
  (``--gds-l1-path``) and the Device-DAX L1 tier (``--l1-devdax-path``); the
  server refuses to start in those configurations.

Configuration
-------------

P2P is enabled per server by the ``--p2p-advertise-url`` flag. The relevant
``lmcache server`` flags are:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Flag
     - Description
   * - ``--p2p-advertise-url HOST:PORT``
     - Transfer-channel endpoint this server advertises to peers. Setting it
       enables P2P. Must be an address other nodes can reach.
   * - ``--p2p-listen-url HOST:PORT``
     - Address the transfer-channel server binds to. Defaults to
       ``--p2p-advertise-url``; set it to bind ``0.0.0.0`` while advertising a
       routable IP.
   * - ``--p2p-lookup-timeout SECONDS``
     - Deadline for a peer lookup before it counts as a miss (default ``30``).
   * - ``--p2p-load-timeout SECONDS``
     - Deadline for a peer KV read before it counts as a failure
       (default ``30``).
   * - ``--p2p-transfer-engine ENGINE``
     - Transfer-channel implementation (default ``nixl``).

P2P also reuses the coordinator connection flags (``--coordinator-url``,
``--coordinator-advertise-ip``, ``--coordinator-heartbeat-interval``); the
heartbeat interval doubles as the peer-discovery poll interval. See
:doc:`/cli/server` and :doc:`/cli/coordinator` for the full flag lists.

.. tip::

   Increase the L1 buffer alignment to at least 64 KB
   (``--l1-align-bytes 65536``) on servers that participate in P2P. A larger
   alignment lets the transfer channel issue bigger, better-aligned RDMA reads
   and noticeably improves transfer performance. The default (4 KB) is fine for
   non-P2P deployments.

Transfer engine backends
------------------------

The transfer engine is the component that performs the remote memory reads,
selected per server with ``--p2p-transfer-engine``.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Engine
     - Description
   * - ``nixl`` (default)
     - RDMA-based transport, shipped as the optional ``lmcache[nixl]`` extra
       (``uv pip install lmcache[nixl]``). Runs over InfiniBand / RoCE
       fabrics.

``nixl`` is the only backend available today. The transfer engine is a
pluggable abstraction, so additional backends can be added in the future
without changing the rest of the P2P stack.

Running a multi-node deployment
-------------------------------

The example below brings up a two-node fleet. Adding more nodes is just more
copies of the per-node step, all pointing at the same coordinator.

**Step 1 — start the coordinator** (on a host all nodes can reach, here
``10.0.0.1``):

.. code-block:: bash

   lmcache coordinator --host 0.0.0.0 --port 9300

**Step 2 — on each node, start the LMCache server with P2P enabled.** Replace
``<NODE_IP>`` with that node's routable address (e.g. ``10.0.0.2``,
``10.0.0.3``, ...):

.. code-block:: bash

   lmcache server \
       --host 0.0.0.0 --port 5555 \
       --http-port 8080 \
       --l1-size-gb 100 --eviction-policy LRU \
       --l1-align-bytes 65536 \
       --coordinator-url http://10.0.0.1:9300 \
       --coordinator-advertise-ip <NODE_IP> \
       --p2p-advertise-url <NODE_IP>:8500

``--coordinator-advertise-ip`` is the address peers use to reach this node's
control plane, and ``--p2p-advertise-url`` is the RDMA transfer endpoint. Here
the server binds ``0.0.0.0`` (all interfaces) and advertises ``<NODE_IP>``.

**Step 3 — on each node, start vLLM** pointed at the *local* LMCache server via
the connector. vLLM never talks to peers directly — the LMCache server it
connects to does P2P on its behalf:

.. code-block:: bash

   vllm serve <model> \
       --port 8000 \
       --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_load_failure_policy":"recompute","kv_connector_extra_config":{"lmcache.mp.port":5555}}'

Once both nodes have registered, a prefix first served on node 2 will be served
from node 2's cache when the same prefix later arrives on node 3 — read over
RDMA instead of recomputed.

Running on a single node (testing & debugging)
----------------------------------------------

You can exercise the entire P2P path on a single multi-GPU machine by running
two LMCache servers and two vLLM servers over ``localhost``, plus one
coordinator. This is the recommended way to develop and debug P2P.

.. code-block:: bash

   # Terminal 1 — coordinator
   lmcache coordinator --host 0.0.0.0 --port 9300

   # Terminal 2 — node "A": LMCache server
   lmcache server \
       --host 127.0.0.1 --port 6555 --http-port 7555 \
       --l1-size-gb 50 --eviction-policy LRU \
       --l1-align-bytes 65536 \
       --instance-id node-a \
       --coordinator-url http://127.0.0.1:9300 \
       --coordinator-advertise-ip 127.0.0.1 \
       --p2p-advertise-url 127.0.0.1:8555

   # Terminal 3 — node "A": vLLM on GPU 0
   CUDA_VISIBLE_DEVICES=0 vllm serve <model> --port 8000 \
       --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_load_failure_policy":"recompute","kv_connector_extra_config":{"lmcache.mp.port":6555}}'

   # Terminal 4 — node "B": LMCache server
   lmcache server \
       --host 127.0.0.1 --port 6556 --http-port 7556 \
       --l1-size-gb 50 --eviction-policy LRU \
       --l1-align-bytes 65536 \
       --instance-id node-b \
       --coordinator-url http://127.0.0.1:9300 \
       --coordinator-advertise-ip 127.0.0.1 \
       --p2p-advertise-url 127.0.0.1:8556

   # Terminal 5 — node "B": vLLM on GPU 1
   CUDA_VISIBLE_DEVICES=1 vllm serve <model> --port 8001 \
       --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_load_failure_policy":"recompute","kv_connector_extra_config":{"lmcache.mp.port":6556}}'

The two servers must differ in **every** port: ZMQ (``--port``), HTTP
(``--http-port``), and the P2P transfer endpoint (``--p2p-advertise-url``).
Give each a distinct ``--instance-id`` so they are easy to tell apart.

To test the path, send a long prompt to vLLM **A** (port ``8000``) and then the
*same* prompt to vLLM **B** (port ``8001``). B has never seen the prompt and its
own cache is empty, so any LMCache hit on B must have been read from A over P2P.

.. note::

   On a single host, ``localhost`` traffic typically uses the loopback/TCP path
   rather than RDMA, so latencies are not representative of a real RDMA fabric.
   Single-node mode is for **functional** testing and debugging; benchmark
   performance on a real multi-node RDMA deployment.

Verifying P2P is working
------------------------

Query a server's status endpoint to see its P2P state and discovered peers:

.. code-block:: bash

   curl -s http://127.0.0.1:7555/status | python3 -m json.tool

Look for ``p2p_state`` (``registered`` once it has joined the coordinator) and
``p2p_peer_count`` (the number of connected peers; ``1`` in the two-node
example). You can also list the fleet from the coordinator:

.. code-block:: bash

   curl -s http://127.0.0.1:9300/instances | python3 -m json.tool

With ``LMCACHE_LOG_LEVEL=DEBUG``, each server logs ``Added P2P adapter ... for
peer <instance-id>`` when it connects to a peer, and ``Removed P2P adapter ...``
when a peer leaves.

Limitations
-----------

* **Read-only.** A node reads KV from its peers; it never writes into a peer's
  memory. This keeps each node the sole owner of its L1.
* **One hop.** A node reads directly from the peer that holds the prefix; reads
  are not chained across multiple peers.
