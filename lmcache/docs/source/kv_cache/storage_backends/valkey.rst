Valkey
======

.. warning::

   This page documents the behavior of LMCache's in-process mode (deprecated). Please consider using :doc:`LMCache MP mode </mp/index>` for better feature support and performance. For the MP mode equivalent of this page, see :doc:`/mp/l2_storage/resp`.


Overview
--------

Valkey is an open source (BSD) high-performance key/value datastore and is a supported option for remote KV Cache offloading in LMCache.
Some other remote backends are :doc:`Mooncake <./mooncake>`, :doc:`Redis <./redis>`, and :doc:`InfiniStore <./infinistore>`.

Prerequisites
-------------

To use this connector, you need valkey-glide 2.0 or higher.

.. code-block:: shell

    # Install Valkey-GLIDE (Minimum 2.0.0 or higher)
    $ pip install valkey-glide

Configuration Reference
-----------------------

The following ``extra_config`` keys are supported:

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Key
     - Default
     - Description
   * - ``valkey_num_workers``
     - ``8``
     - Number of worker threads, each with its own GLIDE client connection.
   * - ``valkey_mode``
     - ``"standalone"``
     - ``"standalone"`` or ``"cluster"``. Cluster mode auto-discovers topology from a seed node.
   * - ``tls_enable``
     - ``false``
     - Enable TLS. Required for ElastiCache Serverless.
   * - ``valkey_username``
     - ``""``
     - Authentication username.
   * - ``valkey_password``
     - ``""``
     - Authentication password.
   * - ``valkey_database``
     - None
     - Database ID (standalone mode only, ignored in cluster mode).
   * - ``valkey_enable_ttl``
     - ``false``
     - Feature flag. When ``true``, every key is written with an expiry (see ``valkey_ttl_sec``) so Valkey/Redis ``volatile-*`` eviction policies can reclaim L2 cache keys once the node reaches ``maxmemory``. When ``false`` (default), keys are persisted without a TTL.
   * - ``valkey_ttl_sec``
     - ``86400``
     - Key TTL in seconds, applied only when ``valkey_enable_ttl`` is ``true``. Must be a positive integer. If the flag is enabled but this key is omitted, it defaults to ``86400`` (24 hours).
   * - ``request_timeout``
     - ``5.0``
     - GLIDE request timeout in seconds. Also used as the Python-side Future timeout.
   * - ``connection_timeout``
     - ``10.0``
     - GLIDE connection timeout in seconds for initial client connections.

Example Configurations
----------------------

Standalone-mode
~~~~~~~~~~~~~~~~

Basic Valkey Configuration (Standalone-mode)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   chunk_size: 256
   remote_url: "valkey://<your host>:6379"
   remote_serde: "naive"
   extra_config:
     valkey_username: "Your username"
     valkey_password: "Your password"

Standalone-mode Valkey Configuration with database
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   chunk_size: 256
   remote_url: "valkey://<your host>:6379"
   remote_serde: "naive"
   extra_config:
     valkey_username: "Your username"
     valkey_password: "Your password"
     valkey_database: 0

Standalone-mode Valkey Configuration with key TTL (volatile eviction)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Enable a per-key TTL so a node configured with a ``volatile-lru`` /
``volatile-lfu`` eviction policy can reclaim L2 cache keys once it reaches
``maxmemory``. Without a TTL such policies never evict the KV cache keys,
which can choke the remote cache.

.. code-block:: yaml

   chunk_size: 256
   remote_url: "valkey://<your host>:6379"
   remote_serde: "naive"
   extra_config:
     valkey_username: "Your username"
     valkey_password: "Your password"
     valkey_enable_ttl: true
     valkey_ttl_sec: 3600   # keys expire after 1 hour

Cluster-mode
~~~~~~~~~~~~~~~~

Cluster-mode Valkey Configuration (Endpoint)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For example, the configuration endpoint in ElastiCache is as follows:

<cache-name>.<identifier>.clustercfg.<region>.cache.amazonaws.com

You need to add this DNS name in the <your host>.


.. code-block:: yaml

   chunk_size: 256
   remote_url: "valkey://<your host>:6379"
   remote_serde: "naive"
   extra_config:
     valkey_mode: "cluster"
     valkey_username: "Your username"
     valkey_password: "Your password"


Cluster-mode Valkey Configuration (Nodes)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Nodes are deployed directly and configured in cluster mode without connecting them via DNS names (CNAME).

In this scenario, you simply input multiple IP hosts and ports.

Example: 172.0.0.1:7001, 172.0.0.2:7002 ... 172.0.0.6:7006

.. code-block:: yaml

   chunk_size: 256
   remote_url: "valkey://<your host 1>:<your port 1>, <your host 2>:<your port 2>, ... <your host N>:<your port N>"
   remote_serde: "naive"
   extra_config:
     valkey_mode: "cluster"
     valkey_username: "Your username"
     valkey_password: "Your password"

Cluster-mode Valkey Configuration with numbered databases (Valkey 9.0+ and Valkey-GLIDE 2.1+)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Valkey connector supports numbered databases in both the Endpoint using DNS and the Nodes method using IP and port pairs.

.. code-block:: yaml

   chunk_size: 256
   remote_url: "valkey://<your host>:6379"
   remote_serde: "naive"
   extra_config:
     valkey_mode: "cluster"
     valkey_username: "Your username"
     valkey_password: "Your password"
     valkey_database: 1

TLS / ElastiCache Serverless
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ElastiCache Serverless requires TLS. Set ``tls_enable: true``:

.. code-block:: yaml

   chunk_size: 256
   remote_url: "valkey://<serverless-endpoint>:6379"
   remote_serde: "naive"
   extra_config:
     valkey_mode: "cluster"
     tls_enable: true
     valkey_num_workers: 32

Performance Tuning
~~~~~~~~~~~~~~~~~~

For large models (e.g., 70B with TP=8), increase the worker count for higher throughput:

.. code-block:: yaml

   chunk_size: 256
   remote_url: "valkey://<your host>:6379"
   remote_serde: "naive"
   pre_caching_hash_algorithm: sha256_cbor_64bit  # Required for TP>1
   extra_config:
     valkey_mode: "cluster"
     valkey_num_workers: 32