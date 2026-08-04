.. _quickstart:

Quickstart
==========

This guide helps you get LMCache running end-to-end in a couple of minutes. Use the tabs below to switch the engine. Steps are the same; only the libraries and launch commands change.

.. tab-set::
   :sync-group: engine

   .. tab-item:: vLLM

      **Install LMCache**

      .. code-block:: bash

         uv venv --python 3.12
         source .venv/bin/activate
         uv pip install lmcache vllm

      LMCache supports two deployment modes with vLLM:

      - **Multiprocess (MP) mode** -- **recommended.** LMCache runs as a
        standalone service and vLLM attaches via ``LMCacheMPConnector``.
        Scales better, exposes management/observability endpoints, and
        supports sharing one cache across multiple engine instances.
      - **In-process mode** -- LMCache runs inside the vLLM process via
        ``LMCacheConnectorV1``. Single command, convenient for quick
        single-node experiments.

      .. tab-set::
         :sync-group: vllm-mode

         .. tab-item:: MP mode (recommended)
            :sync: mp

            Start the LMCache server. ``--host`` / ``--port`` set the ZMQ
            address vLLM connects to; they are spelled out here so the two
            commands line up (these are also the defaults):

            .. code-block:: bash

               # chunk-size 16 is an illustrative demo value so a short
               # prompt produces visible cache traffic; use the default
               # (256) in production.
               lmcache server \
                   --host localhost --port 5555 \
                   --l1-size-gb 20 --eviction-policy LRU --chunk-size 16

            The ZMQ port (``--port``, default **5555**) accepts connections
            from vLLM; the HTTP frontend (default **8080**) serves the
            management and metrics endpoints. See :doc:`../mp/configuration`
            for the full list of ``lmcache server`` and connector options.

            Start vLLM with the MP connector in a separate terminal. Point the
            connector at the server above via ``lmcache.mp.host`` /
            ``lmcache.mp.port`` in ``kv_connector_extra_config``. The host may
            include a ZMQ transport prefix (e.g. ``tcp://``); a bare
            ``host``/``host:port`` is also accepted and is normalized to
            ``tcp://`` automatically:

            .. code-block:: bash

               vllm serve Qwen/Qwen3-8B \
                   --port 8000 --kv-transfer-config \
                   '{"kv_connector":"LMCacheMPConnector", "kv_role":"kv_both", "kv_connector_extra_config": {"lmcache.mp.host": "localhost", "lmcache.mp.port": 5555}}'

            .. note::
               **Where does** ``LMCacheMPConnector`` **resolve to?** This depends on your vLLM version:

               - **vLLM < 0.20.0** -- ``"kv_connector":"LMCacheMPConnector"`` always
                 resolves to vLLM's built-in
                 ``vllm.distributed.kv_transfer.kv_connector.v1.LMCacheMPConnector``;
                 there is no way to redirect it to the LMCache-shipped implementation.

               - **vLLM >= 0.20.0** -- ``"kv_connector":"LMCacheMPConnector"`` still
                 defaults to vLLM's built-in connector, but you can opt in to the
                 LMCache-shipped implementation
                 (:mod:`lmcache.integration.vllm.lmcache_mp_connector`) by adding
                 ``kv_connector_module_path``:

                 .. code-block:: bash

                    vllm serve Qwen/Qwen3-8B \
                        --port 8000 --kv-transfer-config \
                        '{"kv_connector":"LMCacheMPConnector", "kv_connector_module_path":"lmcache.integration.vllm.lmcache_mp_connector", "kv_role":"kv_both"}'

                 The LMCache-shipped connector tracks the latest LMCache server
                 protocol and ships fixes/features ahead of the version vendored
                 into vLLM, so prefer it whenever you are on vLLM 0.20.0 or newer.

            **Test** -- open a new terminal and send two requests whose
            prompts share a prefix:

            **First request**

            .. code-block:: bash

               curl http://localhost:8000/v1/completions \
                 -H "Content-Type: application/json" \
                 -d '{
                   "model": "Qwen/Qwen3-8B",
                   "prompt": "Qwen3 is the latest generation of large language models in Qwen series, offering a comprehensive suite of dense and mixture-of-experts",
                   "max_tokens": 100,
                   "temperature": 0.7
                 }'

            **Second request**

            .. code-block:: bash

               curl http://localhost:8000/v1/completions \
                 -H "Content-Type: application/json" \
                 -d '{
                   "model": "Qwen/Qwen3-8B",
                   "prompt": "Qwen3 is the latest generation of large language models in Qwen series, offering a comprehensive suite of dense and mixture-of-experts (MoE) models",
                   "max_tokens": 100,
                   "temperature": 0.7
                 }'

            **You should see LMCache logs like this** -- in MP mode the
            store/retrieve logs come from the standalone ``lmcache server``
            process, one entry per chunk.

            **First request** -- cache is empty, so every aligned chunk is
            offloaded:

            .. code-block:: text

               [2026-04-22 19:49:56,316] LMCache INFO: Stored 16 tokens in 0.023 seconds (server.py:390:lmcache.v1.multiprocess.server)
               [2026-04-22 19:49:56,555] LMCache INFO: Stored 16 tokens in 0.005 seconds (server.py:390:lmcache.v1.multiprocess.server)
               [2026-04-22 19:49:56,691] LMCache INFO: Stored 16 tokens in 0.005 seconds (server.py:390:lmcache.v1.multiprocess.server)
               ...

            **Second request** -- the shared prefix is retrieved from CPU
            RAM; only the new tail is stored:

            .. code-block:: text

               [2026-04-22 19:50:04,686] LMCache INFO: Retrieved 16 tokens in 0.003 seconds (server.py:573:lmcache.v1.multiprocess.server)
               [2026-04-22 19:50:04,832] LMCache INFO: Stored 16 tokens in 0.005 seconds (server.py:390:lmcache.v1.multiprocess.server)
               [2026-04-22 19:50:04,968] LMCache INFO: Stored 16 tokens in 0.005 seconds (server.py:390:lmcache.v1.multiprocess.server)
               ...

            For request-level statistics (hit ratio, bytes transferred) see
            :doc:`../mp/observability/index`.

         .. tab-item:: In-process mode
            :sync: inproc

            Start vLLM with LMCache embedded in the engine process:

            .. code-block:: bash

               # The chunk size here is only for illustration purpose, use default one (256) later
               LMCACHE_CHUNK_SIZE=8 \
               vllm serve Qwen/Qwen3-8B \
                   --port 8000 --kv-transfer-config \
                   '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'

            .. note::
               To customize further, create a config file. See
               :doc:`../api_reference/configurations` for all options.

            **Alternative simpler command:**

            .. code-block:: bash

               vllm serve <MODEL NAME> \
                   --kv-offloading-backend lmcache \
                   --kv-offloading-size <SIZE IN GB> \
                   --disable-hybrid-kv-cache-manager

            The ``--disable-hybrid-kv-cache-manager`` flag is mandatory.
            All configuration options from the
            :doc:`../api_reference/configurations` page still apply.

            **Test** -- open a new terminal and send two requests whose
            prompts share a prefix:

            **First request**

            .. code-block:: bash

               curl http://localhost:8000/v1/completions \
                 -H "Content-Type: application/json" \
                 -d '{
                   "model": "Qwen/Qwen3-8B",
                   "prompt": "Qwen3 is the latest generation of large language models in Qwen series, offering a comprehensive suite of dense and mixture-of-experts",
                   "max_tokens": 100,
                   "temperature": 0.7
                 }'

            **Second request**

            .. code-block:: bash

               curl http://localhost:8000/v1/completions \
                 -H "Content-Type: application/json" \
                 -d '{
                   "model": "Qwen/Qwen3-8B",
                   "prompt": "Qwen3 is the latest generation of large language models in Qwen series, offering a comprehensive suite of dense and mixture-of-experts (MoE) models",
                   "max_tokens": 100,
                   "temperature": 0.7
                 }'

            **You should see LMCache logs like this** -- in-process mode
            emits the logs inline with the vLLM engine core.

            **First request** -- prompt is offloaded to LMCache:

            .. code-block:: text

               (EngineCore_DP0 pid=458469) [2025-09-30 00:08:43,982] LMCache INFO: Stored 31 out of total 31 tokens. size: 0.0040 gb, cost 1.95 ms, throughput: 1.98 GB/s; offload_time: 1.88 ms, put_time: 0.07 ms

            **Second request** -- hits the cache and stores the new tail:

            .. code-block:: text

               Reqid: cmpl-6709d8795d3c4464b01999c9f3fffede-0, Total tokens 32, LMCache hit tokens: 24, need to load: 8
               (EngineCore_DP0 pid=494270) [2025-09-30 01:12:36,502] LMCache INFO: Retrieved 8 out of 24 required tokens (from 32 total tokens). size: 0.0011 gb, cost 0.55 ms, throughput: 1.98 GB/s;
               (EngineCore_DP0 pid=494270) [2025-09-30 01:12:36,509] LMCache INFO: Storing KV cache for 8 out of 32 tokens (skip_leading_tokens=24)
               (EngineCore_DP0 pid=494270) [2025-09-30 01:12:36,510] LMCache INFO: Stored 8 out of total 8 tokens. size: 0.0011 gb, cost 0.43 ms, throughput: 2.57 GB/s; offload_time: 0.40 ms, put_time: 0.03 ms

            - **Total tokens 32**: The new prompt has 32 tokens after tokenization.
            - **LMCache hit tokens: 24**: 24 tokens (full 8-token chunks) were found in the cache from the first request that stored 31 tokens.
            - **Need to load: 8**: vLLM auto prefix caching uses block size 16; 16 tokens already sit in GPU RAM, so LMCache only loads 24-16=8.
            - **Why 24 hit tokens instead of 31?** LMCache hashes every 8 tokens (8, 16, 24, 31). It matches page-aligned chunks, so it uses the 24-token hash.
            - **Stored another 8 tokens**: The new 8 tokens form a full chunk and are stored for future reuse.

   .. tab-item:: SGLang

      .. note::
         The SGLang integration now defaults to MP (multi-process) mode.
         Please refer to `examples/sgl_integration/README.md`_ for the current setup instructions.

      .. _examples/sgl_integration/README.md: https://github.com/LMCache/LMCache/blob/dev/examples/sgl_integration/README.md

      **Install SGLang**

      .. code-block:: bash

         uv venv --python 3.12
         source .venv/bin/activate
         uv pip install --prerelease=allow lmcache "sglang"

      **Start SGLang with LMCache**

      .. code-block:: bash

         cat > lmc_config.yaml <<'EOF'
         chunk_size: 8  # demo only; use 256 for production
         local_cpu: true
         use_layerwise: true
         max_local_cpu_size: 10  # GB
         EOF

         export LMCACHE_CONFIG_FILE=$PWD/lmc_config.yaml

         python -m sglang.launch_server \
           --model-path Qwen/Qwen3-8B \
           --host 0.0.0.0 \
           --port 30000 \
           --enable-lmcache

      .. note::
         Configure LMCache via the config file. See :doc:`../api_reference/configurations` for the full list.

      **Test** -- open a new terminal and send two requests whose prompts
      share a prefix:

      **First request**

      .. code-block:: bash

         curl http://localhost:30000/v1/chat/completions \
           -H "Content-Type: application/json" \
           -d '{
             "model": "Qwen/Qwen3-8B",
             "messages": [{"role": "user", "content": "Qwen3 is the latest generation of large language models in Qwen series, offering a comprehensive suite of dense and mixture-of-experts"}],
             "max_tokens": 100,
             "temperature": 0.7
           }'

      **Second request**

      .. code-block:: bash

         curl http://localhost:30000/v1/chat/completions \
           -H "Content-Type: application/json" \
           -d '{
             "model": "Qwen/Qwen3-8B",
             "messages": [{"role": "user", "content": "Qwen3 is the latest generation of large language models in Qwen series, offering a comprehensive suite of dense and mixture-of-experts (MoE) models"}],
             "max_tokens": 100,
             "temperature": 0.7
           }'

      **You should see LMCache logs like this:**

      **First request** -- prompt plus generated tokens are stored:

      .. code-block:: text

         Prefill batch, #new-seq: 1, #new-token: 35, #cached-token: 0, token usage: 0.00, #running-req: 0, #queue-req: 0,
         Decode batch, #running-req: 1, #token: 74, token usage: 0.00, cuda graph: True, gen throughput (token/s): 1.63, #queue-req: 0,
         Decode batch, #running-req: 1, #token: 114, token usage: 0.00, cuda graph: True, gen throughput (token/s): 87.95, #queue-req: 0,
         LMCache INFO: Stored 128 out of total 135 tokens. size: 0.0195 GB, cost 12.8890 ms, throughput: 1.5153 GB/s (cache_engine.py:623:lmcache.v1.cache_engine)

      **Second request** -- Radix Cache and LMCache share the prefix; only the new portion is stored:

      .. code-block:: text

         Prefill batch, #new-seq: 1, #new-token: 10, #cached-token: 30, token usage: 0.00, #running-req: 0, #queue-req: 0,
         Decode batch, #running-req: 1, #token: 64, token usage: 0.00, cuda graph: True, gen throughput (token/s): 8.29, #queue-req: 0,
         Decode batch, #running-req: 1, #token: 104, token usage: 0.00, cuda graph: True, gen throughput (token/s): 87.95, #queue-req: 0,
         Decode batch, #running-req: 1, #token: 144, token usage: 0.00, cuda graph: True, gen throughput (token/s): 87.89, #queue-req: 0,
         LMCache INFO: Stored 112 out of total 140 tokens. size: 0.0171 GB, cost 11.1986 ms, throughput: 1.5261 GB/s (cache_engine.py:623:lmcache.v1.cache_engine)

      - **Total tokens 140**: SGLang stores KV cache for both prefill and decode tokens together, so total = 40 prompt + 100 generated = 140 tokens.
      - **Cached tokens: 30**: SGLang's Radix Attention Cache reused 30 tokens from the first request.
      - **LMCache hit tokens: 24**: LMCache detected 24 tokens (3 full 8-token chunks) stored from the first request. Since Radix Cache already provides 30 tokens in GPU memory, these 24 tokens don't need to be loaded from LMCache or stored again.
      - **New tokens: 10**: Only 10 prompt tokens need prefill computation (40 prompt - 30 cached = 10).
      - **Stored 112 out of 140**: 24 tokens (3 full chunks) are already in LMCache and skipped. Of the remaining 116 tokens, 112 (14 full 8-token chunks) are stored.

   .. tab-item:: TensorRT-LLM

      .. note::
         This integration depends on the connector preset registry from
         `NVIDIA/TensorRT-LLM PR #12626
         <https://github.com/NVIDIA/TensorRT-LLM/pull/12626>`_ and the
         matching LMCache adapter, neither of which has shipped in a
         stable release yet. Until they do, install both from source:

         .. code-block:: bash

            uv venv --python 3.12
            source .venv/bin/activate

            # LMCache from source (dev branch)
            uv pip install git+https://github.com/LMCache/LMCache.git@dev

            # TensorRT-LLM from source — see NVIDIA's build guide:
            # https://nvidia.github.io/TensorRT-LLM/installation/build-from-source-linux.html

         Once both ship in a stable release, the install command will be:

         .. code-block:: bash

            uv pip install lmcache "tensorrt_llm>=<version>" \
                --extra-index-url https://pypi.nvidia.com

      LMCache integrates with TensorRT-LLM via TRT-LLM's
      **KV Cache Connector** API and supports two deployment modes:

      - **In-process mode** (``connector: lmcache``) -- LMCache runs as
        a singleton inside the TRT-LLM process. Simplest setup; no
        extra service to manage.
      - **MP mode** (``connector: lmcache-mp``) -- LMCache runs as a
        standalone server. Multiple TRT-LLM workers on the same node
        can share the cache, and the cache survives a TRT-LLM crash.

      .. tab-set::
         :sync-group: trtllm-mode

         .. tab-item:: In-process mode
            :sync: inproc

            Configure LMCache via env vars:

            .. code-block:: bash

               export PYTHONHASHSEED=0  # required — chunk hashing depends on stable hash()
               export LMCACHE_CHUNK_SIZE=256
               export LMCACHE_LOCAL_CPU=True
               export LMCACHE_MAX_LOCAL_CPU_SIZE=2.0  # GiB

            Build the TRT-LLM ``LLM`` with ``connector: lmcache``:

            .. code-block:: python

               from tensorrt_llm import LLM, SamplingParams
               from tensorrt_llm.llmapi.llm_args import (
                   KvCacheConfig, KvCacheConnectorConfig,
               )

               llm = LLM(
                   model="Qwen/Qwen2-1.5B-Instruct",
                   backend="pytorch",
                   kv_cache_config=KvCacheConfig(enable_block_reuse=True),
                   kv_connector_config=KvCacheConnectorConfig(connector="lmcache"),
               )

               out = llm.generate(["Your prompt here"], SamplingParams(max_tokens=64))
               print(out[0].outputs[0].text)

         .. tab-item:: MP mode
            :sync: mp

            ``PYTHONHASHSEED=0`` must be set in **both** terminals --
            chunk hashing depends on a stable ``hash()``, and the
            server and client must agree on the seed.

            Start the LMCache server:

            .. code-block:: bash

               export PYTHONHASHSEED=0
               lmcache server \
                   --l1-size-gb 10 --eviction-policy LRU --chunk-size 256

            In a separate terminal, point TRT-LLM at the server via
            ``server_url``:

            .. code-block:: bash

               export PYTHONHASHSEED=0
               python run_trtllm.py

            where ``run_trtllm.py`` contains:

            .. code-block:: python

               from tensorrt_llm import LLM, SamplingParams
               from tensorrt_llm.llmapi.llm_args import (
                   KvCacheConfig, KvCacheConnectorConfig,
               )

               llm = LLM(
                   model="Qwen/Qwen2-1.5B-Instruct",
                   backend="pytorch",
                   kv_cache_config=KvCacheConfig(enable_block_reuse=True),
                   kv_connector_config=KvCacheConnectorConfig(
                       connector="lmcache-mp",
                       server_url="tcp://localhost:5555",
                   ),
               )

               out = llm.generate(["Your prompt here"], SamplingParams(max_tokens=64))
               print(out[0].outputs[0].text)

      .. note::
         The TRT-LLM adapter reads :class:`LMCacheEngineConfig` the
         same way the vLLM adapter does: ``LMCACHE_CONFIG_FILE`` for
         a YAML file, otherwise individual ``LMCACHE_*`` environment
         variables. See :doc:`../api_reference/configurations` for
         all options.

🎉 **You now have LMCache caching and reusing KV caches across all three engines.**

More MP server options
----------------------

The vLLM MP example above runs ``lmcache server`` locally on the default
ports. Common variations:

**Custom port or remote host** -- by default the connector talks to
``localhost:5555``. To use a different port, or a server on another host,
pass ``lmcache.mp.host`` / ``lmcache.mp.port`` in
``kv_connector_extra_config``:

.. code-block:: bash

   vllm serve Qwen/Qwen3-8B --kv-transfer-config \
     '{"kv_connector":"LMCacheMPConnector", "kv_role":"kv_both", "kv_connector_extra_config": {"lmcache.mp.host": "tcp://10.0.0.1", "lmcache.mp.port": 6555}}'

**CPU-only (no GPU)** -- the server runs with a ``StubCPUDevice`` and shares
KV tensors with vLLM over POSIX shared memory. Start ``lmcache server``
normally, then set ``lmcache.mp.mp_transfer_mode=lmcache_driven`` on the vLLM
side to enable the zero-copy SHM handle path (the default ``auto`` routing
maps non-CUDA devices to ``engine_driven``, which uses the worker-side
gather/scatter copy path instead).

**Docker** -- see :doc:`../production/docker_deployment`.

**HTTP management endpoints** (health, clear-cache, status) -- see
:doc:`../mp/http_api`.

Next Steps
----------

- **Performance Testing**: Try the :doc:`benchmarking` section to experience LMCache's performance benefits with more comprehensive examples
- **Production**: Deploy LMCache with Docker or Kubernetes, plus observability and tuning -- see :doc:`../mp/deployment`