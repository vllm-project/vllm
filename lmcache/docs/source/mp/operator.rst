Kubernetes Operator
===================

The LMCache Kubernetes operator automates the deployment and lifecycle
management of LMCache multiprocess servers.  Instead of hand-writing
DaemonSets, Services, and ConfigMaps (as described in the manual
:doc:`deployment` guide), you declare a single ``LMCacheEngine`` custom
resource and the operator reconciles all underlying Kubernetes objects.

.. contents::
   :local:
   :depth: 2

Why Use the Operator
--------------------

The manual DaemonSet approach works, but it has sharp edges the operator
eliminates:

- **Auto-injected pod settings** -- The operator always sets ``hostIPC: true``
  and ``--host 0.0.0.0``.  Forgetting ``hostIPC`` in a hand-written manifest
  causes silent CUDA IPC failures (``cudaErrorMapBufferObjectFailed``) that are
  hard to debug.
- **Node-local service discovery** -- The operator creates a ClusterIP Service
  with ``internalTrafficPolicy=Local`` and a connection ConfigMap that vLLM
  pods simply mount.  No ``hostNetwork``, no Downward API, no shell variable
  substitution.
- **Auto-computed resource sizing** -- Memory requests and limits are derived
  from ``l1.sizeGB``, avoiding OOM kills (under-provisioned) or wasted node
  capacity (over-provisioned).
- **Declarative Prometheus integration** -- Set
  ``prometheus.serviceMonitor.enabled: true`` and the operator creates a
  ``ServiceMonitor`` CR that the Prometheus Operator discovers automatically.
- **CRD validation** -- OpenAPI schema validation catches misconfigurations
  (e.g., ``l1.sizeGB <= 0``, invalid port range) at ``kubectl apply`` time,
  before any pods are created.

Prerequisites
-------------

- Kubernetes 1.20+
- ``kubectl`` configured to access your cluster
- (Optional) `Prometheus Operator <https://github.com/prometheus-operator/prometheus-operator>`_
  for ServiceMonitor support

Installing the Operator
-----------------------

**Option A: One-line install from release (recommended)**

.. code-block:: bash

    # Latest stable release
    kubectl apply -f https://github.com/LMCache/LMCache/releases/download/operator-latest/install.yaml

    # Or nightly build from the dev branch
    kubectl apply -f https://github.com/LMCache/LMCache/releases/download/operator-nightly-latest/install.yaml

**Option B: Build from source**

.. code-block:: bash

    cd operator
    make build
    make install
    make deploy IMG=<your-registry>/lmcache-operator:latest

Deploying an LMCacheEngine
---------------------------

A minimal CR deploys a DaemonSet with 60 GB L1 cache on every GPU node:

.. code-block:: yaml

    apiVersion: lmcache.lmcache.ai/v1alpha1
    kind: LMCacheEngine
    metadata:
      name: my-cache
    spec:
      l1:
        sizeGB: 60

.. code-block:: bash

    kubectl apply -f lmcache-engine.yaml

The operator automatically:

- Creates a DaemonSet running one LMCache server pod per matched node
- Sets ``hostIPC: true`` and passes ``--host 0.0.0.0`` to the server
- Creates a node-local ClusterIP Service for vLLM discovery
- Creates a connection ConfigMap (``my-cache-connection``) with the
  ``kv-transfer-config`` JSON that vLLM needs
- Auto-computes resource requests/limits from the L1 cache size
- Defaults ``nodeSelector`` to ``nvidia.com/gpu.present: "true"``

.. note::
   The operator defaults the container image to ``lmcache/vllm-openai:latest``.
   Override with ``spec.image.repository`` and ``spec.image.tag`` to pin a
   specific version.

Connecting vLLM
---------------

The operator creates a ConfigMap named ``<engine-name>-connection`` containing
the ``kv-transfer-config`` JSON.  You can either let the operator's mutating
webhook inject it for you (recommended -- keeps your vLLM manifest clean) or
mount it by hand.  See :ref:`mp-operator-connection-injection` below for the
webhook flow; the rest of this section describes the manual mount that is its
equivalent.

Mount it in your vLLM Deployment:

.. code-block:: yaml

    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: vllm
    spec:
      replicas: 1
      selector:
        matchLabels:
          app: vllm
      template:
        metadata:
          labels:
            app: vllm
        spec:
          # Required for CUDA IPC between vLLM and LMCache
          hostIPC: true
          containers:
            - name: vllm
              image: lmcache/vllm-openai:latest
              env:
                # Deterministic hashing required by LMCache
                - name: PYTHONHASHSEED
                  value: "0"
              command: ["/bin/sh", "-c"]
              args:
                - |
                  exec python3 -m vllm.entrypoints.openai.api_server \
                    --model <your-model> \
                    --port 8000 \
                    --gpu-memory-utilization 0.8 \
                    --kv-transfer-config "$(cat /etc/lmcache/kv-transfer-config.json)"
              ports:
                - name: http
                  containerPort: 8000
              volumeMounts:
                - name: kv-transfer-config
                  mountPath: /etc/lmcache
                  readOnly: true
              resources:
                limits:
                  nvidia.com/gpu: "1"
          volumes:
            - name: kv-transfer-config
              configMap:
                name: my-cache-connection  # <engine-name>-connection

Key requirements for vLLM pods:

- **hostIPC: true** -- CUDA IPC (``cudaIpcOpenMemHandle``) needs a shared IPC
  namespace between vLLM and LMCache.
- **PYTHONHASHSEED=0** -- Ensures deterministic token hashing so vLLM and
  LMCache produce consistent cache keys.
- **ConfigMap mount** -- The ``$(cat ...)`` pattern reads the connection JSON
  inline.  The ConfigMap name is always ``<LMCacheEngine name>-connection``.
- **No hostNetwork needed** -- The operator's node-local Service handles
  routing via ``internalTrafficPolicy=Local``.

.. _mp-operator-connection-injection:

Connection Injection (Webhook)
-------------------------------

Hand-wiring the ConfigMap mount and the ``$(cat ...)`` argument substitution
above is repetitive across vLLM Deployments.  A **mutating admission webhook**
shipped with the operator can do it for you so the vLLM manifest stays clean.
It mirrors the CacheBlend webhook (see :ref:`mp-operator-cacheblend`) with an
``lmcache-`` annotation/label discriminator so the two injectors never
cross-fire on the same pod.

When invoked on an opted-in pod whose ``<engine>-connection`` ConfigMap exists,
the webhook mutates the pod at admission time to add:

- ``--kv-transfer-config <JSON>`` -- the ``LMCacheMPConnector`` config, read
  verbatim from the engine's ``<engine>-connection`` ConfigMap and inlined onto
  the vLLM container's ``args`` (no volume mount needed);
- ``hostIPC: true`` on the pod spec (CUDA IPC with the node-local server);
- ``PYTHONHASHSEED=0`` on the vLLM container env, **set-if-absent** -- it
  preserves a value you already set.

Unlike the CacheBlend injector it does **not** consult the engine CR: the
entire connector config lives in the connection ConfigMap, and
``LMCacheEngine`` has no injection sub-spec.  It fails open
(``failurePolicy: Ignore``) and is idempotent (re-admitted pods carrying the
``lmcache.ai/lmcache-injected`` stamp are allowed unchanged).

Prerequisites
~~~~~~~~~~~~~

- **cert-manager** + ``make deploy`` (not ``make run``, which is
  controller-only and disables the webhook via ``ENABLE_WEBHOOKS=false``) --
  same as the CacheBlend webhook; install once per cluster (see
  :ref:`mp-operator-cacheblend` "Additional Prerequisites").
- **Pod Security Standards** -- the injected ``hostIPC`` is rejected by the
  ``baseline`` / ``restricted`` PSS profiles, so the vLLM pod's namespace must
  be labeled ``pod-security.kubernetes.io/enforce=privileged``.
- **Engine reconciled in the same namespace** -- the webhook reads the
  ``<engine>-connection`` ConfigMap directly, so the ``LMCacheEngine`` must
  already exist in the vLLM pod's namespace.

Opting a vLLM Pod In
~~~~~~~~~~~~~~~~~~~~

Add the opt-in label and the engine-binding annotation to the pod template, and
launch vLLM via the image **ENTRYPOINT** (args only) -- a
``command: ["/bin/sh", "-c", ...]`` wrapper is skipped (the webhook stamps
``lmcache.ai/lmcache-skip-reason=command-override`` because appended args would
not reach ``vllm serve``):

.. code-block:: yaml

    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: vllm-lmcache
    spec:
      replicas: 1
      selector:
        matchLabels:
          app: vllm-lmcache
      template:
        metadata:
          labels:
            app: vllm-lmcache
            lmcache.ai/lmcache-inject: "true"        # opt-in (webhook objectSelector)
          annotations:
            lmcache.ai/lmcache-engine: "my-cache"    # bind to the engine (same namespace)
            # Optional -- name the vLLM container if it is not the first one:
            # lmcache.ai/lmcache-container: "vllm"
        spec:
          runtimeClassName: nvidia
          # Do NOT set hostIPC here or mount an emptyDir at /dev/shm -- the
          # webhook injects hostIPC=true; an emptyDir would shadow the host's
          # /dev/shm and break cudaIpcOpenMemHandle.
          containers:
            - name: vllm
              image: lmcache/vllm-openai:latest
              # Args-only launch (image ENTRYPOINT is ["vllm", "serve"]). The
              # webhook appends --kv-transfer-config; do NOT add it yourself
              # (a user-supplied one stamps skip-reason=kv-transfer-config-present).
              args: ["<your-model>", "--port", "8000", "--gpu-memory-utilization", "0.8"]
              ports:
                - name: http
                  containerPort: 8000
              resources:
                limits:
                  nvidia.com/gpu: "1"

A ready-to-edit manifest lives at
``operator/config/samples/vllm_lmcache_deployment.yaml``.

Verifying Injection
~~~~~~~~~~~~~~~~~~~

The webhook mutates **Pods**, not the Deployment, so inspect a pod (not the
Deployment spec):

.. code-block:: bash

    kubectl get pod -l app=vllm-lmcache -o yaml | \
      grep -E "hostIPC|kv-transfer-config|lmcache-injected|lmcache-skip-reason"

If nothing was injected, check the pod's ``lmcache.ai/lmcache-skip-reason``
annotation:

- ``command-override`` -- the pod uses a ``sh -c`` wrapper, so injected args
  would not reach ``vllm serve``.
- ``kv-transfer-config-present`` -- the user already supplied
  ``--kv-transfer-config``; the webhook does not clobber it.
- ``engine-not-found`` -- the ``<engine>-connection`` ConfigMap is missing
  (engine not yet reconciled, or wrong namespace, or wrong name).
- ``target-container-not-found`` -- the
  ``lmcache.ai/lmcache-container`` annotation names a container the pod does
  not have.

With ``failurePolicy: Ignore`` a webhook / cert problem also leaves the pod
un-mutated silently -- confirm the operator pod is ``Running`` and the
``MutatingWebhookConfiguration`` exists.

Using the Latest (or a Pinned) lmcache
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default a vLLM pod runs whatever ``lmcache`` is baked into its image.  To run
a **different** lmcache build instead -- e.g. ship the latest lmcache onto an
older, stable vLLM image, or keep the vLLM client on the exact build its
``LMCacheEngine`` server runs -- set ``spec.injection.payloadImage`` on the
engine.  The webhook then additionally stages that image's ``lmcache`` tree into
each opted-in pod: an ``emptyDir`` + an init container that copies the tree in, a
read-only mount, and ``PYTHONPATH=/lmcache-payload`` so vLLM imports the staged
``lmcache`` instead of the baked-in one.  No vLLM image rebuild.

**1. Build the payload image.** It ships the unpacked ``lmcache`` tree under
``/payload`` and copies it to ``$SHARED_DIR`` on start.  ``docker/Dockerfile.payload``
builds it by extracting an ABI-matched ``lmcache`` from an lmcache image (the
``SOURCE_IMAGE`` build-arg selects the version):

.. code-block:: bash

    docker build -f docker/Dockerfile.payload \
      --build-arg SOURCE_IMAGE=lmcache/vllm-openai:latest-nightly \
      -t <registry>/lmcache-payload:latest .
    docker push <registry>/lmcache-payload:latest

**2. Point the engine at it.** ``payloadImage.repository`` has no valid default
(the inherited image default is not a payload), so set it explicitly; leaving
``injection`` unset keeps connection-only wiring.

.. code-block:: yaml

    apiVersion: lmcache.lmcache.ai/v1alpha1
    kind: LMCacheEngine
    metadata:
      name: my-cache-versioned
    spec:
      l1:
        sizeGB: 60
      injection:
        payloadImage:
          repository: <registry>/lmcache-payload
          tag: latest
          pullPolicy: Always          # :latest moves -- re-pull for the current build
        # imagePullSecrets:            # private payload registry only
        #   - name: my-registry-secret

Opted-in pods bound to this engine (label + annotation as above) need no
changes -- the webhook stages the payload automatically.  Ready-to-apply
samples: ``config/samples/lmcache_v1alpha1_lmcacheengine_injection.yaml`` and
``config/samples/vllm_lmcache_injection_deployment.yaml``.

.. note::
   The payload's ``lmcache`` must be **ABI-compatible** (same Python minor
   version and a compatible torch) with the vLLM image that imports it -- it
   ships compiled extensions.  If they differ, ``import lmcache`` fails with an
   ``undefined symbol`` error in the vLLM pod.  Building the payload from an
   lmcache image close to your vLLM image keeps them compatible.

**3. Verify the swap** on a running pod -- contrast the normal import with one
that ignores the injected ``PYTHONPATH``:

.. code-block:: bash

    POD=$(kubectl get pod -l app=vllm-lmcache-versioned -o name | head -1)

    # imports the STAGED build (from /lmcache-payload):
    kubectl exec $POD -c vllm -- python3 -c \
      "import lmcache; print(lmcache.__version__, lmcache.__file__)"

    # PYTHONPATH stripped -> the image's baked-in build (site-packages):
    kubectl exec $POD -c vllm -- env -u PYTHONPATH python3 -c \
      "import lmcache; print(lmcache.__version__, lmcache.__file__)"

Two different sources for the same module confirms the swap.  If nothing was
staged, check ``lmcache.ai/lmcache-skip-reason`` on the pod.

Verifying the Deployment
------------------------

.. code-block:: bash

    # Check LMCacheEngine status
    kubectl get lmc

Expected output:

.. code-block:: text

    NAME       PHASE     READY   DESIRED   AGE
    my-cache   Running   3       3         5m

.. code-block:: bash

    # Check the connection ConfigMap
    kubectl get configmap my-cache-connection -o yaml

    # Check LMCache pods
    kubectl get pods -l app.kubernetes.io/managed-by=lmcache-operator

    # Check detailed status with endpoints
    kubectl describe lmc my-cache

CRD Spec Reference
-------------------

Image
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 20 45

   * - Field
     - Default
     - Description
   * - ``image.repository``
     - ``lmcache/vllm-openai``
     - Container image repository.
   * - ``image.tag``
     - ``latest``
     - Container image tag.
   * - ``image.pullPolicy``
     - ``IfNotPresent``
     - ``Always``, ``Never``, or ``IfNotPresent``.
   * - ``imagePullSecrets``
     - --
     - Image pull secret references.

Server
~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 20 45

   * - Field
     - Default
     - Description
   * - ``server.port``
     - ``5555``
     - ZMQ listening port (1024--65535).
   * - ``server.chunkSize``
     - ``256``
     - Token chunk size.
   * - ``server.maxWorkers``
     - ``1``
     - Worker threads for ZMQ requests.
   * - ``server.hashAlgorithm``
     - ``blake3``
     - ``builtin``, ``sha256_cbor``, or ``blake3``.
   * - ``server.httpPort``
     - ``8080``
     - HTTP frontend port for health checks and cache admin (1024--65535).

L1 Cache
~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 20 45

   * - Field
     - Default
     - Description
   * - ``l1.sizeGB``
     - *required*
     - L1 cache size in GB.  Must be > 0.

Eviction
~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 20 45

   * - Field
     - Default
     - Description
   * - ``eviction.policy``
     - ``LRU``
     - ``LRU`` or ``noop``.  Use ``noop`` with ``l2Backend.storePolicy: skip_l1``
       for buffer-only mode.
   * - ``eviction.triggerWatermark``
     - ``0.8``
     - Usage ratio (0.0--1.0] to trigger eviction.
   * - ``eviction.evictionRatio``
     - ``0.2``
     - Fraction to evict (0.0--1.0].

Prometheus
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 20 45

   * - Field
     - Default
     - Description
   * - ``prometheus.enabled``
     - ``true``
     - Expose Prometheus metrics.
   * - ``prometheus.port``
     - ``9090``
     - ``/metrics`` endpoint port.
   * - ``prometheus.serviceMonitor.enabled``
     - ``false``
     - Create a ServiceMonitor CR.
   * - ``prometheus.serviceMonitor.interval``
     - ``30s``
     - Scrape interval.
   * - ``prometheus.serviceMonitor.labels``
     - --
     - Extra labels on the ServiceMonitor.

L2 Storage
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 20 45

   * - Field
     - Default
     - Description
   * - ``l2Backend``
     - --
     - List of L2 backends (``type`` + ``config``).
       See :doc:`l2_storage/index`.
   * - ``l2Backend.serde``
     - --
     - Optional serde transform on KV bytes to/from the L2 adapter
       (rendered as the ``serde`` sub-dict of ``--l2-adapter``). Exactly
       one serde type must be set. See :doc:`serde`.
   * - ``l2Backend.serde.aesgcm``
     - --
     - At-rest encryption of L2 KV bytes (AES-GCM, keyed per
       ``cache_salt``). L1 (host RAM) and L0 (GPU) remain plaintext.
   * - ``l2Backend.serde.aesgcm.masterKeySecretRef.name``
     - --
     - Required. User-created Secret in the engine's namespace holding
       the master key under the ``master`` data key; mounted read-only
       into the engine pods at ``/etc/lmcache/keys/master``. The
       operator never generates key material.
   * - ``l2Backend.serde.aesgcm.keyProvider``
     - ``hkdf``
     - How per-``cache_salt`` keys are derived from the master key.
       Only ``hkdf`` (HKDF-SHA256) is implemented.
   * - ``l2Backend.serde.aesgcm.aesBits``
     - ``128``
     - AES key size: ``128`` or ``256``.

GPU & Security
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 20 45

   * - Field
     - Default
     - Description
   * - ``gpuVendor``
     - ``nvidia``
     - GPU vendor: ``nvidia`` (uses the ``nvidia`` RuntimeClass) or ``amd``
       (runs on the default runtime).
   * - ``privileged``
     - ``false``
     - Run the engine container in privileged mode. On most clusters
       ``runtimeClassName: nvidia`` + ``NVIDIA_VISIBLE_DEVICES=all`` already
       grant GPU visibility without it; set ``true`` only where the engine
       cannot otherwise see the GPUs. Required for ``gpuVendor: amd`` (no
       RuntimeClass device injection, so privileged is the only path to
       ``/dev/kfd``/``/dev/dri``). Enabling it requires the namespace to allow
       the ``privileged`` Pod Security Standard.

Scheduling
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 20 45

   * - Field
     - Default
     - Description
   * - ``nodeSelector``
     - GPU nodes
     - Defaults to ``nvidia.com/gpu.present: "true"``.
   * - ``affinity``
     - --
     - Pod affinity rules.
   * - ``tolerations``
     - --
     - Pod tolerations.
   * - ``priorityClassName``
     - --
     - Priority class for pods.

Overrides & Extras
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 20 45

   * - Field
     - Default
     - Description
   * - ``logLevel``
     - ``INFO``
     - ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``.
   * - ``resourceOverrides``
     - --
     - Override auto-computed resources.
   * - ``env``
     - --
     - Extra environment variables.
   * - ``volumes``
     - --
     - Extra volumes.
   * - ``volumeMounts``
     - --
     - Extra volume mounts.
   * - ``podAnnotations``
     - --
     - Extra pod annotations.
   * - ``podLabels``
     - --
     - Extra pod labels.
   * - ``serviceAccountName``
     - --
     - ServiceAccount for pods.
   * - ``extraArgs``
     - --
     - Extra CLI flags (appended last, can override).

Auto-Computed Resources
~~~~~~~~~~~~~~~~~~~~~~~

When ``spec.resourceOverrides`` is not set, the operator derives resources from
``l1.sizeGB``:

- **CPU request**: ``4`` cores
- **Memory request**: ``ceil(l1.sizeGB + 5)`` Gi
- **Memory limit**: ``ceil(memoryRequest * 1.5)`` Gi

For example, ``l1.sizeGB: 60`` produces a 65 Gi request and 98 Gi limit.

Auto-Injected Pod Settings
~~~~~~~~~~~~~~~~~~~~~~~~~~

The operator always injects these into the pod spec (they are not configurable
via the CRD):

- **hostIPC: true** -- Required for CUDA IPC between LMCache and vLLM.
- **--host 0.0.0.0** -- Binds the server to all interfaces so the node-local
  Service can route to it.
- **NVIDIA_VISIBLE_DEVICES=all** -- Ensures GPU access for IPC-based memory
  transfers.
- **NVIDIA_DRIVER_CAPABILITIES=all** -- Exposes all driver capabilities
  (compute, utility, etc.) to the container.
- **TCP socket probes** -- Startup (5s initial, 30 failures), liveness (10s),
  and readiness (5s) probes on the server port.

.. note::
   The operator does **not** mount an emptyDir at ``/dev/shm``.  With
   ``hostIPC: true``, the container sees the host's ``/dev/shm`` directly.
   Mounting an emptyDir would shadow it with a private tmpfs and break CUDA IPC.

Resources Created
~~~~~~~~~~~~~~~~~

For an ``LMCacheEngine`` named ``my-cache``:

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Resource
     - Name
     - Purpose
   * - DaemonSet
     - ``my-cache``
     - Runs LMCache server pods.
   * - Service (ClusterIP)
     - ``my-cache``
     - Node-local discovery (``internalTrafficPolicy=Local``).
   * - Service (headless)
     - ``my-cache-metrics``
     - Prometheus scrape target.
   * - ConfigMap
     - ``my-cache-connection``
     - ``kv-transfer-config`` JSON for vLLM.
   * - ServiceMonitor
     - ``my-cache``
     - Prometheus Operator integration (when enabled).

The connection ConfigMap contains:

.. code-block:: json

    {
      "kv_connector": "LMCacheMPConnector",
      "kv_role": "kv_both",
      "kv_connector_extra_config": {
        "lmcache.mp.host": "tcp://my-cache.default.svc.cluster.local",
        "lmcache.mp.port": "5555"
      }
    }

Status & Conditions
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    kubectl describe lmc my-cache

The status section includes:

- **phase**: ``Pending``, ``Running``, ``Degraded``, or ``Failed``.
- **readyInstances** / **desiredInstances**: Instance counts.
- **endpoints**: Per-node connection info (node name, host IP, pod name, port,
  readiness).
- **conditions**:

  - ``Available`` -- At least one instance is ready.
  - ``AllInstancesReady`` -- All desired instances are ready.
  - ``ConfigValid`` -- Spec validation passed.

Validation Rules
~~~~~~~~~~~~~~~~

The operator validates the CR spec at apply time:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Rule
   * - ``l1.sizeGB``
     - Required, must be > 0.
   * - ``eviction.policy``
     - Must be ``LRU`` or ``noop`` (if set).
   * - ``eviction.triggerWatermark``
     - Must be in (0.0, 1.0].
   * - ``eviction.evictionRatio``
     - Must be in (0.0, 1.0].
   * - ``server.port``
     - Must be in [1024, 65535].
   * - ``l2Backend.serde``
     - Exactly one serde type must be set (only ``aesgcm`` today).
   * - ``l2Backend.serde.aesgcm.masterKeySecretRef.name``
     - Required, must be non-empty.
   * - ``l2Backend.serde`` + ``l2Backend.raw``
     - Rejected if the raw adapter config already sets a ``serde`` key
       (one would silently overwrite the other).

Examples
--------

Target Only GPU Nodes
~~~~~~~~~~~~~~~~~~~~~

Use ``nodeSelector`` to run LMCache only on GPU nodes.  New GPU nodes
automatically get an LMCache pod:

.. code-block:: yaml

    apiVersion: lmcache.lmcache.ai/v1alpha1
    kind: LMCacheEngine
    metadata:
      name: my-cache
    spec:
      nodeSelector:
        nvidia.com/gpu.present: "true"
      l1:
        sizeGB: 60

.. note::
   The operator defaults ``nodeSelector`` to ``nvidia.com/gpu.present: "true"``
   when not specified, so a minimal CR already targets GPU nodes.

Custom Server Port
~~~~~~~~~~~~~~~~~~

If the default port (5555) conflicts with other services:

.. code-block:: yaml

    apiVersion: lmcache.lmcache.ai/v1alpha1
    kind: LMCacheEngine
    metadata:
      name: my-cache
    spec:
      server:
        port: 6555
      l1:
        sizeGB: 60

The connection ConfigMap updates automatically -- vLLM pods pick up the new
port on restart.

Encrypted L2 Backend
~~~~~~~~~~~~~~~~~~~~

Encrypt KV bytes at rest in the L2 tier with the ``aesgcm`` serde. Create
the master-key Secret first, in the engine's namespace (the operator never
generates keys):

.. code-block:: bash

    head -c 16 /dev/urandom > master.key   # 16 bytes for AES-128, 32 for AES-256
    kubectl create secret generic lmcache-l2-master-key \
        --from-file=master=master.key

.. code-block:: yaml

    apiVersion: lmcache.lmcache.ai/v1alpha1
    kind: LMCacheEngine
    metadata:
      name: my-cache
    spec:
      l1:
        sizeGB: 60
      l2Backend:
        raw:
          type: fs
          config:
            base_path: /data/lmcache/l2
        serde:
          aesgcm:
            masterKeySecretRef:
              name: lmcache-l2-master-key

The Secret is mounted directly into the engine pods (read-only, only the
``master`` data key). A missing Secret or missing key surfaces as a pod
mount event and self-heals once the Secret is fixed. See :doc:`serde` for
the key model and threat model.

Production with Prometheus Monitoring
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    apiVersion: lmcache.lmcache.ai/v1alpha1
    kind: LMCacheEngine
    metadata:
      name: production-cache
      namespace: llm-serving
    spec:
      nodeSelector:
        nvidia.com/gpu.present: "true"
      image:
        repository: lmcache/standalone
        tag: v0.1.0
      server:
        port: 6555
        chunkSize: 256
        maxWorkers: 4
      l1:
        sizeGB: 60
      eviction:
        triggerWatermark: 0.8
        evictionRatio: 0.2
      prometheus:
        enabled: true
        port: 9090
        serviceMonitor:
          enabled: true
          labels:
            release: kube-prometheus-stack
      podAnnotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9090"
      priorityClassName: system-node-critical

See :doc:`observability/index` for metric names and Grafana configuration.

Override Auto-Computed Resources
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    apiVersion: lmcache.lmcache.ai/v1alpha1
    kind: LMCacheEngine
    metadata:
      name: my-cache
    spec:
      l1:
        sizeGB: 60
      resourceOverrides:
        requests:
          memory: "70Gi"
          cpu: "8"
        limits:
          memory: "100Gi"

.. _mp-operator-cacheblend:

CacheBlend
----------

CacheBlend reuses cached KV at shifted (non-prefix) positions by recomputing a
small subset of tokens.  The operator manages it as a second CRD,
``CacheBlendEngine``, plus a **mutating admission webhook** that injects the
pure-Python ``lmcache-cacheblend`` vLLM plugin into your serving pods -- so you
do **not** rebuild the vLLM image.  See :doc:`/kv_cache_optimizations/blending`
for the technique itself.

It has two halves the operator runs together:

- a GPU-resident CacheBlend V3 engine (``lmcache server --engine-type blend``),
  deployed as a DaemonSet with the **same GPU model as** ``LMCacheEngine``
  (``runtimeClassName: nvidia`` + ``NVIDIA_VISIBLE_DEVICES=all`` + ``hostIPC``,
  plus ``privileged`` when ``spec.privileged`` is set, and **no**
  ``nvidia.com/gpu`` claim) so it shares the vLLM GPU for same-device CUDA IPC;
  and
- the vLLM-side plugin, injected into opted-in pods by the webhook.

Additional Prerequisites
~~~~~~~~~~~~~~~~~~~~~~~~~~

Beyond the operator prerequisites above:

- **cert-manager** -- the webhook's serving certificate is issued by a
  cert-manager ``Issuer`` + ``Certificate``.  Install it before ``make deploy``:

  .. code-block:: bash

      kubectl apply -f https://github.com/cert-manager/cert-manager/releases/latest/download/cert-manager.yaml
      kubectl -n cert-manager wait --for=condition=Available deploy --all --timeout=180s

- **Deploy with the webhook** -- use ``make deploy`` (not ``make run``, which is
  controller-only and disables the webhook via ``ENABLE_WEBHOOKS=false``).
- **Pod Security Standards** -- the webhook injects ``hostIPC``/``privileged``,
  which the ``baseline``/``restricted`` profiles reject, so label the engine's
  and the vLLM pod's namespaces ``pod-security.kubernetes.io/enforce=privileged``.

Deploying a CacheBlendEngine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    apiVersion: lmcache.lmcache.ai/v1alpha1
    kind: CacheBlendEngine
    metadata:
      name: my-cacheblend
    spec:
      l1:
        sizeGB: 60
      injection:
        # The (private) cacheblend-plugin init-container image -- repository/tag/
        # pullPolicy, like spec.image.  Set repository to YOUR image; the
        # inherited engine-image default is not a valid payload.
        payloadImage:
          repository: <registry>/cacheblend-plugin
          tag: <tag>
        # Appended to the vLLM pod so the private payload image can pull; the
        # Secret must exist in the vLLM pod's namespace.
        imagePullSecrets:
          - name: my-registry-secret

The engine runs ``lmcache server --engine-type blend`` as a DaemonSet and
emits a ``my-cacheblend-connection`` ConfigMap with the ``CBKVConnector``
``kv-transfer-config`` (the operator wires the node-local Service host/port and
the ``cb.*`` tunables).

Opting a vLLM Pod In
~~~~~~~~~~~~~~~~~~~~~

Label the pod template for the webhook and bind it to an engine by name.  Launch
vLLM via the image **ENTRYPOINT** (args only) -- a
``command: ["/bin/sh", "-c", ...]`` wrapper is skipped, since appended args would
not reach ``vllm serve``:

.. code-block:: yaml

    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: vllm-cacheblend
    spec:
      replicas: 1
      selector:
        matchLabels:
          app: vllm-cacheblend
      template:
        metadata:
          labels:
            app: vllm-cacheblend
            lmcache.ai/cacheblend-inject: "true"          # opt-in (webhook objectSelector)
          annotations:
            lmcache.ai/cacheblend-engine: "my-cacheblend" # bind to the engine
        spec:
          runtimeClassName: nvidia
          containers:
            - name: vllm
              image: lmcache/vllm-openai:<pinned-tag>
              args: ["<your-model>", "--port", "8000", "--gpu-memory-utilization", "0.8"]
              resources:
                limits:
                  nvidia.com/gpu: "1"

The webhook injects the plugin init container, ``PYTHONPATH``, ``hostIPC``, the
private-image pull secret, and the required CacheBlend vLLM flags
(``--attention-backend CUSTOM``, ``--kv-transfer-config`` from the engine's
connection ConfigMap, ``--block-size 64``, ``--pipeline-parallel-size 1``,
``--no-enable-chunked-prefill``, ``--no-async-scheduling``, ``--enforce-eager``).
You supply only the model and your non-CacheBlend flags.

Verifying Injection
~~~~~~~~~~~~~~~~~~~~~

The webhook mutates **Pods**, not the Deployment, so inspect a pod:

.. code-block:: bash

    kubectl get pod -l app=vllm-cacheblend -o yaml | \
      grep -E "initContainers|cb-plugin|PYTHONPATH|attention-backend|cacheblend-injected|skip-reason"

If nothing was injected, check the pod's ``lmcache.ai/cacheblend-skip-reason``
annotation: ``command-override`` (a ``sh -c`` wrapper was used),
``kv-transfer-config-present`` (you set your own), ``engine-not-found`` (the
``<name>-connection`` ConfigMap is missing), ``payload-image-unset`` (the
engine's ``injection.payloadImage`` has no repository), or
``target-container-not-found`` (the requested ``targetContainer`` /
``cacheblend-container`` annotation names a container the pod does not have).
With ``failurePolicy: Ignore`` a
webhook/cert problem also leaves the pod un-mutated silently -- confirm the
operator pod is ``Running`` and the ``MutatingWebhookConfiguration`` exists.

CacheBlendEngine Fields
~~~~~~~~~~~~~~~~~~~~~~~~~

``CacheBlendEngineSpec`` mirrors ``LMCacheEngineSpec`` (every field in the CRD
Spec Reference above) and adds:

.. list-table::
   :header-rows: 1
   :widths: 35 20 45

   * - Field
     - Default
     - Description
   * - ``blend.checkLayer``
     - ``1``
     - Layer at which token importance is scored (``cb.check_layer``).
   * - ``blend.recompRatio``
     - ``0.15``
     - Fraction of non-prefix-hit tokens recomputed (``cb.recomp_ratio``).
   * - ``injection.payloadImage``
     - *required*
     - The (private) cacheblend-plugin init-container image
       (``repository`` / ``tag`` / ``pullPolicy``).  Set ``repository`` -- the
       inherited engine-image default is not a valid payload.
   * - ``injection.imagePullSecrets``
     - --
     - Pull secrets appended to the vLLM pod for the private payload image.
   * - ``injection.targetContainer``
     - first container
     - Name of the vLLM container to inject into.
   * - ``injection.cudagraph``
     - ``eager``
     - ``eager`` | ``piecewise`` | ``full_decode_only`` (never ``full``).

``server.chunkSize`` defaults to ``256`` and must equal 256 (the blend matcher
requires ``chunk_size == vLLM --block-size * 4``).

LMCacheCoordinator
------------------

The ``LMCacheCoordinator`` CRD runs the **mp coordinator** -- a fleet-wide HTTP
service that tracks mp server instances, evicts those whose heartbeats lapse,
performs L2 quota eviction, and hosts the global CacheBlend fingerprint
directory.  It is a plain (non-GPU) ``Deployment`` exposed through a ClusterIP
Service; engines reach it via ``coordinator.ref`` or ``coordinator.url``.

Deploying a Coordinator
~~~~~~~~~~~~~~~~~~~~~~~~~

A ready-to-edit manifest lives at
``config/samples/lmcache_v1alpha1_lmcachecoordinator.yaml`` in the operator
repo.  A minimal coordinator:

.. code-block:: yaml

    apiVersion: lmcache.lmcache.ai/v1alpha1
    kind: LMCacheCoordinator
    metadata:
      name: my-coordinator
    spec:
      port: 9300

.. code-block:: bash

    kubectl get lmcc my-coordinator   # shortName: lmcc

Connecting an Engine
~~~~~~~~~~~~~~~~~~~~~

Point an ``LMCacheEngine`` / ``CacheBlendEngine`` at the coordinator through its
``coordinator`` block.  Use ``ref`` to name a coordinator in the same namespace
(the operator resolves it to the in-cluster Service URL), or ``url`` for an
explicit endpoint:

.. code-block:: yaml

    spec:
      coordinator:
        ref:
          name: my-coordinator       # or: url: http://my-coordinator.default.svc:9300
        heartbeatInterval: 5          # seconds; must be > 0
        l2EventReporting: false       # report L2 store/lookup events for fleet eviction

Coordinator CRD Spec Reference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Topology
^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 35 20 45

   * - Field
     - Default
     - Description
   * - ``replicas``
     - ``1``
     - Coordinator pods.  The registry is per-process in-memory, so >1 only
       makes sense behind a shared durable backend.  Must be >= 0.
   * - ``image.repository`` / ``image.tag`` / ``image.pullPolicy``
     - shared engine image
     - Runs the same lmcache binary as the engines.
   * - ``imagePullSecrets``
     - --
     - Image pull secret references.

HTTP Server
^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 35 20 45

   * - Field
     - Default
     - Description
   * - ``host``
     - ``0.0.0.0``
     - Address the coordinator's HTTP server binds to.
   * - ``port``
     - ``9300``
     - HTTP port (1--65535).

Membership & Health
^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 35 20 45

   * - Field
     - Default
     - Description
   * - ``instanceTimeout``
     - ``30``
     - Seconds without a heartbeat after which an instance is evicted.  Set
       comfortably above the engines' ``coordinator.heartbeatInterval``.
   * - ``healthCheckInterval``
     - ``10``
     - Seconds between health-check sweeps; ``0`` disables the loop.

L2 Quota Eviction
^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 35 20 45

   * - Field
     - Default
     - Description
   * - ``evictionCheckInterval``
     - ``5``
     - Seconds between L2 eviction sweeps; ``0`` disables the loop.
   * - ``evictionRatio``
     - ``0.2``
     - Fraction of tracked keys (by count) to evict per cycle, [0.0, 1.0].
   * - ``triggerWatermark``
     - ``1.0``
     - Usage fraction of the quota that fires eviction, (0.0, 1.0].

Global CacheBlend Directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 35 20 45

   * - Field
     - Default
     - Description
   * - ``blendChunkSize``
     - ``256``
     - Tokens per chunk for the global CacheBlend directory (the match unit).
       **Must equal** the LMCache chunk size the blend servers use.  Must be > 0.
   * - ``blendProbeStride``
     - ``1``
     - Positions between match probes.  ``1`` probes every offset for full
       recall; raise it to trade recall for coordinator CPU.  Must be > 0.

Prometheus, Scheduling & Overrides
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 35 20 45

   * - Field
     - Default
     - Description
   * - ``prometheus.enabled``
     - ``true``
     - Expose the metrics container port.  See the note below.
   * - ``prometheus.port``
     - ``9090``
     - Metrics port.
   * - ``prometheus.serviceMonitor.enabled``
     - ``false``
     - Create a ServiceMonitor CR (and headless metrics Service).
   * - ``prometheus.serviceMonitor.interval``
     - ``30s``
     - Scrape interval.
   * - ``logLevel``
     - ``INFO``
     - ``DEBUG`` | ``INFO`` | ``WARNING`` | ``ERROR``.
   * - ``resourceOverrides``
     - --
     - Pod resource requests/limits (no auto-compute; the coordinator is
       CPU/memory light).
   * - ``nodeSelector`` / ``affinity`` / ``tolerations`` / ``priorityClassName``
     - --
     - Pod scheduling controls.
   * - ``env`` / ``volumes`` / ``volumeMounts`` / ``podAnnotations`` / ``podLabels`` / ``serviceAccountName``
     - --
     - Standard pod-shaping fields.
   * - ``extraArgs``
     - --
     - Extra CLI flags (appended last, can override any auto-generated flag).

.. note::
   The coordinator process does **not** yet expose a ``/metrics`` endpoint.  The
   Prometheus wiring is present for parity but is only useful once metrics are
   added; ``serviceMonitor.enabled`` defaults to ``false``.

Coordinator Resources Created
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For an ``LMCacheCoordinator`` named ``my-coordinator``:

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Resource
     - Name
     - Purpose
   * - Deployment
     - ``my-coordinator``
     - Runs the coordinator HTTP server pods.
   * - Service (ClusterIP)
     - ``my-coordinator``
     - Fleet-wide discovery on the HTTP port.
   * - Service (headless)
     - ``my-coordinator-metrics``
     - Prometheus scrape target (when ``serviceMonitor.enabled``).
   * - ServiceMonitor
     - ``my-coordinator``
     - Prometheus Operator integration (when ``serviceMonitor.enabled``).

The status ``endpoint`` other components use to reach the coordinator is
``http://<name>.<namespace>.svc:<port>`` (e.g.
``http://my-coordinator.default.svc:9300``).

Coordinator Status & Conditions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The status section includes:

- **phase**: ``Pending``, ``Running``, ``Degraded``, or ``Failed``.
- **replicas** / **readyReplicas**: Pod counts from the Deployment.
- **endpoint**: In-cluster URL for reaching the coordinator.
- **observedGeneration**: Most recent reconciled generation.
- **conditions**:

  - ``Available`` -- At least one replica is ready.
  - ``AllInstancesReady`` -- All desired replicas are ready.
  - ``ConfigValid`` -- Spec validation passed.

Coordinator Validation Rules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Rule
   * - ``port``
     - Must be in [1, 65535].
   * - ``replicas``
     - Must be >= 0.
   * - ``instanceTimeout``
     - Must be > 0.
   * - ``healthCheckInterval`` / ``evictionCheckInterval``
     - Must be >= 0.
   * - ``evictionRatio``
     - Must be in [0.0, 1.0].
   * - ``triggerWatermark``
     - Must be in (0.0, 1.0].
   * - ``blendChunkSize`` / ``blendProbeStride``
     - Must be > 0.

Operator vs Manual Deployment
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Concern
     - Manual DaemonSet
     - LMCacheEngine Operator
   * - hostIPC
     - Must set manually
     - Auto-injected
   * - ``--host 0.0.0.0``
     - Must set manually
     - Auto-injected
   * - Service discovery
     - ``hostNetwork`` + ``status.hostIP``
     - Node-local ClusterIP Service + ConfigMap
   * - vLLM config
     - Copy JSON into Deployment
     - Mount ``<name>-connection`` ConfigMap
   * - Resource sizing
     - Manual calculation
     - Auto-computed from ``l1.sizeGB``
   * - Prometheus
     - Manual ServiceMonitor
     - ``serviceMonitor.enabled: true``
   * - Validation
     - Runtime errors only
     - ``kubectl apply`` rejects invalid specs
   * - New GPU nodes
     - DaemonSet handles it
     - DaemonSet handles it (same)

Security Considerations
-----------------------

**hostIPC** exposes the host's IPC namespace (System V IPC, POSIX message
queues) to the container.  Any process in the container can interact with IPC
resources from other processes on the same host.

- Deploy only in trusted environments.
- Clusters using Pod Security Standards must allow the ``privileged`` profile
  for the LMCache namespace -- the ``baseline`` and ``restricted`` profiles
  reject ``hostIPC``.
- ``spec.privileged`` defaults to ``false``. When enabled (required for
  ``gpuVendor: amd``), the engine container additionally runs privileged,
  granting it full device access -- enable it only where GPU visibility
  requires it.

Development
-----------

.. code-block:: bash

    make generate     # Generate DeepCopy methods
    make manifests    # Generate CRD YAML + RBAC
    make build        # Compile operator binary
    make fmt          # go fmt
    make vet          # go vet
    make test         # Run unit tests
    make lint         # Run golangci-lint

Pushing a custom operator image:

.. code-block:: bash

    # Docker Hub
    make docker-build docker-push IMG=docker.io/<your-user>/lmcache-operator:latest
    make deploy IMG=docker.io/<your-user>/lmcache-operator:latest

    # Multi-platform (amd64 + arm64)
    make docker-buildx IMG=<your-registry>/lmcache-operator:latest

If your cluster needs pull credentials:

.. code-block:: bash

    kubectl create secret docker-registry regcred \
      --docker-server=<your-registry> \
      --docker-username=<username> \
      --docker-password=<password> \
      -n lmcache-operator-system
