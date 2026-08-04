Hugging Face Buckets Backend
============================

The Hugging Face Buckets backend stores LMCache chunks in a Hugging Face Bucket
using LMCache's built-in remote storage plugin framework. This is a persistent
remote backend that fits warm and cold KV cache persistence better than the
hottest local tiers.

When to use it
--------------

Use the HFBucket backend when you want:

* A Hub-native persistent store for KV cache data.
* A remote backend that can be configured through ``remote_storage_plugins``.
* Multiple named bucket instances in one LMCache deployment.

Avoid using it as the primary hot path for the lowest-latency cache lookups.
Local CPU, local disk, and other lower-latency backends are a better fit for
the hottest cache tier.


Requirements and limitations
----------------------------

* LMCache uses ``huggingface_hub`` bucket APIs for uploads, downloads, listing,
  and deletes.
* The first built-in release is intentionally conservative:

  * Only full chunks are supported.
  * Partial chunk uploads are rejected.
  * Downloads are rejected when the stored object size does not match the
    expected full LMCache chunk size.
  * Chunk metadata is not stored in the bucket objects.


Minimal configuration
---------------------

.. code-block:: yaml

   chunk_size: 256
   local_cpu: false
   save_unfull_chunk: false
   remote_serde: "naive"
   blocking_timeout_secs: 10
   remote_storage_plugins: ["hfbucket"]
   extra_config:
     remote_storage_plugin.hfbucket.bucket_handle: "hf://buckets/my-org/lmcache-kv/prod"
     remote_storage_plugin.hfbucket.token_env: "HF_TOKEN"
     remote_storage_plugin.hfbucket.create_bucket_if_missing: false
     remote_storage_plugin.hfbucket.download_tmp_dir: "/tmp/lmcache-hfbucket"
     remote_storage_plugin.hfbucket.metadata_cache_ttl_secs: 30


Multiple instances
------------------

Use instance-qualified plugin names to configure more than one bucket-backed
remote store in the same LMCache config.

.. code-block:: yaml

   remote_storage_plugins: ["hfbucket.us", "hfbucket.eu"]
   extra_config:
     remote_storage_plugin.hfbucket.us.bucket_handle: "hf://buckets/my-org/lmcache-kv/us"
     remote_storage_plugin.hfbucket.us.token_env: "HF_US_TOKEN"
     remote_storage_plugin.hfbucket.eu.bucket_handle: "hf://buckets/my-org/lmcache-kv/eu"
     remote_storage_plugin.hfbucket.eu.token_env: "HF_EU_TOKEN"


Configuration reference
-----------------------

All configuration keys live under
``extra_config.remote_storage_plugin.<plugin_name>.*`` where ``plugin_name`` is
either ``hfbucket`` or an instance-qualified name such as ``hfbucket.prod``.

* ``bucket_handle`` (required): Hugging Face Bucket handle in
  ``hf://buckets/<namespace>/<bucket>[/<prefix>]`` format.
* ``token_env`` (optional, default ``HF_TOKEN``): Environment variable used to
  resolve the Hugging Face access token.
* ``token`` (optional): Direct token override. ``token_env`` takes precedence
  when both are set.
* ``create_bucket_if_missing`` (optional, default ``false``): Lazily create the
  bucket on the first write path.
* ``download_tmp_dir`` (optional): Root directory for connector-local download
  scratch space. On Linux, pointing this at a tmpfs mount such as
  ``/dev/shm/lmcache-hfbucket`` avoids the disk write on the download path.
* ``metadata_cache_ttl_secs`` (optional, default ``30``): TTL for cached exact
  existence and size metadata.


MP Mode Configuration
---------------------

In multi-process (MP) mode, Hugging Face Buckets are configured as an L2
adapter through a JSON spec passed to the LMCache server. This is separate from
the non-MP ``remote_storage_plugins`` configuration above. Each
``--l2-adapter`` argument takes a JSON object whose ``"type": "hfbucket"``
field selects the HFBucket adapter.

.. code-block:: json

   {
     "type": "hfbucket",
     "bucket_handle": "hf://buckets/my-org/lmcache-kv/prod",
     "token_env": "HF_TOKEN",
     "create_bucket_if_missing": false,
     "download_tmp_dir": "/tmp/lmcache-hfbucket-mp",
     "metadata_cache_ttl_secs": 30,
     "num_workers": 4,
     "max_capacity_gb": 500,
     "eviction": {
       "eviction_policy": "LRU",
       "trigger_watermark": 0.85,
       "eviction_ratio": 0.2
     }
   }

HFBucket L2 Adapter Fields
~~~~~~~~~~~~~~~~~~~~~~~~~~

* **type** (required): must be ``"hfbucket"``.
* **bucket_handle** (required): Hugging Face Bucket handle in
  ``hf://buckets/<namespace>/<bucket>[/<prefix>]`` format.
* **token_env**: environment variable used to resolve the Hugging Face access
  token (default ``"HF_TOKEN"``).
* **token**: optional direct token fallback. ``token_env`` takes precedence
  when the environment variable is set. Prefer ``token_env`` for production
  deployments so secrets do not live in adapter JSON.
* **create_bucket_if_missing**: lazily create the bucket on the first store
  operation (default ``false``). This only helps when the bucket is missing and
  the token has permission to create it; it does not fix invalid credentials,
  invalid handles, or network failures.
* **download_tmp_dir**: root directory for temporary load downloads (default
  ``/tmp/lmcache-hfbucket-mp``). The MP adapter downloads bucket files into
  per-task temporary files and then copies their bytes into the destination
  ``MemoryObj`` buffers supplied by the MP controller.
* **metadata_cache_ttl_secs**: TTL for cached exact path-size metadata (default
  ``30``). Set this lower when another process may modify the same bucket
  prefix outside LMCache and fresher metadata is more important than reducing
  Hugging Face metadata calls.
* **num_workers**: number of worker threads used for blocking Hugging Face Hub
  bucket API calls (default ``4``). The HFBucket Python APIs are synchronous,
  so MP mode runs upload, lookup, load, and delete work on a bounded thread
  pool behind the adapter's eventfd-based completion interface.
* **max_capacity_gb**: capacity used by ``get_usage()`` for watermark-based L2
  eviction. Set to ``0`` (default) to disable aggregate capacity tracking;
  ``get_usage()`` then reports the adapter as not providing an eviction signal.
* **eviction**: optional sub-dict enabling the L2 eviction controller for this
  adapter. When present, keys that are currently being loaded are protected by
  the lookup-and-lock path and skipped by ``delete()`` until they are unlocked.

Differences vs Non-MP HFBucket
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Hugging Face bucket operations are synchronous but the adapter makes submission
  non-blocking by running the blocking calls on worker threads.
* MP loads do not allocate and return new memory. The MP controller provides
  destination ``MemoryObj`` buffers, and the adapter copies downloaded bytes
  into those buffers.
* Keys are identified by ``ObjectKey`` (``model_name`` + ``kv_rank`` +
  ``chunk_hash`` + optional ``cache_salt``) rather than ``CacheEngineKey``.
  The serialized MP object name is
  ``<model>@<kv_rank_hex>@<chunk_hash_hex>[@<cache_salt>]`` and is then
  encoded for the bucket path. This naming is not compatible with the non-MP
  HFBucket connector's ``CacheEngineKey`` object names, so a bucket prefix
  populated by non-MP LMCache cannot be read directly by MP LMCache and vice
  versa.
* Full object writes are batch based. Hugging Face batch writes are not
  transactional, so a failed store task may still leave some objects in the
  bucket. The MP adapter reconciles backend metadata after such failures so
  any objects that actually landed are counted for usage and later deletion 
  (submitted store task is still reported as failed).


Notes
-----

* The backend stores objects under the configured bucket prefix using a
  reversible encoding of LMCache keys, so ``list()`` returns LMCache key strings
  instead of raw bucket object paths.
