HF Bucket
=========

An L2 adapter that stores KV cache objects in a `Hugging Face Bucket
<https://huggingface.co/docs/hub/storage-backends>`_ using the
``huggingface_hub`` bucket APIs.  Blocking Hub calls run on a bounded thread
pool driven by an asyncio loop on a daemon thread, so the L2 controller thread
is never blocked on network I/O.

Object names are derived from the MP ``ObjectKey`` as
``<model>@<kv_rank_hex>@<chunk_hash_hex>[@<cache_salt>]`` and then encoded with
the standard HFBucket object-name encoding plus the optional bucket prefix.
Because Hugging Face batch writes are not transactional, a store task that
partially fails reconciles backend metadata so that any objects that actually
landed are still counted for usage accounting and later deletion.

This is a persistent remote backend best suited to warm and cold KV cache
tiers; prefer a lower-latency local adapter for the hottest cache tier.

**Required fields:**

- ``bucket_handle``: Bucket location in the form
  ``hf://buckets/<namespace>/<bucket>[/<prefix>]``.

**Optional fields:**

- ``token_env`` (string, default ``"HF_TOKEN"``): Environment variable used to
  resolve the Hugging Face access token.
- ``token`` (string): Direct token fallback used when ``token_env`` is unset.
- ``create_bucket_if_missing`` (bool, default ``false``): Create the bucket
  lazily on the first store instead of requiring it to exist.
- ``download_tmp_dir`` (string): Root directory for temporary load downloads.
- ``metadata_cache_ttl_secs`` (float, default ``30.0``): TTL for the
  path-size metadata cache that backs lookups and usage accounting.
- ``num_workers`` (int, default ``4``): Number of worker threads for blocking
  Hugging Face Hub API calls.
- ``max_capacity_gb`` (float, default ``0.0``): Aggregate capacity used by
  ``get_usage()``.  A value of ``0`` disables aggregate eviction.
- ``eviction`` (dict): Optional eviction policy, see ``L2AdapterConfigBase``.

**Configuration examples:**

.. code-block:: bash

    # Minimal: use an existing bucket with a token from $HF_TOKEN
    --l2-adapter '{"type": "hfbucket", "bucket_handle": "hf://buckets/my-org/lmcache-kv/prod"}'

    # Create the bucket on first store and bound the worker pool
    --l2-adapter '{"type": "hfbucket", "bucket_handle": "hf://buckets/my-org/lmcache-kv/prod", "create_bucket_if_missing": true, "num_workers": 8}'

    # Enable aggregate eviction with a capacity cap
    --l2-adapter '{"type": "hfbucket", "bucket_handle": "hf://buckets/my-org/lmcache-kv/prod", "max_capacity_gb": 50, "eviction": {"eviction_policy": "LRU", "trigger_watermark": 0.9, "eviction_ratio": 0.1}}'
