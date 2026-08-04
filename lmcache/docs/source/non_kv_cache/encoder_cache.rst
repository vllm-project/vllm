Encoder caching
===============

.. warning::

   This page documents the behavior of LMCache's in-process mode (deprecated). Please consider using :doc:`LMCache MP mode </mp/index>` for better feature support and performance.


The **Encoder Cache (EC)** stores the output of a multimodal model's
encoder stage, keyed by vLLM's per-input ``mm_hash``. When two
requests share a multimodal input — the same image, video, or audio
clip — the second request loads the encoder output from the cache
and the encoder does not run.

This applies to any modality vLLM exposes through its encoder-cache
extension point: vision encoders (CLIP / ViT-style towers used for
images and sampled video frames), audio encoders (Whisper-style
towers used for raw waveforms), and combined-modality encoders such
as Qwen2.5-Omni. The connector is modality-agnostic — it caches a
tensor of shape ``[num_tokens, hidden_dim]`` keyed by ``mm_hash``,
without knowing which encoder produced it.

vLLM exposes the encoder-cache extension point via
``ECConnectorBase`` (vLLM v1 only). LMCache provides an
``LMCacheECConnector`` shim on the vLLM side and an ``ECCacheEngine``
on the LMCache side; together they back the encoder cache with any of
LMCache's storage backends (local CPU, local disk, remote, NIXL).

Enabling it
-----------

Pass ``--ec-transfer-config`` to ``vllm serve``:

.. code-block:: bash

   vllm serve <model> \
       --ec-transfer-config '{
         "ec_connector": "LMCacheECConnector",
         "ec_role": "ec_both",
         "ec_connector_module_path": "vllm.distributed.ec_transfer.ec_connector.lmcache_connector"
       }'

``ec_role`` choices: ``ec_producer`` (saves only), ``ec_consumer``
(reads only), ``ec_both`` (single-instance default).

Set ``LMCACHE_CONFIG_FILE`` to point at a YAML with at least one
storage backend configured for EC:

.. code-block:: yaml

   chunk_size: 256
   local_cpu: true
   max_local_cpu_size: 2          # GiB
   local_disk: "file:///var/lmcache/ec"
   max_local_disk_size: 16        # GiB

To size EC storage independently from the (separate) KV cache, prefix
overrides with ``ec_`` in YAML or ``LMCACHE_EC_`` in the environment
(e.g. ``ec_max_local_disk_size: 64`` or
``LMCACHE_EC_MAX_LOCAL_DISK_SIZE=64``). EC and KV always run with
**separate** ``StorageManager`` instances so one cannot evict the
other.

If you don't set ``local_disk`` (or its EC override) the engine still
starts, but EC entries live in CPU memory only and do not survive
process restart. Set ``local_disk`` (or ``ec_local_disk``) to a real
path if you want cache persistence — there is no implicit on-disk
default location.

Verifying it's working
----------------------

Three independent signals:

1. **vLLM metric.** ``loggers.py`` reports
   ``MM cache hit rate: X%`` after warm requests.
2. **LMCache log line.** Cold (first-time) requests emit
   ``LMCache INFO: EC put: stored N bytes for mm_hash=H``. Warm
   requests emit no ``EC put``.
3. **On-disk file.** Under ``local_disk`` an entry of the form
   ``<model>@1@0@<chunk_hash>@<dtype>.pt`` appears after the first
   request and is reused thereafter. The ``@1@0@`` prefix reflects
   sentinel ``world_size=1, worker_id=0`` in the EC cache key, so all
   tensor-parallel ranks share one entry.

Design notes (user-visible)
---------------------------

- **Cache key uses sentinel TP shape.** Encoder outputs are
  replicated across TP ranks, so the EC key uses
  ``world_size=1, worker_id=0`` regardless of the deployment's actual
  TP. Concurrent puts from N ranks land on the same key with
  identical contents.
- **Dtype decoupled from KV quant.** The dtype field of the EC cache
  key is the encoder output dtype (``vllm_config.model_config.dtype``),
  not ``metadata.kv_dtype``. Changing KV quantization does not
  invalidate EC entries.
- **Separate StorageManager from KV.** KV and EC have very different
  access patterns (KV chunked / layerwise / high-volume; EC
  single-tensor / request-scoped). Sharing one allocator pool would
  let one cache evict the other in unpredictable ways. Per-cache
  sizing knobs (``ec_max_local_*``) are explicit instead.
- **Connector role pinned to "worker".** vLLM's ``ECConnectorBase``
  is dual-role (scheduler and worker). The LMCache connector calls
  ``create_lmcache_metadata(..., role="worker")`` regardless, because
  the scheduler-side ``has_cache_item`` needs a fully constructed
  ``StorageManager`` and LMCache currently aborts disk-backend setup
  when ``metadata.role == "scheduler"``.

The full internal design (class layering, code paths, follow-up work)
lives at :file:`docs/design/v1/encoder-cache.md` in the source tree.

Benchmark
---------

Live measurement on a single H100 80GB with
``Qwen/Qwen2.5-VL-7B-Instruct`` (bf16) and Big Buck Bunny (10:34,
720p, ≈ 60 MB MP4). Same chat-completion request sent 1 cold + N
warm times against the same vLLM server.

Two configurations, varying only ``num_frames`` (how many frames vLLM
samples from the video):

.. list-table::
   :header-rows: 1
   :widths: 22 14 18 18 14 14

   * - num_frames
     - EC entry
     - Cold TTFT (s)
     - Warm TTFT mean (s)
     - Saved
     - Speedup
   * - 32 (vLLM default)
     - 34.3 MB
     - 3.923
     - 3.125
     - 798 ms
     - **1.26×**
   * - 128
     - 130.8 MB
     - 5.895
     - 3.375
     - 2.52 s
     - **1.75×**

Speedup grows with ``num_frames`` because the encoder workload
scales linearly with frame count while the rest of prefill (LM
forward over the resulting multimodal tokens + the short text prompt)
scales sublinearly. The same principle applies to other modalities:
the win is largest when the encoder is the dominant share of prefill
(long videos at high frame counts, long audio clips, large images at
high resolution) and smallest when text prefill dominates.

Reproducing
~~~~~~~~~~~

Server (heavier-encoder configuration):

.. code-block:: bash

   vllm serve <Qwen2.5-VL-7B-Instruct path> \
       --port 8000 \
       --gpu-memory-utilization 0.85 \
       --max-model-len 32768 \
       --max-num-seqs 8 \
       --limit-mm-per-prompt '{"video": 1}' \
       --media-io-kwargs '{"video": {"num_frames": 128}}' \
       --enforce-eager \
       --ec-transfer-config '{"ec_connector": "LMCacheECConnector",
                              "ec_role": "ec_both",
                              "ec_connector_module_path": "vllm.distributed.ec_transfer.ec_connector.lmcache_connector"}'

Client: any streaming OpenAI-compatible client that re-sends the same
multimodal payload. The benchmark measures TTFT (time to first
token) because the encoder runs during prefill — any encoder savings
show up there. Decode tokens-per-second is unaffected by EC.
