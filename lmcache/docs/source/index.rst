.. LMCache documentation master file, created by
   sphinx-quickstart on Mon Sep 30 10:39:18 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. role:: raw-html(raw)
    :format: html

Welcome to LMCache!
=====================

.. figure:: ./assets/lmcache-logo_crop.png
  :width: 60%
  :align: center
  :alt: LMCache
  :class: no-scaled-link

.. rst-class:: hero-tagline

**A KV Cache Management Layer for Scalable LLM Inference.**

.. note::
   We are currently in the process of upgrading our documentation to provide better guidance and examples. Some sections may be under construction. Thank you for your patience!

.. raw:: html

   <p style="text-align:center">
   <script async defer src="https://buttons.github.io/buttons.js"></script>
   <a class="github-button" href="https://github.com/LMCache/LMCache" data-show-count="true" data-size="large" aria-label="Star">Star</a>
   <a class="github-button" href="https://github.com/LMCache/LMCache/subscription" data-icon="octicon-eye" data-size="large" aria-label="Watch">Watch</a>
   <a class="github-button" href="https://github.com/LMCache/LMCache/fork" data-show-count="true" data-icon="octicon-repo-forked" data-size="large" aria-label="Fork">Fork</a>
   </p>

.. container:: hero-description

   LMCache is a **KV cache management layer** for LLM inference. It turns KV cache from a temporary state into reusable *AI-native knowledge* that can be *stored* persistently, *reused* across multiple serving engines, *monitored* with an observability stack, and *transformed* for better generation quality. As a result, LMCache **reduces TTFT** (time-to-first-token) and **improves throughput**, especially for long-context agentic, multi-turn conversation, and knowledge-augmented workloads (e.g., RAG).

   LMCache is **vendor-neutral**. It can be used as a KV cache layer for a range of mainstream open-source serving engines, inference frameworks, hardware vendors, storage systems, and infrastructure providers. The vendor neutrality allows users to freely switch between serving engines and storage vendors, while reusing the stored KV caches.

.. image:: ./assets/deployment_modes_light.png
  :width: 90%
  :align: center
  :alt: LMCache Deployment Modes
  :class: only-light no-scaled-link

.. image:: ./assets/deployment_modes_dark.png
  :width: 90%
  :align: center
  :alt: LMCache Deployment Modes
  :class: only-dark no-scaled-link

:raw-html:`<br />`

Key features
------------

- **Engine-independent deployment**: LMCache, as a standalone daemon process, manages KV cache independently from the inference engine process, so that KV cache will not be lost even if the inference engine crashes (i.e., no fate-sharing with engines).

- **Persistent, tiered KV cache offloading and reuse**: Move KV caches out of GPU memory into a tiered storage hierarchy spanning CPU memory, local storage, and remote backends, enabling reuse across requests, sessions, and engine instances to reduce repeated prefill computation and improve TTFT.

- **Production-level KV cache observability**: LMCache provides a rich set of KV cache observability metrics, including typical Kubernetes metrics (health monitoring, performance diagnostics), KV-cache-specific metrics (request-level and token-level prefix cache hits, lifecycle, request-level KV cache performance), management metrics (user-specific usage), and more.

- **Pluggable storage and transport backends**: Easily integrate remote storage and KV transfer backends through a unified interface, enabling KV cache offloading and sharing across storage providers. Through this interface, LMCache supports storage backends including CPU RAM, local disk (SSD), Redis/Valkey, Mooncake, InfiniStore, S3-compatible object storage, NIXL, and GDS.

- **Non-prefix KV reuse**: Extend KV reuse beyond prefix caching by reusing cached KV blocks at any position in the prompt. This leverages CacheBlend to selectively recompute tokens for quality recovery.

- **PD disaggregation and KV transfer**: Support KV cache transfer from prefill workers to decode workers over NVLink, RDMA, or TCP through transport layers such as NIXL.

- **Pluggable KV transformation**: A simple interface for researchers to write compression, token dropping, and custom serialization through a flexible SERDE interface.

LMCache is becoming an integral layer in the LLM inference *ecosystem*, with *community*-driven integration with serving engines, inference frameworks, hardware vendors, storage systems, and infrastructure providers:

.. figure:: ./assets/ecosystem.png
  :width: 90%
  :align: center
  :alt: LMCache ecosystem
  :class: no-scaled-link

:raw-html:`<br />`

Updates
-------

- [2026/06] 🔥 LMCache multi-platform development support with no GPU required (`blog <https://blog.lmcache.ai/en/2026/06/23/vllm-lmcache-a-starter-guide-no-gpu-required/>`__, `blog <https://blog.lmcache.ai/en/2026/06/15/understanding-lmcache-mp-mode-transfer-paths-a-beginners-guide/>`__).
- [2026/05] 🔥 Agentic workload benchmark on AMD MI300X (`blog <https://blog.lmcache.ai/en/2026/05/12/benchmarking-lmcache-for-multi-turn-agentic-workloads-on-amd-mi300x/>`__).
- [2026/04] 🔥 LMCache's new multiprocess (MP) architecture release (`blog <https://blog.lmcache.ai/en/2026/04/03/lmcaches-new-architecture-boosts-moe-inference-performance-by-10x/>`__).
- [2026/03] LMCache at GTC 2026 (`post <https://www.linkedin.com/posts/lmcache-lab_llm-opensource-nvidiagtc-activity-7442721875664826369-pMAu>`__).
- [2026/01] LMCache multi-node P2P CPU memory sharing, from experimental feature to production (`blog <https://blog.lmcache.ai/en/2026/01/21/p2p-1/>`__).

.. dropdown:: More

   - [2025/11] LMCache x CoreWeave accelerate efficient LLM inference for Cohere (`blog <https://blog.lmcache.ai/en/2025/10/29/breaking-the-memory-barrier-how-lmcache-and-coreweave-power-efficient-llm-inference-for-cohere/>`__).
   - [2025/10] LMCache joins the PyTorch Foundation and Tensormesh unveiled (`blog <https://blog.lmcache.ai/en/2025/10/31/tensormesh-unveiled-and-lmcache-joins-the-pytorch-foundation/>`__, `PyTorch <https://pytorch.org/blog/lmcache-joins-pytorch-ecosystem/>`__).
   - [2025/09] NVIDIA Dynamo integrates LMCache, accelerating LLM inference (`blog <https://blog.lmcache.ai/en/2025/09/18/nvidia-dynamo-integrates-lmcache-accelerating-llm-inference/>`__).
   - [2025/08] 🎉 LMCache hits 5,000+ GitHub stars (`blog <https://blog.lmcache.ai/en/2025/08/28/%f0%9f%8e%89-lmcache-hits-5000-github-stars-thank-you-community/>`__).
   - [2025/08] LMCache supports gpt-oss (20B/120B) on day 1 (`blog <https://blog.lmcache.ai/en/2025/08/05/lmcache-supports-gpt-oss-20b-120b-on-day-1/>`__).
   - [2025/07] Get faster LLM inference and cheaper responses with LMCache and Redis (`Redis blog <https://redis.io/blog/get-faster-llm-inference-and-cheaper-responses-with-lmcache-and-redis/>`__).
   - [2025/07] LMCache extends its turbo-boost to multimodal models in vLLM V1 (`blog <https://blog.lmcache.ai/en/2025/07/03/lmcache-extends-its-turbo-boost-to-multimodal-models-in-vllm-v1/>`__).
   - [2025/06] LLM Production Stack goes cross-hardware: AMD, Arm and Ascend (`blog <https://blog.lmcache.ai/en/2025/06/20/llm-production-stack-goes-cross-hardware-ascend-arm-and-amd-support-incoming/>`__).

:raw-html:`<br />`


For more information, check out the following:

* `LMCache blogs <https://lmcache.github.io>`_
* `Join LMCache slack workspace <https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-3zxjao8h0-lRfBfnLqbALOtLsWn2ITxA>`_
* Our papers:

  * `CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving <https://dl.acm.org/doi/10.1145/3651890.3672274>`_
  * `CacheBlend: Fast Large Language Model Serving with Cached Knowledge Fusion <https://arxiv.org/abs/2405.16444>`_
  * `Do Large Language Models Need a Content Delivery Network? <https://arxiv.org/abs/2409.13761>`_

:raw-html:`<br />`


Documentation
-------------


.. toctree::
   :maxdepth: 2

   getting_started/index

:raw-html:`<br />`

.. toctree::
   :maxdepth: 3

   interacting_with_server

:raw-html:`<br />`

.. toctree::
   :maxdepth: 3

   recipes/index

:raw-html:`<br />`

.. toctree::
   :maxdepth: 4

   mp/l2_storage/index

:raw-html:`<br />`

.. toctree::
   :maxdepth: 2

   distributed_kv_cache

:raw-html:`<br />`

.. toctree::
   :maxdepth: 2

   production/index

:raw-html:`<br />`

.. toctree::
   :maxdepth: 2

   mp/observability/index

:raw-html:`<br />`

.. toctree::
   :maxdepth: 2

   community/index

:raw-html:`<br />`

.. toctree::
   :maxdepth: 2

   kv_cache_optimizations/index

:raw-html:`<br />`

.. toctree::
   :maxdepth: 2

   developer_guide/index

:raw-html:`<br />`

.. toctree::
   :maxdepth: 2

   non_kv_cache/index

:raw-html:`<br />`

.. toctree::
   :maxdepth: 2

   legacy/index

:raw-html:`<br />`