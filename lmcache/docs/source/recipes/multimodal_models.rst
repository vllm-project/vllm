.. _multimodal_models:

Multimodal Models
=================

Recipes for multimodal (vision-language) architectures validated end-to-end
with LMCache, with a recipe page per architecture covering only the
LMCache-specific configuration that diverges from defaults.

Multimodal models run a separate encoder (vision/audio) whose output
embeddings are injected into the decoder at placeholder token positions; the
decoder itself uses the ordinary paged KV cache, so LMCache caches multimodal
requests through the same KV pathway as text-only models.

The one multimodal-specific concern is **cache keying**: vLLM emits identical
placeholder token ids for every image, so raw token ids cannot distinguish
images. LMCache handles this automatically by overwriting each placeholder
span with a value derived from the image's content hash (``mm_hash``) before
key hashing -- same text with different images gets distinct cache entries,
and no configuration is required. This applies to both the in-process
connector and MP mode.

Vision-encoder outputs are a separate, optional cache -- see
:doc:`../non_kv_cache/encoder_cache` (in-process mode only; not yet available
in MP mode).

Recipe page contents
--------------------

Each recipe page is intentionally minimal:

- **Validated models** -- exact HF repo IDs that have been tested.
- **Engine tabs** -- one tab per serving engine (vLLM, SGLang, TRT-LLM). Each
  tab links to the engine's own documentation for the model and shows the
  exact ``lmcache server`` and engine launch commands. Tabs for engines that
  are not yet validated state so explicitly.
- **CacheBlend support** -- validation status (may be empty).
- **Compression support** -- table of compression methods (CacheGen, etc.)
  with per-method validation status. Extensible: new methods get a row.
- **Caveats** -- known limitations, if any.

For the generic LMCache + engine wiring (ports, remote hosts, sending a first
request), see :doc:`../getting_started/quickstart`. Recipes assume that page
as a prerequisite.

Supported architectures
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 28 30 10 10 10 12

   * - Model
     - Example HF model
     - vLLM
     - SGLang
     - TRT-LLM
     - Recipe

   * - Qwen2.5-VL
     - ``Qwen/Qwen2.5-VL-3B-Instruct``
     - ✓
     - —
     - —
     - :doc:`qwen2_5_vl`

Legend: ``✓`` validated, ``—`` not validated. The **Model** column is the model
family; each recipe page lists the exact vLLM architecture class it covers.

Multimodal models whose decoder interleaves attention types keep their recipe
under :doc:`/mp/hybrid_models` -- e.g. Gemma 3 (:doc:`gemma3`).

Contributing a recipe
---------------------

To add a new multimodal architecture:

1. Copy an existing page (e.g. ``qwen2_5_vl.rst``) to
   ``recipes/<architecture_snake_case>.rst``.
2. Fill in **Validated models**, **Engines**, **LMCache configuration**, and
   **Caveats**. Keep each section terse -- if a field has nothing to say, say
   so in one line rather than padding it.
3. Add a row to the table above and an entry to the hidden toctree below.

.. toctree::
   :hidden:
   :maxdepth: 1

   qwen2_5_vl
