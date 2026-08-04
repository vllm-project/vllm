.. _uniform_attention_models:

Uniform Attention Models
========================

Recipes for standard (uniform) attention transformer architectures validated
end-to-end with LMCache, with a recipe page per architecture covering only the
LMCache-specific configuration that diverges from defaults.

These models use a **single attention type** across all layers, so vLLM serves
them with one KV cache group. Models that interleave multiple attention types
(sliding-window + full, or Mamba / linear-attention + full) are covered under
:doc:`/mp/hybrid_models`.

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

For the generic LMCache + engine wiring (ports, remote hosts,
sending a first request), see :doc:`../getting_started/quickstart`. Recipes assume that
page as a prerequisite.

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

   * - MiniMax M2 series
     - ``MiniMaxAI/MiniMax-M2``
     - ✓
     - —
     - —
     - :doc:`minimax_m2`

   * - Mistral / Devstral
     - ``mistralai/Devstral-2-123B-Instruct-2512``
     - ✓
     - —
     - —
     - :doc:`devstral`

   * - Qwen3 MoE
     - ``Qwen/Qwen3-235B-A22B``
     - ✓
     - —
     - —
     - :doc:`qwen3`

   * - Llama
     - ``meta-llama/Meta-Llama-3.1-70B-Instruct``
     - ✓
     - —
     - —
     - :doc:`llama`

   * - Phi-3 / Phi-4
     - ``microsoft/Phi-4-mini-instruct``
     - ✓
     - —
     - —
     - :doc:`phi3`

   * - Mixtral
     - ``mistralai/Mixtral-8x7B-Instruct-v0.1``
     - ✓
     - —
     - —
     - :doc:`mixtral`

Legend: ``✓`` validated, ``—`` not validated. The **Model** column is the model
family; each recipe page lists the exact vLLM architecture class it covers.

Contributing a recipe
---------------------

To add a new uniform-attention architecture:

1. Copy an existing page (e.g. ``minimax_m2.rst``) to
   ``recipes/<architecture_snake_case>.rst``.
2. Fill in **Validated models**, **Engines**, **LMCache configuration**, and
   **Caveats**. Keep each section terse -- if a field has nothing to say, say
   so in one line rather than padding it.
3. Add a row to the table above and an entry to the hidden toctree below.

(For models that interleave attention types, add the page under
:doc:`/mp/hybrid_models` instead.)

.. toctree::
   :hidden:
   :maxdepth: 1

   minimax_m2
   devstral
   qwen3
   llama
   phi3
   mixtral
