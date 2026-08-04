.. _recipes:

Recipes
=======

This section lists model architectures validated end-to-end with LMCache, with a
recipe page per architecture covering only the LMCache-specific configuration
that diverges from defaults. Engine-side documentation (how to serve the model
itself) lives with the serving engine; recipe pages link out rather than
duplicate.

For the generic LMCache + engine wiring (ports, remote hosts, sending a first
request), see :doc:`../getting_started/quickstart` -- recipes assume that page
as a prerequisite.

Recipes are grouped by architecture:

.. toctree::
   :maxdepth: 1

   uniform_attention_models
   /mp/hybrid_models
   multimodal_models
